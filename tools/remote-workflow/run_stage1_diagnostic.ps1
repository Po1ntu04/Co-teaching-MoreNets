param(
    [string]$Branch = "",
    [string]$CommitMessage = "exp: stage1 target utility diagnostic",
    [string]$LocalPython = "",
    [int]$PollSeconds = 120,
    [switch]$SkipGitSync,
    [switch]$SmokeOnly,
    [switch]$NoWait
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\lib.ps1"

function Get-LocalPython {
    param([string]$Preferred)
    if ($Preferred) {
        return $Preferred
    }
    $repoDefault = "D:\anaconda3\envs\py3.6\python.exe"
    if (Test-Path $repoDefault) {
        return $repoDefault
    }
    return "python"
}

function Invoke-LocalCheck {
    param([string]$Python)
    & $Python -m py_compile main.py model.py
    if ($LASTEXITCODE -ne 0) {
        throw "Local py_compile failed."
    }
    $helpText = & $Python main.py --help
    if ($LASTEXITCODE -ne 0) {
        throw "main.py --help failed."
    }
    foreach ($flag in @("--diag_alignment", "--diag_every_epoch", "--diag_output_dir")) {
        if (-not ($helpText -match [regex]::Escape($flag))) {
            throw "Missing CLI flag in --help: $flag"
        }
    }
}

function Resolve-Stage1SshTarget {
    param([hashtable]$Config)
    $options = @(
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=accept-new",
        "-p", [string]$Config.Port
    )
    if ($Config.ContainsKey("SshHostAlias") -and -not [string]::IsNullOrWhiteSpace([string]$Config.SshHostAlias)) {
        $oldErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & ssh @options ([string]$Config.SshHostAlias) "true" *> $null
        $ErrorActionPreference = $oldErrorActionPreference
        if ($LASTEXITCODE -eq 0) {
            return [string]$Config.SshHostAlias
        }
    }
    $directTarget = "$($Config.User)@$($Config.Host)"
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & ssh @options $directTarget "true" *> $null
    $ErrorActionPreference = $oldErrorActionPreference
    if ($LASTEXITCODE -ne 0) {
        throw "SSH precheck failed for alias and direct target."
    }
    return $directTarget
}

function Invoke-Stage1RemoteScript {
    param(
        [hashtable]$Config,
        [string]$Target,
        [string]$Script,
        [string[]]$Arguments = @(),
        [switch]$CaptureOutput
    )
    $encodedArgs = @()
    foreach ($argument in $Arguments) {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes([string]$argument)
        $encodedArgs += [Convert]::ToBase64String($bytes)
    }
    $wrapper = @'
decode_arg() {
    printf '%s' "$1" | base64 --decode
}
'@ + "`n" + $Script
    $wrapper = $wrapper -replace "`r", ""
    $sshArgs = @(
        "-o", "StrictHostKeyChecking=accept-new",
        "-p", [string]$Config.Port,
        $Target,
        "bash", "-s", "--"
    ) + $encodedArgs
    if ($CaptureOutput) {
        $output = $wrapper | & ssh @sshArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Remote command failed with exit code $LASTEXITCODE."
        }
        return $output
    }
    $wrapper | & ssh @sshArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Remote command failed with exit code $LASTEXITCODE."
    }
}

function Invoke-SelectiveGitSync {
    param(
        [string]$Branch,
        [string]$Message
    )
    $files = @(
        "main.py",
        "model.py",
        "experiments/STAGE1_TARGET_UTILITY_DIAGNOSTIC_CN.md",
        "scripts/analyze_stage1_diagnostics.py",
        "tools/remote-workflow/run_stage1_diagnostic.ps1"
    )
    foreach ($file in $files) {
        if (-not (Test-Path $file)) {
            throw "Required sync file not found: $file"
        }
    }
    & git add -- $files
    if ($LASTEXITCODE -ne 0) {
        throw "git add whitelist failed."
    }
    $staged = & git diff --cached --name-only
    if ($staged) {
        & git commit -m $Message
        if ($LASTEXITCODE -ne 0) {
            throw "git commit failed."
        }
    }
    & git push origin $Branch
    if ($LASTEXITCODE -ne 0) {
        throw "git push failed."
    }
}

function Get-RemoteGpuSelection {
    param(
        [hashtable]$Config,
        [string]$Target
    )
    $query = "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
    $output = & ssh -o StrictHostKeyChecking=accept-new -p $Config.Port $Target $query
    if ($LASTEXITCODE -ne 0) {
        throw "Remote nvidia-smi query failed."
    }
    $gpus = @()
    foreach ($line in $output) {
        $parts = ([string]$line).Split(",", 5)
        if ($parts.Length -lt 5) {
            continue
        }
        $name = $parts[1].Trim()
        $priority = 3
        if ($name -match "4090") {
            $priority = 0
        }
        elseif ($name -match "3090") {
            $priority = 1
        }
        elseif ($name -match "3080 Ti|3080Ti") {
            $priority = 2
        }
        $used = [int]$parts[2].Trim()
        $util = [int]$parts[4].Trim()
        $free = ($used -lt 1500 -and $util -lt 20)
        $gpus += [pscustomobject]@{
            Priority = $priority
            Index = [int]$parts[0].Trim()
            Name = $name
            Used = $used
            Total = [int]$parts[3].Trim()
            Util = $util
            Free = $free
        }
    }
    $candidate = $gpus | Where-Object { $_.Free } | Sort-Object Priority, Index | Select-Object -First 1
    if (-not $candidate) {
        $summary = ($gpus | Sort-Object Priority, Index | ForEach-Object {
            "GPU $($_.Index) $($_.Name) used=$($_.Used)MiB total=$($_.Total)MiB util=$($_.Util)% free=$($_.Free)"
        }) -join "`n"
        throw "No free GPU under threshold.`n$summary"
    }
    $batchE1 = 384
    $batchE2 = 320
    if ($candidate.Priority -eq 2) {
        $batchE1 = 192
        $batchE2 = 160
    }
    return "$($candidate.Index),$($candidate.Name),$batchE1,$batchE2"
}

function Invoke-RemotePrecheck {
    param(
        [hashtable]$Config,
        [string]$Target
    )
    $script = @'
set -euo pipefail
REPO_DIR="$(decode_arg "$1")"
CONDA_INIT="$(decode_arg "$2")"
CONDA_ENV="$(decode_arg "$3")"

test -d "$REPO_DIR"
cd "$REPO_DIR"
$CONDA_INIT
conda activate "$CONDA_ENV"
python --version
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
test -d data/cifar-10-batches-py || test -f data/cifar-10-python.tar.gz || echo "WARN_DATA_MISSING"
'@
    Invoke-Stage1RemoteScript -Config $Config -Target $Target -Script $script -Arguments @(
        $Config.RepoDir,
        $Config.CondaInit,
        $Config.DefaultCondaEnv
    )
}

function Update-RemoteBranch {
    param(
        [hashtable]$Config,
        [string]$Target,
        [string]$Branch
    )
    $script = @'
set -euo pipefail
BRANCH="$(decode_arg "$1")"
REPO_DIR="$(decode_arg "$2")"
GIT_REMOTE="$(decode_arg "$3")"
cd "$REPO_DIR"
git fetch "$GIT_REMOTE"
git checkout "$BRANCH"
git pull --ff-only "$GIT_REMOTE" "$BRANCH"
git status --short --branch
git log -1 --format=%H
'@
    Invoke-Stage1RemoteScript -Config $Config -Target $Target -Script $script -Arguments @(
        $Branch,
        $Config.RepoDir,
        $Config.GitRemote
    )
}

function Start-RemoteRun {
    param(
        [hashtable]$Config,
        [string]$Target,
        [string]$Session,
        [string]$LogFile,
        [string]$RunCommand
    )
    $script = @'
set -euo pipefail
REPO_DIR="$(decode_arg "$1")"
CONDA_INIT="$(decode_arg "$2")"
CONDA_ENV="$(decode_arg "$3")"
SESSION="$(decode_arg "$4")"
LOG_FILE="$(decode_arg "$5")"
RUN_CMD="$(decode_arg "$6")"

cd "$REPO_DIR"
mkdir -p "$(dirname "$LOG_FILE")"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION" >&2
    exit 20
fi

RUNNER="$REPO_DIR/.workflow_${SESSION}.sh"
cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_DIR"
$CONDA_INIT
conda activate "$CONDA_ENV"
$RUN_CMD
EOF
chmod +x "$RUNNER"
tmux new-session -d -s "$SESSION" "bash '$RUNNER' > '$LOG_FILE' 2>&1"
echo "STARTED $SESSION $LOG_FILE"
'@
    Invoke-Stage1RemoteScript -Config $Config -Target $Target -Script $script -Arguments @(
        $Config.RepoDir,
        $Config.CondaInit,
        $Config.DefaultCondaEnv,
        $Session,
        $LogFile,
        $RunCommand
    )
}

function Wait-RemoteRun {
    param(
        [hashtable]$Config,
        [string]$Target,
        [string]$Session,
        [string]$LogFile,
        [int]$PollSeconds
    )
    while ($true) {
        $remote = "cd '$($Config.RepoDir)' && if tmux has-session -t '$Session' 2>/dev/null; then echo RUNNING; tail -n 20 '$LogFile' 2>/dev/null || true; else echo DONE; tail -n 80 '$LogFile' 2>/dev/null || true; fi"
        $output = & ssh -o StrictHostKeyChecking=accept-new -p $Config.Port $Target $remote
        if ($LASTEXITCODE -ne 0) {
            throw "Remote wait command failed."
        }
        $output | ForEach-Object { Write-Host $_ }
        if (($output | Select-Object -First 1) -eq "DONE") {
            break
        }
        Start-Sleep -Seconds $PollSeconds
    }
}

function Copy-RemoteResults {
    param(
        [hashtable]$Config,
        [string]$Target,
        [string]$RemotePath,
        [string]$LocalPath
    )
    New-Item -ItemType Directory -Force -Path $LocalPath | Out-Null
    & scp -o StrictHostKeyChecking=accept-new -P $Config.Port -r "${Target}:$RemotePath" $LocalPath
    if ($LASTEXITCODE -ne 0) {
        throw "scp failed: $RemotePath"
    }
}

function Invoke-RunAudit {
    param(
        [string]$Python,
        [string]$Root,
        [string]$Output
    )
    & $Python scripts/analyze_stage1_diagnostics.py --root $Root --output $Output
    if ($LASTEXITCODE -ne 0) {
        throw "Stage-1 diagnostic analysis failed."
    }
}

$repoRoot = Get-RepoRoot
Set-Location $repoRoot
$config = Get-WorkflowConfig
if (-not $Branch) {
    $Branch = Get-CurrentBranch
}
$python = Get-LocalPython -Preferred $LocalPython

Write-Host "== local checks =="
Invoke-LocalCheck -Python $python

if (-not $SkipGitSync) {
    Write-Host "== selective git sync =="
    Invoke-SelectiveGitSync -Branch $Branch -Message $CommitMessage
}

Write-Host "== ssh target =="
$target = Resolve-Stage1SshTarget -Config $config
Write-Host "SSH target: $target"

Write-Host "== remote precheck =="
Invoke-RemotePrecheck -Config $config -Target $target
Update-RemoteBranch -Config $config -Target $target -Branch $Branch

Write-Host "== gpu selection =="
$gpuLine = Get-RemoteGpuSelection -Config $config -Target $target
$gpuParts = $gpuLine.Split(",", 4)
if ($gpuParts.Length -lt 4) {
    throw "Unexpected GPU selection output: $gpuLine"
}
$dev = $gpuParts[0].Trim()
$gpuName = $gpuParts[1].Trim()
$batchE1 = [int]$gpuParts[2].Trim()
$batchE2 = [int]$gpuParts[3].Trim()
Write-Host "Selected GPU: $dev $gpuName; E1 batch=$batchE1 E2 batch=$batchE2"

$localResultRoot = Join-Path $repoRoot "remote_results\stage1_target_utility"
New-Item -ItemType Directory -Force -Path $localResultRoot | Out-Null

$runs = @(
    @{
        Name = "stage1_smoke"
        Session = "stage1_smoke"
        Log = "logs/stage1/stage1_smoke.log"
        Result = "results_diag/stage1_smoke"
        Command = "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$dev python -u main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.4 --num_models 3 --q_mode loss --mstep_mode hard --sam_rho 0 --replay_size 0 --replay_ratio 0 --lambda_mode accuracy --lambda_patience 9999 --batch_size $batchE1 --num_workers 8 --prefetch_factor 4 --drop_last --n_epoch 2 --num_iter_per_epoch 20 --diag_alignment --diag_every_epoch 1 --diag_batches 2 --diag_val_batches 1 --diag_target both --result_dir results_diag/stage1_smoke --diag_output_dir results_diag/stage1_smoke/diag"
    }
)

if (-not $SmokeOnly) {
    foreach ($seed in @(1, 2, 3)) {
        $runs += @{
            Name = "stage1_e1_seed$seed"
            Session = "stage1_e1_s$seed"
            Log = "logs/stage1/stage1_e1_seed$seed.log"
            Result = "results_diag/stage1_e1_seed$seed"
            Command = "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$dev python -u main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.4 --num_models 3 --q_mode loss --mstep_mode hard --sam_rho 0 --replay_size 0 --replay_ratio 0 --lambda_mode accuracy --lambda_patience 9999 --batch_size $batchE1 --num_workers 8 --prefetch_factor 4 --drop_last --n_epoch 80 --seed $seed --diag_alignment --diag_every_epoch 5 --diag_batches 4 --diag_val_batches 2 --diag_target both --result_dir results_diag/stage1_e1_seed$seed --diag_output_dir results_diag/stage1_e1_seed$seed/diag"
        }
    }
    foreach ($seed in @(1, 2, 3)) {
        $runs += @{
            Name = "stage1_e2_sam005_seed$seed"
            Session = "stage1_e2_s$seed"
            Log = "logs/stage1/stage1_e2_sam005_seed$seed.log"
            Result = "results_diag/stage1_e2_sam_seed$seed"
            Command = "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$dev python -u main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.4 --num_models 3 --q_mode loss --mstep_mode hard --sam_rho 0.05 --replay_size 0 --replay_ratio 0 --lambda_mode accuracy --lambda_patience 9999 --batch_size $batchE2 --num_workers 8 --prefetch_factor 4 --drop_last --n_epoch 80 --seed $seed --diag_alignment --diag_every_epoch 5 --diag_batches 4 --diag_val_batches 2 --diag_target both --result_dir results_diag/stage1_e2_sam_seed$seed --diag_output_dir results_diag/stage1_e2_sam_seed$seed/diag"
        }
    }
}

foreach ($run in $runs) {
    Write-Host "== start $($run.Name) =="
    Start-RemoteRun -Config $config -Target $target -Session $run.Session -LogFile $run.Log -RunCommand $run.Command
    if (-not $NoWait) {
        Wait-RemoteRun -Config $config -Target $target -Session $run.Session -LogFile $run.Log -PollSeconds $PollSeconds
        $localRunRoot = Join-Path $localResultRoot $run.Name
        Copy-RemoteResults -Config $config -Target $target -RemotePath "$($config.RepoDir)/$($run.Result)" -LocalPath $localRunRoot
        $auditOut = Join-Path $localRunRoot "stage1_audit.json"
        Invoke-RunAudit -Python $python -Root $localRunRoot -Output $auditOut
    }
}

if (-not $NoWait) {
    $summaryOut = Join-Path $localResultRoot "stage1_audit_all.json"
    Invoke-RunAudit -Python $python -Root $localResultRoot -Output $summaryOut
    Write-Host "Stage-1 audit written to $summaryOut"
}
