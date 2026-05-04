param(
    [ValidateSet("smoke", "e31", "baseline", "target_s025", "target_s05", "target_s010", "target_rerank", "diag_variants")]
    [string]$Mode = "smoke",
    [int]$Seed = 1,
    [string]$Branch = "",
    [string]$Session = "",
    [string]$LocalPython = "",
    [int]$PollSeconds = 120,
    [switch]$SkipGitSync,
    [switch]$NoWait,
    [switch]$FetchResults
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

function Invoke-LocalStage3Check {
    param([string]$Python)
    & $Python -m py_compile main.py model.py scripts/analyze_stage3_target_construction.py
    if ($LASTEXITCODE -ne 0) {
        throw "Local py_compile failed."
    }
    $helpText = & $Python main.py --help
    if ($LASTEXITCODE -ne 0) {
        throw "main.py --help failed."
    }
    foreach ($flag in @("--diag_target_construction", "--diag_target_sources", "--diag_target_output_dir", "--target_align_mode")) {
        if (-not ($helpText -match [regex]::Escape($flag))) {
            throw "Missing CLI flag in --help: $flag"
        }
    }
}

function Invoke-Stage3SelectiveGitSync {
    param(
        [string]$Branch,
        [string]$Message
    )
    $files = @(
        "main.py",
        "experiments/STAGE3_TARGET_CONSTRUCTION_DIAGNOSTIC_CN.md",
        "experiments/STAGE3_5_CONSERVATIVE_TARGET_ALIGN_CN.md",
        "scripts/analyze_stage3_target_construction.py",
        "tools/remote-workflow/lib.ps1",
        "tools/remote-workflow/run_stage3_target_construction.ps1"
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

function Get-Stage3SshTarget {
    param([hashtable]$Config)
    $options = @(
        "-o", "ClearAllForwardings=yes",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ExitOnForwardFailure=no",
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

function Invoke-Stage3RemoteScript {
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
        "-o", "ClearAllForwardings=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ExitOnForwardFailure=no",
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

function Get-RemoteGpuSelection {
    param(
        [hashtable]$Config,
        [string]$Target
    )
    $query = "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
    $output = & ssh -o ClearAllForwardings=yes -o StrictHostKeyChecking=accept-new -o ExitOnForwardFailure=no -p $Config.Port $Target $query
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
    return "$($candidate.Index),$($candidate.Name)"
}

function Get-Stage3BatchSize {
    param([string]$GpuName)
    if ($GpuName -match "3080 Ti|3080Ti") {
        return 256
    }
    return 512
}

function Get-Stage3RunName {
    param(
        [string]$Mode,
        [int]$Seed
    )
    if ($Mode -eq "e31") {
        return "e31_seed$Seed"
    }
    if ($Mode -eq "baseline") {
        return "algo_baseline_seed$Seed"
    }
    if ($Mode -eq "target_s025") {
        return "algo_target_s025_seed$Seed"
    }
    if ($Mode -eq "target_s05") {
        return "algo_target_s05_seed$Seed"
    }
    if ($Mode -eq "target_s010") {
        return "stage35_target_s010_seed$Seed"
    }
    if ($Mode -eq "target_rerank") {
        return "stage35_target_rerank_seed$Seed"
    }
    if ($Mode -eq "diag_variants") {
        return "stage35_diag_variants_seed$Seed"
    }
    return "smoke_seed$Seed"
}

function Get-Stage3Command {
    param(
        [string]$Mode,
        [int]$Seed,
        [int]$Gpu,
        [int]$BatchSize,
        [string]$RunName
    )
    $epochs = 4
    $iters = 20
    $diagEvery = 1
    $diagBatches = 1
    $diagCandidates = 64
    $diagArgs = @(
        "--diag_target_construction --diag_target_every_epoch $diagEvery",
        "--diag_target_batches $diagBatches --diag_target_val_batches 1 --diag_target_candidates $diagCandidates",
        "--diag_target_sources clean_val,noisy_val,peer_consensus,ema_teacher,purified_buffer,purified_buffer_balanced,purified_buffer_moderate,purified_buffer_coverage,ema_purified"
    ) -join " "
    $utilityArgs = "--utility_mode none"

    if ($Mode -eq "e31") {
        $epochs = 31
        $iters = 100
        $diagEvery = 5
        $diagBatches = 2
        $diagCandidates = 128
        $diagArgs = @(
            "--diag_target_construction --diag_target_every_epoch $diagEvery",
            "--diag_target_batches $diagBatches --diag_target_val_batches 1 --diag_target_candidates $diagCandidates",
            "--diag_target_sources clean_val,noisy_val,peer_consensus,ema_teacher,purified_buffer"
        ) -join " "
    }
    elseif ($Mode -eq "diag_variants") {
        $epochs = 31
        $iters = 100
        $diagEvery = 5
        $diagBatches = 2
        $diagCandidates = 128
        $diagArgs = @(
            "--diag_target_construction --diag_target_every_epoch $diagEvery",
            "--diag_target_batches $diagBatches --diag_target_val_batches 1 --diag_target_candidates $diagCandidates",
            "--diag_target_sources clean_val,noisy_val,peer_consensus,ema_teacher,purified_buffer,purified_buffer_balanced,purified_buffer_moderate,purified_buffer_coverage,ema_purified"
        ) -join " "
    }
    elseif ($Mode -in @("baseline", "target_s025", "target_s05", "target_s010", "target_rerank")) {
        $epochs = 31
        $iters = 100
        $diagArgs = ""
        if ($Mode -eq "target_s025") {
            $utilityArgs = "--utility_mode target_align --target_align_mode weighted --utility_strength 0.25 --target_align_source purified_buffer --target_align_min_source 16 --target_align_max_source 128"
        }
        elseif ($Mode -eq "target_s05") {
            $utilityArgs = "--utility_mode target_align --target_align_mode weighted --utility_strength 0.5 --target_align_source purified_buffer --target_align_min_source 16 --target_align_max_source 128"
        }
        elseif ($Mode -eq "target_s010") {
            $utilityArgs = "--utility_mode target_align --target_align_mode weighted --utility_strength 0.10 --target_align_source purified_buffer --target_align_min_source 16 --target_align_max_source 128"
        }
        elseif ($Mode -eq "target_rerank") {
            $utilityArgs = "--utility_mode target_align --target_align_mode rerank_only --target_align_rerank_frac 0.75 --utility_strength 1.0 --target_align_source purified_buffer --target_align_min_source 16 --target_align_max_source 128"
        }
    }

    $parts = @(
        "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$Gpu python -u main.py",
        "--dataset cifar10 --noise_type symmetric --noise_rate 0.4",
        "--num_models 2 --q_mode loss --mstep_mode hard",
        "--sam_rho 0.05 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001",
        "--replay_size 2000 --replay_ratio 0",
        "--lambda_mode accuracy --lambda_patience 9999 --min_active 2",
        "--batch_size $BatchSize --num_workers 8 --prefetch_factor 4 --drop_last",
        "--n_epoch $epochs --num_iter_per_epoch $iters --num_gradual 10 --epoch_decay_start 80",
        "--val_split 0.1 --seed $Seed",
        $utilityArgs,
        $diagArgs,
        "--result_dir results_stage3/target_construction_$RunName",
        "--diag_target_output_dir results_diag/stage3_target_construction/$RunName"
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    return ($parts -join " ")
}

$repoRoot = Get-RepoRoot
Set-Location $repoRoot
$config = Get-WorkflowConfig
if (-not $Branch) {
    $Branch = Get-CurrentBranch
}
if (-not $Session) {
    $Session = "stage3-target-$Mode-seed$Seed"
}
$python = Get-LocalPython -Preferred $LocalPython
Invoke-LocalStage3Check -Python $python
if (-not $SkipGitSync) {
    Invoke-Stage3SelectiveGitSync -Branch $Branch -Message "exp: stage3 target construction diagnostic"
}

$target = Get-Stage3SshTarget -Config $config
$gpuInfo = Get-RemoteGpuSelection -Config $config -Target $target
$gpuParts = $gpuInfo.Split(",", 2)
$gpuIndex = [int]$gpuParts[0]
$gpuName = $gpuParts[1]
$batchSize = Get-Stage3BatchSize -GpuName $gpuName
$runName = Get-Stage3RunName -Mode $Mode -Seed $Seed
$diagDir = "results_diag/stage3_target_construction/$runName"
$runCmd = Get-Stage3Command -Mode $Mode -Seed $Seed -Gpu $gpuIndex -BatchSize $batchSize -RunName $runName
$logFile = "logs/stage3/$Session.log"

$remoteScript = @'
set -euo pipefail
BRANCH="$(decode_arg "$1")"
REPO_DIR="$(decode_arg "$2")"
GIT_REMOTE="$(decode_arg "$3")"
CONDA_INIT="$(decode_arg "$4")"
CONDA_ENV="$(decode_arg "$5")"
SESSION="$(decode_arg "$6")"
LOG_FILE="$(decode_arg "$7")"
RUN_CMD="$(decode_arg "$8")"
DIAG_DIR="$(decode_arg "$9")"

cd "$REPO_DIR"
git fetch "$GIT_REMOTE"
git checkout "$BRANCH"
git pull --ff-only "$GIT_REMOTE" "$BRANCH"

mkdir -p "$(dirname "$LOG_FILE")"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists." >&2
    exit 20
fi

RUNNER="$REPO_DIR/.workflow_${SESSION}.sh"
{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd \"$REPO_DIR\""
    echo "$CONDA_INIT"
    echo "conda activate \"$CONDA_ENV\""
    echo "echo \"== command ==\""
    printf 'echo %q\n' "$RUN_CMD"
    echo "$RUN_CMD"
    echo "python scripts/analyze_stage3_target_construction.py \"$DIAG_DIR\" --output \"$DIAG_DIR/stage3_${SESSION}_summary.json\""
} > "$RUNNER"
chmod +x "$RUNNER"
tmux new-session -d -s "$SESSION" "bash '$RUNNER' > '$LOG_FILE' 2>&1"
echo "Started tmux session: $SESSION"
echo "Log file: $LOG_FILE"
'@

Invoke-Stage3RemoteScript -Config $config -Target $target -Script $remoteScript -Arguments @(
    $Branch,
    $config.RepoDir,
    $config.GitRemote,
    $config.CondaInit,
    $config.DefaultCondaEnv,
    $Session,
    $logFile,
    $runCmd,
    $diagDir
)

Write-Host "Started Stage-3 $Mode seed $Seed on GPU $gpuIndex ($gpuName), batch_size=$batchSize."
Write-Host "Session: $Session"
Write-Host "Log: $logFile"
Write-Host "Command: $runCmd"

if ($NoWait) {
    return
}

while ($true) {
    Start-Sleep -Seconds $PollSeconds
    $statusScript = @'
set -euo pipefail
SESSION="$(decode_arg "$1")"
LOG_FILE="$(decode_arg "$2")"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "RUNNING"
    tail -n 20 "$LOG_FILE" || true
else
    echo "DONE"
    tail -n 120 "$LOG_FILE" || true
fi
'@
    $status = Invoke-Stage3RemoteScript -Config $config -Target $target -Script $statusScript -Arguments @($Session, $logFile) -CaptureOutput
    $status | ForEach-Object { Write-Host $_ }
    if ($status -contains "DONE") {
        break
    }
}

if ($FetchResults) {
    $localDir = Join-Path $repoRoot "remote_results\stage3_target_construction"
    New-Item -ItemType Directory -Force -Path $localDir | Out-Null
    & scp -o ClearAllForwardings=yes -o StrictHostKeyChecking=accept-new -P $config.Port -r "$target`:$($config.RepoDir)/results_diag/stage3_target_construction" $localDir
    if ($LASTEXITCODE -ne 0) {
        throw "scp results fetch failed."
    }
    Write-Host "Fetched Stage-3 diagnostics to $localDir"
}
