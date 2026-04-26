param(
    [ValidateSet("coteaching", "srit_like")]
    [string]$Mode = "coteaching",
    [int]$Seed = 1,
    [string]$Branch = "",
    [string]$Session = "",
    [string]$LocalPython = "",
    [int]$PollSeconds = 120,
    [switch]$SkipGitSync,
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

function Invoke-LocalBenchmarkCheck {
    param([string]$Python)
    & $Python -m py_compile main.py model.py scripts/analyze_benchmark_accuracy.py
    if ($LASTEXITCODE -ne 0) {
        throw "Local py_compile failed."
    }
    $helpText = & $Python main.py --help
    if ($LASTEXITCODE -ne 0) {
        throw "main.py --help failed."
    }
    foreach ($flag in @("--optimizer", "--momentum", "--weight_decay")) {
        if (-not ($helpText -match [regex]::Escape($flag))) {
            throw "Missing CLI flag in --help: $flag"
        }
    }
}

function Invoke-BenchmarkSelectiveGitSync {
    param(
        [string]$Branch,
        [string]$Message
    )
    $files = @(
        "main.py",
        "experiments/SRIT_REPRODUCTION_BENCHMARK_CN.md",
        "scripts/analyze_benchmark_accuracy.py",
        "tools/remote-workflow/run_srit_benchmark.ps1"
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

function Get-BenchmarkSshTarget {
    param([hashtable]$Config)
    $options = @(
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

function Invoke-BenchmarkRemoteScript {
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
    $output = & ssh -o StrictHostKeyChecking=accept-new -o ExitOnForwardFailure=no -p $Config.Port $Target $query
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

function Get-BenchmarkCommand {
    param(
        [string]$Mode,
        [int]$Seed,
        [int]$Gpu
    )
    $common = @(
        "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$Gpu python -u main.py",
        "--dataset cifar10 --noise_type symmetric --noise_rate 0.4",
        "--num_models 2 --q_mode loss --mstep_mode hard",
        "--replay_size 0 --replay_ratio 0",
        "--lambda_mode accuracy --lambda_patience 9999 --min_active 2",
        "--batch_size 128 --num_workers 8 --prefetch_factor 4 --drop_last",
        "--n_epoch 200 --num_gradual 10 --epoch_decay_start 80",
        "--val_split 0 --seed $Seed"
    ) -join " "

    if ($Mode -eq "coteaching") {
        return "$common --sam_rho 0 --optimizer adam --lr 0.001 --weight_decay 0 --result_dir results_benchmark/srit_repro/coteaching_seed$Seed"
    }
    return "$common --sam_rho 0.05 --optimizer sgd --lr 0.1 --momentum 0.9 --weight_decay 0.0001 --result_dir results_benchmark/srit_repro/srit_like_seed$Seed"
}

$repoRoot = Get-RepoRoot
Set-Location $repoRoot
$config = Get-WorkflowConfig
if (-not $Branch) {
    $Branch = Get-CurrentBranch
}
if (-not $Session) {
    $Session = "srit-benchmark-$Mode-seed$Seed"
}
$python = Get-LocalPython -Preferred $LocalPython
Invoke-LocalBenchmarkCheck -Python $python
if (-not $SkipGitSync) {
    Invoke-BenchmarkSelectiveGitSync -Branch $Branch -Message "exp: srit benchmark runner"
}

$target = Get-BenchmarkSshTarget -Config $config
$gpuInfo = Get-RemoteGpuSelection -Config $config -Target $target
$gpuParts = $gpuInfo.Split(",", 2)
$gpuIndex = [int]$gpuParts[0]
$gpuName = $gpuParts[1]
$runCmd = Get-BenchmarkCommand -Mode $Mode -Seed $Seed -Gpu $gpuIndex
$logFile = "logs/benchmark/$Session.log"

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
cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_DIR"
$CONDA_INIT
conda activate "$CONDA_ENV"
echo "== command =="
echo "$RUN_CMD"
$RUN_CMD
EOF
chmod +x "$RUNNER"
tmux new-session -d -s "$SESSION" "bash '$RUNNER' > '$LOG_FILE' 2>&1"
echo "Started tmux session: $SESSION"
echo "GPU: $RUN_CMD"
echo "Log file: $LOG_FILE"
'@

Invoke-BenchmarkRemoteScript -Config $config -Target $target -Script $remoteScript -Arguments @(
    $Branch,
    $config.RepoDir,
    $config.GitRemote,
    $config.CondaInit,
    $config.DefaultCondaEnv,
    $Session,
    $logFile,
    $runCmd
)

Write-Host "Started $Mode seed $Seed on GPU $gpuIndex ($gpuName)."
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
    tail -n 80 "$LOG_FILE" || true
fi
'@
    $status = Invoke-BenchmarkRemoteScript -Config $config -Target $target -Script $statusScript -Arguments @($Session, $logFile) -CaptureOutput
    $status | ForEach-Object { Write-Host $_ }
    if ($status -contains "DONE") {
        break
    }
}
