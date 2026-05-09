param(
    [ValidateSet("smoke", "diagnostic", "prior", "selection", "weight")]
    [string]$Mode = "smoke",
    [ValidateSet(2, 3, 5)]
    [int]$NumModels = 3,
    [int]$Seed = 1,
    [string]$Branch = "",
    [string]$Session = "",
    [int]$Gpu = -1,
    [switch]$SkipGitSync,
    [switch]$SkipRemoteGitSync,
    [switch]$NoWait,
    [switch]$FetchResults
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\lib.ps1"

function Invoke-QIsolationSelectiveGitSync {
    param(
        [string]$Branch,
        [string]$Message
    )
    $files = @(
        "main.py",
        "experiments/RELIABILITY_UTILITY_RESEARCH_PROGRAM_CN.md",
        "experiments/Q_COLLAPSE_ISOLATION_CN.md",
        "scripts/analyze_q_collapse.py",
        "tools/remote-workflow/run_q_collapse_isolation.ps1"
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

function Get-QIsolationSshTarget {
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
        & ssh @options ([string]$Config.SshHostAlias) "true" *> $null
        if ($LASTEXITCODE -eq 0) {
            return [string]$Config.SshHostAlias
        }
    }
    $directTarget = "$($Config.User)@$($Config.Host)"
    & ssh @options $directTarget "true" *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "SSH precheck failed for alias and direct target."
    }
    return $directTarget
}

function Invoke-QIsolationRemoteScript {
    param(
        [hashtable]$Config,
        [string]$Target,
        [string]$Script,
        [string[]]$Arguments = @()
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
    $wrapper | & ssh @sshArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Remote command failed with exit code $LASTEXITCODE."
    }
}

function Get-FreeGpu {
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
        $gpus += [pscustomobject]@{
            Priority = $priority
            Index = [int]$parts[0].Trim()
            Name = $name
            Used = $used
            Total = [int]$parts[3].Trim()
            Util = $util
            Free = ($used -lt 1500 -and $util -lt 20)
        }
    }
    $candidate = $gpus | Where-Object { $_.Free } | Sort-Object Priority, Index | Select-Object -First 1
    if (-not $candidate) {
        $summary = ($gpus | Sort-Object Priority, Index | ForEach-Object {
            "GPU $($_.Index) $($_.Name) used=$($_.Used)MiB total=$($_.Total)MiB util=$($_.Util)% free=$($_.Free)"
        }) -join "`n"
        throw "No free GPU under threshold.`n$summary"
    }
    return [int]$candidate.Index
}

function Get-QUsageMode {
    param([string]$Mode)
    if ($Mode -eq "diagnostic" -or $Mode -eq "smoke") { return "diagnostic_only" }
    if ($Mode -eq "prior") { return "prior_only" }
    if ($Mode -eq "selection") { return "selection_only" }
    if ($Mode -eq "weight") { return "weight_only" }
    throw "Unsupported mode: $Mode"
}

function Get-QMstepMode {
    param([string]$Mode)
    if ($Mode -eq "weight") { return "soft" }
    return "hard"
}

function Get-QCommand {
    param(
        [string]$Mode,
        [int]$NumModels,
        [int]$Seed,
        [int]$Gpu,
        [string]$RunName
    )
    $epochs = 30
    $iters = 100
    if ($Mode -eq "smoke") {
        $epochs = 3
        $iters = 20
    }
    $qUsage = Get-QUsageMode -Mode $Mode
    $mstep = Get-QMstepMode -Mode $Mode
    $parts = @(
        "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$Gpu python -u main.py",
        "--dataset cifar10 --noise_type symmetric --noise_rate 0.4",
        "--num_models $NumModels --q_mode hybrid --q_usage_mode $qUsage --mstep_mode $mstep",
        "--sam_rho 0 --replay_size 0 --replay_ratio 0",
        "--lambda_mode accuracy --lambda_patience 9999 --min_active 2",
        "--batch_size 512 --num_workers 8 --prefetch_factor 4 --drop_last",
        "--n_epoch $epochs --num_iter_per_epoch $iters --num_gradual 10 --epoch_decay_start 80",
        "--val_split 0.1 --seed $Seed",
        "--result_dir results_diag/q_isolation/$RunName"
    )
    return ($parts -join " ")
}

$repoRoot = Get-RepoRoot
Set-Location $repoRoot
$config = Get-WorkflowConfig
if (-not $Branch) {
    $Branch = Get-CurrentBranch
}

& python -m py_compile main.py model.py scripts/analyze_q_collapse.py
if ($LASTEXITCODE -ne 0) {
    throw "Local py_compile failed."
}

if (-not $SkipGitSync) {
    Invoke-QIsolationSelectiveGitSync -Branch $Branch -Message "exp: q collapse isolation controls"
}

$target = Get-QIsolationSshTarget -Config $config
if ($Gpu -lt 0) {
    $Gpu = Get-FreeGpu -Config $config -Target $target
}

$runName = "qiso_${Mode}_m${NumModels}_seed${Seed}"
if (-not $Session) {
    $Session = $runName
}
$logFile = "logs/q_isolation/$Session.log"
$runCmd = Get-QCommand -Mode $Mode -NumModels $NumModels -Seed $Seed -Gpu $Gpu -RunName $runName

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
RESULT_ROOT="$(decode_arg "$9")"
SKIP_REMOTE_GIT="$(decode_arg "${10}")"

cd "$REPO_DIR"
if [ "$SKIP_REMOTE_GIT" != "1" ]; then
    git fetch "$GIT_REMOTE"
    git checkout "$BRANCH"
    git pull --ff-only "$GIT_REMOTE" "$BRANCH"
else
    git checkout "$BRANCH"
fi

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
    echo "python scripts/analyze_q_collapse.py \"$RESULT_ROOT\" --out \"$RESULT_ROOT/${SESSION}_summary.csv\""
} > "$RUNNER"
chmod +x "$RUNNER"
tmux new-session -d -s "$SESSION" "bash '$RUNNER' > '$LOG_FILE' 2>&1"
echo "Started tmux session: $SESSION"
echo "Log file: $LOG_FILE"
'@

Invoke-QIsolationRemoteScript -Config $config -Target $target -Script $remoteScript -Arguments @(
    $Branch,
    $config.RepoDir,
    $config.GitRemote,
    $config.CondaInit,
    $config.DefaultCondaEnv,
    $Session,
    $logFile,
    $runCmd,
    "results_diag/q_isolation/$runName",
    $(if ($SkipRemoteGitSync) { "1" } else { "0" })
)

Write-Host "Started Q isolation $Mode m=$NumModels seed=$Seed on GPU $Gpu."
Write-Host "Session: $Session"
Write-Host "Log: $logFile"
Write-Host "Command: $runCmd"

if ($NoWait) {
    return
}

while ($true) {
    Start-Sleep -Seconds 60
    $statusCommand = "cd '$($config.RepoDir)' && if tmux has-session -t '$Session' 2>/dev/null; then echo RUNNING; tail -n 20 '$logFile' || true; else echo DONE; tail -n 120 '$logFile' || true; fi"
    $status = & ssh -o ClearAllForwardings=yes -o StrictHostKeyChecking=accept-new -o ExitOnForwardFailure=no -p $config.Port $target $statusCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Remote status command failed with exit code $LASTEXITCODE."
    }
    $status | ForEach-Object { Write-Host $_ }
    if ($status -contains "DONE") {
        break
    }
}

if ($FetchResults) {
    $localDir = Join-Path $repoRoot "remote_results\q_isolation"
    New-Item -ItemType Directory -Force -Path $localDir | Out-Null
    & scp -o ClearAllForwardings=yes -o StrictHostKeyChecking=accept-new -P $config.Port -r "$target`:$($config.RepoDir)/results_diag/q_isolation/$runName" $localDir
    if ($LASTEXITCODE -ne 0) {
        throw "scp results fetch failed."
    }
    Write-Host "Fetched Q isolation diagnostics to $localDir"
}
