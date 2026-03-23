param(
    [string]$Branch = "",
    [string]$Session = "",
    [string]$CondaEnv = "",
    [string]$LogFile = "",
    [string]$Command = "python -u main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.5",
    [switch]$SkipSync
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\lib.ps1"

$config = Get-WorkflowConfig
if (-not $Branch) {
    $Branch = Get-CurrentBranch
}
if (-not $Session) {
    $Session = $config.DefaultTmuxSession
}
if (-not $CondaEnv) {
    $CondaEnv = $config.DefaultCondaEnv
}
if (-not $LogFile) {
    $LogFile = $config.DefaultTmuxLog
}

if (-not $SkipSync) {
    & "$PSScriptRoot\sync_branch.ps1" -Branch $Branch
    if ($LASTEXITCODE -ne 0) {
        throw "Local sync failed."
    }
}

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

RUNNER="$REPO_DIR/.workflow_last_tmux_run.sh"
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

echo "Started tmux session: $SESSION"
echo "Log file: $LOG_FILE"
'@

Invoke-RemoteScript -Config $config -Script $remoteScript -Arguments @(
    $Branch,
    $config.RepoDir,
    $config.GitRemote,
    $config.CondaInit,
    $CondaEnv,
    $Session,
    $LogFile,
    $Command
)
