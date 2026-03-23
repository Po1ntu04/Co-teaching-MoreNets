param(
    [ValidateSet("tmux", "slurm")]
    [string]$Mode = "tmux",
    [string]$Session = "",
    [string]$JobId = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\lib.ps1"

$config = Get-WorkflowConfig
if (-not $Session) {
    $Session = $config.DefaultTmuxSession
}

if ($Mode -eq "slurm" -and [string]::IsNullOrWhiteSpace($JobId)) {
    throw "JobId is required when Mode=slurm."
}

$remoteScript = @'
set -euo pipefail

MODE="$(decode_arg "$1")"
SESSION="$(decode_arg "$2")"
JOB_ID="$(decode_arg "$3")"

if [ "$MODE" = "tmux" ]; then
    tmux kill-session -t "$SESSION"
    echo "Stopped tmux session: $SESSION"
else
    scancel "$JOB_ID"
    echo "Cancelled slurm job: $JOB_ID"
fi
'@

Invoke-RemoteScript -Config $config -Script $remoteScript -Arguments @(
    $Mode,
    $Session,
    $JobId
)
