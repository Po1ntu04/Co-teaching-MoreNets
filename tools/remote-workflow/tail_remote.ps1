param(
    [string]$LogFile = "",
    [int]$Lines = 120,
    [switch]$LatestSlurm
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\lib.ps1"

$config = Get-WorkflowConfig
if (-not $LogFile) {
    $LogFile = $config.DefaultTmuxLog
}

$remoteScript = @'
set -euo pipefail

REPO_DIR="$(decode_arg "$1")"
LOG_FILE="$(decode_arg "$2")"
LINES="$(decode_arg "$3")"
LATEST_SLURM="$(decode_arg "$4")"

cd "$REPO_DIR"

if [ "$LATEST_SLURM" = "1" ]; then
    LOG_FILE=$(ls -t logs/*.out 2>/dev/null | head -n 1 || true)
fi

if [ -z "$LOG_FILE" ]; then
    echo "No log file resolved." >&2
    exit 21
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found: $LOG_FILE" >&2
    exit 22
fi

echo "== tail: $LOG_FILE =="
tail -n "$LINES" "$LOG_FILE"
'@

Invoke-RemoteScript -Config $config -Script $remoteScript -Arguments @(
    $config.RepoDir,
    $LogFile,
    [string]$Lines,
    $(if ($LatestSlurm) { "1" } else { "0" })
)
