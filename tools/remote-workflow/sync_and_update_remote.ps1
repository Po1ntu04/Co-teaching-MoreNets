param(
    [string]$Branch = "",
    [switch]$AutoCommit,
    [string]$Message = "chore: sync before remote update"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\lib.ps1"

$config = Get-WorkflowConfig
if (-not $Branch) {
    $Branch = Get-CurrentBranch
}

& "$PSScriptRoot\sync_branch.ps1" -Branch $Branch -AutoCommit:$AutoCommit -Message $Message
if ($LASTEXITCODE -ne 0) {
    throw "Local sync failed."
}

$remoteScript = @'
set -euo pipefail

BRANCH="$(decode_arg "$1")"
REPO_DIR="$(decode_arg "$2")"
GIT_REMOTE="$(decode_arg "$3")"

cd "$REPO_DIR"
git fetch "$GIT_REMOTE"
git checkout "$BRANCH"
git pull --ff-only "$GIT_REMOTE" "$BRANCH"
echo "== remote branch =="
git rev-parse --abbrev-ref HEAD
echo "== remote commit =="
git rev-parse HEAD
'@

Invoke-RemoteScript -Config $config -Script $remoteScript -Arguments @(
    $Branch,
    $config.RepoDir,
    $config.GitRemote
)
