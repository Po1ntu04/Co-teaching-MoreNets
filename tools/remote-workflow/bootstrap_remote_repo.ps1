param(
    [string]$RepoDir = "",
    [switch]$ForceOriginUpdate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\lib.ps1"

$config = Get-WorkflowConfig
if (-not $RepoDir) {
    $RepoDir = $config.RepoDir
}

$remoteScript = @'
set -euo pipefail

REPO_PARENT_DIR="$(decode_arg "$1")"
REPO_DIR="$(decode_arg "$2")"
REMOTE_GIT_URL="$(decode_arg "$3")"
FORCE_ORIGIN_UPDATE="$(decode_arg "$4")"

mkdir -p "$REPO_PARENT_DIR"

if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR"
    if [ "$FORCE_ORIGIN_UPDATE" = "1" ]; then
        git remote set-url origin "$REMOTE_GIT_URL"
    fi
    echo "Remote repo already exists: $REPO_DIR"
    git remote -v
    exit 0
fi

cd "$REPO_PARENT_DIR"
git clone "$REMOTE_GIT_URL" "$(basename "$REPO_DIR")"
cd "$REPO_DIR"
git remote -v
'@

Invoke-RemoteScript -Config $config -Script $remoteScript -Arguments @(
    $config.RepoParentDir,
    $RepoDir,
    $config.RemoteGitUrl,
    $(if ($ForceOriginUpdate) { "1" } else { "0" })
)
