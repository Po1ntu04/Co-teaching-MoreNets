param(
    [string]$Branch = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\lib.ps1"

$config = Get-WorkflowConfig
if (-not $Branch) {
    $Branch = Get-CurrentBranch
}

$remoteScript = @'
set -euo pipefail

BRANCH="$(decode_arg "$1")"
REPO_DIR="$(decode_arg "$2")"
GIT_REMOTE="$(decode_arg "$3")"

echo "== host =="
hostname
echo

echo "== repo =="
if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR"
    pwd
    git rev-parse --is-inside-work-tree
    git remote -v
    echo
    echo "== branch visibility =="
    git ls-remote "$GIT_REMOTE" "refs/heads/$BRANCH" | head -n 1 || true
    echo
    echo "== local branch state =="
    git rev-parse --abbrev-ref HEAD
    git status --short
else
    echo "Repo not initialized yet: $REPO_DIR"
fi
echo

echo "== tools =="
git --version
python --version || true
tmux -V || true
sbatch --version || true
'@

Invoke-RemoteScript -Config $config -Script $remoteScript -Arguments @(
    $Branch,
    $config.RepoDir,
    $config.GitRemote
)
