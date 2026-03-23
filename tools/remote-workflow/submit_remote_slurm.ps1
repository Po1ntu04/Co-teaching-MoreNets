param(
    [string]$Branch = "",
    [string]$SbatchFile = "slurm/run_coteaching.sbatch",
    [switch]$SkipSync
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\lib.ps1"

$config = Get-WorkflowConfig
if (-not $Branch) {
    $Branch = Get-CurrentBranch
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
SBATCH_FILE="$(decode_arg "$4")"

cd "$REPO_DIR"
git fetch "$GIT_REMOTE"
git checkout "$BRANCH"
git pull --ff-only "$GIT_REMOTE" "$BRANCH"

mkdir -p logs
sbatch "$SBATCH_FILE"
'@

Invoke-RemoteScript -Config $config -Script $remoteScript -Arguments @(
    $Branch,
    $config.RepoDir,
    $config.GitRemote,
    $SbatchFile
) | Write-Host
