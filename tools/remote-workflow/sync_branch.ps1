param(
    [string]$Branch = "",
    [string]$Remote = "",
    [switch]$AutoCommit,
    [string]$Message = "chore: sync before remote run"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\lib.ps1"

$repoRoot = Get-RepoRoot
Set-Location $repoRoot

$config = Get-WorkflowConfig
if (-not $Branch) {
    $Branch = Get-CurrentBranch
}
if (-not $Remote) {
    $Remote = $config.GitRemote
}

$status = (& git status --short)
if ($status) {
    if (-not $AutoCommit) {
        throw "Working tree is not clean. Commit manually, or rerun with -AutoCommit."
    }

    & git add -A
    & git commit -m $Message
    if ($LASTEXITCODE -ne 0) {
        throw "git commit failed."
    }
}

& git push $Remote $Branch
if ($LASTEXITCODE -ne 0) {
    throw "git push failed."
}

Write-Host "Synced branch '$Branch' to remote '$Remote'."
