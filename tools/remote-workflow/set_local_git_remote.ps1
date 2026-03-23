param(
    [string]$RemoteName = "origin"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\lib.ps1"

$config = Get-WorkflowConfig
& git remote set-url $RemoteName $config.LocalGitUrl
if ($LASTEXITCODE -ne 0) {
    throw "Failed to set local git remote URL."
}

git remote -v
