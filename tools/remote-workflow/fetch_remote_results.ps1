param(
    [string]$RemotePath = "",
    [string]$LocalPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\lib.ps1"

$config = Get-WorkflowConfig
$repoRoot = Get-RepoRoot

if (-not $RemotePath) {
    $RemotePath = "$($config.RepoDir)/$($config.DefaultResultsDir)"
}
if (-not $LocalPath) {
    $LocalPath = Join-Path $repoRoot "remote_results"
}

New-Item -ItemType Directory -Force -Path $LocalPath | Out-Null

$target = Get-SshTarget -Config $config
$sourceSpec = "${target}:$RemotePath"

& scp -P $config.Port -r $sourceSpec $LocalPath
if ($LASTEXITCODE -ne 0) {
    throw "scp failed."
}

Write-Host "Fetched '$RemotePath' into '$LocalPath'."
