Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    $root = (& git rev-parse --show-toplevel 2>$null)
    if (-not $root) {
        throw "Not inside a git repository."
    }
    return $root.Trim()
}

function Get-CurrentBranch {
    $branch = (& git branch --show-current 2>$null)
    if (-not $branch) {
        throw "Could not determine current git branch."
    }
    return $branch.Trim()
}

function Get-WorkflowConfig {
    $repoRoot = Get-RepoRoot
    $configPath = Join-Path $repoRoot "tools\remote-workflow\config.psd1"
    if (-not (Test-Path $configPath)) {
        $examplePath = Join-Path $repoRoot "tools\remote-workflow\config.example.psd1"
        throw "Missing $configPath. Copy $examplePath to config.psd1 and fill in your server settings."
    }

    $config = Import-PowerShellDataFile $configPath
    $required = @(
        "Host",
        "Port",
        "User",
        "RepoParentDir",
        "RepoDir",
        "GitRemote",
        "LocalGitUrl",
        "RemoteGitUrl",
        "CondaInit",
        "DefaultCondaEnv",
        "DefaultTmuxSession",
        "DefaultTmuxLog",
        "DefaultResultsDir"
    )

    foreach ($key in $required) {
        if (-not $config.ContainsKey($key) -or [string]::IsNullOrWhiteSpace([string]$config[$key])) {
            throw "Config key '$key' is missing or empty in tools/remote-workflow/config.psd1."
        }
    }

    return $config
}

function Get-SshTarget {
    param(
        [hashtable]$Config
    )
    if ($Config.ContainsKey("SshHostAlias") -and -not [string]::IsNullOrWhiteSpace([string]$Config.SshHostAlias)) {
        return [string]$Config.SshHostAlias
    }
    return "$($Config.User)@$($Config.Host)"
}

function Invoke-RemoteScript {
    param(
        [hashtable]$Config,
        [string]$Script,
        [string[]]$Arguments = @(),
        [switch]$CaptureOutput
    )

    $target = Get-SshTarget -Config $Config
    $encodedArgs = @()
    foreach ($argument in $Arguments) {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes([string]$argument)
        $encodedArgs += [Convert]::ToBase64String($bytes)
    }

    $wrapper = @'
decode_arg() {
    printf '%s' "$1" | base64 --decode
}
'@ + "`n" + $Script
    $wrapper = $wrapper -replace "`r", ""

    $sshArgs = @("-p", [string]$Config.Port, $target, "bash", "-s", "--") + $encodedArgs

    if ($CaptureOutput) {
        $output = $wrapper | & ssh @sshArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Remote command failed with exit code $LASTEXITCODE."
        }
        return $output
    }

    $wrapper | & ssh @sshArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Remote command failed with exit code $LASTEXITCODE."
    }
}

function Assert-CleanWorkingTree {
    $status = (& git status --short)
    if ($status) {
        throw "Working tree is not clean. Commit or stash your changes, or rerun sync_branch.ps1 with -AutoCommit."
    }
}
