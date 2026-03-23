param(
    [string]$PythonVersion = "3.9",
    [string]$CondaEnv = "",
    [string]$TorchInstallCommand = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. "$PSScriptRoot\lib.ps1"

$config = Get-WorkflowConfig
if (-not $CondaEnv) {
    $CondaEnv = $config.DefaultCondaEnv
}

$remoteScript = @'
set -euo pipefail

REPO_DIR="$(decode_arg "$1")"
CONDA_INIT="$(decode_arg "$2")"
CONDA_ENV="$(decode_arg "$3")"
PYTHON_VERSION="$(decode_arg "$4")"
TORCH_INSTALL_COMMAND="$(decode_arg "$5")"

cd "$REPO_DIR"
eval "$CONDA_INIT"

if [ ! -d "$CONDA_ENV" ]; then
    conda create -y -p "$CONDA_ENV" python="$PYTHON_VERSION"
fi

conda activate "$CONDA_ENV"
python -m pip install --upgrade pip
python -m pip install -r requirements-py39.txt

if [ -n "$TORCH_INSTALL_COMMAND" ]; then
    eval "$TORCH_INSTALL_COMMAND"
else
    echo "Torch install command not provided. Install torch/torchvision manually for your CUDA runtime."
fi

echo "== python =="
python --version
echo "== pip packages =="
python -m pip show numpy scipy pillow six | sed -n "1,20p" || true
'@

Invoke-RemoteScript -Config $config -Script $remoteScript -Arguments @(
    $config.RepoDir,
    $config.CondaInit,
    $CondaEnv,
    $PythonVersion,
    $TorchInstallCommand
)
