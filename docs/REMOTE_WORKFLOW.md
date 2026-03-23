# Remote Workflow

This repository uses a Windows local development + Linux remote execution workflow.

Core path:

1. Edit locally in VS Code.
2. Push the current branch from local to GitHub.
3. Pull the same branch on the remote server.
4. Run remotely with `tmux` or `Slurm`.
5. Tail logs and fetch outputs back to local.

Git moves source code. SSH controls the remote machine. `scp` only pulls results or logs.

## This Repo's Concrete Setup

- SSH alias: `b101`
- Remote host: `b101.guhk.cc`
- Remote user: `yuzhixiang`
- Recommended remote repo path: `/data1/yuzhixiang/work/Co-Teaching`
- Local Git URL: `git@github.com:Po1ntu04/Co-teaching-MoreNets.git`
- Remote Git URL: `https://github.com/Po1ntu04/Co-teaching-MoreNets.git`
- Main remote execution mode: `tmux`
- Default conda env path: `/data1/yuzhixiang/.conda/envs/coteaching-py39`

## Files

- `tools/remote-workflow/config.psd1`
- `tools/remote-workflow/check_remote.ps1`
- `tools/remote-workflow/bootstrap_remote_repo.ps1`
- `tools/remote-workflow/set_local_git_remote.ps1`
- `tools/remote-workflow/sync_branch.ps1`
- `tools/remote-workflow/sync_and_update_remote.ps1`
- `tools/remote-workflow/setup_remote_env.ps1`
- `tools/remote-workflow/run_remote.ps1`
- `tools/remote-workflow/submit_remote_slurm.ps1`
- `tools/remote-workflow/tail_remote.ps1`
- `tools/remote-workflow/fetch_remote_results.ps1`
- `tools/remote-workflow/stop_remote.ps1`

## One-Time Setup

### 1. Local Git switches to SSH

```powershell
powershell -ExecutionPolicy Bypass -File tools/remote-workflow/set_local_git_remote.ps1
```

### 2. Bootstrap the remote repository

```powershell
powershell -ExecutionPolicy Bypass -File tools/remote-workflow/bootstrap_remote_repo.ps1
```

If the remote repo exists but its `origin` is wrong:

```powershell
powershell -ExecutionPolicy Bypass -File tools/remote-workflow/bootstrap_remote_repo.ps1 -ForceOriginUpdate
```

### 3. Verify remote toolchain

```powershell
powershell -ExecutionPolicy Bypass -File tools/remote-workflow/check_remote.ps1
```

### 4. Create or align the Python 3.9 conda environment

Base dependencies only:

```powershell
powershell -ExecutionPolicy Bypass -File tools/remote-workflow/setup_remote_env.ps1
```

If you already know the exact torch install command for your server CUDA runtime:

```powershell
powershell -ExecutionPolicy Bypass -File tools/remote-workflow/setup_remote_env.ps1 `
  -TorchInstallCommand "python -m pip install torch torchvision --index-url <your-torch-wheel-index>"
```

## Recommended Daily Flow

### A. One-click sync local branch and update remote checkout

If your working tree is already committed:

```powershell
powershell -ExecutionPolicy Bypass -File tools/remote-workflow/sync_and_update_remote.ps1
```

If you want the script to auto-commit first:

```powershell
powershell -ExecutionPolicy Bypass -File tools/remote-workflow/sync_and_update_remote.ps1 -AutoCommit -Message "wip: sync before remote update"
```

### B. Run remotely with tmux

```powershell
powershell -ExecutionPolicy Bypass -File tools/remote-workflow/run_remote.ps1 `
  -Command "python -u main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.5"
```

### C. Submit with Slurm when needed

```powershell
powershell -ExecutionPolicy Bypass -File tools/remote-workflow/submit_remote_slurm.ps1 `
  -SbatchFile "slurm/run_coteaching.sbatch"
```

### D. Tail logs and fetch outputs

```powershell
powershell -ExecutionPolicy Bypass -File tools/remote-workflow/tail_remote.ps1
powershell -ExecutionPolicy Bypass -File tools/remote-workflow/fetch_remote_results.ps1
```

## Notes

- Remote pulls always use `git pull --ff-only`. If the remote checkout diverges, the script stops instead of overwriting state.
- `run_remote.ps1` is the default path for your workflow because you said you mainly use `tmux`.
- `setup_remote_env.ps1` installs the repo's Python requirements, but torch/torchvision should still match the real CUDA runtime on the server.
