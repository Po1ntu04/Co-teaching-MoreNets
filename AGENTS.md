# AGENTS.md

## Working Modes

- This repository supports two execution modes:
  - local editing and quick verification on the Windows developer machine
  - remote Linux execution for GPU jobs, long experiments, and environment-specific reproduction

- Do not invent ad hoc remote commands when a repository script already exists.
- Prefer the scripts under `tools/remote-workflow/` for branch sync, remote launch, log tailing, and result fetching.

## Git And Sync

- Treat Git as the source of truth for code transfer between local and remote.
- Before remote execution, make sure the target branch is committed and pushed.
- Use fast-forward pulls on the remote host. Do not overwrite unrelated remote changes.
- Do not force-push unless explicitly requested.

## Remote Execution

- Prefer `submit_remote_slurm.ps1` for scheduled GPU jobs on clusters with Slurm.
- Prefer `run_remote.ps1` for direct `tmux`-based long runs.
- After starting a remote job, inspect logs with `tail_remote.ps1` and summarize current state.
- Stop only the session or job that belongs to the current workflow.

## Safety

- Never print secrets, tokens, SSH private keys, or proxy credentials.
- Ask before destructive actions such as deleting remote outputs or killing unrelated sessions/jobs.
- Keep local machine-specific settings in `tools/remote-workflow/config.psd1`, which must stay untracked.
