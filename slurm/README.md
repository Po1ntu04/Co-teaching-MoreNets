# Slurm scripts

- `slurm/run_coteaching.sbatch` uses `#SBATCH --output=logs/%x_%j.out` and `#SBATCH --error=logs/%x_%j.err`.
- Slurm requires the `logs/` directory to exist *before* `sbatch` submission.

Submit:

```bash
bash slurm/submit_coteaching.sh
```

Or manually:

```bash
mkdir -p logs
sbatch slurm/run_coteaching.sbatch
```
