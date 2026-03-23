#!/bin/bash
set -euo pipefail

# Create log directory BEFORE submitting the job.
mkdir -p logs

sbatch slurm/run_coteaching.sbatch
