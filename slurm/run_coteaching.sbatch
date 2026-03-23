#!/bin/bash
#SBATCH --job-name=coteaching
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

# IMPORTANT:
# If you write logs to a subfolder, make sure it exists BEFORE sbatch submission.
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Example: activate conda env
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate coteaching-py37-torch16

echo "Job started on $(hostname) at $(date)"

python -u main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.5

echo "Job finished at $(date)"
