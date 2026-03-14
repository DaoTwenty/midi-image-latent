#!/bin/bash
#SBATCH --job-name=exp1b-full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/triana24/midi-image-latent/outputs/logs/exp1b_full_%j.log
#SBATCH --error=/scratch/triana24/midi-image-latent/outputs/logs/exp1b_full_%j.log

# Experiment 1B — Detection Method Comparison
# 1 VAE x 8 detection methods
# Expected runtime: ~30 min on H100 MIG
# NOTE: Update the VAE in exp_1b_detection_methods.yaml to the best from Exp 1A
#       before submitting this job.

set -euo pipefail

REPO=/scratch/triana24/midi-image-latent

echo "=========================================="
echo "Job:        $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start:      $(date)"
echo "Experiment: exp_1b_detection_methods (full)"
echo "Conditions: 1 VAE x 8 detection methods"
echo "=========================================="

module purge
module load python/3.11 gcc/12 cuda/12.6 arrow/23.0.1

export HF_HOME="$SCRATCH/.cache/huggingface"

if [[ -f "$REPO/.env" ]]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"
        export "$key"="$value"
    done < <(grep -v '^#' "$REPO/.env" | grep '=')
fi

source "$REPO/.venv/bin/activate"

cd "$REPO"

echo "Python: $(which python)"
echo "HF_HOME: $HF_HOME"
echo "CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count(), "GPU(s)")')"

python scripts/run_experiment.py \
    configs/experiments/exp_1b_detection_methods.yaml \
    --data-root data/maestro/maestro-v3.0.0 \
    --sweep-detectors \
    --log-level INFO

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
