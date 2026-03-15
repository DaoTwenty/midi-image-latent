#!/bin/bash
#SBATCH --job-name=exp1a-full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/triana24/midi-image-latent/outputs/logs/exp1a_full_%j.log
#SBATCH --error=/scratch/triana24/midi-image-latent/outputs/logs/exp1a_full_%j.log

# Experiment 1A — Full VAE Comparison
# 12 VAEs x 3 channel strategies = 36 conditions
# Expected runtime: ~2h on H100 MIG (10.5 GB VRAM)

set -euo pipefail

REPO=/scratch/triana24/midi-image-latent

echo "=========================================="
echo "Job:        $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start:      $(date)"
echo "Experiment: exp_1a_vae_comparison (full)"
echo "Conditions: 12 VAEs x 3 channel strategies"
echo "=========================================="

module purge
module load python/3.11 gcc/12 cuda/12.6 arrow/23.0.1

export HF_HOME="$SCRATCH/.cache/huggingface"

# Load .env (HF_TOKEN, WANDB_API_KEY, etc.)
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

# Wandb: offline on compute nodes (no internet), sync later from login node
export WANDB_MODE=offline
export WANDB_DIR="$REPO/outputs"

source "$REPO/.venv/bin/activate"

cd "$REPO"

echo "Python: $(which python)"
echo "HF_HOME: $HF_HOME"
echo "WANDB_MODE: $WANDB_MODE"
echo "CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count(), "GPU(s)")')"

python scripts/run_experiment.py \
    configs/experiments/exp_1a_vae_comparison.yaml \
    --data-root data/maestro/maestro-v3.0.0 \
    --sweep-strategies \
    --log-level INFO

echo ""
echo "[wandb] Offline runs saved. Sync from login node:"
echo "  bash scripts/wandb_sync.sh"
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
