#!/bin/bash
#SBATCH --job-name=exp-all-mini
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/triana24/midi-image-latent/outputs/logs/exp_all_mini_%j.log
#SBATCH --error=/scratch/triana24/midi-image-latent/outputs/logs/exp_all_mini_%j.log

# All experiments in mini mode — sequential, for smoke-testing the full pipeline.
# Uses bars_per_instrument=20, max_files=5, first 2 VAEs only.
# Expected runtime: ~30 min total on H100 MIG.

set -euo pipefail

REPO=/scratch/triana24/midi-image-latent

echo "=========================================="
echo "Job:        $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start:      $(date)"
echo "Mode:       ALL EXPERIMENTS (mini)"
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

DATA_ROOT="data/maestro/maestro-v3.0.0"
MINI_FLAGS="--mini --data-root $DATA_ROOT --log-level INFO"

# ------------------------------------------------------------------ Exp 1A
echo ""
echo "=== Exp 1A: VAE Comparison (mini) ==="
python scripts/run_experiment.py \
    configs/experiments/exp_1a_vae_comparison.yaml \
    $MINI_FLAGS \
    --sweep-strategies

# ------------------------------------------------------------------ Exp 1B
echo ""
echo "=== Exp 1B: Detection Methods (mini) ==="
python scripts/run_experiment.py \
    configs/experiments/exp_1b_detection_methods.yaml \
    $MINI_FLAGS \
    --sweep-detectors

# ------------------------------------------------------------------ Exp 2
echo ""
echo "=== Exp 2: Resolution Study (mini) ==="
python scripts/run_experiment.py \
    configs/experiments/exp_2_resolution_study.yaml \
    $MINI_FLAGS

# ------------------------------------------------------------------ Exp 3
echo ""
echo "=== Exp 3: Channel Strategy (mini) ==="
python scripts/run_experiment.py \
    configs/experiments/exp_3_channel_strategy.yaml \
    $MINI_FLAGS \
    --sweep-strategies

# ------------------------------------------------------------------ Exp 4 (latent analysis)
echo ""
echo "=== Exp 4: Latent Analysis (mini) ==="
python scripts/run_experiment.py \
    configs/experiments/exp_4_latent_analysis.yaml \
    $MINI_FLAGS

# ------------------------------------------------------------------ Exp 4B
echo ""
echo "=== Exp 4B: Latent Structure (mini) ==="
python scripts/run_experiment.py \
    configs/experiments/exp_4b_latent_structure.yaml \
    $MINI_FLAGS

# ------------------------------------------------------------------ Exp 4C
echo ""
echo "=== Exp 4C: Sub-Latent Compression (mini) ==="
python scripts/run_experiment.py \
    configs/experiments/exp_4c_sublatent.yaml \
    $MINI_FLAGS

# ------------------------------------------------------------------ Exp 4D
echo ""
echo "=== Exp 4D: Conditioning (mini) ==="
python scripts/run_experiment.py \
    configs/experiments/exp_4d_conditioning.yaml \
    $MINI_FLAGS

# ------------------------------------------------------------------ Exp 5
echo ""
echo "=== Exp 5: Sequence Generation (mini) ==="
python scripts/run_experiment.py \
    configs/experiments/exp_5_sequence_generation.yaml \
    $MINI_FLAGS

echo ""
echo "[wandb] Offline runs saved. Sync from login node:"
echo "  bash scripts/wandb_sync.sh"
echo ""
echo "=========================================="
echo "All mini experiments complete: $(date)"
echo "=========================================="
