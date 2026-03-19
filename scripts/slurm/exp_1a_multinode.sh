#!/bin/bash
#SBATCH --job-name=exp1a-multinode
#SBATCH --account=def-pasquier
#SBATCH --nodes=2
#SBATCH --gpus-per-node=h100:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/triana24/midi-image-latent/outputs/logs/exp1a_multinode_%j.log
#SBATCH --error=/scratch/triana24/midi-image-latent/outputs/logs/exp1a_multinode_%j.log

# Experiment 1A — Multi-node job (2 nodes x 2 GPUs = 4 GPUs total)
# 12 VAEs x 3 channel strategies = 36 conditions
# Global ranks 0..3 each handle a round-robin subset of conditions:
#   rank 0 -> conditions 0,  4,  8, 12, 16, 20, 24, 28, 32
#   rank 1 -> conditions 1,  5,  9, 13, 17, 21, 25, 29, 33
#   rank 2 -> conditions 2,  6, 10, 14, 18, 22, 26, 30, 34
#   rank 3 -> conditions 3,  7, 11, 15, 19, 23, 27, 31, 35
# run_experiment_distributed.py implements this via SLURM_PROCID + SLURM_NTASKS.
# Expected runtime: ~1h on 4x H100

set -euo pipefail

REPO=/scratch/triana24/midi-image-latent

echo "=========================================="
echo "Job:        $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start:      $(date)"
echo "Experiment: exp_1a_vae_comparison (multi-node)"
echo "Strategy:   2 nodes x 2 GPUs, 4 total tasks"
echo "Conditions: 12 VAEs x 3 channel strategies = 36 total"
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

echo "Python:       $(which python)"
echo "HF_HOME:      $HF_HOME"
echo "WANDB_MODE:   $WANDB_MODE"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo ""

# Ensure per-rank output directories exist (all four ranks)
for RANK in 0 1 2 3; do
    mkdir -p "$REPO/outputs/exp_1a/rank_$RANK"
done
mkdir -p "$REPO/outputs/exp_1a/merged"

# srun distributes tasks across both nodes (2 tasks per node x 2 nodes = 4 tasks).
# SLURM_PROCID is the global rank (0..3).
# SLURM_LOCALID is the per-node local rank (0..1), used for CUDA_VISIBLE_DEVICES.
# run_experiment_distributed.py auto-detects SLURM_PROCID and SLURM_NTASKS and
# distributes conditions round-robin: rank r handles [r, r+N, r+2N, ...].
srun --output="$REPO/outputs/logs/exp1a_multinode_%j_rank%t.log" \
    bash -c '
        export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
        echo "[rank $SLURM_PROCID / $SLURM_NTASKS] node=$SLURMD_NODENAME local_gpu=$SLURM_LOCALID"
        python scripts/run_experiment_distributed.py \
            configs/experiments/exp_1a_vae_comparison.yaml \
            --data-root data/lakh \
            --output-dir outputs/exp_1a/rank_$SLURM_PROCID \
            --log-level INFO
    '

# ------------------------------------------------------------------ Merge step
# srun is synchronous — all tasks across all nodes have finished before here.
# Run the merge from the primary task (this script runs on the first node, rank 0).
echo ""
echo "=== Merging per-rank results ==="
python scripts/run_experiment_distributed.py \
    configs/experiments/exp_1a_vae_comparison.yaml \
    --merge-results \
    --ranks-dir "$REPO/outputs/exp_1a" \
    --num-ranks "$SLURM_NTASKS" \
    --output-dir "$REPO/outputs/exp_1a/merged" \
    --log-level INFO

echo ""
echo "[wandb] Offline runs saved. Sync from login node:"
echo "  bash scripts/wandb_sync.sh"
echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
