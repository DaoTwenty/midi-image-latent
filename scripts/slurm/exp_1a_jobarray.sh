#!/bin/bash
#SBATCH --job-name=exp1a-array
#SBATCH --account=def-pasquier
#SBATCH --array=0-35
#SBATCH --gpus-per-node=h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/triana24/midi-image-latent/outputs/logs/exp1a_array_%A_%a.log
#SBATCH --error=/scratch/triana24/midi-image-latent/outputs/logs/exp1a_array_%A_%a.log

# Experiment 1A — SLURM job array (one job per condition)
# 12 VAEs x 3 channel strategies = 36 conditions (array indices 0-35)
#
# Condition layout (consistent with run_experiment_distributed.py):
#   condition_index = vae_index * 3 + strategy_index
#   vae_index      in [0..11] — the 12 VAE models
#   strategy_index in [0..2]  — velocity_only, vo_split, vos
#
# This is the most scheduler-friendly approach:
#   - Each array task is independent; failures don't affect other conditions
#   - The scheduler can backfill tasks as GPUs become free
#   - Suitable when GPUs are scarce or when runtimes are heterogeneous
#   - Re-run specific conditions by requeuing individual array tasks
#
# Requeue a single failed task:
#   scontrol requeue <ARRAY_JOB_ID>_<TASK_ID>
#
# Check array status:
#   squeue -j <ARRAY_JOB_ID> --array
#
# Merge results after all tasks complete:
#   sbatch scripts/slurm/exp_1a_array_merge.sh <ARRAY_JOB_ID>
#
# Expected runtime per task: ~20-30 min on H100

set -euo pipefail

REPO=/scratch/triana24/midi-image-latent
CONDITION_IDX=$SLURM_ARRAY_TASK_ID

echo "=========================================="
echo "Job:          $SLURM_JOB_ID (array $SLURM_ARRAY_JOB_ID)"
echo "Array task:   $SLURM_ARRAY_TASK_ID"
echo "Condition:    $CONDITION_IDX / 35"
echo "Node:         $SLURMD_NODENAME"
echo "Start:        $(date)"
echo "Experiment:   exp_1a_vae_comparison (job array)"
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

# Each task writes to its own output directory to avoid conflicts
OUTPUT_DIR="$REPO/outputs/exp_1a/condition_$CONDITION_IDX"
mkdir -p "$OUTPUT_DIR"

source "$REPO/.venv/bin/activate"

cd "$REPO"

echo "Python:        $(which python)"
echo "HF_HOME:       $HF_HOME"
echo "WANDB_MODE:    $WANDB_MODE"
echo "Output dir:    $OUTPUT_DIR"
echo "GPU count:     $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Each array task runs exactly one condition identified by --condition-indices.
# CUDA_VISIBLE_DEVICES is not set explicitly — each task gets the one GPU
# allocated by SLURM (gpus-per-node=1) and sees it as device 0.
python scripts/run_experiment_distributed.py \
    configs/experiments/exp_1a_vae_comparison.yaml \
    --condition-indices "$CONDITION_IDX" \
    --data-root data/lakh \
    --output-dir "$OUTPUT_DIR" \
    --log-level INFO

echo ""
echo "[wandb] Offline run saved. Sync from login node:"
echo "  bash scripts/wandb_sync.sh"
echo ""
echo "=========================================="
echo "Condition $CONDITION_IDX done: $(date)"
echo "=========================================="
