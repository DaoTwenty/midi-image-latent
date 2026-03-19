#!/bin/bash
# Experiment 1A — Distributed Multi-GPU VAE Comparison
# 36 conditions (12 VAEs x 3 channel strategies) split across N GPUs.
#
# Single-node 4-GPU example:
#   sbatch --gpus-per-node=h100:4 --cpus-per-task=32 scripts/slurm/exp_1a_distributed.sh
#
# Multi-node 2x4-GPU example (8 ranks total, 4.5 conditions each):
#   sbatch --nodes=2 --ntasks-per-node=4 --gpus-per-task=1 \
#          --cpus-per-task=8 scripts/slurm/exp_1a_distributed.sh
#
# Defaults below are for a single node with 4 H100 GPUs.

#SBATCH --job-name=exp1a-dist
#SBATCH --account=def-pasquier
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/triana24/midi-image-latent/outputs/logs/exp1a_dist_%j.log
#SBATCH --error=/scratch/triana24/midi-image-latent/outputs/logs/exp1a_dist_%j.log

set -euo pipefail

REPO=/scratch/triana24/midi-image-latent
OUTPUT_ROOT=$SCRATCH/midi-image-latent/outputs/exp_1a_distributed

echo "=========================================="
echo "Job:        $SLURM_JOB_ID"
echo "Node(s):    $SLURMD_NODENAME"
echo "Tasks:      $SLURM_NTASKS (${SLURM_NTASKS_PER_NODE:-?} per node)"
echo "Start:      $(date)"
echo "Experiment: exp_1a_vae_comparison (distributed)"
echo "Conditions: 12 VAEs x 3 channel strategies = 36"
echo "Output:     $OUTPUT_ROOT"
echo "=========================================="

module purge
module load python/3.11 gcc/12 cuda/12.6 arrow/23.0.1

export HF_HOME="$SCRATCH/.cache/huggingface"
export WANDB_DIR="$REPO/outputs"

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

source "$REPO/.venv/bin/activate"

echo "Python: $(which python)"
echo "HF_HOME: $HF_HOME"

mkdir -p "$OUTPUT_ROOT/logs"

# ------------------------------------------------------------------
# Launch one srun rank per GPU task.
# Each rank reads SLURM_PROCID and SLURM_LOCALID to determine
# its GPU assignment and condition subset.
# ------------------------------------------------------------------
srun --output="$OUTPUT_ROOT/logs/rank_%t.log" \
     --error="$OUTPUT_ROOT/logs/rank_%t.log" \
     python "$REPO/scripts/run_experiment_distributed.py" \
         "$REPO/configs/experiments/exp_1a_vae_comparison.yaml" \
         --data-root "$REPO/data/lakh" \
         --output-root "$OUTPUT_ROOT" \
         --sweep-strategies \
         --log-level INFO \
         --multi-node \
         --assign-strategy round-robin

echo ""
echo "All ranks complete."
echo "Merging results..."

python "$REPO/scripts/merge_sweep_results.py" \
    "$OUTPUT_ROOT" \
    --scan-rank-dirs \
    --verbose \
    --output "$OUTPUT_ROOT/sweep_summary.json"

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "Output: $OUTPUT_ROOT/sweep_summary.json"
echo "=========================================="
