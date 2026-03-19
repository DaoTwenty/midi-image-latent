#!/bin/bash

#SBATCH --job-name=exp1a-dist
#SBATCH --account=def-pasquier
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/triana24/midi-image-latent/outputs/logs/exp1a_dist_%j_head.log
#SBATCH --error=/scratch/triana24/midi-image-latent/outputs/logs/exp1a_dist_%j_head.log

set -euo pipefail

REPO=/scratch/triana24/midi-image-latent
PYTHON="$REPO/.venv/bin/python"
CONFIG="$REPO/configs/experiments/exp_1a_vae_comparison.yaml"

# Pass any extra CLI args (e.g. --mini) through to the distributed runner.
EXTRA_ARGS="$@"

# ---------------------------------------------------------------------------
# Job header
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Head node:    $SLURMD_NODENAME"
echo "Nodes:        $SLURM_NNODES"
echo "Tasks total:  $SLURM_NTASKS (${SLURM_NTASKS_PER_NODE:-?} per node)"
echo "CPUs/task:    ${SLURM_CPUS_PER_TASK:-8}"
echo "Start:        $(date)"
echo "Experiment:   exp_1a_vae_comparison (distributed)"
echo "Conditions:   12 VAEs x 3 channel strategies = 36"
echo "Config:       $CONFIG"
echo "Extra args:   ${EXTRA_ARGS:-<none>}"
echo "=========================================="

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
module purge
module load python/3.11 gcc/12 cuda/12.6 arrow/23.0.1

# Thread limits — prevent over-subscription of CPU cores per rank.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

export HF_HOME="$SCRATCH/.cache/huggingface"
export WANDB_MODE=offline
export WANDB_DIR="$SCRATCH/midi-image-latent/outputs"

# Load project secrets
if [[ -f "$REPO/.env" ]]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"
        [[ -n "$key" ]] && export "$key"="$value"
    done < <(grep -v '^#' "$REPO/.env" | grep '=')
fi

source "$REPO/.venv/bin/activate"

echo "Python:     $(which python)"
echo "HF_HOME:    $HF_HOME"
echo ""

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
OUTPUT_ROOT="$SCRATCH/midi-image-latent/outputs/exp_1a_vae_comparison_distributed"
mkdir -p "$OUTPUT_ROOT/logs"

# ---------------------------------------------------------------------------
# Distributed sweep
# ---------------------------------------------------------------------------
T_START=$(date +%s)

srun \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    --gpus-per-task=1 \
    --output="$OUTPUT_ROOT/logs/rank_%t.log" \
    --error="$OUTPUT_ROOT/logs/rank_%t.log" \
    "$PYTHON" "$REPO/scripts/run_experiment_distributed.py" \
        "$CONFIG" \
        --output-root "$OUTPUT_ROOT" \
        --sweep-strategies \
        --multi-node \
        --assign-strategy cost-balanced \
        --log-level INFO \
        $EXTRA_ARGS

T_SRUN=$(date +%s)
echo ""
echo "All ranks complete in $(( T_SRUN - T_START ))s."

# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
echo ""
echo "Merging per-rank results..."
"$PYTHON" "$REPO/scripts/merge_sweep_results.py" \
    "$OUTPUT_ROOT" \
    --scan-rank-dirs \
    --verbose \
    --output "$OUTPUT_ROOT/sweep_summary.json"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
T_END=$(date +%s)
ELAPSED=$(( T_END - T_START ))

echo ""
echo "=========================================="
echo "Done:       $(date)"
echo "Elapsed:    ${ELAPSED}s"
echo "Summary:    $OUTPUT_ROOT/sweep_summary.json"
echo "Rank logs:  $OUTPUT_ROOT/logs/"
echo ""
echo "To sync wandb offline runs from a login node:"
echo "  bash $REPO/scripts/wandb_sync.sh"
echo "=========================================="
