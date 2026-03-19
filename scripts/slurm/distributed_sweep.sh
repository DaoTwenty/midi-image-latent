#!/bin/bash
# =============================================================================
# distributed_sweep.sh — General-purpose distributed experiment sweep template
# =============================================================================
#
# Runs any experiment config across multiple GPUs (single or multi-node).
# One srun task per GPU; each task runs an independent subset of conditions.
#
# USAGE
# -----
#   sbatch [slurm-opts] scripts/slurm/distributed_sweep.sh <config.yaml> [extra-args...]
#
# The first positional argument is the path to the experiment YAML config
# (absolute or relative to $REPO). All remaining arguments are forwarded
# verbatim to run_experiment_distributed.py.
#
# RESOURCE DEFAULTS (override with sbatch flags)
# -----------------------------------------------
#   --nodes=1              one node; use --nodes=N for multi-node
#   --ntasks-per-node=4    four tasks per node == four GPUs per node
#   --gpus-per-task=1      exactly one GPU per task (required for binding)
#   --cpus-per-task=8      eight CPU cores per GPU process
#   --mem-per-cpu=4G       4 GB per core -> 32 GB per GPU process by default
#
# EXAMPLES
# --------
#   # Single node, 4 GPUs, default settings:
#   sbatch scripts/slurm/distributed_sweep.sh configs/experiments/exp_1a_vae_comparison.yaml
#
#   # Single node, 8 GPUs, cost-balanced assignment:
#   sbatch --ntasks-per-node=8 \
#          scripts/slurm/distributed_sweep.sh \
#          configs/experiments/exp_1a_vae_comparison.yaml \
#          --assign-strategy cost-balanced
#
#   # Two nodes, 4 GPUs each (8 ranks total):
#   sbatch --nodes=2 --ntasks-per-node=4 \
#          scripts/slurm/distributed_sweep.sh \
#          configs/experiments/exp_1a_vae_comparison.yaml \
#          --sweep-strategies
#
#   # Mini test with explicit VAE subset:
#   sbatch --ntasks-per-node=2 \
#          scripts/slurm/distributed_sweep.sh \
#          configs/experiments/exp_1a_vae_comparison.yaml \
#          --mini
#
# OUTPUT LAYOUT
# -------------
#   $OUTPUT_ROOT/
#     logs/rank_0.log, rank_1.log, ...   per-rank stdout/stderr
#     rank_0000/sweep_summary_rank0000.json
#     rank_0001/sweep_summary_rank0001.json
#     ...
#     sweep_summary.json                 merged results (written after srun)
#
# NOTES
# -----
# - All nodes share $SCRATCH via the cluster network filesystem.
# - No internet access on compute nodes; HF weights must be pre-cached.
# - WANDB runs in offline mode; sync from a login node after the job.
# - The merge step (merge_sweep_results.py) runs on the head node after
#   srun returns, so it executes only once regardless of node count.

#SBATCH --job-name=midi-vae-sweep
#SBATCH --account=def-pasquier
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/triana24/midi-image-latent/outputs/logs/sweep_%j_head.log
#SBATCH --error=/scratch/triana24/midi-image-latent/outputs/logs/sweep_%j_head.log

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------
CONFIG=${1:?Usage: sbatch scripts/slurm/distributed_sweep.sh <config.yaml> [extra-args...]}
shift
EXTRA_ARGS="$@"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO=/scratch/triana24/midi-image-latent
PYTHON="$REPO/.venv/bin/python"

# Derive a short experiment name from the config filename for output dir naming.
# e.g. configs/experiments/exp_1a_vae_comparison.yaml -> exp_1a_vae_comparison
EXP_NAME=$(basename "${CONFIG}" .yaml)

OUTPUT_ROOT="$SCRATCH/midi-image-latent/outputs/${EXP_NAME}_distributed"

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
echo "Config:       $CONFIG"
echo "Extra args:   ${EXTRA_ARGS:-<none>}"
echo "Output root:  $OUTPUT_ROOT"
echo "=========================================="

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
module purge
module load python/3.11 gcc/12 cuda/12.6 arrow/23.0.1

# Per-process thread limits — set BEFORE srun so each rank inherits them.
# These prevent NumPy/MKL/OpenBLAS from over-subscribing CPU cores.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# HuggingFace model cache (shared across nodes via $SCRATCH)
export HF_HOME="$SCRATCH/.cache/huggingface"

# Wandb: no outbound internet on compute nodes — use offline mode and sync later
export WANDB_MODE=offline
export WANDB_DIR="$SCRATCH/midi-image-latent/outputs"

# Load project secrets (HF_TOKEN, WANDB_API_KEY, etc.)
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

# Activate the project virtualenv
source "$REPO/.venv/bin/activate"

echo "Python:     $(which python)"
echo "HF_HOME:    $HF_HOME"
echo "WANDB_MODE: $WANDB_MODE"
echo ""

# ---------------------------------------------------------------------------
# Output directories (head node creates them; all nodes share $SCRATCH)
# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_ROOT/logs"

# ---------------------------------------------------------------------------
# Distributed sweep via srun
# ---------------------------------------------------------------------------
# Each srun rank is assigned SLURM_PROCID (global rank) and SLURM_LOCALID
# (per-node rank). run_experiment_distributed.py reads these to:
#   1. Determine total world size (SLURM_NTASKS)
#   2. Assign a round-robin condition subset to this rank
#   3. Set CUDA_VISIBLE_DEVICES=$SLURM_LOCALID (one GPU per rank)
#
# --cpus-per-task is passed explicitly so srun respects the binding even when
# the user overrides the #SBATCH default.
# --gpus-per-task=1 ensures each rank sees exactly one GPU device.
# Per-rank logs go to rank_%t.log where %t is the task index (0-based).

T_START=$(date +%s)

srun \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    --gpus-per-task=1 \
    --output="$OUTPUT_ROOT/logs/rank_%t.log" \
    --error="$OUTPUT_ROOT/logs/rank_%t.log" \
    "$PYTHON" "$REPO/scripts/run_experiment_distributed.py" \
        "$CONFIG" \
        --output-root "$OUTPUT_ROOT" \
        --multi-node \
        --assign-strategy cost-balanced \
        --log-level INFO \
        $EXTRA_ARGS

T_SRUN=$(date +%s)
echo ""
echo "All ranks complete in $(( T_SRUN - T_START ))s."

# ---------------------------------------------------------------------------
# Merge per-rank results
# ---------------------------------------------------------------------------
# srun is synchronous — all tasks across all nodes have finished before here.
# merge_sweep_results.py scans rank_XXXX/ subdirectories produced by each rank.
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
echo "Done:        $(date)"
echo "Elapsed:     ${ELAPSED}s"
echo "Summary:     $OUTPUT_ROOT/sweep_summary.json"
echo "Rank logs:   $OUTPUT_ROOT/logs/"
echo ""
echo "To sync wandb offline runs from a login node:"
echo "  bash $REPO/scripts/wandb_sync.sh"
echo "=========================================="
