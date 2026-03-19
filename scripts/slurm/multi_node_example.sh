#!/bin/bash
# =============================================================================
# multi_node_example.sh — Multi-node SLURM example (2 nodes x 4 GPUs = 8 ranks)
# =============================================================================
#
# This script is a fully-documented reference for multi-node distributed sweeps.
# It targets Experiment 1A (36 conditions) but the pattern applies to any config.
#
# TOPOLOGY
# --------
#   2 nodes x 4 GPUs per node = 8 GPU ranks total
#   36 conditions split round-robin across 8 ranks:
#     rank 0 -> conditions  0,  8, 16, 24, 32   (5 conditions)
#     rank 1 -> conditions  1,  9, 17, 25, 33   (5 conditions)
#     rank 2 -> conditions  2, 10, 18, 26, 34   (4 conditions)
#     rank 3 -> conditions  3, 11, 19, 27, 35   (4 conditions)
#     rank 4 -> conditions  4, 12, 20, 28        (4 conditions)
#     rank 5 -> conditions  5, 13, 21, 29        (4 conditions)
#     rank 6 -> conditions  6, 14, 22, 30        (4 conditions)
#     rank 7 -> conditions  7, 15, 23, 31        (4 conditions)
#   (--assign-strategy cost-balanced may reorder for more even wall-time)
#
# HOW SLURM DISTRIBUTES WORK
# ---------------------------
# SLURM allocates 2 nodes and launches 4 tasks on each node.
# Each task sees exactly 1 GPU via --gpus-per-task=1.
# SLURM sets these env vars inside each task:
#   SLURM_PROCID   — global rank (0..7)
#   SLURM_LOCALID  — per-node rank (0..3), used as local GPU index
#   SLURM_NTASKS   — total world size (8)
#   SLURM_NNODES   — node count (2)
# run_experiment_distributed.py reads SLURM_PROCID and SLURM_NTASKS to
# compute its condition slice, then sets CUDA_VISIBLE_DEVICES=$SLURM_LOCALID.
#
# SHARED FILESYSTEM
# -----------------
# Both nodes mount $SCRATCH over the cluster network filesystem (NFS/Lustre).
# Every rank can read the experiment config and write its rank output directory
# without any explicit MPI communication. The merge step (run on the head node
# after srun returns) simply reads all rank_XXXX/ directories from $SCRATCH.
#
# CHECKING PROGRESS
# -----------------
# 1. Check job status:
#      squeue -j $SLURM_JOB_ID
#      squeue -u $USER
#
# 2. Tail per-rank logs (replace JOBID and RANK):
#      tail -f $SCRATCH/midi-image-latent/outputs/exp_1a_multinode_JOBID/logs/rank_0.log
#      tail -f $SCRATCH/midi-image-latent/outputs/exp_1a_multinode_JOBID/logs/rank_7.log
#
# 3. Watch rank output directories appear:
#      watch -n 10 ls -lh $SCRATCH/midi-image-latent/outputs/exp_1a_multinode_JOBID/
#
# 4. Check SLURM job accounting after completion:
#      sacct -j $SLURM_JOB_ID --format=JobID,State,Elapsed,MaxRSS,MaxVMSize
#
# MANUAL MERGE (if the job failed before the merge step)
# -------------------------------------------------------
# If the job exits before merge_sweep_results.py runs, you can merge manually:
#
#   OUTPUT_ROOT="$SCRATCH/midi-image-latent/outputs/exp_1a_multinode_<JOBID>"
#   /scratch/triana24/midi-image-latent/.venv/bin/python \
#       /scratch/triana24/midi-image-latent/scripts/merge_sweep_results.py \
#       "$OUTPUT_ROOT" \
#       --scan-rank-dirs \
#       --verbose \
#       --output "$OUTPUT_ROOT/sweep_summary.json"
#
# SUBMIT THIS SCRIPT
# ------------------
#   sbatch scripts/slurm/multi_node_example.sh
#
#   # Override node count without editing this file:
#   sbatch --nodes=4 --ntasks-per-node=4 scripts/slurm/multi_node_example.sh

#SBATCH --job-name=exp1a-multinode
#SBATCH --account=def-pasquier
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/triana24/midi-image-latent/outputs/logs/exp1a_multinode_%j_head.log
#SBATCH --error=/scratch/triana24/midi-image-latent/outputs/logs/exp1a_multinode_%j_head.log

set -euo pipefail

REPO=/scratch/triana24/midi-image-latent
PYTHON="$REPO/.venv/bin/python"
CONFIG="$REPO/configs/experiments/exp_1a_vae_comparison.yaml"

# Job-specific output root (job ID in path keeps runs from clobbering each other)
OUTPUT_ROOT="$SCRATCH/midi-image-latent/outputs/exp_1a_multinode_${SLURM_JOB_ID}"

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
echo "Experiment:   exp_1a_vae_comparison (multi-node)"
echo "Topology:     ${SLURM_NNODES} nodes x ${SLURM_NTASKS_PER_NODE:-?} GPUs"
echo "Conditions:   12 VAEs x 3 channel strategies = 36"
echo "Output root:  $OUTPUT_ROOT"
echo "=========================================="

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
module purge
module load python/3.11 gcc/12 cuda/12.6 arrow/23.0.1

# Thread limits — each GPU process should use exactly its allocated CPU cores.
# Setting these BEFORE srun ensures every rank inherits the same values.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# HuggingFace model cache — shared across all nodes via $SCRATCH.
# Weights must be pre-downloaded on a login node (no outbound internet here).
export HF_HOME="$SCRATCH/.cache/huggingface"

# Wandb offline mode — sync from login node after the job completes.
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

source "$REPO/.venv/bin/activate"

echo "Python:       $(which python)"
echo "HF_HOME:      $HF_HOME"
echo "WANDB_MODE:   $WANDB_MODE"
echo ""

# ---------------------------------------------------------------------------
# Create output directories
# ---------------------------------------------------------------------------
# The head node creates the directory structure; all nodes share $SCRATCH so
# every rank can write to rank_XXXX/ subdirectories immediately.
mkdir -p "$OUTPUT_ROOT/logs"

# ---------------------------------------------------------------------------
# srun: launch one rank per GPU across all nodes
# ---------------------------------------------------------------------------
# SLURM distributes 4 tasks to each of the 2 nodes.
# Inside each task:
#   SLURM_PROCID  = global rank (0..7)
#   SLURM_LOCALID = per-node local rank (0..3) -> GPU index on that node
#   SLURM_NTASKS  = 8 (world size)
#
# run_experiment_distributed.py --multi-node reads these variables and:
#   1. Assigns this rank's condition slice via _assign_conditions()
#   2. Sets CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
#   3. Runs only its assigned conditions
#   4. Writes rank_XXXX/sweep_summary_rankXXXX.json to OUTPUT_ROOT

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
        --log-level INFO

T_SRUN=$(date +%s)
echo ""
echo "All ${SLURM_NTASKS} ranks finished in $(( T_SRUN - T_START ))s."
echo ""
echo "Rank log files:"
ls -lh "$OUTPUT_ROOT/logs/" || true

# ---------------------------------------------------------------------------
# Merge step — runs on the head node after srun returns
# ---------------------------------------------------------------------------
# srun is synchronous: all tasks across all nodes have exited before we reach
# this line. The head node reads every rank_XXXX/sweep_summary_rankXXXX.json
# and merges them into a single sweep_summary.json.
echo ""
echo "Merging per-rank results..."
"$PYTHON" "$REPO/scripts/merge_sweep_results.py" \
    "$OUTPUT_ROOT" \
    --scan-rank-dirs \
    --verbose \
    --output "$OUTPUT_ROOT/sweep_summary.json"

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
T_END=$(date +%s)
ELAPSED=$(( T_END - T_START ))

echo ""
echo "=========================================="
echo "Done:        $(date)"
echo "Elapsed:     ${ELAPSED}s (wall time including merge)"
echo "Nodes used:  $SLURM_NNODES"
echo "Ranks used:  $SLURM_NTASKS"
echo "Summary:     $OUTPUT_ROOT/sweep_summary.json"
echo "Rank logs:   $OUTPUT_ROOT/logs/rank_*.log"
echo ""
echo "Next steps:"
echo "  Inspect results:"
echo "    cat $OUTPUT_ROOT/sweep_summary.json | python -m json.tool | less"
echo ""
echo "  Sync wandb offline runs (from a login node):"
echo "    bash $REPO/scripts/wandb_sync.sh"
echo ""
echo "  Check SLURM accounting:"
echo "    sacct -j $SLURM_JOB_ID --format=JobID,State,Elapsed,MaxRSS,NCPUS,AllocGRES"
echo "=========================================="
