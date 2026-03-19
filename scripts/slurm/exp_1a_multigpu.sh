#!/bin/bash
#SBATCH --job-name=exp1a-multigpu
#SBATCH --account=def-pasquier
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/triana24/midi-image-latent/outputs/logs/exp1a_multigpu_%j.log
#SBATCH --error=/scratch/triana24/midi-image-latent/outputs/logs/exp1a_multigpu_%j.log

# Experiment 1A — Single-node, multi-GPU job
# 12 VAEs x 3 channel strategies = 36 conditions
# 4 GPUs on one node, each task handles a ~9-condition subset (round-robin)
# Each process gets SLURM_PROCID=0..3, sets CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# Expected runtime: ~1h on 4x H100

set -euo pipefail

REPO=/scratch/triana24/midi-image-latent

echo "=========================================="
echo "Job:        $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start:      $(date)"
echo "Experiment: exp_1a_vae_comparison (multi-GPU)"
echo "Strategy:   single-node, 4 GPUs, 4 tasks"
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

echo "Python:     $(which python)"
echo "HF_HOME:    $HF_HOME"
echo "WANDB_MODE: $WANDB_MODE"
echo "Total GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo ""

# Ensure per-rank output directories exist
mkdir -p "$REPO/outputs/exp_1a/rank_0"
mkdir -p "$REPO/outputs/exp_1a/rank_1"
mkdir -p "$REPO/outputs/exp_1a/rank_2"
mkdir -p "$REPO/outputs/exp_1a/rank_3"

# srun launches one process per task (ntasks-per-node=4).
# Each process inherits SLURM_PROCID (0..3) and SLURM_LOCALID (local GPU index).
# run_experiment_distributed.py auto-detects SLURM_PROCID and SLURM_NTASKS when
# --rank and --world-size are omitted. CUDA_VISIBLE_DEVICES is set per task so
# each process sees exactly one GPU.
srun --output="$REPO/outputs/logs/exp1a_multigpu_%j_rank%t.log" \
    bash -c '
        export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
        echo "[rank $SLURM_PROCID] node=$SLURMD_NODENAME gpu=$SLURM_LOCALID"
        python scripts/run_experiment_distributed.py \
            configs/experiments/exp_1a_vae_comparison.yaml \
            --data-root data/lakh \
            --output-dir outputs/exp_1a/rank_$SLURM_PROCID \
            --log-level INFO
    '

# ------------------------------------------------------------------ Merge step
# Only rank 0 (the first task in this job) merges per-rank results.
# srun above is synchronous — all tasks have finished before we reach here.
echo ""
echo "=== Merging per-rank results (rank 0 only) ==="
python scripts/run_experiment_distributed.py \
    configs/experiments/exp_1a_vae_comparison.yaml \
    --merge-results \
    --ranks-dir "$REPO/outputs/exp_1a" \
    --num-ranks 4 \
    --output-dir "$REPO/outputs/exp_1a/merged" \
    --log-level INFO

echo ""
echo "[wandb] Offline runs saved. Sync from login node:"
echo "  bash scripts/wandb_sync.sh"
echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
