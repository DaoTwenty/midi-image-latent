#!/bin/bash
#SBATCH --job-name=exp1a-full
#SBATCH --account=def-pasquier
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/triana24/midi-image-latent/outputs/logs/exp1a_full_%j.log
#SBATCH --error=/scratch/triana24/midi-image-latent/outputs/logs/exp1a_full_%j.log

# Experiment 1A — Full VAE Comparison (single-GPU, sequential)
# 12 VAEs x 3 channel strategies = 36 conditions
# Expected runtime: ~2h on H100 MIG (10.5 GB VRAM)
#
# Multi-GPU alternatives (all use scripts/run_experiment_distributed.py):
#
#   Single-node, 4 GPUs — ~4x speedup, requires 1 node with 4x H100:
#     sbatch scripts/slurm/exp_1a_multigpu.sh
#     Each srun task gets SLURM_PROCID=0..3 and CUDA_VISIBLE_DEVICES=$SLURM_LOCALID.
#     Conditions distributed round-robin across tasks.
#
#   Multi-node, 2x2 GPUs — 4 GPUs across 2 nodes, same round-robin distribution:
#     sbatch scripts/slurm/exp_1a_multinode.sh
#     SLURM_PROCID is the global rank; SLURM_LOCALID is the per-node GPU index.
#
#   Job array — one SLURM job per condition (36 jobs total), most scheduler-friendly:
#     sbatch scripts/slurm/exp_1a_jobarray.sh
#     Each array task receives --condition-indices $SLURM_ARRAY_TASK_ID.
#     Best for heterogeneous GPU availability or partial reruns.
#     Merge results after completion:
#       sbatch --dependency=afterok:<ARRAY_JOB_ID> scripts/slurm/exp_1a_array_merge.sh

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

export WANDB_MODE=offline
export WANDB_DIR="$REPO/outputs"

source "$REPO/.venv/bin/activate"

cd "$REPO"

echo "Python: $(which python)"
echo "HF_HOME: $HF_HOME"
echo "CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count(), "GPU(s)")')"

python scripts/run_experiment.py \
    configs/experiments/exp_1a_vae_comparison.yaml \
    --sweep-strategies \
    --log-level INFO

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
