#!/bin/bash
#SBATCH --job-name=exp2-full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/triana24/midi-image-latent/outputs/logs/exp2_full_%j.log
#SBATCH --error=/scratch/triana24/midi-image-latent/outputs/logs/exp2_full_%j.log

# Experiment 2 — Resolution & Orientation Study
# 1 VAE x 6 rendering variants (2 orientations x 3 resolutions)
# Expected runtime: ~30 min on H100 MIG
#
# Each of the 6 render variants is run as a separate pipeline invocation with
# its own output directory, using per-variant override YAMLs generated at runtime.
#
# NOTE: Update the VAE in exp_2_resolution_study.yaml to the best performer
#       from Exp 1A before submitting this job.

set -euo pipefail

REPO=/scratch/triana24/midi-image-latent

echo "=========================================="
echo "Job:        $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start:      $(date)"
echo "Experiment: exp_2_resolution_study (full)"
echo "Conditions: 1 VAE x 6 render variants"
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

VARIANTS_DIR="$REPO/outputs/exp_2/tmp_variant_configs"
mkdir -p "$VARIANTS_DIR"

# Each entry: "variant_name pitch_axis resolution"
VARIANTS=(
    "height_64x64  height 64"
    "height_128x128 height 128"
    "height_256x256 height 256"
    "width_64x64   width  64"
    "width_128x128  width  128"
    "width_256x256  width  256"
)

for entry in "${VARIANTS[@]}"; do
    read -r name pitch_axis res <<< "$entry"

    override_yaml="$VARIANTS_DIR/${name}.yaml"

    # Write a minimal override YAML for this variant
    cat > "$override_yaml" << YAML
paths:
  data_root: data/lakh
  output_root: outputs/exp_2/${name}/
  cache_dir: outputs/exp_2/${name}/cache/

render:
  pitch_axis: ${pitch_axis}

data:
  target_resolution: [${res}, ${res}]

tracking:
  experiment_name: exp_2_${name}
YAML

    echo ""
    echo "--- Variant: $name (pitch_axis=$pitch_axis, resolution=${res}x${res}) ---"

    python scripts/run_experiment.py \
        configs/experiments/exp_2_resolution_study.yaml \
        --override-config "$override_yaml" \
        --log-level INFO
done

echo ""
echo "[wandb] Offline runs saved. Sync from login node:"
echo "  bash scripts/wandb_sync.sh"
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
