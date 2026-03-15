#!/bin/bash
#SBATCH --job-name=midi_vae_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/triana24/midi-image-latent/logs/gpu_session_%j.out
#SBATCH --error=/scratch/triana24/midi-image-latent/logs/gpu_session_%j.err

set -e

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "=== Start: $(date) ==="

# Load modules
module --force purge
module restore midi_vae_gpu

# Activate venv
cd /scratch/triana24/midi-image-latent
source .venv/bin/activate

# Verify GPU
nvidia-smi
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'PyTorch {torch.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
props = torch.cuda.get_device_properties(0)
print(f'VRAM: {props.total_mem / 1e9:.1f} GB')
"

echo ""
echo "=== Phase 1: Single VAE smoke test ==="
python scripts/gpu_smoke_test.py smoke

echo ""
echo "=== Phase 2: All 12 VAEs ==="
python scripts/gpu_smoke_test.py vaes

echo ""
echo "=== Phase 3: GPU integration tests ==="
python scripts/gpu_smoke_test.py tests

echo ""
echo "=== Finished: $(date) ==="
