#!/bin/bash
# GPU session setup script
# Source this after getting an interactive GPU allocation:
#   salloc --partition=gpubase_interac --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=4 --mem=16G --time=03:00:00
#   source scripts/gpu_session.sh

set -e

echo "=== Loading modules ==="
module restore midi_vae_gpu

echo "=== Activating venv ==="
source /scratch/triana24/midi-image-latent/.venv/bin/activate

echo "=== Verifying GPU ==="
nvidia-smi
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'PyTorch {torch.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo ""
echo "=== Environment ready! ==="
echo "Run GPU tasks with:"
echo "  python -m pytest tests/ -m gpu -v --tb=long"
echo "  python scripts/gpu_smoke_test.py"
