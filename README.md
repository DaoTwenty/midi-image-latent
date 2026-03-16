# MIDI Image VAE

Investigating whether frozen pretrained image VAEs (Stable Diffusion, FLUX, CogView4 families) can encode MIDI piano-roll images into musically useful latent spaces.

The core idea: render MIDI bars as images, encode them through image VAEs, then decode and evaluate whether the resulting latent spaces preserve musical structure (note accuracy, harmony, rhythm, dynamics).

## Overview

- **12 pretrained VAEs** from HuggingFace diffusers (SD 1.5, SDXL, SD3, FLUX, CogView4, etc.)
- **3 channel strategies** for piano-roll rendering (velocity-only, velocity/onset split, velocity/onset/sustain)
- **8 note detection methods** for reconstructing MIDI from decoded images
- **45+ evaluation metrics** across 9 categories (reconstruction, harmony, rhythm, dynamics, information theory, latent space, conditioning, generative)
- **5 experiments** comparing VAE quality, detection methods, resolution/orientation, channel strategies, and latent structure

## Quick Start

### 1. Environment Setup

```bash
# Clone the repo
git clone <repo-url> && cd midi-image-latent

# Create virtual environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev,extra]"
```

### 2. Configure Secrets

```bash
cp .env.example .env
# Edit .env and fill in:
#   HF_TOKEN     — from https://huggingface.co/settings/tokens (required for gated models)
#   WANDB_API_KEY — from https://wandb.ai/authorize (for experiment tracking)
```

### 3. Download Data

```bash
bash scripts/download_data.sh            # all three datasets
bash scripts/download_data.sh lakh       # Lakh MIDI (~1.7 GB, 178K .mid files)
bash scripts/download_data.sh maestro    # MAESTRO v3 (~58 MB, 1276 .midi files)
bash scripts/download_data.sh pop909     # POP909 (~20 MB, 909 songs)
```

### 4. Preprocess

```bash
# Preprocess MAESTRO (piano only, classical)
python scripts/preprocess_dataset.py \
    --config configs/base.yaml configs/data/maestro.yaml \
    --data-root data/lakh \
    --output-dir outputs/cache/maestro

# Preprocess Lakh (multi-instrument, 178K songs)
python scripts/preprocess_dataset.py \
    --config configs/base.yaml configs/data/lakh.yaml \
    --data-root data/lakh \
    --output-dir outputs/cache/lakh \
    --max-files 1000

# Preprocess POP909 (pop songs, piano/melody/bridge)
python scripts/preprocess_dataset.py \
    --config configs/base.yaml configs/data/pop909.yaml \
    --data-root data/pop909 \
    --output-dir outputs/cache/pop909
```

### 5. Run an Experiment

```bash
# Quick test (2 VAEs, 20 bars, ~5 min on GPU)
python scripts/run_experiment.py \
    configs/experiments/exp_1a_vae_comparison.yaml \
    --mini \
    --data-root data/lakh

# Full run (12 VAEs, 5000 bars, ~2h on H100)
python scripts/run_experiment.py \
    configs/experiments/exp_1a_vae_comparison.yaml \
    --data-root data/lakh \
    --sweep-strategies

# Dry run (list conditions without executing)
python scripts/run_experiment.py \
    configs/experiments/exp_1a_vae_comparison.yaml \
    --dry-run
```

## Experiments

| Experiment | Config | Description |
|-----------|--------|-------------|
| **1A** | `exp_1a_vae_comparison.yaml` | Compare all 12 VAEs on reconstruction quality |
| **1B** | `exp_1b_detection_methods.yaml` | Compare 8 note detection methods on best VAE |
| **2** | `exp_2_resolution_study.yaml` | Resolution (64/128/256) and orientation (height/width) sweep |
| **3** | `exp_3_channel_strategy.yaml` | Compare velocity-only, VO-split, and VOS channel strategies |
| **4** | `exp_4_latent_analysis.yaml` | Latent space structure analysis (probes, clustering, interpolation) |
| **4B** | `exp_4b_latent_structure.yaml` | Detailed latent structure (PCA, t-SNE, instrument separability) |
| **4C** | `exp_4c_sublatent.yaml` | Sub-latent compression (PCA, MLP, sub-VAE) |
| **4D** | `exp_4d_conditioning.yaml` | Conditional generation with instrument/tempo features |
| **5** | `exp_5_sequence_generation.yaml` | Sequence-level generation with Transformer |

For a detailed breakdown of every config field, sweep mechanics, and per-experiment instructions, see **[EXPERIMENTS.md](EXPERIMENTS.md)**.

## Running on a SLURM Cluster

Pre-built SLURM scripts are in `scripts/slurm/`. They handle module loading, environment setup, and offline W&B logging.

```bash
# Submit a single experiment
sbatch scripts/slurm/exp_1a_full.sh

# Run all experiments in mini mode (smoke test)
sbatch scripts/slurm/exp_all_mini.sh

# Check job status
squeue -u $USER
```

### W&B on Compute Nodes (No Internet)

Compute nodes on HPC clusters typically have no internet access. The SLURM scripts automatically set `WANDB_MODE=offline`, which logs locally. After the job finishes, sync from a login node:

```bash
# List offline runs waiting to be synced
bash scripts/wandb_sync.sh --list

# Sync all offline runs to wandb cloud
bash scripts/wandb_sync.sh

# Sync and clean up local data
bash scripts/wandb_sync.sh --clean

# Sync a specific run
bash scripts/wandb_sync.sh outputs/wandb/offline-run-20260314_120000-abc123
```

## Project Structure

```
midi_vae/                    # Core library
  config.py                  # OmegaConf + Pydantic config system
  registry.py                # Component registry with @register decorator
  data/
    types.py                 # Data contracts (BarData, PianoRollImage, etc.)
    preprocessing.py         # MIDI ingest, bar segmentation
    rendering.py             # Channel strategies (velocity_only, vo_split, vos)
    transforms.py            # Resize, normalize, pad
    datasets.py              # Dataset classes (Lakh, MAESTRO, Pop909)
  models/
    vae_wrapper.py           # FrozenImageVAE abstract base
    vae_registry.py          # 12 concrete VAE wrappers
    sublatent/               # Sub-latent models (PCA, MLP, sub-VAE)
    sequence/                # Bar-level Transformer for generation
  note_detection/            # 8 detection methods (threshold, HMM, CNN, etc.)
  metrics/                   # 45+ metrics across 9 categories
  pipelines/                 # Pipeline stages and sweep executor
  tracking/                  # Experiment tracking, W&B integration
  visualization/             # Piano-roll plots, latent visualizations
  utils/                     # Seed, device, I/O, logging utilities
configs/
  base.yaml                 # Default config (all experiments inherit from this)
  experiments/               # Per-experiment YAML configs
  data/                      # Dataset-specific configs
  overrides/                 # Override configs for variants
tests/                       # Test suite
scripts/
  run_experiment.py          # Main entry point for running experiments
  preprocess_dataset.py      # Dataset preprocessing
  download_data.sh           # Data download script
  wandb_sync.sh              # Sync offline W&B runs to cloud
  slurm/                     # SLURM job scripts
```

## Configuration

All parameters are defined in YAML configs. The system uses OmegaConf for loading/merging and Pydantic for validation.

```yaml
# configs/base.yaml (defaults)
paths:
  data_root: data/
  output_root: outputs/

data:
  dataset: lakh
  bars_per_instrument: 5000
  target_resolution: [128, 128]

render:
  channel_strategy: velocity_only

tracking:
  wandb_enabled: true
  wandb_mode: offline
  wandb_project: midi-image-vae

seed: 42
device: cuda
```

Override via CLI:

```bash
python scripts/run_experiment.py config.yaml \
    --data-root /path/to/data \
    --output-root /path/to/outputs \
    --mini
```

Or via override YAML:

```bash
python scripts/run_experiment.py config.yaml \
    --override-config my_overrides.yaml
```

## CLI Reference

```
python scripts/run_experiment.py <config.yaml> [options]

Options:
  --mini              Run with 2 VAEs, 20 bars, 5 files (quick test)
  --dry-run           List sweep conditions without running
  --data-root PATH    Override data directory
  --output-root PATH  Override output directory
  --max-files N       Limit files loaded per instrument
  --sweep-detectors   Sweep over detection_methods list in config
  --sweep-strategies  Sweep over channel_strategies list in config
  --override-config   Merge a second YAML over the primary config
  --resume-from STAGE Resume pipeline from a specific stage
  --log-level LEVEL   DEBUG, INFO, WARNING, or ERROR
```

## Testing

```bash
# Run all tests
.venv/bin/python -m pytest tests/ -x --tb=short

# With coverage
.venv/bin/python -m pytest tests/ --cov=midi_vae --cov-report=term-missing

# Skip GPU tests (default)
.venv/bin/python -m pytest tests/ -m "not gpu"
```

## Tech Stack

- Python 3.11+, PyTorch >= 2.2, HuggingFace diffusers >= 0.28
- pypianoroll + pretty_midi for MIDI parsing
- OmegaConf + Pydantic v2 for configuration
- wandb for experiment tracking (optional)
- h5py for efficient data storage
- matplotlib + seaborn for visualization

## Architecture Principles

- **Configuration-first**: All parameters in YAML. No magic constants.
- **Registry pattern**: Swappable components via `@ComponentRegistry.register(type, name)`.
- **Pipeline composition**: Experiments are DAGs of typed stages.
- **Typed contracts**: All inter-stage data uses frozen dataclasses.
- **GPU-ready**: All code supports CPU and CUDA via configurable device.
