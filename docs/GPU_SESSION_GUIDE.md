# GPU Session Guide

Comprehensive reference for bringing this project onto GPU hardware. Read this at the start of the GPU session.

---

## Pre-Flight Checklist

### 1. Verify GPU Environment
```bash
# Check CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Verify .venv has CUDA-enabled PyTorch (CPU-only torch was installed during development)
.venv/bin/python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

**If torch reports `CUDA: False`**, reinstall with CUDA support:
```bash
.venv/bin/pip install torch>=2.2 --index-url https://download.pytorch.org/whl/cu121
# Or cu118, cu124 depending on your driver version
```

### 2. HuggingFace Auth
Several models require accepting license agreements on HuggingFace:
- `CompVis/stable-diffusion-v1-4` (community license)
- `stabilityai/stable-diffusion-3-medium-diffusers` (gated)
- `black-forest-labs/FLUX.1-dev` (gated)
- `black-forest-labs/FLUX.1-Kontext-dev` (gated)
- `black-forest-labs/FLUX.2-dev` (gated)
- `THUDM/CogView4-6B` (may require agreement)

```bash
# Login once
huggingface-cli login
# Accept licenses at https://huggingface.co/<model_id> for each gated model
```

### 3. Disk Space for Model Weights
Each VAE downloads ~300MB-2GB of weights. The 12 models combined need roughly **10-15 GB** in `~/.cache/huggingface/`. The 16-channel models (FLUX, SD3, CogView4) are larger.

### 4. Verify Codebase is Clean
```bash
.venv/bin/python -m pytest tests/ -x --tb=short   # Should show 517 passed, 4 skipped
git status                                          # Should be clean on main
```

---

## VRAM Budget by Model

All VAEs are encoder-only at inference (frozen, no gradients). Approximate VRAM:

| Model | Latent Ch | dtype | Approx VRAM | batch_size=32 safe? |
|-------|-----------|-------|-------------|---------------------|
| `sd_vae_ft_mse` | 4 | fp32 | ~300 MB | Yes (even on 8GB) |
| `sdxl_vae` | 4 | fp32 | ~300 MB | Yes |
| `eq_vae_ema` | 4 | fp32 | ~300 MB | Yes |
| `eq_sdxl_vae` | 4 | fp32 | ~300 MB | Yes |
| `sd_v1_4` | 4 | fp32 | ~300 MB | Yes |
| `playground_v25` | 4 | fp32 | ~300 MB | Yes |
| `sd3_medium` | 16 | fp32 | ~800 MB | Yes on 16GB+ |
| `flux1_dev` | 16 | bf16 | ~500 MB | Yes on 16GB+ |
| `flux1_kontext` | 16 | bf16 | ~500 MB | Yes on 16GB+ |
| `flux2_dev` | 16 | bf16 | ~500 MB | Yes on 16GB+ |
| `flux2_tiny` | 16 | bf16 | ~200 MB | Yes |
| `cogview4` | 16 | bf16 | ~500 MB | Yes on 16GB+ |

Input images are 128x128x3 (tiny by image VAE standards), so batch memory is minimal. The bottleneck is model weight loading — only one VAE needs to be in VRAM at a time.

**If you have < 16 GB VRAM**: Reduce `batch_size` to 16 for 16-channel models in the experiment YAML configs. The SweepExecutor processes one VAE at a time, so it will never load two simultaneously.

**If you have < 8 GB VRAM**: Use `dtype: bfloat16` for all models (not just FLUX/CogView4). Edit `configs/experiments/exp_1a_vae_comparison.yaml` to set `dtype: bfloat16` on SD-family models too.

---

## Execution Order (Recommended)

Work through these phases sequentially. Each phase validates the previous one.

### Phase 1: Smoke-Test a Single VAE (~10 min)

Goal: Confirm one VAE loads, encodes, and decodes correctly.

```python
# Quick smoke test script
import torch
from midi_vae.registry import ComponentRegistry
from midi_vae.config import VAEConfig

# Import to trigger registration
import midi_vae.models.vae_registry

cfg = VAEConfig(
    model_id="stabilityai/sd-vae-ft-mse",
    name="sd_vae_ft_mse",
    latent_type="mean",
    dtype="float32",
    batch_size=4,
)

vae = ComponentRegistry.get("vae", "sd_vae_ft_mse")(config=cfg, device="cuda")
vae.load_model()

# Synthetic piano-roll-shaped input (128x128 image, 3 channels, normalized to [-1,1])
x = torch.randn(2, 3, 128, 128).cuda()
z_mu, z_sigma = vae.encode(x)
print(f"Latent shape: {z_mu.shape}")  # Expect (2, 4, 16, 16)

recon = vae.decode(z_mu)
print(f"Recon shape: {recon.shape}")  # Expect (2, 3, 128, 128)
print(f"Recon range: [{recon.min():.2f}, {recon.max():.2f}]")
```

**What to verify:**
- Latent spatial dims = input_size / 8 (so 128/8 = 16 -> shape (B, C, 16, 16))
- 4-channel models produce `z_mu` of shape (B, 4, 16, 16)
- 16-channel models produce `z_mu` of shape (B, 16, 16, 16)
- Reconstruction has same spatial shape as input
- No NaN or Inf values

### Phase 2: Test All 12 VAEs (~30-60 min)

Load each VAE one at a time, verify shapes. Use `torch.cuda.empty_cache()` between models:

```python
import gc, torch

VAES_TO_TEST = [
    ("sd_vae_ft_mse", "stabilityai/sd-vae-ft-mse", "float32", None, 4),
    ("sdxl_vae", "stabilityai/sdxl-vae", "float32", None, 4),
    ("eq_vae_ema", "zelaki/eq-vae-ema", "float32", None, 4),
    ("eq_sdxl_vae", "KBlueLeaf/EQ-SDXL-VAE", "float32", None, 4),
    ("sd_v1_4", "CompVis/stable-diffusion-v1-4", "float32", "vae", 4),
    ("playground_v25", "playgroundai/playground-v2.5-1024px-aesthetic", "float32", "vae", 4),
    ("sd3_medium", "stabilityai/stable-diffusion-3-medium-diffusers", "float32", "vae", 16),
    ("flux1_dev", "black-forest-labs/FLUX.1-dev", "bfloat16", "vae", 16),
    ("flux1_kontext", "black-forest-labs/FLUX.1-Kontext-dev", "bfloat16", "vae", 16),
    ("flux2_dev", "black-forest-labs/FLUX.2-dev", "bfloat16", "vae", 16),
    ("flux2_tiny", "fal/FLUX.2-Tiny-AutoEncoder", "bfloat16", None, 16),
    ("cogview4", "THUDM/CogView4-6B", "bfloat16", "vae", 16),
]

# Test each, free memory between
for name, model_id, dtype, subfolder, expected_ch in VAES_TO_TEST:
    print(f"\n--- Testing {name} ---")
    try:
        cfg = VAEConfig(model_id=model_id, name=name, dtype=dtype, batch_size=2)
        vae = ComponentRegistry.get("vae", name)(config=cfg, device="cuda")
        vae.load_model()
        x = torch.randn(1, 3, 128, 128).cuda()
        z_mu, z_sigma = vae.encode(x)
        recon = vae.decode(z_mu)
        assert z_mu.shape[1] == expected_ch, f"Expected {expected_ch} channels, got {z_mu.shape[1]}"
        print(f"  OK: latent={z_mu.shape}, recon={recon.shape}")
    except Exception as e:
        print(f"  FAILED: {e}")
    finally:
        del vae
        gc.collect()
        torch.cuda.empty_cache()
```

**Known issues to watch for:**
- `flux2_tiny` uses `AutoModel` (transformers) not `AutoencoderKL` (diffusers) — the encode API may differ. If it fails, the `Flux2Tiny` wrapper may need adjustment.
- `cogview4` is a large model; if VRAM is tight, try `batch_size=1`.
- Some gated models will raise `401 Unauthorized` if you haven't accepted their HF license.

### Phase 3: Run the GPU Integration Test

```bash
.venv/bin/python -m pytest tests/test_integration_encode_decode.py -m gpu -v --tb=long
```

This test (at line 409 of the file) runs the full pipeline with a real VAE. It may need the smallest VAE (`sd_vae_ft_mse`) to be available.

### Phase 4: Run Experiment 1A (The Big One)

This is the core experiment: 12 VAEs x 5 instruments x 3 channel strategies.

```bash
# Start with a small subset first
.venv/bin/python -m midi_vae.pipelines.sweep configs/experiments/exp_1a_vae_comparison.yaml
```

**But first, you need data.** Download with the provided script:
```bash
bash scripts/download_data.sh          # all three datasets
bash scripts/download_data.sh maestro  # just MAESTRO v3
bash scripts/download_data.sh lakh     # just Lakh MIDI (~1.7 GB)
bash scripts/download_data.sh pop909   # just POP909 (~20 MB)
```

**Tip**: For initial validation, use the ``--mini`` flag with ``run_experiment.py``
to test with 2 VAEs and 20 bars before running the full dataset.

### Phase 5: Remaining Experiments (Order of Dependencies)

```
Exp 1A (VAE comparison)          -- finds best VAE
  |
  +-- Exp 1B (detection methods) -- uses best VAE from 1A
  +-- Exp 2 (resolution study)   -- uses best VAE from 1A
  +-- Exp 3 (channel strategy)   -- uses all VAEs
  |
  +-- Exp 4A (latent analysis)   -- existing config: exp_4_latent_analysis.yaml
  +-- Exp 4B (latent structure)  -- PCA/t-SNE/clustering on latents
  +-- Exp 4C (sub-latent)        -- trains sub-latent models
  |     |
  |     +-- Exp 4D (conditioning) -- adds conditioning to best sub-latent
  |
  +-- Exp 5 (sequence generation) -- trains bar Transformer
```

---

## Code Touchpoints When Moving to GPU

### 1. Update `CLAUDE.md` GPU Section
Change "CPU-ONLY" to active GPU. Remove the constraint about not downloading weights.

### 2. Config `device` Field
All experiment YAMLs already have `device: cuda`. The `get_device()` utility in `midi_vae/utils/device.py` auto-falls back to CPU if CUDA is unavailable, so no code changes needed.

### 3. The `Flux2Tiny` Wrapper (Potential Fix Needed)
This wrapper uses `transformers.AutoModel` instead of `diffusers.AutoencoderKL`. The encode/decode API may differ from the other 11 VAEs. If it fails:
- Check if `self._model.encode(chunk).latent_dist` exists
- The model may use a different API like `self._model.encode(chunk)` returning a tensor directly
- Fix in `midi_vae/models/vae_registry.py` lines 430-481

### 4. Batch Size Tuning
The experiment configs set `batch_size: 32` for 4-channel VAEs and `batch_size: 16` for 16-channel VAEs. These are conservative for 128x128 inputs. You can likely increase them on 24GB+ GPUs, but the speedup is minimal since encoding is already fast for small images.

---

## Known Issues to Fix During GPU Session

### 1. sklearn `multi_class` Deprecation (FIXED)
Already patched in `midi_vae/metrics/conditioning.py` with try/except fallback.

### 2. Missing `__main__` Entry Point
There's no `__main__.py` in `midi_vae/` — you may need to create a CLI entry point:
```python
# midi_vae/__main__.py or scripts/run_experiment.py
from midi_vae.pipelines.sweep import SweepExecutor
from midi_vae.config import load_config
import sys

cfg = load_config(sys.argv[1])
executor = SweepExecutor(cfg)
results = executor.run()
```

### 3. HDF5 Caching
The pipeline caching uses pickle by default. For large latent tensors (5000 bars x 12 VAEs), consider enabling HDF5 storage:
- `h5py` is already in dependencies
- The `midi_vae/utils/io.py` module likely has helpers for this

### 4. Memory Management Between VAEs
The SweepExecutor should explicitly call `del vae; gc.collect(); torch.cuda.empty_cache()` between VAE conditions. Check `midi_vae/pipelines/sweep.py` to verify this happens — if not, it needs adding.

---

## Timing Estimates

For 5000 bars/instrument x 5 instruments = 25,000 images at 128x128:

| Operation | Per-VAE Time (est.) | Notes |
|-----------|-------------------|-------|
| Model download (first time) | 1-10 min | Cached after first download |
| Encode 25k images | 2-5 min | batch_size=32 on A100/4090 |
| Decode 25k latents | 2-5 min | Same |
| Note detection (all methods) | ~1 min | CPU-bound, fast |
| Metrics computation | ~2 min | Mostly numpy, some GPU |
| **Full Exp 1A (12 VAEs)** | **45-90 min** | Serial VAE processing |

Sub-latent training (Exp 4C): ~30 min per approach x 5 dims = 2-3 hours total.
Transformer training (Exp 5): ~1-2 hours depending on sequence length and dataset size.

---

## Quick Command Reference

```bash
# Run all tests including GPU
.venv/bin/python -m pytest tests/ -m "gpu or not gpu" -v --tb=long

# Run just GPU tests
.venv/bin/python -m pytest tests/ -m gpu -v --tb=long

# Run a specific experiment
.venv/bin/python -c "
from midi_vae.config import load_config
from midi_vae.pipelines.sweep import SweepExecutor
cfg = load_config('configs/experiments/exp_1a_vae_comparison.yaml')
results = SweepExecutor(cfg).run()
"

# Monitor GPU during run
watch -n 1 nvidia-smi

# Check disk usage of model cache
du -sh ~/.cache/huggingface/hub/
```
