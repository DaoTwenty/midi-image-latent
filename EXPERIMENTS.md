# Experiment Configuration Guide

This document explains every field in the experiment YAML configs, how the sweep system works, and how to run each experiment.

## Table of Contents

- [Config Architecture](#config-architecture)
- [Config Field Reference](#config-field-reference)
  - [paths](#paths)
  - [data](#data)
  - [render](#render)
  - [vaes](#vaes)
  - [note_detection](#note_detection)
  - [sublatent](#sublatent)
  - [metrics](#metrics)
  - [tracking](#tracking)
  - [Top-level fields](#top-level-fields)
- [Sweep System](#sweep-system)
  - [How sweeps work](#how-sweeps-work)
  - [Sweep-only keys](#sweep-only-keys)
- [Experiments](#experiments)
  - [Exp 1A: VAE Comparison](#exp-1a-vae-comparison)
  - [Exp 1B: Detection Methods](#exp-1b-detection-methods)
  - [Exp 2: Resolution & Orientation](#exp-2-resolution--orientation)
  - [Exp 3: Channel Strategy](#exp-3-channel-strategy)
  - [Exp 4: Latent Space Analysis](#exp-4-latent-space-analysis)
  - [Exp 4B: Latent Structure](#exp-4b-latent-structure)
  - [Exp 4C: Sub-Latent Compression](#exp-4c-sub-latent-compression)
  - [Exp 4D: Conditioned Sub-Latent](#exp-4d-conditioned-sub-latent)
  - [Exp 5: Sequence Generation](#exp-5-sequence-generation)
- [CLI Reference](#cli-reference)
- [Tips](#tips)

---

## Config Architecture

Every experiment is defined by a single YAML file in `configs/experiments/`. The system uses two layers:

1. **OmegaConf** loads and merges YAML files (base config + experiment config + CLI overrides)
2. **Pydantic** validates the merged result into a typed `ExperimentConfig` object

```
configs/base.yaml          <-- defaults for all experiments
configs/experiments/*.yaml  <-- per-experiment overrides
CLI flags (--mini, etc.)    <-- runtime overrides
```

The experiment YAML overrides `base.yaml` values. You don't need to repeat fields that match the defaults.

### Schema vs sweep keys

Each YAML file has two kinds of keys:

| Kind | Examples | Validated by Pydantic? | Purpose |
|------|----------|----------------------|---------|
| **Schema fields** | `render`, `note_detection`, `vaes` | Yes | Config for a single pipeline run |
| **Sweep-only keys** | `detection_methods`, `render_variants`, `channel_strategies` | No (stripped before validation) | Lists of variants the CLI iterates over |

Sweep-only keys are documentation + data for the sweep executor. They get stripped by `_build_pydantic_config()` before Pydantic ever sees them.

---

## Config Field Reference

### `paths`

Where data lives and where outputs go.

```yaml
paths:
  data_root: data/                    # Root directory for datasets
  output_root: outputs/exp_1a/        # Where results, metrics, reconstructions are saved
  cache_dir: outputs/exp_1a/cache/    # Stage-level caching (avoids re-encoding)
```

- `data_root` can be overridden with `--data-root` at the CLI
- `output_root` can be overridden with `--output-root`
- `cache_dir` stores intermediate results so re-runs skip completed stages

### `data`

Dataset selection and preprocessing parameters.

```yaml
data:
  dataset: lakh              # Which dataset: lakh | maestro | pop909
  instruments:               # Which instruments to extract from MIDI
    - drums
    - bass
    - guitar
    - piano
    - strings
  bars_per_instrument: 5000  # Max bars to collect per instrument
  min_notes_per_bar: 2       # Skip bars with fewer notes than this
  time_steps: 96             # Time resolution per bar (64 | 96 | 128)
  target_resolution:         # Image size for rendered piano rolls
    - 128
    - 128
  max_files: null            # Limit files loaded (null = no limit; set by --mini or --max-files)
```

| Field | Type | Description |
|-------|------|-------------|
| `dataset` | string | `lakh` (178K multi-instrument MIDI), `maestro` (1276 classical piano), `pop909` (909 pop songs) |
| `instruments` | list[string] | Instrument families to extract. Names map to General MIDI programs. MAESTRO only has `piano`. POP909 only has `piano`. |
| `bars_per_instrument` | int | Cap on bars per instrument. Set low for quick tests. |
| `min_notes_per_bar` | int | Minimum note count to accept a bar. Filters out silent/rest bars. |
| `time_steps` | int | Temporal resolution of the piano-roll grid. 96 = 24 ticks/beat x 4 beats. |
| `target_resolution` | [int, int] | [height, width] of the rendered image in pixels. Must be compatible with the VAE (multiples of 8). |
| `max_files` | int or null | Limit how many MIDI files are loaded. `--mini` sets this to 5. |

### `render`

How piano-roll grids are converted to 3-channel images.

```yaml
render:
  channel_strategy: velocity_only    # How to use the 3 RGB channels
  pitch_axis: height                 # Which image axis represents pitch
  normalize_range: [-1.0, 1.0]      # Pixel value range (VAEs expect [-1, 1])
  resize_method: bilinear            # Interpolation for resizing
```

| Field | Values | Description |
|-------|--------|-------------|
| `channel_strategy` | `velocity_only` | All 3 channels = velocity. Simple, baseline approach. |
| | `vo_split` | R=velocity, G=onset, B=zero. Encodes note attacks separately. |
| | `vos` | R=velocity, G=onset, B=sustain. Richest encoding, separates attack from hold. |
| `pitch_axis` | `height` / `width` | `height` = pitch on Y-axis (low notes at bottom), `width` = pitch on X-axis. |
| `normalize_range` | [float, float] | Output pixel range. Image VAEs expect [-1, 1]. |
| `resize_method` | `bilinear` / `nearest` | Interpolation when resizing piano-roll to `target_resolution`. |

### `vaes`

List of frozen pretrained VAEs to encode through. Each entry defines one VAE.

```yaml
vaes:
  - model_id: stabilityai/sd-vae-ft-mse   # HuggingFace model ID
    name: sd_vae_ft_mse                     # Short name (used in logs, output dirs, registry)
    latent_type: mean                       # What to extract from the latent distribution
    dtype: float32                          # Model precision
    batch_size: 32                          # Images per forward pass
    subfolder: null                         # HF repo subfolder (some models store VAE in 'vae/')
```

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | string | HuggingFace model identifier. Used by `AutoencoderKL.from_pretrained()`. |
| `name` | string | Short identifier. Must match the registered name in `vae_registry.py`. |
| `latent_type` | `mean` / `sample` / `both` | `mean` = use z_mu (deterministic), `sample` = reparameterized sample, `both` = store both. |
| `dtype` | `float32` / `bfloat16` | Model weight precision. SD-family uses float32, FLUX/CogView use bfloat16. |
| `batch_size` | int | Encoding batch size. 16-channel VAEs need smaller batches (16) vs 4-channel (32). |
| `subfolder` | string or null | Some HF repos store the VAE in a subfolder (e.g., `CompVis/stable-diffusion-v1-4` has it under `vae/`). |

**The 12 VAEs:**

| Name | Family | Latent Channels | Latent Shape | dtype |
|------|--------|----------------|--------------|-------|
| `sd_vae_ft_mse` | SD 1.x | 4 | (B,4,16,16) | float32 |
| `sdxl_vae` | SDXL | 4 | (B,4,16,16) | float32 |
| `eq_vae_ema` | EQ-VAE | 4 | (B,4,16,16) | float32 |
| `eq_sdxl_vae` | EQ-VAE SDXL | 4 | (B,4,16,16) | float32 |
| `sd_v1_4` | SD 1.4 | 4 | (B,4,16,16) | float32 |
| `playground_v25` | Playground | 4 | (B,4,16,16) | float32 |
| `sd3_medium` | SD3 | 16 | (B,16,16,16) | float32 |
| `flux1_dev` | FLUX.1 | 16 | (B,16,16,16) | bfloat16 |
| `flux1_kontext` | FLUX.1 Kontext | 16 | (B,16,16,16) | bfloat16 |
| `cogview4` | CogView4 | 16 | (B,16,16,16) | bfloat16 |
| `flux2_dev` | FLUX.2 | 32 | (B,32,16,16) | bfloat16 |
| `flux2_tiny` | FLUX.2 Tiny | 128 | (B,128,8,8) | bfloat16 |

### `note_detection`

How to convert decoded (reconstructed) images back into discrete MIDI notes.

```yaml
note_detection:
  method: global_threshold    # Detection algorithm name
  params:                     # Algorithm-specific parameters
    threshold: 0.5
```

**Available methods:**

| Method | Key Params | Description |
|--------|-----------|-------------|
| `global_threshold` | `threshold` (0-1) | Simple binary threshold on pixel values. |
| `per_pitch_adaptive` | `percentile` (0-100) | Per-pitch-row adaptive threshold based on percentile. |
| `hysteresis` | `threshold_on`, `threshold_off` | Two-threshold hysteresis (like Canny edge detection). |
| `velocity_aware` | `velocity_weight`, `threshold` | Weights velocity channel differently from onset. |
| `morphological` | `erosion_size`, `dilation_size` | Morphological open/close to clean up noise. |
| `hmm_tracker` | `n_states` | Hidden Markov Model for note state tracking. |
| `cnn_segmenter` | `hidden_dim` | Learned CNN-based segmentation (requires training). |
| `gmm_detector` | `n_components` | Gaussian Mixture Model for note/background separation. |

### `sublatent`

Optional sub-latent compression model that reduces the VAE latent to a smaller dimension.

```yaml
sublatent:
  enabled: false          # Whether to train/use a sub-latent model
  approach: mlp           # Compression method
  target_dim: 64          # Output dimensionality
  training:               # Training hyperparameters
    learning_rate: 0.0001
    epochs: 100
    batch_size: 64
    weight_decay: 0.00001
    pixel_weight: 1.0     # Weight for pixel reconstruction loss
    onset_weight: 5.0     # Weight for onset reconstruction loss
    kl_weight: 0.001      # Weight for KL divergence (sub_vae only)
    patience: 10          # Early stopping patience
  conditioning:           # Optional conditioning features
    features: [instrument]
    embed_dim: 32
    fusion: concat        # concat | add | film
```

| Field | Values | Description |
|-------|--------|-------------|
| `enabled` | bool | Must be `true` for sub-latent training/inference to run. |
| `approach` | `pca` / `mlp` / `sub_vae` / `umap` | `pca` = linear, `mlp` = learned nonlinear, `sub_vae` = VAE-in-VAE, `umap` = manifold. |
| `target_dim` | int | Compressed latent dimensionality (e.g., 64 vs original 1024 for 4-ch VAE). |
| `conditioning.features` | list[string] | Musical features to condition on: `instrument`, `pitch_mean`, `note_density`, `onset_rate`, `tempo`, etc. |
| `conditioning.fusion` | `concat` / `add` / `film` | How to fuse conditioning: concatenation, additive, or FiLM (feature-wise linear modulation). |

### `metrics`

Which metric categories to compute.

```yaml
metrics:
  - reconstruction    # Pixel MSE, SSIM, PSNR
  - harmony           # Chroma accuracy, key detection, interval histogram
  - rhythm            # Onset F1, timing deviation, groove similarity
  - dynamics          # Velocity MSE, dynamic range preservation
  - latent_space      # Probe accuracy, silhouette score, interpolation smoothness
  - information       # Mutual information, entropy preservation
  - conditioning      # Conditioning effectiveness metrics
  - generative        # Self-BLEU, coverage, novelty (Exp 5 only)
```

Use `- all` to enable every category. Most experiments use a subset relevant to their research question.

### `tracking`

Experiment tracking, artifact saving, and W&B integration.

```yaml
tracking:
  experiment_name: exp_1a_vae_comparison   # Unique name for this experiment
  wandb_enabled: true                       # Enable W&B logging
  wandb_project: midi-image-vae            # W&B project name
  wandb_entity: null                        # W&B team/user (null = personal default)
  wandb_mode: offline                       # online | offline | disabled
  wandb_dir: outputs/                       # Where offline run data is stored
  wandb_tags: []                            # Tags for filtering runs in W&B UI
  save_reconstructions: true                # Save decoded images to disk
  save_latents: true                        # Save latent vectors to disk
  checkpoint_every_n: 500                   # Checkpoint interval (in bars processed)
```

| Field | Description |
|-------|-------------|
| `wandb_mode` | `offline` for HPC compute nodes (no internet). Sync later with `bash scripts/wandb_sync.sh`. |
| `save_reconstructions` | Saves decoded piano-roll images for visual inspection. |
| `save_latents` | Saves z_mu / z_sigma tensors for downstream analysis (Exp 4). |
| `checkpoint_every_n` | How often to save progress. Enables `--resume-from` after interrupts. |

### Top-level fields

```yaml
seed: 42          # Global random seed (Python, NumPy, PyTorch, CUDA)
device: cuda      # cuda | cpu — falls back to cpu if CUDA unavailable
num_workers: 4    # DataLoader worker processes
```

---

## Sweep System

### How sweeps work

A single experiment config can define a **sweep** over multiple conditions. The `SweepExecutor` takes the base config and creates one pipeline run per condition, varying one axis at a time.

The default sweep axis is the `vaes` list — every VAE in the list gets its own run. Additional sweep axes are activated with CLI flags:

| CLI Flag | Sweep Axis | Iterates Over |
|----------|-----------|---------------|
| *(default)* | VAEs | Each entry in `vaes:` list |
| `--sweep-strategies` | Channel strategies | Each entry in `channel_strategies:` list |
| `--sweep-detectors` | Detection methods | Each entry in `detection_methods:` list |

When multiple axes are active, the sweep is a **Cartesian product**:
- Exp 1A with `--sweep-strategies`: 12 VAEs x 3 strategies = 36 conditions
- Exp 1B with `--sweep-detectors`: 1 VAE x 8 methods = 8 conditions

### Sweep-only keys

These keys exist only in the YAML file for the sweep system to read. They are **not** part of the Pydantic schema and get stripped before validation.

| Key | Used By | Description |
|-----|---------|-------------|
| `detection_methods` | Exp 1B | List of note detection configs to sweep over |
| `render_variants` | Exp 2 | List of render configs (resolution/orientation) to sweep over |
| `channel_strategies` | Exp 3 | List of channel strategy configs to sweep over |
| `encoding_variants` | Exp 4 | Mean vs sample encoding variants |
| `latent_analysis` | Exp 4, 4B | PCA/UMAP/t-SNE/clustering parameters |
| `sublatent_variants` | Exp 4C | Approaches x target_dims grid |
| `sublatent_base` | Exp 4D | Best sub-latent from Exp 4C (placeholder) |
| `conditioning_variants` | Exp 4D | Conditioning families to sweep over |
| `sequence_variants` | Exp 5 | Raw latents vs sub-latent input conditions |
| `transformer` | Exp 5 | Transformer architecture config |
| `generation` | Exp 5 | Generation parameters (temperature, top-k, etc.) |
| `sequence_training` | Exp 5 | Transformer training hyperparameters |

Each pipeline run receives the base schema fields (`render`, `note_detection`, etc.) with the current sweep variant's values overriding the defaults.

---

## Experiments

### Exp 1A: VAE Comparison

**The flagship experiment.** Compares all 12 VAEs on reconstruction quality across 5 instruments and 3 channel strategies.

```
Config:      configs/experiments/exp_1a_vae_comparison.yaml
Conditions:  12 VAEs x 3 channel strategies = 36 (with --sweep-strategies)
             12 VAEs x 1 strategy = 12 (without)
Pipeline:    Ingest -> Render -> Encode -> Decode -> Detect -> Evaluate
Metrics:     reconstruction, harmony, rhythm, dynamics
Runtime:     ~2h on H100 MIG (full), ~5 min (--mini)
```

```bash
# Quick test
python scripts/run_experiment.py configs/experiments/exp_1a_vae_comparison.yaml \
    --mini --data-root data/lakh

# Full run with channel strategy sweep
python scripts/run_experiment.py configs/experiments/exp_1a_vae_comparison.yaml \
    --data-root data/lakh --sweep-strategies

# SLURM
sbatch scripts/slurm/exp_1a_full.sh
```

### Exp 1B: Detection Methods

Compares all 8 note detection methods using the best VAE from Exp 1A.

```
Config:      configs/experiments/exp_1b_detection_methods.yaml
Conditions:  1 VAE x 8 detection methods = 8 (with --sweep-detectors)
Pipeline:    Ingest -> Render -> Encode -> Decode -> Detect -> Evaluate
Metrics:     reconstruction, harmony, rhythm
Depends on:  Exp 1A results (update the VAE entry with the winner)
```

```bash
# After updating the VAE to the Exp 1A winner:
python scripts/run_experiment.py configs/experiments/exp_1b_detection_methods.yaml \
    --sweep-detectors --data-root data/lakh

# SLURM
sbatch scripts/slurm/exp_1b_full.sh
```

**Note:** The `detection_methods` list in the YAML defines the 8 methods to sweep. The `note_detection` field is the default (used if you don't pass `--sweep-detectors`). Update the `vaes` entry with the best-performing VAE from Exp 1A before running.

### Exp 2: Resolution & Orientation

Tests how image resolution and pitch axis orientation affect reconstruction quality.

```
Config:      configs/experiments/exp_2_resolution_study.yaml
Conditions:  2 orientations x 3 resolutions = 6 render variants
Pipeline:    Ingest -> Render -> Encode -> Decode -> Detect -> Evaluate
Metrics:     reconstruction, harmony
Depends on:  Exp 1A results (update the VAE entry)
```

```bash
# Run each variant via override configs
python scripts/run_experiment.py configs/experiments/exp_2_resolution_study.yaml \
    --data-root data/lakh

# SLURM
sbatch scripts/slurm/exp_2_full.sh
```

**Note:** The `render_variants` list defines 6 render configurations (64x64, 128x128, 256x256 for both height and width pitch axes). The SLURM script iterates over these by generating per-variant override YAMLs.

### Exp 3: Channel Strategy

Compares the 3 channel strategies across all 12 VAEs.

```
Config:      configs/experiments/exp_3_channel_strategy.yaml
Conditions:  12 VAEs x 3 channel strategies = 36 (with --sweep-strategies)
Pipeline:    Ingest -> Render -> Encode -> Decode -> Detect -> Evaluate
Metrics:     reconstruction, harmony, rhythm, dynamics
```

```bash
python scripts/run_experiment.py configs/experiments/exp_3_channel_strategy.yaml \
    --sweep-strategies --data-root data/lakh

# SLURM
sbatch scripts/slurm/exp_3_full.sh
```

### Exp 4: Latent Space Analysis

Umbrella experiment with 4 sub-experiments analyzing the VAE latent space.

```
Config:      configs/experiments/exp_4_latent_analysis.yaml
Sub-exps:    4A (mu vs sample), 4B (manifold), 4C (sub-latent), 4D (conditioning)
Metrics:     reconstruction, latent_space, information
Depends on:  Exp 1A results (update the VAE entry)
```

**4A — Probabilistic Encoding**: Compares using z_mu (deterministic) vs z_sample (stochastic) from the latent distribution. Defined via `encoding_variants`.

**4B — Manifold Analysis**: Collects all z_mu vectors and runs PCA, UMAP, t-SNE, k-means clustering. Defined via `latent_analysis` parameters.

### Exp 4B: Latent Structure

Dedicated experiment for detailed latent space visualization and clustering.

```
Config:      configs/experiments/exp_4b_latent_structure.yaml
Pipeline:    Ingest -> Render -> Encode -> LatentAnalysis
Metrics:     latent_space, information
```

```yaml
# Key parameters
latent_analysis:
  n_pca_components: 50        # PCA dimensions before UMAP/t-SNE
  n_umap_components: 2        # Final UMAP dimensions
  umap_n_neighbors: 15        # UMAP neighborhood size
  cluster_k: 20               # k-means clusters
  colour_by:                   # Scatter plot coloring
    - instrument
    - cluster_label
    - note_density
  reduction_methods:           # Which reductions to compute
    - pca_2d
    - umap
    - tsne
```

### Exp 4C: Sub-Latent Compression

Trains compact sub-latent models that compress VAE latents while preserving musical structure.

```
Config:      configs/experiments/exp_4c_sublatent.yaml
Conditions:  4 approaches x 5 target dims = 20
Pipeline:    Ingest -> Render -> Encode -> TrainSubLatent -> SubDecode -> Detect -> Evaluate
Metrics:     reconstruction, harmony, rhythm, dynamics, latent_space, information
Runtime:     ~2-3 hours
```

```yaml
# Sweep grid
sublatent_variants:
  approaches: [pca, mlp, sub_vae, umap]
  target_dims: [16, 32, 64, 128, 256]
```

The `sublatent` schema field holds the default training config. The sweep executor overrides `approach` and `target_dim` for each condition.

### Exp 4D: Conditioned Sub-Latent

Tests whether conditioning on musical meta-features improves sub-latent quality.

```
Config:      configs/experiments/exp_4d_conditioning.yaml
Conditions:  3 conditioning families
Pipeline:    Ingest -> Render -> Encode -> TrainSubLatent(conditioned) -> SubDecode -> Detect -> Evaluate
Metrics:     reconstruction, harmony, rhythm, dynamics, latent_space, information, conditioning
Depends on:  Exp 4C results (update sublatent_base with best approach/dim)
```

```yaml
# 3 conditioning families
conditioning_variants:
  - name: instrument_conditioning
    features: [instrument]
    fusion: concat

  - name: pitch_range_conditioning
    features: [pitch_mean, pitch_std, pitch_range]
    fusion: concat

  - name: rhythm_conditioning
    features: [note_density, onset_rate, tempo]
    fusion: film          # Feature-wise Linear Modulation
```

### Exp 5: Sequence Generation

Trains a bar-level Transformer on latent sequences and generates new music.

```
Config:      configs/experiments/exp_5_sequence_generation.yaml
Conditions:  2 (raw latents vs sub-latent input)
Pipeline:    Ingest -> Render -> Encode -> PoolLatents -> TrainTransformer -> Generate -> Decode -> Detect -> Evaluate
Metrics:     reconstruction, harmony, rhythm, dynamics, generative
Runtime:     ~1-2 hours
```

```yaml
# Transformer architecture
transformer:
  d_model: 256              # Model dimension
  nhead: 8                  # Attention heads
  num_encoder_layers: 6
  num_decoder_layers: 6
  dim_feedforward: 1024
  dropout: 0.1
  max_seq_len: 64           # Max bars in a sequence
  positional_encoding: sinusoidal

# Generation parameters
generation:
  num_bars: 32              # Bars per generated sequence
  temperature: 1.0
  top_p: 0.9                # Nucleus sampling threshold
  num_samples: 100          # Sequences to generate

# Two input variants
sequence_variants:
  - name: raw_latents
    use_sublatent: false
    latent_dim: 1024        # 4ch x 16 x 16 flattened

  - name: sublatent_64
    use_sublatent: true
    sublatent_dim: 64
```

---

## CLI Reference

```
python scripts/run_experiment.py <config.yaml> [options]

Positional:
  config                Path to experiment YAML config file

Options:
  --mini                Quick test mode: 2 VAEs, 20 bars, 5 files
  --dry-run             List all sweep conditions without executing
  --data-root PATH      Override paths.data_root
  --output-root PATH    Override paths.output_root
  --max-files N         Limit files per instrument (overrides --mini default of 5)
  --override-config P   Merge a second YAML over the primary config
  --resume-from STAGE   Resume pipeline from a named stage (skip earlier stages)
  --sweep-detectors     Sweep over detection_methods list in config
  --sweep-strategies    Sweep over channel_strategies list in config
  --log-level LEVEL     DEBUG | INFO | WARNING | ERROR (default: INFO)
```

### Examples

```bash
# Dry run: see what conditions would execute
python scripts/run_experiment.py configs/experiments/exp_1a_vae_comparison.yaml --dry-run

# Mini mode with specific dataset
python scripts/run_experiment.py configs/experiments/exp_1a_vae_comparison.yaml \
    --mini --data-root data/lakh

# Full Exp 1A with strategy sweep
python scripts/run_experiment.py configs/experiments/exp_1a_vae_comparison.yaml \
    --data-root data/lakh --sweep-strategies --log-level INFO

# Resume from decode stage after a crash
python scripts/run_experiment.py configs/experiments/exp_1a_vae_comparison.yaml \
    --data-root data/lakh --resume-from decode

# Override config for a custom variant
python scripts/run_experiment.py configs/experiments/exp_2_resolution_study.yaml \
    --override-config configs/overrides/256x256.yaml
```

---

## Tips

### Experiment dependency chain

Experiments are designed to run in order. Later experiments use results from earlier ones:

```
Exp 1A (VAE comparison)
  |
  +--> Exp 1B (detection methods) — uses best VAE from 1A
  +--> Exp 2  (resolution study)  — uses best VAE from 1A
  +--> Exp 3  (channel strategy)  — independent but benefits from 1A insights
  +--> Exp 4  (latent analysis)   — uses best VAE from 1A
         |
         +--> Exp 4C (sub-latent)    — uses best VAE from 1A
                |
                +--> Exp 4D (conditioning) — uses best sub-latent from 4C
                +--> Exp 5  (generation)   — uses best sub-latent from 4C
```

After Exp 1A, update the `vaes` entry in downstream configs with the winning VAE.

### Mini mode for smoke testing

Always run `--mini` first to verify the pipeline works end-to-end before submitting a full SLURM job:

```bash
python scripts/run_experiment.py configs/experiments/exp_1a_vae_comparison.yaml \
    --mini --data-root data/lakh
```

Mini mode: 2 VAEs, 20 bars/instrument, 5 files max. Takes ~5 minutes on GPU.

### GPU memory management

- The `SweepExecutor` calls `gc.collect()` + `torch.cuda.empty_cache()` between every condition
- `EncodeStage` passes loaded VAE instances to `DecodeStage` via context, avoiding loading the same multi-GB model twice per condition
- If you hit OOM, reduce `batch_size` in the VAE config (try 8 or 4)
- 16-channel VAEs (FLUX, SD3, CogView4) use more VRAM than 4-channel ones — they default to `batch_size: 16`

### W&B on HPC clusters

Compute nodes typically have no internet. The workflow is:

1. SLURM scripts set `WANDB_MODE=offline` automatically
2. Runs are logged to local files in `outputs/wandb/`
3. After the job finishes, sync from a login node: `bash scripts/wandb_sync.sh`

### Caching

Pipeline stages cache their outputs in `cache_dir`. If you change the config and want a fresh run, delete the cache:

```bash
rm -rf outputs/exp_1a/cache/
```

---

## Changing the Dataset for an Experiment

The experiment configs default to `dataset: lakh`. To run an experiment on a different dataset (e.g., MAESTRO or POP909), you need to change **3 things** that must stay in sync: the `dataset` name, the `--data-root` path, and the `instruments` list.

### What you need to change

| # | What | Where | Why |
|---|------|-------|-----|
| 1 | `data.dataset` | Experiment YAML | Controls the file glob pattern (`lakh` = `**/*.mid`, `maestro` = `**/*.midi`, `pop909` = `**/*.mid`) |
| 2 | `--data-root` | CLI or SLURM script | Points the pipeline at the actual directory containing the MIDI files |
| 3 | `data.instruments` | Experiment YAML | Must match what the dataset actually contains (MAESTRO/POP909 = piano only) |

### Dataset quick reference

| Dataset | `dataset` value | `--data-root` path | Available instruments | File extension | Notes |
|---------|----------------|-------------------|----------------------|---------------|-------|
| Lakh MIDI | `lakh` | `data/lakh` | drums, bass, guitar, piano, strings | `.mid` | 178K multi-instrument files |
| MAESTRO v3 | `maestro` | `data/maestro/maestro-v3.0.0` | piano | `.midi` | 1276 classical piano performances |
| POP909 | `pop909` | `data/pop909` | piano | `.mid` | 909 pop songs, mostly 2/4 time |

### Step by step

**Option A: Edit the experiment YAML directly**

1. Open the experiment config (e.g., `configs/experiments/exp_1a_vae_comparison.yaml`)

2. Change the `data` section:

   ```yaml
   # Before (Lakh)
   data:
     dataset: lakh
     instruments:
       - drums
       - bass
       - guitar
       - piano
       - strings

   # After (MAESTRO)
   data:
     dataset: maestro
     instruments:
       - piano
   ```

3. Update the `--data-root` in your run command or SLURM script:

   ```bash
   # Before
   python scripts/run_experiment.py configs/experiments/exp_1a_vae_comparison.yaml \
       --data-root data/lakh

   # After
   python scripts/run_experiment.py configs/experiments/exp_1a_vae_comparison.yaml \
       --data-root data/maestro/maestro-v3.0.0
   ```

4. Clear the cache (old dataset's cached bars would be stale):

   ```bash
   rm -rf outputs/exp_1a/cache/
   ```

**Option B: Use an override config (recommended — keeps the original clean)**

1. Create an override YAML, e.g., `configs/overrides/use_maestro.yaml`:

   ```yaml
   data:
     dataset: maestro
     instruments:
       - piano
   paths:
     data_root: data/maestro/maestro-v3.0.0
   ```

2. Run with `--override-config`:

   ```bash
   python scripts/run_experiment.py configs/experiments/exp_1a_vae_comparison.yaml \
       --override-config configs/overrides/use_maestro.yaml
   ```

   The override YAML merges on top of the experiment config, so you only specify what changes.

3. You can also combine with `--data-root` (CLI flags override YAML):

   ```bash
   python scripts/run_experiment.py configs/experiments/exp_1a_vae_comparison.yaml \
       --override-config configs/overrides/use_maestro.yaml \
       --data-root /some/other/path
   ```

**Option C: Use the pre-made dataset configs**

The `configs/data/` directory has ready-made configs for each dataset. You can use them as override configs directly:

```bash
# Run Exp 1A on MAESTRO
python scripts/run_experiment.py configs/experiments/exp_1a_vae_comparison.yaml \
    --override-config configs/data/maestro.yaml \
    --data-root data/maestro/maestro-v3.0.0

# Run Exp 1A on POP909
python scripts/run_experiment.py configs/experiments/exp_1a_vae_comparison.yaml \
    --override-config configs/data/pop909.yaml \
    --data-root data/pop909
```

Note: `--data-root` still needed because `configs/data/*.yaml` don't set `paths.data_root`.

### Common mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Changed `--data-root` but not `dataset` | Wrong glob pattern, finds 0 files (e.g., searching for `*.mid` in MAESTRO which has `*.midi`) | Set `data.dataset` to match the actual dataset |
| Changed `dataset` but not `instruments` | Config asks for drums/bass/guitar but MAESTRO only has piano — gets 0 bars for 4/5 instruments | Set `instruments` to only what the dataset contains |
| Changed dataset but didn't clear cache | Pipeline loads stale cached bars from the old dataset | `rm -rf outputs/<exp>/cache/` |
| Forgot `--data-root` | Pipeline looks in default `data/` with recursive glob — may accidentally find files from multiple datasets | Always pass `--data-root` pointing to the specific dataset directory |

### For SLURM scripts

If you want to change the dataset for a SLURM job, edit both the `--data-root` line in the script and the experiment YAML (or add `--override-config`):

```bash
# In scripts/slurm/exp_1a_full.sh, change:
python scripts/run_experiment.py \
    configs/experiments/exp_1a_vae_comparison.yaml \
    --override-config configs/data/maestro.yaml \
    --data-root data/maestro/maestro-v3.0.0 \
    --sweep-strategies \
    --log-level INFO
```
