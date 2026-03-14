# Running Experiments

This guide explains how to run MIDI Image VAE experiments on the Fir HPC cluster.
All compute must go through the SLURM scheduler — never train on a login node.

---

## Prerequisites

### 1. Modules

The project requires the following modules.  Always start clean:

```bash
module purge
module load python/3.11 gcc/12 cuda/12.6 arrow/23.0.1
```

Save the environment so you do not have to repeat this step:

```bash
module save midi_vae_env
# Restore later with:
module restore midi_vae_env
```

### 2. Virtual environment

The project ships with a `.venv` at the repo root.  **Never** use system Python
or create a new venv.  All scripts activate `.venv` automatically.

```bash
source /scratch/triana24/midi-image-latent/.venv/bin/activate
```

### 3. HuggingFace authentication

All VAEs are fetched from HuggingFace Hub.  Your token must be set before the
first run.

Create `.env` in the repo root (one-time setup):

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

The runner script parses this file and exports `HF_TOKEN` automatically.
Compute nodes have no outbound internet; all model weights must be pre-cached.

Cache models on a login node before submitting jobs:

```bash
export HF_HOME=$SCRATCH/.cache/huggingface
source /scratch/triana24/midi-image-latent/.venv/bin/activate
python -c "
from diffusers import AutoencoderKL
AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')
"
```

Repeat for each VAE in the experiment config.  See `docs/GPU_DEFERRED_TASKS.md`
for a full list of VAEs to cache.

### 4. Data

MAESTRO v3 piano data is at:

```
/scratch/triana24/midi-image-latent/data/maestro/maestro-v3.0.0/
```

Pass this path with `--data-root` when the YAML lists `data_root: data/`.

---

## Quick start — mini experiment in 5 minutes

The `--mini` flag cuts each experiment to 20 bars per instrument, 5 files, and
the first 2 VAEs.  Use it to verify the pipeline end-to-end before submitting
full runs.

### Dry run (enumerate conditions only, no compute)

```bash
cd /scratch/triana24/midi-image-latent
source .venv/bin/activate

python scripts/run_experiment.py \
    configs/experiments/exp_1a_vae_comparison.yaml \
    --dry-run \
    --data-root data/maestro/maestro-v3.0.0
```

### Mini run (CPU, login node — quick sanity check)

```bash
python scripts/run_experiment.py \
    configs/experiments/exp_1a_vae_comparison.yaml \
    --mini \
    --data-root data/maestro/maestro-v3.0.0
```

### Mini run via SLURM (GPU, recommended)

```bash
cd /scratch/triana24/midi-image-latent
sbatch scripts/slurm/exp_1a_mini.sh
```

Monitor progress:

```bash
squeue -u $USER
tail -f outputs/logs/exp1a_mini_<JOBID>.log
```

---

## Full experiments — SLURM submission

All full-run scripts are in `scripts/slurm/`.  Submit them with `sbatch`:

| Script | Experiment | Conditions | Time |
|---|---|---|---|
| `exp_1a_mini.sh` | Exp 1A mini | 2 VAEs x 3 strategies | ~5 min |
| `exp_1a_full.sh` | Exp 1A full | 12 VAEs x 3 strategies | ~2 h |
| `exp_1b_full.sh` | Exp 1B full | 1 VAE x 8 detectors | ~30 min |
| `exp_2_full.sh` | Exp 2 full | 1 VAE x 6 render variants | ~30 min |
| `exp_3_full.sh` | Exp 3 full | 12 VAEs x 3 strategies | ~2 h |
| `exp_all_mini.sh` | All experiments (mini) | Sequential, all configs | ~30 min |

Submit a full run:

```bash
sbatch scripts/slurm/exp_1a_full.sh
```

Submit all experiments in mini mode at once (smoke test):

```bash
sbatch scripts/slurm/exp_all_mini.sh
```

---

## CLI reference — `scripts/run_experiment.py`

```
python scripts/run_experiment.py <CONFIG> [OPTIONS]

Positional:
  CONFIG                  Path to experiment YAML config

Options:
  --mini                  Mini mode: 20 bars/instrument, 5 files, 2 VAEs
  --dry-run               Print conditions without running pipelines
  --data-root PATH        Override paths.data_root in the config
  --output-root PATH      Override paths.output_root in the config
  --max-files N           Cap files per instrument (overrides --mini default)
  --resume-from STAGE     Resume pipeline from a specific stage name
  --sweep-detectors       Sweep all detection_methods from the YAML
  --sweep-strategies      Sweep all channel_strategies from the YAML
  --log-level LEVEL       DEBUG | INFO | WARNING | ERROR  (default: INFO)
```

Examples:

```bash
# Run Exp 1A with 3 channel strategies swept
python scripts/run_experiment.py \
    configs/experiments/exp_1a_vae_comparison.yaml \
    --data-root data/maestro/maestro-v3.0.0 \
    --sweep-strategies

# Run Exp 1B sweeping all 8 detection methods
python scripts/run_experiment.py \
    configs/experiments/exp_1b_detection_methods.yaml \
    --data-root data/maestro/maestro-v3.0.0 \
    --sweep-detectors

# Resume a run that was interrupted at the Detect stage
python scripts/run_experiment.py \
    configs/experiments/exp_1a_vae_comparison.yaml \
    --data-root data/maestro/maestro-v3.0.0 \
    --resume-from detect
```

---

## Experiment descriptions

### Experiment 1A — VAE Comparison
- **Goal**: Which VAE family produces the most musically useful latent space?
- **Config**: `configs/experiments/exp_1a_vae_comparison.yaml`
- **Conditions**: 12 VAEs x 3 channel strategies = 36 conditions
- **VAE families**: SD 1.x / SDXL / EQ-VAE (4 channels), SD3 / FLUX / CogView4 (16 channels)
- **Outputs**: `outputs/exp_1a/`
- **Metrics**: reconstruction, harmony, rhythm, dynamics

### Experiment 1B — Detection Method Comparison
- **Goal**: Which note detection method best recovers MIDI from reconstructed piano rolls?
- **Config**: `configs/experiments/exp_1b_detection_methods.yaml`
- **Conditions**: 1 VAE x 8 detection methods
- **Methods**: global_threshold, per_pitch_adaptive, hysteresis, velocity_aware, morphological, hmm_tracker, cnn_segmenter, gmm_detector
- **Prerequisite**: Update the VAE to the best performer from Exp 1A before running.
- **Outputs**: `outputs/exp_1b/`

### Experiment 2 — Resolution & Orientation Study
- **Goal**: Best piano-roll image layout (pitch_axis orientation x image resolution)?
- **Config**: `configs/experiments/exp_2_resolution_study.yaml`
- **Conditions**: 1 VAE x 6 variants (height/width x 64/128/256)
- **Prerequisite**: Update the VAE to the best performer from Exp 1A.
- **Outputs**: `outputs/exp_2/`

### Experiment 3 — Channel Strategy Comparison
- **Goal**: Does richer channel encoding (VOS vs velocity_only) improve reconstruction?
- **Config**: `configs/experiments/exp_3_channel_strategy.yaml`
- **Conditions**: 12 VAEs x 3 channel strategies = 36 conditions
- **Strategies**: velocity_only (1 ch), vo_split (2 ch), vos (3 ch)
- **Outputs**: `outputs/exp_3/`

### Experiment 4 — Latent Space Analysis (sub-experiments)

#### 4A — Probabilistic Encoding (`exp_4_latent_analysis.yaml`)
- **Goal**: Mean (mu) vs sample from the posterior — which gives better reconstruction?
- **Conditions**: 1 VAE x 2 latent_type variants

#### 4B — Latent Structure (`exp_4b_latent_structure.yaml`)
- **Goal**: PCA / UMAP / t-SNE visualization of the latent space; k-means clustering.
- **Conditions**: 1 VAE, analysis only (no reconstruction sweep)

#### 4C — Sub-Latent Compression (`exp_4c_sublatent.yaml`)
- **Goal**: Compress VAE latents with 4 methods x 5 target dims = 20 conditions.
- **Methods**: pca, mlp, sub_vae, umap
- **Target dims**: 16, 32, 64, 128, 256

#### 4D — Conditioned Sub-Latent (`exp_4d_conditioning.yaml`)
- **Goal**: Does conditioning on instrument/pitch/rhythm features improve compression?
- **Conditions**: 3 conditioning families

### Experiment 5 — Sequence Generation (`exp_5_sequence_generation.yaml`)
- **Goal**: Autoregressive music generation from VAE latent sequences.
- **Architecture**: Bar Transformer (256-dim, 6 layers) over latent sequences.
- **Conditions**: raw VAE latents vs compressed sub-latent (dim 64).
- **Outputs**: `outputs/exp_5/`

---

## Output structure

```
outputs/
  exp_1a/
    sweep_summary.json          # Aggregated metrics for all conditions
    cache/                      # Cached pipeline stage results
      exp_1a_vae_comparison__cond0000__sd_vae_ft_mse__velocity_only__global_threshold/
        ingest.pkl
        render.pkl
        encode.pkl
        ...
  logs/
    exp1a_mini_<JOBID>.log      # SLURM stdout/stderr
```

The `sweep_summary.json` contains:

```json
{
  "experiment": "exp_1a_vae_comparison",
  "elapsed_seconds": 3600.0,
  "num_conditions": 36,
  "conditions": ["sd_vae_ft_mse__velocity_only__global_threshold", ...],
  "metrics_summary": {
    "sd_vae_ft_mse__velocity_only__global_threshold": {
      "sd_vae_ft_mse/pixel_mse": 0.023,
      ...
    }
  }
}
```

---

## Troubleshooting

### Out of memory (OOM)

The H100 MIG slice has 10.5 GB VRAM.  If you see CUDA OOM:

1. Reduce `batch_size` in the VAE config section of the YAML (SD-family: 32 -> 16; FLUX: 16 -> 8).
2. Switch to `dtype: bfloat16` for FLUX / CogView4 models if you are using float32.
3. Enable `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` before the script.

### Gated models (403 / access denied)

Some models (FLUX.1-dev, FLUX.2-dev, CogView4) are gated on HuggingFace.  You
must accept the license on the model card page while logged in, then re-generate
your `HF_TOKEN` with the correct permissions.

```bash
huggingface-cli login --token $HF_TOKEN
```

### Models not found on compute nodes

Compute nodes have no internet.  Pre-download all weights on a login node:

```bash
export HF_HOME=$SCRATCH/.cache/huggingface
python -c "
from diffusers import AutoencoderKL
models = [
    ('stabilityai/sd-vae-ft-mse', None),
    ('stabilityai/sdxl-vae', None),
    # ... add all 12 VAEs
]
for model_id, subfolder in models:
    kw = {'subfolder': subfolder} if subfolder else {}
    AutoencoderKL.from_pretrained(model_id, **kw)
    print(f'Cached: {model_id}')
"
```

### Config validation error

If the runner fails with a Pydantic validation error, check that:

- The YAML `vaes:` list is not empty.
- `paths.data_root` points to a real directory.
- `data.target_resolution` is a list of 2 integers, e.g. `[128, 128]`.

### Resuming an interrupted run

Use `--resume-from <STAGE>` where `<STAGE>` is one of:
`ingest`, `render`, `encode`, `decode`, `detect`, `evaluate`.

The PipelineRunner loads cached outputs from earlier stages automatically.

---

## Adding a new experiment

1. Copy an existing config as a starting point:
   ```bash
   cp configs/experiments/exp_1a_vae_comparison.yaml configs/experiments/exp_6_my_experiment.yaml
   ```

2. Edit the YAML.  Fields that affect the sweep:
   - `vaes:` — list of VAEConfig dicts; each becomes a sweep axis.
   - `render.channel_strategy:` — default strategy; use `--sweep-strategies` to sweep.
   - `note_detection.method:` — default detector; use `--sweep-detectors` to sweep.
   - `tracking.experiment_name:` — update to a unique name.

3. Create a SLURM script in `scripts/slurm/`:
   ```bash
   cp scripts/slurm/exp_1a_full.sh scripts/slurm/exp_6_full.sh
   # Edit job-name, time, and the python command at the bottom.
   ```

4. Test with `--dry-run` before submitting:
   ```bash
   python scripts/run_experiment.py configs/experiments/exp_6_my_experiment.yaml --dry-run
   ```

5. Run mini first, then submit the full job:
   ```bash
   sbatch scripts/slurm/exp_6_full.sh
   ```
