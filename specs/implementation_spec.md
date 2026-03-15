# Implementation Specification — Quick Reference

> Full document: `midi_image_vae_implementation_spec.docx`
> This is a condensed reference for Claude Code agents. See the full doc for detailed tables and rationale.

## Config Schema (Section 3)

```python
class PathsConfig(BaseModel):
    data_root: str
    output_root: str = 'outputs'
    cache_dir: str = 'outputs/cache'

class DataConfig(BaseModel):
    dataset: str = 'lakh'           # lakh | pop909 | maestro
    instruments: list[str] = ['drums','bass','guitar','piano','strings']
    bars_per_instrument: int = 5000
    min_notes_per_bar: int = 2
    time_steps: int = 96            # 64 | 96 | 128
    target_resolution: tuple[int,int] = (128, 128)

class RenderConfig(BaseModel):
    channel_strategy: str = 'velocity_only'  # velocity_only | vo_split | vos
    pitch_axis: str = 'height'               # height | width
    normalize_range: tuple[float,float] = (-1.0, 1.0)
    resize_method: str = 'bilinear'

class VAEConfig(BaseModel):
    model_id: str                    # HuggingFace model ID
    name: str                        # Short name for logging
    latent_type: str = 'mean'        # mean | sample | both
    dtype: str = 'float32'           # float32 | bfloat16
    batch_size: int = 32

class NoteDetectionConfig(BaseModel):
    method: str = 'global_threshold'
    params: dict = {}

class SubLatentConfig(BaseModel):
    enabled: bool = False
    approach: str = 'mlp'            # pca | umap | mlp | sub_vae
    target_dim: int = 64
    training: TrainingConfig = TrainingConfig()
    conditioning: ConditioningConfig | None = None

class TrackingConfig(BaseModel):
    experiment_name: str
    wandb_enabled: bool = False
    wandb_project: str = 'midi-image-vae'
    save_reconstructions: bool = True
    save_latents: bool = True
    checkpoint_every_n: int = 500

class ExperimentConfig(BaseModel):     # TOP-LEVEL
    paths: PathsConfig
    data: DataConfig
    render: RenderConfig
    vaes: list[VAEConfig]
    note_detection: NoteDetectionConfig
    sublatent: SubLatentConfig
    metrics: list[str] = ['all']
    tracking: TrackingConfig
    seed: int = 42
    device: str = 'cuda'
    num_workers: int = 4
```

## Data Types (Section 4.2)

```python
@dataclass(frozen=True)
class BarData:
    bar_id: str                    # {song_id}_{track}_{bar_num}
    song_id: str
    instrument: str                # drums | bass | guitar | piano | strings
    program_number: int
    piano_roll: np.ndarray         # (128, T) velocity matrix
    onset_mask: np.ndarray         # (128, T) binary onset positions
    sustain_mask: np.ndarray       # (128, T) binary sustain positions
    tempo: float
    time_signature: tuple[int,int]
    metadata: dict

@dataclass(frozen=True)
class PianoRollImage:
    bar_id: str
    image: torch.Tensor            # (3, H, W) normalized to [-1, 1]
    channel_strategy: str
    resolution: tuple[int, int]
    pitch_axis: str

@dataclass(frozen=True)
class LatentEncoding:
    bar_id: str
    vae_name: str
    z_mu: torch.Tensor             # (C, H_lat, W_lat)
    z_sigma: torch.Tensor          # (C, H_lat, W_lat)
    z_sample: torch.Tensor | None

@dataclass(frozen=True)
class ReconstructedBar:
    bar_id: str
    vae_name: str
    recon_image: torch.Tensor      # (3, H, W) continuous-valued
    detected_notes: list[MidiNote]
    detection_method: str

@dataclass(frozen=True)
class MidiNote:
    pitch: int       # 0-127
    onset_step: int
    offset_step: int
    velocity: int    # 0-127
```

## ABCs (Section 4.3)

| ABC | Key Methods | Responsibility |
|-----|-------------|----------------|
| `ChannelStrategy` | `render(bar: BarData) → Tensor(3,H,W)` | Piano-roll matrix → 3-channel image |
| `FrozenImageVAE` | `encode(img) → (mu, sigma)`, `decode(z) → img` | Frozen HF VAE wrapper |
| `NoteDetector` | `detect(recon_image, channel_strategy) → list[MidiNote]`, `fit(validation_bars)` | Continuous image → MIDI notes |
| `Metric` | `compute(gt: BarData, recon: ReconstructedBar) → dict[str, float]` | Evaluate reconstruction quality |
| `SubLatentModel` | `encode(z_mu) → s`, `decode(s) → z_hat`, `train_step(batch)` | Compact sub-latent projection |

## Channel Strategies (Section 5.2)

| Name | R channel | G channel | B channel |
|------|-----------|-----------|-----------|
| `velocity_only` | velocity | velocity | velocity |
| `vo_split` | velocity | onset_mask | zeros |
| `vos` | velocity | onset_mask | sustain_mask |

## VAE Registry (Section 6.2)

| Name | HF Model ID | Channels | Loading |
|------|------------|----------|---------|
| `sd_v1_4` | CompVis/stable-diffusion-v1-4 | 4 | subfolder='vae' |
| `eq_vae_ema` | zelaki/eq-vae-ema | 4 | direct |
| `eq_sdxl_vae` | KBlueLeaf/EQ-SDXL-VAE | 4 | direct |
| `sd_vae_ft_mse` | stabilityai/sd-vae-ft-mse | 4 | direct |
| `sdxl_vae` | stabilityai/sdxl-vae | 4 | direct |
| `playground_v25` | playgroundai/playground-v2.5-1024px-aesthetic | 4 | subfolder='vae' |
| `sd3_medium` | stabilityai/stable-diffusion-3-medium-diffusers | 16 | subfolder='vae' |
| `flux1_dev` | black-forest-labs/FLUX.1-dev | 16 | subfolder='vae' |
| `flux1_kontext` | black-forest-labs/FLUX.1-Kontext-dev | 16 | subfolder='vae' |
| `flux2_dev` | black-forest-labs/FLUX.2-dev | 16 | subfolder='vae' |
| `flux2_tiny` | fal/FLUX.2-Tiny-AutoEncoder | 16 | custom (bfloat16) |
| `cogview4` | THUDM/CogView4-6B | 16 | subfolder='vae' |

## Note Detection Methods (Section 7.2)

| Name | Type | Needs Fitting |
|------|------|---------------|
| `global_threshold` | Binary τ applied uniformly | No |
| `per_pitch_adaptive` | Per-pitch τ from activation stats | Yes |
| `hysteresis` | High τ_on for onset, low τ_off for sustain | No |
| `velocity_aware` | Onset weighted by velocity channel | No |
| `morphological` | Erosion + dilation post-processing | No |
| `hmm_tracker` | 2-state HMM per pitch, Viterbi | Yes |
| `cnn_segmenter` | 1D CNN onset/offset boundaries | Yes (train) |
| `gmm_detector` | Gaussian mixture per pitch | Yes |

## Pipeline Stage → Experiment Mapping (Section 10.3)

| Experiment | Stages | Conditions |
|-----------|--------|------------|
| Exp 1A | Ingest→Render→Encode→Decode→Detect→Evaluate | 12 VAEs × 5 instruments × 3 channels = 180 |
| Exp 1B | ...→[Detect × 8 methods]→Evaluate | Best VAE, 8 detection methods |
| Exp 2 | ...→Render(6 variants)→... | 2 orientations × 3 resolutions |
| Exp 3 | ...→Render(3 strategies)→... | 3 channels × 12 VAEs |
| Exp 4A | ...→Encode(mu+sample)→... | Best VAE, mu vs sample |
| Exp 4B | ...→Encode→LatentAnalysis | Full dataset z_mu |
| Exp 4C | ...→TrainSubLatent→SubDecode→... | 4 approaches × 5 target dims |
| Exp 4D | Same as 4C + FeatureConditioner | 3 conditioning families |
| Exp 5 | ...→PoolLatents→TrainTransformer→Generate→... | Raw vs sub-latent |

## PipelineRunner (Section 10.2)

```python
class PipelineRunner:
    def __init__(self, stages, config, tracker): ...
    def run(self, resume_from=None):
        for stage in topological_order():
            if cached: load_and_skip()
            else: execute_and_cache()
```

## ExperimentTracker (Section 11.1)

- ID format: `{experiment_name}_{YYYYMMDD_HHMMSS}_{4char_hash}`
- Saves: config.yaml, environment.json, metrics/, artifacts/, figures/, logs/, jobs/
- Environment: git hash, pip freeze, torch version, CUDA, GPU, hostname

## Sub-Latent Loss (Section 9.3)

```python
total = pixel_weight * MSE(recon, target) + onset_weight * BCE(soft_threshold(recon_G), onset_gt)
if sub_vae: total += kl_weight * KL(q(s|z) || N(0,I))
# defaults: pixel=1.0, onset=5.0, kl=0.001
```
