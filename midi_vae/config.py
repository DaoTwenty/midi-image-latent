"""Configuration system using OmegaConf for loading and Pydantic for validation.

Provides typed config schemas and a load_config function that merges
multiple YAML files with CLI overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, DictConfig
from pydantic import BaseModel, Field, model_validator


class PathsConfig(BaseModel):
    """Filesystem paths for data and outputs."""

    data_root: str
    output_root: str = "outputs"
    cache_dir: str = "outputs/cache"


class SubsetConfig(BaseModel):
    """Controls which files from a dataset are selected for processing.

    Filters are applied in this order: subdirectory, glob_pattern, file_list,
    then fraction.  subdirectory and glob_pattern are mutually exclusive.
    """

    subdirectory: str | None = None
    """Only files whose path relative to data_root starts with this prefix."""

    glob_pattern: str | None = None
    """Custom glob pattern relative to data_root (e.g. 'a/**/*.mid').
    Mutually exclusive with subdirectory."""

    file_list: str | None = None
    """Path to a text file containing one filepath per line.
    Paths are resolved relative to data_root."""

    fraction: float | None = None
    """Random fraction of files to retain, e.g. 0.1 for 10%.  Must be in (0.0, 1.0]."""

    fraction_seed: int | None = None
    """Separate RNG seed for fraction sampling; defaults to the top-level experiment seed."""

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _validate_exclusivity(self) -> SubsetConfig:
        """Ensure subdirectory and glob_pattern are not both set, and fraction is valid."""
        if self.subdirectory and self.glob_pattern:
            raise ValueError(
                "subset.subdirectory and subset.glob_pattern are mutually exclusive"
            )
        if self.fraction is not None and not (0.0 < self.fraction <= 1.0):
            raise ValueError("subset.fraction must be in (0.0, 1.0]")
        return self


class DataConfig(BaseModel):
    """Dataset selection and preprocessing parameters."""

    dataset: str = "lakh"  # lakh | maestro | pop909
    instruments: list[str] = Field(
        default_factory=lambda: ["drums", "bass", "guitar", "piano", "strings"]
    )
    bars_per_instrument: int = 5000
    min_notes_per_bar: int = 2
    time_steps: int = 96  # 64 | 96 | 128
    target_resolution: tuple[int, int] = (128, 128)
    max_files: int | None = None  # Limit number of files loaded (for debugging/mini runs)
    subset: SubsetConfig | None = None  # Optional subset filtering of source files


class RenderConfig(BaseModel):
    """Piano-roll rendering parameters."""

    channel_strategy: str = "velocity_only"  # velocity_only | vo_split | vos
    pitch_axis: str = "height"  # height | width
    normalize_range: tuple[float, float] = (-1.0, 1.0)
    resize_method: str = "bilinear"


class VAEConfig(BaseModel):
    """Configuration for a single VAE model."""

    model_id: str  # HuggingFace model ID
    name: str  # Short name for logging
    latent_type: str = "mean"  # mean | sample | both
    dtype: str = "float32"  # float32 | bfloat16
    batch_size: int = 32
    subfolder: str | None = None  # HF subfolder (e.g., 'vae')


class NoteDetectionConfig(BaseModel):
    """Note detection method configuration."""

    method: str = "global_threshold"
    params: dict[str, Any] = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    """Training hyperparameters for sub-latent models."""

    learning_rate: float = 1e-4
    epochs: int = 100
    batch_size: int = 64
    weight_decay: float = 1e-5
    pixel_weight: float = 1.0
    onset_weight: float = 5.0
    kl_weight: float = 0.001
    patience: int = 10


class ConditioningConfig(BaseModel):
    """Conditioning configuration for sub-latent models."""

    features: list[str] = Field(default_factory=list)
    embed_dim: int = 32
    fusion: str = "concat"  # concat | add | film


class SubLatentConfig(BaseModel):
    """Sub-latent model configuration."""

    enabled: bool = False
    approach: str = "mlp"  # pca | umap | mlp | sub_vae
    target_dim: int = 64
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    conditioning: ConditioningConfig | None = None


class TrackingConfig(BaseModel):
    """Experiment tracking and artifact storage configuration."""

    experiment_name: str = "default"
    wandb_enabled: bool = False
    wandb_project: str = "midi-image-vae"
    wandb_entity: str | None = None  # wandb team/user; None = personal default
    wandb_mode: str = "online"  # online | offline | disabled
    wandb_dir: str | None = None  # directory for offline run data; None = default
    wandb_tags: list[str] = Field(default_factory=list)
    save_reconstructions: bool = True
    save_latents: bool = True
    checkpoint_every_n: int = 500


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration.

    Composes all sub-configs into a single validated structure.
    """

    paths: PathsConfig
    data: DataConfig = Field(default_factory=DataConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    vaes: list[VAEConfig] = Field(default_factory=list)
    note_detection: NoteDetectionConfig = Field(default_factory=NoteDetectionConfig)
    sublatent: SubLatentConfig = Field(default_factory=SubLatentConfig)
    metrics: list[str] = Field(default_factory=lambda: ["all"])
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4

    model_config = {"frozen": True}


def load_config(
    paths: list[str] | None = None,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    """Load and merge configuration from YAML files with CLI overrides.

    Args:
        paths: List of YAML file paths to load and merge (later files override earlier).
        overrides: List of dotlist overrides (e.g., ['data.dataset=pop909', 'seed=123']).

    Returns:
        A validated, frozen ExperimentConfig instance.

    Raises:
        FileNotFoundError: If a config file doesn't exist.
        pydantic.ValidationError: If the merged config fails validation.
    """
    configs: list[DictConfig] = []

    for p in (paths or []):
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        configs.append(OmegaConf.load(str(path)))

    # Merge all YAML configs (later overrides earlier)
    if configs:
        merged = OmegaConf.merge(*configs)
    else:
        merged = OmegaConf.create({})

    # Apply CLI overrides
    if overrides:
        cli_conf = OmegaConf.from_dotlist(overrides)
        merged = OmegaConf.merge(merged, cli_conf)

    # Convert to plain dict for Pydantic
    config_dict = OmegaConf.to_container(merged, resolve=True)

    return ExperimentConfig(**config_dict)
