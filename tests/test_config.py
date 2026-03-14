"""Tests for the config system in midi_vae/config.py.

Verifies loading base.yaml, config validation, sub-config schemas,
and config merging/overrides.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from midi_vae.config import (
    ConditioningConfig,
    DataConfig,
    ExperimentConfig,
    NoteDetectionConfig,
    PathsConfig,
    RenderConfig,
    SubLatentConfig,
    TrackingConfig,
    TrainingConfig,
    VAEConfig,
    load_config,
)


# Path to the base YAML in the repo
BASE_YAML = Path(__file__).parent.parent / "configs" / "base.yaml"


# ---------------------------------------------------------------------------
# PathsConfig tests
# ---------------------------------------------------------------------------


class TestPathsConfig:
    """Tests for PathsConfig schema."""

    def test_requires_data_root(self) -> None:
        """PathsConfig requires data_root."""
        with pytest.raises(ValidationError):
            PathsConfig()  # type: ignore[call-arg]

    def test_defaults(self) -> None:
        """output_root and cache_dir have sensible defaults."""
        cfg = PathsConfig(data_root="/data")
        assert cfg.output_root == "outputs"
        assert cfg.cache_dir == "outputs/cache"

    def test_custom_paths(self) -> None:
        """Custom paths are stored correctly."""
        cfg = PathsConfig(
            data_root="/custom/data",
            output_root="/custom/out",
            cache_dir="/custom/cache",
        )
        assert cfg.data_root == "/custom/data"
        assert cfg.output_root == "/custom/out"
        assert cfg.cache_dir == "/custom/cache"


# ---------------------------------------------------------------------------
# DataConfig tests
# ---------------------------------------------------------------------------


class TestDataConfig:
    """Tests for DataConfig schema."""

    def test_default_dataset_is_lakh(self) -> None:
        """Default dataset is 'lakh'."""
        cfg = DataConfig()
        assert cfg.dataset == "lakh"

    def test_default_instruments(self) -> None:
        """Default instruments list includes standard 5."""
        cfg = DataConfig()
        assert set(cfg.instruments) == {"drums", "bass", "guitar", "piano", "strings"}

    def test_default_time_steps(self) -> None:
        """Default time_steps is 96."""
        cfg = DataConfig()
        assert cfg.time_steps == 96

    def test_default_target_resolution(self) -> None:
        """Default target_resolution is (128, 128)."""
        cfg = DataConfig()
        assert cfg.target_resolution == (128, 128)

    def test_custom_dataset(self) -> None:
        """Custom dataset name is stored correctly."""
        cfg = DataConfig(dataset="pop909")
        assert cfg.dataset == "pop909"


# ---------------------------------------------------------------------------
# RenderConfig tests
# ---------------------------------------------------------------------------


class TestRenderConfig:
    """Tests for RenderConfig schema."""

    def test_default_channel_strategy(self) -> None:
        """Default channel_strategy is 'velocity_only'."""
        cfg = RenderConfig()
        assert cfg.channel_strategy == "velocity_only"

    def test_default_pitch_axis(self) -> None:
        """Default pitch_axis is 'height'."""
        cfg = RenderConfig()
        assert cfg.pitch_axis == "height"

    def test_default_normalize_range(self) -> None:
        """Default normalize_range is (-1.0, 1.0)."""
        cfg = RenderConfig()
        assert cfg.normalize_range == (-1.0, 1.0)


# ---------------------------------------------------------------------------
# VAEConfig tests
# ---------------------------------------------------------------------------


class TestVAEConfig:
    """Tests for VAEConfig schema."""

    def test_requires_model_id(self) -> None:
        """VAEConfig requires model_id."""
        with pytest.raises(ValidationError):
            VAEConfig(name="test")  # type: ignore[call-arg]

    def test_requires_name(self) -> None:
        """VAEConfig requires name."""
        with pytest.raises(ValidationError):
            VAEConfig(model_id="stabilityai/sd-vae-ft-mse")  # type: ignore[call-arg]

    def test_defaults(self) -> None:
        """VAEConfig has sensible defaults."""
        cfg = VAEConfig(model_id="stabilityai/sd-vae-ft-mse", name="sd_vae")
        assert cfg.latent_type == "mean"
        assert cfg.dtype == "float32"
        assert cfg.batch_size == 32
        assert cfg.subfolder is None

    def test_custom_values(self) -> None:
        """Custom values are stored correctly."""
        cfg = VAEConfig(
            model_id="black-forest-labs/FLUX.1-dev",
            name="flux",
            latent_type="sample",
            dtype="bfloat16",
            batch_size=16,
            subfolder="vae",
        )
        assert cfg.latent_type == "sample"
        assert cfg.dtype == "bfloat16"
        assert cfg.subfolder == "vae"


# ---------------------------------------------------------------------------
# NoteDetectionConfig tests
# ---------------------------------------------------------------------------


class TestNoteDetectionConfig:
    """Tests for NoteDetectionConfig schema."""

    def test_default_method(self) -> None:
        """Default method is 'global_threshold'."""
        cfg = NoteDetectionConfig()
        assert cfg.method == "global_threshold"

    def test_default_params_is_empty(self) -> None:
        """Default params is an empty dict."""
        cfg = NoteDetectionConfig()
        assert cfg.params == {}

    def test_custom_params(self) -> None:
        """Custom params are stored correctly."""
        cfg = NoteDetectionConfig(method="adaptive", params={"threshold": 0.5, "min_duration": 2})
        assert cfg.method == "adaptive"
        assert cfg.params["threshold"] == 0.5


# ---------------------------------------------------------------------------
# SubLatentConfig tests
# ---------------------------------------------------------------------------


class TestSubLatentConfig:
    """Tests for SubLatentConfig schema."""

    def test_default_disabled(self) -> None:
        """Sub-latent is disabled by default."""
        cfg = SubLatentConfig()
        assert cfg.enabled is False

    def test_default_approach(self) -> None:
        """Default approach is 'mlp'."""
        cfg = SubLatentConfig()
        assert cfg.approach == "mlp"

    def test_nested_training_config(self) -> None:
        """SubLatentConfig includes a TrainingConfig with defaults."""
        cfg = SubLatentConfig()
        assert isinstance(cfg.training, TrainingConfig)
        assert cfg.training.epochs == 100
        assert cfg.training.learning_rate == pytest.approx(1e-4)

    def test_conditioning_optional(self) -> None:
        """conditioning is optional (None by default)."""
        cfg = SubLatentConfig()
        assert cfg.conditioning is None


# ---------------------------------------------------------------------------
# TrackingConfig tests
# ---------------------------------------------------------------------------


class TestTrackingConfig:
    """Tests for TrackingConfig schema."""

    def test_defaults(self) -> None:
        """TrackingConfig has sensible defaults."""
        cfg = TrackingConfig()
        assert cfg.experiment_name == "default"
        assert cfg.wandb_enabled is False
        assert cfg.save_reconstructions is True
        assert cfg.save_latents is True


# ---------------------------------------------------------------------------
# ExperimentConfig tests
# ---------------------------------------------------------------------------


class TestExperimentConfig:
    """Tests for top-level ExperimentConfig schema."""

    def test_requires_paths(self) -> None:
        """ExperimentConfig requires paths."""
        with pytest.raises((ValidationError, TypeError)):
            ExperimentConfig()  # type: ignore[call-arg]

    def test_all_sub_configs_have_defaults(self) -> None:
        """All sub-configs get populated with defaults."""
        cfg = ExperimentConfig(
            paths=PathsConfig(data_root="/data"),
        )
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.render, RenderConfig)
        assert isinstance(cfg.note_detection, NoteDetectionConfig)
        assert isinstance(cfg.sublatent, SubLatentConfig)
        assert isinstance(cfg.tracking, TrackingConfig)

    def test_default_seed(self) -> None:
        """Default seed is 42."""
        cfg = ExperimentConfig(paths=PathsConfig(data_root="/data"))
        assert cfg.seed == 42

    def test_default_device(self) -> None:
        """Default device is 'cuda'."""
        cfg = ExperimentConfig(paths=PathsConfig(data_root="/data"))
        assert cfg.device == "cuda"

    def test_vaes_list(self) -> None:
        """vaes is a list of VAEConfig."""
        cfg = ExperimentConfig(
            paths=PathsConfig(data_root="/data"),
            vaes=[VAEConfig(model_id="stub", name="stub")],
        )
        assert len(cfg.vaes) == 1
        assert cfg.vaes[0].name == "stub"

    def test_is_frozen(self) -> None:
        """ExperimentConfig is immutable (model_config frozen=True)."""
        cfg = ExperimentConfig(paths=PathsConfig(data_root="/data"))
        with pytest.raises(Exception):
            cfg.seed = 99  # type: ignore[misc]

    def test_tmp_config_fixture(self, tmp_config: ExperimentConfig) -> None:
        """tmp_config fixture provides a valid ExperimentConfig."""
        assert isinstance(tmp_config, ExperimentConfig)
        assert "tmp" in tmp_config.paths.data_root or tmp_config.paths.data_root.startswith("/")


# ---------------------------------------------------------------------------
# load_config tests
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_base_yaml(self) -> None:
        """load_config loads base.yaml and returns a valid ExperimentConfig."""
        if not BASE_YAML.exists():
            pytest.skip("base.yaml not found")
        cfg = load_config(paths=[str(BASE_YAML)])
        assert isinstance(cfg, ExperimentConfig)

    def test_load_base_yaml_values(self) -> None:
        """Values from base.yaml are correctly parsed."""
        if not BASE_YAML.exists():
            pytest.skip("base.yaml not found")
        cfg = load_config(paths=[str(BASE_YAML)])
        assert cfg.data.dataset == "lakh"
        assert cfg.seed == 42
        assert cfg.render.channel_strategy == "velocity_only"

    def test_missing_file_raises(self) -> None:
        """load_config raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_config(paths=["/nonexistent/path/config.yaml"])

    def test_cli_overrides(self) -> None:
        """CLI dotlist overrides take effect."""
        if not BASE_YAML.exists():
            pytest.skip("base.yaml not found")
        cfg = load_config(
            paths=[str(BASE_YAML)],
            overrides=["seed=999", "data.dataset=pop909"],
        )
        assert cfg.seed == 999
        assert cfg.data.dataset == "pop909"

    def test_empty_config_with_overrides(self) -> None:
        """load_config with no files but paths override works."""
        cfg = load_config(
            overrides=[
                "paths.data_root=/tmp/data",
                "seed=7",
            ]
        )
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.seed == 7

    def test_yaml_merge_later_overrides_earlier(self, tmp_path: Path) -> None:
        """Later YAML files override earlier ones during merge."""
        yaml1 = tmp_path / "a.yaml"
        yaml2 = tmp_path / "b.yaml"
        yaml1.write_text("paths:\n  data_root: /data_a\nseed: 1\n")
        yaml2.write_text("seed: 2\n")

        cfg = load_config(paths=[str(yaml1), str(yaml2)])
        assert cfg.seed == 2
        assert cfg.paths.data_root == "/data_a"

    def test_validation_error_on_invalid_config(self, tmp_path: Path) -> None:
        """Validation error raised when required fields are missing."""
        # paths.data_root is required; omitting it should fail Pydantic validation
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("seed: 42\n")  # no paths section

        with pytest.raises(ValidationError):
            load_config(paths=[str(bad_yaml)])
