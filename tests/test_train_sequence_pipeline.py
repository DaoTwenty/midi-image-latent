"""Tests for the train_sequence pipeline stage.

Covers TrainSequenceStage instantiation, IO declarations, basic run() behaviour,
and integration with the PipelineRunner.

The train_sequence stage is expected at midi_vae/pipelines/train_sequence.py
(CHARLIE Sprint 4).  Tests use ``@pytest.mark.skipif`` guards and CPU-only
synthetic data — no GPU or pretrained weights required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from midi_vae.config import ExperimentConfig, PathsConfig
from midi_vae.data.types import LatentEncoding
from midi_vae.pipelines.base import PipelineStage, PipelineRunner, StageIO

# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------

train_sequence_available = False
TrainSequenceStage = None  # type: ignore[assignment]

try:
    from midi_vae.pipelines.train_sequence import TrainSequenceStage  # type: ignore[assignment]
    train_sequence_available = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

LATENT_DIM = 64       # flattened latent dimension used in all tests
N_LATENTS = 20        # small dataset for fast CPU tests


def _make_config(tmp_path: Path) -> ExperimentConfig:
    """Return a minimal ExperimentConfig pointing to tmp directories."""
    return ExperimentConfig(
        paths=PathsConfig(
            data_root=str(tmp_path / "data"),
            output_root=str(tmp_path / "outputs"),
            cache_dir=str(tmp_path / "cache"),
        ),
    )


def _make_latent_encodings(n: int = N_LATENTS) -> list[LatentEncoding]:
    """Create a list of synthetic LatentEncoding objects (4, 4, 4 → flat 64)."""
    torch.manual_seed(0)
    encodings = []
    for i in range(n):
        z = torch.randn(4, 4, 4)  # 4*4*4 = 64 == LATENT_DIM
        encodings.append(LatentEncoding(
            bar_id=f"song_{i}_piano_0",
            vae_name="stub_vae",
            z_mu=z,
            z_sigma=torch.ones(4, 4, 4) * 0.1,
            z_sample=z,
        ))
    return encodings


# ---------------------------------------------------------------------------
# Instantiation tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not train_sequence_available,
    reason="midi_vae.pipelines.train_sequence not yet implemented (CHARLIE Sprint 4)",
)
class TestTrainSequenceStageInstantiation:
    """Tests for TrainSequenceStage construction."""

    def test_instantiation_with_config(self, tmp_path) -> None:
        """TrainSequenceStage can be constructed with a valid ExperimentConfig."""
        cfg = _make_config(tmp_path)
        stage = TrainSequenceStage(config=cfg)  # type: ignore[operator]
        assert stage is not None

    def test_is_pipeline_stage_subclass(self, tmp_path) -> None:
        """TrainSequenceStage is a PipelineStage subclass."""
        cfg = _make_config(tmp_path)
        stage = TrainSequenceStage(config=cfg)  # type: ignore[operator]
        assert isinstance(stage, PipelineStage)

    def test_config_is_stored(self, tmp_path) -> None:
        """config attribute is stored on the instance."""
        cfg = _make_config(tmp_path)
        stage = TrainSequenceStage(config=cfg)  # type: ignore[operator]
        assert stage.config is cfg

    def test_custom_checkpoint_dir_is_accepted(self, tmp_path) -> None:
        """TrainSequenceStage accepts a custom checkpoint_dir argument."""
        cfg = _make_config(tmp_path)
        ckpt_dir = str(tmp_path / "seq_checkpoints")
        try:
            stage = TrainSequenceStage(config=cfg, checkpoint_dir=ckpt_dir)  # type: ignore[operator]
            assert stage is not None
        except TypeError:
            pass  # Optional parameter — acceptable if not supported


# ---------------------------------------------------------------------------
# IO declaration tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not train_sequence_available,
    reason="midi_vae.pipelines.train_sequence not yet implemented (CHARLIE Sprint 4)",
)
class TestTrainSequenceStageIO:
    """Tests for the io() method — input/output key declarations."""

    @pytest.fixture
    def stage(self, tmp_path) -> "TrainSequenceStage":  # type: ignore[name-defined]
        return TrainSequenceStage(config=_make_config(tmp_path))  # type: ignore[operator]

    def test_io_returns_stage_io(self, stage) -> None:
        """io() returns a StageIO instance."""
        result = stage.io()
        assert isinstance(result, StageIO)

    def test_io_inputs_is_tuple_of_strings(self, stage) -> None:
        """io().inputs is a tuple of strings."""
        io = stage.io()
        assert isinstance(io.inputs, tuple)
        for key in io.inputs:
            assert isinstance(key, str)

    def test_io_outputs_is_tuple_of_strings(self, stage) -> None:
        """io().outputs is a tuple of strings."""
        io = stage.io()
        assert isinstance(io.outputs, tuple)
        for key in io.outputs:
            assert isinstance(key, str)

    def test_io_inputs_include_latent_encodings(self, stage) -> None:
        """io().inputs should include 'latent_encodings' as a required input."""
        io = stage.io()
        assert "latent_encodings" in io.inputs, (
            f"Expected 'latent_encodings' in inputs, got: {io.inputs}"
        )

    def test_io_outputs_include_model_path(self, stage) -> None:
        """io().outputs includes a path key for the saved model checkpoint."""
        io = stage.io()
        path_keys = [k for k in io.outputs if "path" in k.lower() or "checkpoint" in k.lower()]
        assert len(path_keys) >= 1, (
            f"Expected a checkpoint path key in outputs, got: {io.outputs}"
        )

    def test_io_outputs_include_train_stats(self, stage) -> None:
        """io().outputs includes a stats/metrics key."""
        io = stage.io()
        stats_keys = [k for k in io.outputs if "stats" in k.lower() or "metrics" in k.lower()]
        assert len(stats_keys) >= 1, (
            f"Expected a stats key in outputs, got: {io.outputs}"
        )


# ---------------------------------------------------------------------------
# run() tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not train_sequence_available,
    reason="midi_vae.pipelines.train_sequence not yet implemented (CHARLIE Sprint 4)",
)
class TestTrainSequenceStageRun:
    """Tests for the run() method with small synthetic data."""

    @pytest.fixture
    def stage(self, tmp_path) -> "TrainSequenceStage":  # type: ignore[name-defined]
        return TrainSequenceStage(config=_make_config(tmp_path))  # type: ignore[operator]

    def test_run_with_empty_encodings_does_not_crash(self, stage) -> None:
        """run() with empty latent_encodings returns without raising."""
        context: dict[str, Any] = {"latent_encodings": []}
        result = stage.run(context)
        assert isinstance(result, dict)

    def test_run_returns_dict(self, stage) -> None:
        """run() always returns a dict."""
        context: dict[str, Any] = {"latent_encodings": _make_latent_encodings(n=5)}
        result = stage.run(context)
        assert isinstance(result, dict)

    def test_run_outputs_have_expected_keys(self, stage) -> None:
        """run() output contains the keys declared in io().outputs."""
        context: dict[str, Any] = {"latent_encodings": _make_latent_encodings(n=5)}
        result = stage.run(context)
        io = stage.io()
        for key in io.outputs:
            assert key in result, (
                f"Expected key '{key}' in run() output, got: {list(result.keys())}"
            )

    def test_run_produces_checkpoint_file_or_empty_string(self, stage) -> None:
        """The checkpoint path key points to an existing file or is an empty string."""
        context: dict[str, Any] = {"latent_encodings": _make_latent_encodings(n=5)}
        result = stage.run(context)
        # Find the checkpoint path key
        io = stage.io()
        path_keys = [k for k in io.outputs if "path" in k.lower() or "checkpoint" in k.lower()]
        if path_keys:
            ckpt_path = result.get(path_keys[0], "")
            if ckpt_path:
                assert Path(ckpt_path).exists(), (
                    f"Checkpoint path '{ckpt_path}' does not exist"
                )

    def test_run_stats_are_dict(self, stage) -> None:
        """The stats output is a dict."""
        context: dict[str, Any] = {"latent_encodings": _make_latent_encodings(n=5)}
        result = stage.run(context)
        io = stage.io()
        stats_keys = [k for k in io.outputs if "stats" in k.lower() or "metrics" in k.lower()]
        if stats_keys:
            stats = result.get(stats_keys[0])
            assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# PipelineRunner integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not train_sequence_available,
    reason="midi_vae.pipelines.train_sequence not yet implemented (CHARLIE Sprint 4)",
)
class TestTrainSequenceInPipelineRunner:
    """Tests for TrainSequenceStage inside PipelineRunner."""

    class _ProducerStage(PipelineStage):
        """Minimal upstream stage that injects synthetic latent encodings."""

        def io(self) -> StageIO:
            return StageIO(inputs=(), outputs=("latent_encodings",))

        def run(self, context: dict[str, Any]) -> dict[str, Any]:
            return {"latent_encodings": _make_latent_encodings(n=5)}

    def test_pipeline_runner_with_train_sequence_stage(self, tmp_path) -> None:
        """PipelineRunner successfully runs with TrainSequenceStage downstream."""
        cfg = _make_config(tmp_path)
        producer = self._ProducerStage(config=cfg)
        trainer = TrainSequenceStage(config=cfg)  # type: ignore[operator]

        runner = PipelineRunner(
            stages=[producer, trainer],
            config=cfg,
            cache_dir=None,
        )
        context = runner.run()

        assert isinstance(context, dict)
        # Both stage outputs should be in the final context
        assert "latent_encodings" in context
        io = trainer.io()
        for key in io.outputs:
            assert key in context, (
                f"Expected key '{key}' in PipelineRunner context, got: {list(context.keys())}"
            )

    def test_pipeline_runner_topological_order(self, tmp_path) -> None:
        """PipelineRunner resolves dependency order correctly."""
        cfg = _make_config(tmp_path)
        producer = self._ProducerStage(config=cfg)
        trainer = TrainSequenceStage(config=cfg)  # type: ignore[operator]

        runner = PipelineRunner(
            stages=[trainer, producer],  # reversed order — runner should fix it
            config=cfg,
            cache_dir=None,
        )
        # Should not raise even with reversed stage order
        context = runner.run()
        assert isinstance(context, dict)


# ---------------------------------------------------------------------------
# Name property and cache_key
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not train_sequence_available,
    reason="midi_vae.pipelines.train_sequence not yet implemented (CHARLIE Sprint 4)",
)
class TestTrainSequenceStageMisc:
    """Tests for name, cache_key, and other PipelineStage interface methods."""

    @pytest.fixture
    def stage(self, tmp_path) -> "TrainSequenceStage":  # type: ignore[name-defined]
        return TrainSequenceStage(config=_make_config(tmp_path))  # type: ignore[operator]

    def test_stage_name_is_string(self, stage) -> None:
        """name property returns a non-empty string."""
        assert isinstance(stage.name, str)
        assert len(stage.name) > 0

    def test_cache_key_returns_none_or_string(self, stage) -> None:
        """cache_key() returns None or a hex-digest string."""
        context: dict[str, Any] = {"latent_encodings": _make_latent_encodings(n=5)}
        key = stage.cache_key(context)
        assert key is None or isinstance(key, str)

    def test_cache_key_with_empty_encodings_returns_none(self, stage) -> None:
        """cache_key() returns None when there are no latent_encodings."""
        context: dict[str, Any] = {"latent_encodings": []}
        key = stage.cache_key(context)
        assert key is None

    def test_cache_key_is_reproducible(self, stage) -> None:
        """Same inputs produce the same cache key on repeated calls."""
        encodings = _make_latent_encodings(n=5)
        context: dict[str, Any] = {"latent_encodings": encodings}
        key1 = stage.cache_key(context)
        key2 = stage.cache_key(context)
        assert key1 == key2
