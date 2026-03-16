"""Stub tests for evaluation metrics in midi_vae/metrics/.

These tests define the contracts that DELTA's metric implementations must
satisfy. Tests for specific metrics are conditional on their modules existing.
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

from midi_vae.data.types import BarData, PianoRollImage, ReconstructedBar, MidiNote
from midi_vae.metrics.base import Metric, MetricsEngine
from midi_vae.registry import ComponentRegistry


# ---------------------------------------------------------------------------
# Skip guards for specific metric modules
# ---------------------------------------------------------------------------

reconstruction_available = False
try:
    from midi_vae.metrics import reconstruction as _recon_module  # noqa: F401
    reconstruction_available = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Metric ABC contract tests (using a concrete stub)
# ---------------------------------------------------------------------------


class StubMetric(Metric):
    """Minimal concrete Metric for testing the ABC interface."""

    @property
    def name(self) -> str:
        return "stub_metric"

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
        gt_image: PianoRollImage | None = None,
    ) -> dict[str, float]:
        return {"stub_metric/value": 0.5}


class TestMetricABC:
    """Tests for the Metric ABC contract."""

    def test_concrete_metric_name(self) -> None:
        """Metric.name returns the expected identifier."""
        m = StubMetric()
        assert m.name == "stub_metric"

    def test_compute_returns_dict(
        self,
        synthetic_bar: BarData,
    ) -> None:
        """compute() returns a dict mapping str -> float."""
        m = StubMetric()
        recon = ReconstructedBar(
            bar_id=synthetic_bar.bar_id,
            vae_name="stub",
            recon_image=torch.zeros(3, 128, 128),
            detected_notes=[],
            detection_method="global_threshold",
        )
        result = m.compute(synthetic_bar, recon)
        assert isinstance(result, dict)
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, float)

    def test_requires_notes_default_false(self) -> None:
        """Metric.requires_notes defaults to False."""
        m = StubMetric()
        assert m.requires_notes is False


# ---------------------------------------------------------------------------
# MetricsEngine tests (using stub metric in registry)
# ---------------------------------------------------------------------------


class TestMetricsEngine:
    """Tests for MetricsEngine in midi_vae/metrics/base.py."""

    @pytest.fixture(autouse=True)
    def register_stub_metric(self):
        """Register a stub metric for testing, then clean up."""
        # Save original metric registrations
        original_metrics = dict(ComponentRegistry._registry.get("metric", {}))

        # Register the stub metric (skip if already registered)
        if "test_stub_metric" not in ComponentRegistry._registry.get("metric", {}):
            @ComponentRegistry.register("metric", "test_stub_metric")
            class TestStub(Metric):
                @property
                def name(self) -> str:
                    return "test_stub_metric"

                def compute(
                    self,
                    gt: BarData,
                    recon: ReconstructedBar,
                    gt_image: PianoRollImage | None = None,
                ) -> dict[str, float]:
                    return {"test_stub_metric/value": 1.0}

        yield

        # Restore the metric registry to its original state
        if "metric" in ComponentRegistry._registry:
            ComponentRegistry._registry["metric"] = original_metrics
        elif original_metrics:
            ComponentRegistry._registry["metric"] = original_metrics

    def test_load_named_metrics(self) -> None:
        """MetricsEngine loads metrics by name from registry."""
        engine = MetricsEngine(["test_stub_metric"])
        assert len(engine.metrics) == 1

    def test_evaluate_returns_dict(self, synthetic_bar: BarData) -> None:
        """evaluate() returns a dict of metric results."""
        engine = MetricsEngine(["test_stub_metric"])
        recon = ReconstructedBar(
            bar_id=synthetic_bar.bar_id,
            vae_name="stub",
            recon_image=torch.zeros(3, 128, 128),
            detected_notes=[],
            detection_method="global_threshold",
        )
        result = engine.evaluate(synthetic_bar, recon)
        assert isinstance(result, dict)
        assert "test_stub_metric/value" in result

    def test_evaluate_batch(self, synthetic_bar: BarData) -> None:
        """evaluate_batch() processes a list of (gt, recon) pairs."""
        engine = MetricsEngine(["test_stub_metric"])
        recon = ReconstructedBar(
            bar_id=synthetic_bar.bar_id,
            vae_name="stub",
            recon_image=torch.zeros(3, 128, 128),
            detected_notes=[],
            detection_method="global_threshold",
        )
        pairs = [(synthetic_bar, recon), (synthetic_bar, recon)]
        results = engine.evaluate_batch(pairs)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, dict)

    def test_all_metrics_loaded_with_all_keyword(self) -> None:
        """MetricsEngine(['all']) loads all registered metrics."""
        engine = MetricsEngine(["all"])
        # Should include the test_stub_metric we registered
        metric_names = [m.name for m in engine.metrics]
        assert "test_stub_metric" in metric_names

    def test_requires_notes_metric_skipped_when_no_notes(
        self,
        synthetic_bar: BarData,
    ) -> None:
        """Metrics requiring notes are skipped when detected_notes is empty."""
        # Register a note-requiring metric
        @ComponentRegistry.register("metric", "note_requiring_metric")
        class NoteRequiringMetric(Metric):
            @property
            def name(self) -> str:
                return "note_requiring_metric"

            @property
            def requires_notes(self) -> bool:
                return True

            def compute(
                self,
                gt: BarData,
                recon: ReconstructedBar,
                gt_image: PianoRollImage | None = None,
            ) -> dict[str, float]:
                return {"note_requiring_metric/f1": 1.0}

        engine = MetricsEngine(["note_requiring_metric"])
        recon = ReconstructedBar(
            bar_id=synthetic_bar.bar_id,
            vae_name="stub",
            recon_image=torch.zeros(3, 128, 128),
            detected_notes=[],  # Empty — metric should be skipped
            detection_method="global_threshold",
        )
        result = engine.evaluate(synthetic_bar, recon)
        # Metric was skipped, so its key is absent
        assert "note_requiring_metric/f1" not in result


# ---------------------------------------------------------------------------
# PixelMSE tests (conditional)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not reconstruction_available,
    reason="midi_vae.metrics.reconstruction not yet implemented (DELTA Sprint pending)",
)
class TestPixelMSE:
    """Tests for PixelMSE metric in midi_vae/metrics/reconstruction.py."""

    @pytest.fixture
    def pixel_mse(self):
        """Get PixelMSE metric from registry."""
        MetricClass = ComponentRegistry.get("metric", "pixel_mse")
        return MetricClass()

    def test_identical_images_give_zero_mse(
        self,
        pixel_mse,
        synthetic_bar: BarData,
        synthetic_image,
    ) -> None:
        """Identical ground truth and reconstruction yield MSE = 0."""
        recon = ReconstructedBar(
            bar_id=synthetic_bar.bar_id,
            vae_name="stub",
            recon_image=synthetic_image.image.clone(),
            detected_notes=[],
            detection_method="global_threshold",
        )
        result = pixel_mse.compute(synthetic_bar, recon, gt_image=synthetic_image)
        # When images match exactly, MSE should be 0
        assert result.get("pixel_mse", float("inf")) == pytest.approx(0.0, abs=1e-6)

    def test_different_images_give_positive_mse(
        self,
        pixel_mse,
        synthetic_bar: BarData,
        synthetic_image,
    ) -> None:
        """Different images yield positive MSE."""
        recon = ReconstructedBar(
            bar_id=synthetic_bar.bar_id,
            vae_name="stub",
            recon_image=torch.ones(3, 128, 128),
            detected_notes=[],
            detection_method="global_threshold",
        )
        result = pixel_mse.compute(synthetic_bar, recon, gt_image=synthetic_image)
        # With a non-zero image vs zero piano roll, MSE > 0
        assert result.get("pixel_mse", 0.0) > 0.0

    def test_result_is_dict_of_floats(
        self,
        pixel_mse,
        synthetic_bar: BarData,
    ) -> None:
        """Result is a dict mapping str -> float."""
        recon = ReconstructedBar(
            bar_id=synthetic_bar.bar_id,
            vae_name="stub",
            recon_image=torch.zeros(3, 128, 128),
            detected_notes=[],
            detection_method="global_threshold",
        )
        result = pixel_mse.compute(synthetic_bar, recon, gt_image=PianoRollImage(
            bar_id=synthetic_bar.bar_id,
            image=torch.zeros(3, 128, 128),
            channel_strategy="velocity_only",
            resolution=(128, 128),
            pitch_axis="height",
        ))
        assert isinstance(result, dict)
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, float)

    def test_metric_name(self, pixel_mse) -> None:
        """PixelMSE metric has the expected name."""
        assert pixel_mse.name == "pixel_mse"


@pytest.mark.skipif(
    not reconstruction_available,
    reason="midi_vae.metrics.reconstruction not yet implemented (DELTA Sprint pending)",
)
class TestPSNR:
    """Tests for PSNR metric in midi_vae/metrics/reconstruction.py."""

    @pytest.fixture
    def psnr_metric(self):
        """Get PSNR metric from registry."""
        MetricClass = ComponentRegistry.get("metric", "psnr")
        return MetricClass()

    def test_identical_images_give_high_psnr(
        self,
        psnr_metric,
        synthetic_bar: BarData,
        synthetic_image,
    ) -> None:
        """Identical images yield very high (or inf) PSNR."""
        recon = ReconstructedBar(
            bar_id=synthetic_bar.bar_id,
            vae_name="stub",
            recon_image=synthetic_image.image.clone(),
            detected_notes=[],
            detection_method="global_threshold",
        )
        result = psnr_metric.compute(synthetic_bar, recon, gt_image=synthetic_image)
        psnr_val = result.get("psnr", 0.0)
        # Identical images → PSNR is very large or inf
        assert psnr_val > 40.0 or psnr_val == float("inf")

    def test_different_images_give_lower_psnr(
        self,
        psnr_metric,
        synthetic_bar: BarData,
        synthetic_image,
    ) -> None:
        """Different images yield lower PSNR than identical images."""
        recon = ReconstructedBar(
            bar_id=synthetic_bar.bar_id,
            vae_name="stub",
            recon_image=torch.ones(3, 128, 128),
            detected_notes=[],
            detection_method="global_threshold",
        )
        result = psnr_metric.compute(synthetic_bar, recon, gt_image=synthetic_image)
        psnr_val = result.get("psnr", float("inf"))
        assert psnr_val < float("inf")


# ---------------------------------------------------------------------------
# OnsetF1 tests (conditional)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not reconstruction_available,
    reason="midi_vae.metrics.reconstruction not yet implemented (DELTA Sprint pending)",
)
class TestOnsetF1:
    """Tests for OnsetF1 metric in midi_vae/metrics/reconstruction.py."""

    @pytest.fixture
    def onset_f1(self):
        """Get OnsetF1 metric from registry."""
        MetricClass = ComponentRegistry.get("metric", "onset_f1")
        return MetricClass()

    def test_requires_notes(self, onset_f1) -> None:
        """OnsetF1 requires detected notes."""
        assert onset_f1.requires_notes is True

    def test_perfect_match_gives_f1_one(
        self,
        onset_f1,
        synthetic_bar: BarData,
        synthetic_notes,
    ) -> None:
        """Perfect match between GT and detected notes gives F1 = 1.0."""
        recon = ReconstructedBar(
            bar_id=synthetic_bar.bar_id,
            vae_name="stub",
            recon_image=torch.zeros(3, 128, 128),
            detected_notes=synthetic_notes,
            detection_method="global_threshold",
        )
        result = onset_f1.compute(synthetic_bar, recon)
        f1 = result.get("onset_f1", 0.0)
        assert f1 == pytest.approx(1.0, abs=0.01)

    def test_no_overlap_gives_f1_zero(
        self,
        onset_f1,
        synthetic_bar: BarData,
    ) -> None:
        """No matching notes between GT and detected gives F1 = 0.0."""
        # Detected notes at completely different pitches/times than GT
        wrong_notes = [
            MidiNote(pitch=10, onset_step=10, offset_step=12, velocity=50),
            MidiNote(pitch=20, onset_step=35, offset_step=37, velocity=50),
        ]
        recon = ReconstructedBar(
            bar_id=synthetic_bar.bar_id,
            vae_name="stub",
            recon_image=torch.zeros(3, 128, 128),
            detected_notes=wrong_notes,
            detection_method="global_threshold",
        )
        result = onset_f1.compute(synthetic_bar, recon)
        f1 = result.get("onset_f1", 1.0)
        assert f1 == pytest.approx(0.0, abs=0.01)

    def test_result_has_precision_recall_f1(
        self,
        onset_f1,
        synthetic_bar: BarData,
        synthetic_notes,
    ) -> None:
        """Result dict contains precision, recall, and f1 keys."""
        recon = ReconstructedBar(
            bar_id=synthetic_bar.bar_id,
            vae_name="stub",
            recon_image=torch.zeros(3, 128, 128),
            detected_notes=synthetic_notes,
            detection_method="global_threshold",
        )
        result = onset_f1.compute(synthetic_bar, recon)
        assert any("precision" in k for k in result)
        assert any("recall" in k for k in result)
        assert any("f1" in k for k in result)


# ---------------------------------------------------------------------------
# NoteDensityPearson tests (conditional)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not reconstruction_available,
    reason="midi_vae.metrics.reconstruction not yet implemented (DELTA Sprint pending)",
)
class TestNoteDensityPearson:
    """Tests for NoteDensityPearson metric in midi_vae/metrics/reconstruction.py."""

    @pytest.fixture
    def density_metric(self):
        """Get NoteDensityPearson metric from registry."""
        MetricClass = ComponentRegistry.get("metric", "note_density_pearson")
        return MetricClass()

    def test_perfect_match_gives_correlation_one(
        self,
        density_metric,
    ) -> None:
        """Identical note density profile gives Pearson r = 1.0."""
        # Create a bar with VARYING density (needed for non-degenerate Pearson)
        T = 96
        piano_roll = np.zeros((128, T), dtype=np.float32)
        onset_mask = np.zeros((128, T), dtype=np.float32)
        sustain_mask = np.zeros((128, T), dtype=np.float32)
        # 2 overlapping notes in first half, 1 note in second half → varying density
        piano_roll[60, 0:48] = 100
        onset_mask[60, 0] = 1.0
        sustain_mask[60, 1:48] = 1.0
        piano_roll[64, 10:40] = 80
        onset_mask[64, 10] = 1.0
        sustain_mask[64, 11:40] = 1.0
        piano_roll[67, 60:90] = 90
        onset_mask[67, 60] = 1.0
        sustain_mask[67, 61:90] = 1.0

        bar = BarData(
            bar_id="density_test_0",
            song_id="density_test",
            instrument="piano",
            program_number=0,
            piano_roll=piano_roll,
            onset_mask=onset_mask,
            sustain_mask=sustain_mask,
            tempo=120.0,
            time_signature=(4, 4),
            metadata={},
        )
        # Detected notes exactly match the GT
        notes = [
            MidiNote(pitch=60, onset_step=0, offset_step=48, velocity=100),
            MidiNote(pitch=64, onset_step=10, offset_step=40, velocity=80),
            MidiNote(pitch=67, onset_step=60, offset_step=90, velocity=90),
        ]
        recon = ReconstructedBar(
            bar_id="density_test_0",
            vae_name="stub",
            recon_image=torch.zeros(3, 128, 128),
            detected_notes=notes,
            detection_method="global_threshold",
        )
        result = density_metric.compute(bar, recon)
        r = result.get("note_density_pearson", 0.0)
        assert r == pytest.approx(1.0, abs=0.01)

    def test_result_is_float(
        self,
        density_metric,
        synthetic_bar: BarData,
        synthetic_notes,
    ) -> None:
        """Result value is a float in [-1, 1]."""
        recon = ReconstructedBar(
            bar_id=synthetic_bar.bar_id,
            vae_name="stub",
            recon_image=torch.zeros(3, 128, 128),
            detected_notes=synthetic_notes,
            detection_method="global_threshold",
        )
        result = density_metric.compute(synthetic_bar, recon)
        for k, v in result.items():
            assert isinstance(v, float)
            assert -1.0 <= v <= 1.0 or v != v  # allow NaN for edge cases
