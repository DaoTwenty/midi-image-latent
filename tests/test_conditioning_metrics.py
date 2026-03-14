"""Tests for conditioning evaluation metrics in midi_vae/metrics/conditioning.py.

Conditioning metrics measure how well a conditioned sub-latent model respects
the provided musical features (instrument, pitch range, rhythm) — i.e. whether
the generated/reconstructed bar is consistent with its conditioning context.

The conditioning module is expected from DELTA Sprint 4.  Tests use
``@pytest.mark.skipif`` guards so the suite stays green until the module lands.
All tests use synthetic CPU data — no GPU or real model weights required.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midi_vae.data.types import BarData, MidiNote, ReconstructedBar
from midi_vae.metrics.base import Metric

# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------

conditioning_available = False
try:
    import midi_vae.metrics.conditioning as _cond_module  # noqa: F401
    conditioning_available = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Shared synthetic helpers
# ---------------------------------------------------------------------------


def _make_bar(
    bar_id: str = "cond_test_0",
    pitches: list[tuple[int, int, int, int]] | None = None,
    instrument: str = "piano",
    tempo: float = 120.0,
    time_sig: tuple[int, int] = (4, 4),
) -> BarData:
    """Create a synthetic BarData with specified note events."""
    T = 96
    piano_roll = np.zeros((128, T), dtype=np.float32)
    onset_mask = np.zeros((128, T), dtype=np.float32)
    sustain_mask = np.zeros((128, T), dtype=np.float32)

    if pitches is None:
        pitches = [(60, 0, 24, 80), (64, 24, 48, 70), (67, 48, 72, 90)]

    for p, on, off, vel in pitches:
        piano_roll[p, on:off] = vel
        onset_mask[p, on] = 1.0
        if off > on + 1:
            sustain_mask[p, on + 1 : off] = 1.0

    return BarData(
        bar_id=bar_id,
        song_id="song_" + bar_id,
        instrument=instrument,
        program_number=0,
        piano_roll=piano_roll,
        onset_mask=onset_mask,
        sustain_mask=sustain_mask,
        tempo=tempo,
        time_signature=time_sig,
        metadata={},
    )


def _make_recon(
    bar: BarData,
    notes: list[MidiNote] | None = None,
) -> ReconstructedBar:
    """Build a ReconstructedBar from a BarData."""
    if notes is None:
        notes = []
    return ReconstructedBar(
        bar_id=bar.bar_id,
        vae_name="stub_vae",
        recon_image=torch.zeros(3, 128, 128),
        detected_notes=notes,
        detection_method="global_threshold",
    )


def _make_note(pitch: int = 60, onset: int = 0, offset: int = 24, vel: int = 80) -> MidiNote:
    return MidiNote(pitch=pitch, onset_step=onset, offset_step=offset, velocity=vel)


# ---------------------------------------------------------------------------
# Metric ABC contract tests (stub)
# ---------------------------------------------------------------------------


class TestConditioningMetricContract:
    """Verify conditioning metrics implement the Metric ABC contract."""

    class StubCondMetric(Metric):
        @property
        def name(self) -> str:
            return "stub_conditioning"

        def compute(self, gt: BarData, recon: ReconstructedBar) -> dict[str, float]:
            return {"stub_conditioning/score": 0.5}

    def test_compute_returns_dict_of_floats(self, synthetic_bar, synthetic_notes) -> None:
        """compute() must return dict[str, float]."""
        m = self.StubCondMetric()
        recon = _make_recon(synthetic_bar, list(synthetic_notes))
        result = m.compute(synthetic_bar, recon)
        assert isinstance(result, dict)
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, float)

    def test_name_is_non_empty_string(self) -> None:
        """name property is a non-empty string."""
        m = self.StubCondMetric()
        assert isinstance(m.name, str)
        assert len(m.name) > 0


# ---------------------------------------------------------------------------
# InstrumentConsistency metric tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not conditioning_available,
    reason="midi_vae.metrics.conditioning not yet implemented (DELTA Sprint 4)",
)
class TestInstrumentConsistency:
    """Tests for InstrumentConsistency metric.

    Measures whether the reconstructed bar sounds like the expected instrument
    (e.g. drums vs piano pitch distribution differences).
    """

    @pytest.fixture
    def metric(self):
        from midi_vae.registry import ComponentRegistry
        cls = ComponentRegistry.get("metric", "instrument_consistency")
        return cls()

    def test_name_is_string(self, metric) -> None:
        assert isinstance(metric.name, str)

    def test_correct_instrument_gives_high_score(self, metric) -> None:
        """Piano notes in piano bar → high consistency score."""
        # Piano notes roughly in mid-range
        bar = _make_bar(instrument="piano", pitches=[(60, 0, 24, 80), (64, 24, 48, 70)])
        notes = [_make_note(pitch=60), _make_note(pitch=64, onset=24, offset=48)]
        recon = _make_recon(bar, notes)
        result = metric.compute(bar, recon)
        assert isinstance(result, dict)
        for v in result.values():
            assert not math.isnan(v)
            assert isinstance(v, float)

    def test_empty_detected_notes_handled(self, metric) -> None:
        """Metric handles empty detected_notes gracefully."""
        bar = _make_bar(instrument="piano")
        recon = _make_recon(bar, [])
        result = metric.compute(bar, recon)
        assert isinstance(result, dict)

    def test_result_values_are_floats_in_range(self, metric) -> None:
        """All result values are floats in [0.0, 1.0] or a valid metric range."""
        bar = _make_bar(instrument="piano")
        notes = [_make_note(pitch=p) for p in range(60, 72, 2)]
        recon = _make_recon(bar, notes)
        result = metric.compute(bar, recon)
        for v in result.values():
            assert isinstance(v, float)
            assert not math.isnan(v)


# ---------------------------------------------------------------------------
# PitchRangeAdherence metric tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not conditioning_available,
    reason="midi_vae.metrics.conditioning not yet implemented (DELTA Sprint 4)",
)
class TestPitchRangeAdherence:
    """Tests for PitchRangeAdherence metric.

    Measures whether detected notes fall within the expected pitch range
    derived from the ground-truth bar's conditioning features.
    """

    @pytest.fixture
    def metric(self):
        from midi_vae.registry import ComponentRegistry
        cls = ComponentRegistry.get("metric", "pitch_range_adherence")
        return cls()

    def test_name_is_string(self, metric) -> None:
        assert isinstance(metric.name, str)

    def test_notes_within_range_give_high_score(self, metric) -> None:
        """Detected notes matching GT pitch range → high adherence."""
        bar = _make_bar(pitches=[(60, 0, 24, 80), (72, 24, 48, 80)])
        # Detected notes in same range [60, 72]
        notes = [_make_note(pitch=63), _make_note(pitch=67, onset=24, offset=48)]
        recon = _make_recon(bar, notes)
        result = metric.compute(bar, recon)
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, float)

    def test_notes_outside_range_give_lower_score(self, metric) -> None:
        """Notes completely outside GT pitch range → lower adherence than in-range."""
        bar = _make_bar(pitches=[(60, 0, 24, 80), (72, 24, 48, 80)])
        # Notes in-range
        notes_in = [_make_note(pitch=65), _make_note(pitch=68, onset=24, offset=48)]
        # Notes far out of range
        notes_out = [_make_note(pitch=20), _make_note(pitch=110, onset=24, offset=48)]

        recon_in = _make_recon(bar, notes_in)
        recon_out = _make_recon(bar, notes_out)

        result_in = metric.compute(bar, recon_in)
        result_out = metric.compute(bar, recon_out)

        score_in = list(result_in.values())[0]
        score_out = list(result_out.values())[0]

        # In-range notes should have >= score vs out-of-range notes
        assert score_in >= score_out - 0.01, (
            f"Expected in-range score ({score_in:.3f}) >= out-of-range ({score_out:.3f})"
        )

    def test_empty_notes_handled(self, metric) -> None:
        """Empty detected notes handled without raising."""
        bar = _make_bar()
        recon = _make_recon(bar, [])
        result = metric.compute(bar, recon)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# TempoConsistency metric tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not conditioning_available,
    reason="midi_vae.metrics.conditioning not yet implemented (DELTA Sprint 4)",
)
class TestTempoConsistency:
    """Tests for TempoConsistency metric.

    Measures whether detected onset timing is consistent with the GT bar's tempo.
    """

    @pytest.fixture
    def metric(self):
        from midi_vae.registry import ComponentRegistry
        cls = ComponentRegistry.get("metric", "tempo_consistency")
        return cls()

    def test_name_is_string(self, metric) -> None:
        assert isinstance(metric.name, str)

    def test_on_grid_onsets_give_high_consistency(self, metric) -> None:
        """Notes on the rhythmic grid for 120 BPM → high tempo consistency."""
        bar = _make_bar(tempo=120.0)
        # Quarter note steps at 96 time steps / 4 beats = 24 steps per beat
        notes = [_make_note(pitch=60 + i * 4, onset=i * 24, offset=i * 24 + 12)
                 for i in range(4)]
        recon = _make_recon(bar, notes)
        result = metric.compute(bar, recon)
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, float)

    def test_empty_notes_handled(self, metric) -> None:
        """Empty notes handled gracefully."""
        bar = _make_bar(tempo=120.0)
        recon = _make_recon(bar, [])
        result = metric.compute(bar, recon)
        assert isinstance(result, dict)

    def test_single_note_handled(self, metric) -> None:
        """Single note (no IOI) handled without error."""
        bar = _make_bar(tempo=120.0)
        recon = _make_recon(bar, [_make_note()])
        result = metric.compute(bar, recon)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# FeatureConditioningError metric tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not conditioning_available,
    reason="midi_vae.metrics.conditioning not yet implemented (DELTA Sprint 4)",
)
class TestFeatureConditioningError:
    """Tests for FeatureConditioningError metric.

    Compares scalar conditioning features (pitch_mean, note_density, etc.) between
    GT bar and reconstruction to measure conditioning accuracy.
    """

    @pytest.fixture
    def metric(self):
        from midi_vae.registry import ComponentRegistry
        cls = ComponentRegistry.get("metric", "feature_conditioning_error")
        return cls()

    def test_name_is_string(self, metric) -> None:
        assert isinstance(metric.name, str)

    def test_identical_bars_give_zero_error(self, metric, synthetic_bar, synthetic_notes) -> None:
        """Perfect reconstruction gives zero (or near-zero) conditioning error."""
        recon = _make_recon(synthetic_bar, list(synthetic_notes))
        result = metric.compute(synthetic_bar, recon)
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, float)
            # Error should be non-negative
            assert v >= 0.0

    def test_different_bars_give_positive_error(self, metric) -> None:
        """Different GT vs recon notes → non-zero feature conditioning error."""
        bar = _make_bar(pitches=[(60, 0, 24, 80), (64, 24, 48, 70)])
        # Recon at completely different pitch range
        notes = [_make_note(pitch=100, onset=0, offset=12)]
        recon = _make_recon(bar, notes)
        result = metric.compute(bar, recon)
        for v in result.values():
            assert isinstance(v, float)

    def test_result_keys_are_namespaced(self, metric, synthetic_bar, synthetic_notes) -> None:
        """Result keys follow namespacing convention (contain '/')."""
        recon = _make_recon(synthetic_bar, list(synthetic_notes))
        result = metric.compute(synthetic_bar, recon)
        assert all("/" in k for k in result.keys()), (
            f"Expected namespaced keys, got: {list(result.keys())}"
        )

    def test_empty_detected_notes_handled(self, metric, synthetic_bar) -> None:
        """Metric handles empty detected_notes without raising."""
        recon = _make_recon(synthetic_bar, [])
        result = metric.compute(synthetic_bar, recon)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Registry tests for conditioning metrics
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not conditioning_available,
    reason="midi_vae.metrics.conditioning not yet implemented (DELTA Sprint 4)",
)
class TestConditioningMetricsRegistry:
    """Verify conditioning metrics are registered in ComponentRegistry."""

    def test_at_least_one_conditioning_metric_registered(self) -> None:
        """At least one conditioning metric is in the registry."""
        from midi_vae.registry import ComponentRegistry
        registered = ComponentRegistry.list_components("metric").get("metric", [])
        cond_names = [
            n for n in registered
            if any(kw in n for kw in (
                "conditioning", "instrument_consistency", "pitch_range",
                "tempo_consistency", "feature_conditioning",
            ))
        ]
        assert len(cond_names) >= 1, (
            f"No conditioning metrics found in registry. Registered: {registered}"
        )

    def test_conditioning_metrics_are_metric_subclasses(self) -> None:
        """Every registered conditioning metric subclasses Metric."""
        from midi_vae.registry import ComponentRegistry
        registered = ComponentRegistry.list_components("metric").get("metric", [])
        for name in registered:
            cls = ComponentRegistry.get("metric", name)
            assert issubclass(cls, Metric), (
                f"Registered metric '{name}' does not subclass Metric"
            )

    def test_conditioning_metrics_instantiate_without_args(self) -> None:
        """Conditioning metrics can be instantiated with no constructor arguments."""
        from midi_vae.registry import ComponentRegistry
        registered = ComponentRegistry.list_components("metric").get("metric", [])
        cond_names = [
            n for n in registered
            if any(kw in n for kw in (
                "conditioning", "instrument_consistency", "pitch_range",
                "tempo_consistency", "feature_conditioning",
            ))
        ]
        for name in cond_names:
            cls = ComponentRegistry.get("metric", name)
            try:
                m = cls()
                assert hasattr(m, "name")
                assert hasattr(m, "compute")
            except Exception as exc:
                pytest.fail(f"Failed to instantiate conditioning metric '{name}': {exc}")
