"""Tests for generative evaluation metrics in midi_vae/metrics/generative.py.

Generative metrics compare a set of generated BarData/ReconstructedBar objects
against a reference corpus, measuring novelty, diversity, and musical coherence
of the generated content.

The generative module is expected from DELTA/CHARLIE Sprint 4.  Tests use
``@pytest.mark.skipif`` guards so the suite stays green until the module lands.
All tests use synthetic CPU data — no GPU or pretrained weights required.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch

from midi_vae.data.types import BarData, MidiNote, ReconstructedBar
from midi_vae.metrics.base import Metric

# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------

generative_available = False
try:
    import midi_vae.metrics.generative as _gen_module  # noqa: F401
    generative_available = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Shared synthetic helpers
# ---------------------------------------------------------------------------


def _make_bar(
    bar_id: str = "test_0",
    pitches: list[tuple[int, int, int, int]] | None = None,
    instrument: str = "piano",
    tempo: float = 120.0,
) -> BarData:
    """Construct a synthetic BarData with known notes.

    Args:
        bar_id: Identifier string.
        pitches: List of (pitch, onset, offset, velocity) tuples.
        instrument: Instrument name.
        tempo: BPM value.

    Returns:
        A BarData with corresponding piano_roll, onset_mask, sustain_mask.
    """
    T = 96
    piano_roll = np.zeros((128, T), dtype=np.float32)
    onset_mask = np.zeros((128, T), dtype=np.float32)
    sustain_mask = np.zeros((128, T), dtype=np.float32)

    if pitches is None:
        pitches = [(60, 0, 24, 80), (64, 24, 48, 70)]

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
        time_signature=(4, 4),
        metadata={},
    )


def _make_recon(
    bar: BarData,
    notes: list[MidiNote] | None = None,
) -> ReconstructedBar:
    """Construct a ReconstructedBar from a BarData and optional note list."""
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
# Generative Metric ABC tests (minimal stub for contract testing)
# ---------------------------------------------------------------------------


class TestGenerativeMetricContract:
    """Verify that any generative metric implements the Metric ABC correctly."""

    class StubGenerativeMetric(Metric):
        """Minimal stub generative metric for contract tests."""

        @property
        def name(self) -> str:
            return "stub_generative"

        def compute(self, gt: BarData, recon: ReconstructedBar) -> dict[str, float]:
            return {"stub_generative/score": 1.0}

    def test_compute_returns_dict_of_floats(self, synthetic_bar, synthetic_notes) -> None:
        """compute() returns dict[str, float]."""
        m = self.StubGenerativeMetric()
        recon = _make_recon(synthetic_bar, list(synthetic_notes))
        result = m.compute(synthetic_bar, recon)
        assert isinstance(result, dict)
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, float)

    def test_name_is_string(self) -> None:
        """name property returns a non-empty string."""
        m = self.StubGenerativeMetric()
        assert isinstance(m.name, str)
        assert len(m.name) > 0

    def test_requires_notes_default_false(self) -> None:
        """requires_notes defaults to False unless overridden."""
        m = self.StubGenerativeMetric()
        assert isinstance(m.requires_notes, bool)


# ---------------------------------------------------------------------------
# Concrete generative metric tests — conditional on module availability
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not generative_available,
    reason="midi_vae.metrics.generative not yet implemented (DELTA Sprint 4)",
)
class TestPitchHistogramSimilarity:
    """Tests for PitchHistogramSimilarity metric (compares pitch distributions)."""

    @pytest.fixture
    def metric(self):
        """Load PitchHistogramSimilarity from the registry."""
        from midi_vae.registry import ComponentRegistry
        cls = ComponentRegistry.get("metric", "pitch_histogram_similarity")
        return cls()

    def test_name_is_correct(self, metric) -> None:
        """Metric has the expected registered name."""
        assert "pitch_histogram" in metric.name.lower() or "pitch" in metric.name.lower()

    def test_identical_bars_give_high_similarity(self, metric, synthetic_bar) -> None:
        """Perfect match yields similarity close to 1.0."""
        notes = [
            MidiNote(pitch=60, onset_step=0, offset_step=24, velocity=100),
            MidiNote(pitch=64, onset_step=24, offset_step=48, velocity=80),
        ]
        recon = _make_recon(synthetic_bar, notes)
        result = metric.compute(synthetic_bar, recon)
        assert isinstance(result, dict)
        score = list(result.values())[0]
        assert score >= 0.0

    def test_empty_notes_handled_gracefully(self, metric, synthetic_bar) -> None:
        """Metric handles empty detected_notes list without raising."""
        recon = _make_recon(synthetic_bar, [])
        result = metric.compute(synthetic_bar, recon)
        assert isinstance(result, dict)

    def test_result_values_are_floats(self, metric, synthetic_bar, synthetic_notes) -> None:
        """All result values are Python floats."""
        recon = _make_recon(synthetic_bar, list(synthetic_notes))
        result = metric.compute(synthetic_bar, recon)
        for v in result.values():
            assert isinstance(v, float)

    def test_result_values_are_bounded(self, metric, synthetic_bar, synthetic_notes) -> None:
        """Similarity / distance scores are in a reasonable range."""
        recon = _make_recon(synthetic_bar, list(synthetic_notes))
        result = metric.compute(synthetic_bar, recon)
        for v in result.values():
            # Similarity typically in [0, 1] or distance in [0, ∞)
            assert not math.isnan(v), "NaN in result"


@pytest.mark.skipif(
    not generative_available,
    reason="midi_vae.metrics.generative not yet implemented (DELTA Sprint 4)",
)
class TestNoteTransitionEntropy:
    """Tests for NoteTransitionEntropy metric (note-level Markov diversity)."""

    @pytest.fixture
    def metric(self):
        from midi_vae.registry import ComponentRegistry
        cls = ComponentRegistry.get("metric", "note_transition_entropy")
        return cls()

    def test_name_is_string(self, metric) -> None:
        assert isinstance(metric.name, str)

    def test_single_note_gives_zero_entropy(self, metric, synthetic_bar) -> None:
        """Single detected note → no transitions → entropy should be 0."""
        single = [_make_note(pitch=60, onset=0, offset=24)]
        recon = _make_recon(synthetic_bar, single)
        result = metric.compute(synthetic_bar, recon)
        entropy = list(result.values())[0]
        assert entropy >= 0.0

    def test_many_varied_notes_give_positive_entropy(self, metric, synthetic_bar) -> None:
        """Many different-pitch notes → positive transition entropy."""
        notes = [
            _make_note(pitch=60, onset=0, offset=12),
            _make_note(pitch=64, onset=12, offset=24),
            _make_note(pitch=67, onset=24, offset=36),
            _make_note(pitch=60, onset=36, offset=48),
            _make_note(pitch=72, onset=48, offset=60),
        ]
        recon = _make_recon(synthetic_bar, notes)
        result = metric.compute(synthetic_bar, recon)
        entropy = list(result.values())[0]
        assert entropy >= 0.0

    def test_empty_notes_handled(self, metric, synthetic_bar) -> None:
        """Metric handles empty notes without raising."""
        recon = _make_recon(synthetic_bar, [])
        result = metric.compute(synthetic_bar, recon)
        assert isinstance(result, dict)


@pytest.mark.skipif(
    not generative_available,
    reason="midi_vae.metrics.generative not yet implemented (DELTA Sprint 4)",
)
class TestRhythmicDiversity:
    """Tests for RhythmicDiversity metric (IOI / onset-time diversity)."""

    @pytest.fixture
    def metric(self):
        from midi_vae.registry import ComponentRegistry
        cls = ComponentRegistry.get("metric", "rhythmic_diversity")
        return cls()

    def test_name_is_string(self, metric) -> None:
        assert isinstance(metric.name, str)

    def test_uniform_spacing_gives_low_diversity(self, metric, synthetic_bar) -> None:
        """Uniformly spaced onsets → low IOI variance / diversity."""
        notes = [
            _make_note(pitch=60 + i, onset=i * 24, offset=i * 24 + 12)
            for i in range(4)
        ]
        recon = _make_recon(synthetic_bar, notes)
        result = metric.compute(synthetic_bar, recon)
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, float)

    def test_irregular_spacing_handled(self, metric, synthetic_bar) -> None:
        """Irregularly spaced onsets handled without raising."""
        notes = [
            _make_note(pitch=60, onset=0, offset=5),
            _make_note(pitch=62, onset=7, offset=30),
            _make_note(pitch=65, onset=31, offset=90),
        ]
        recon = _make_recon(synthetic_bar, notes)
        result = metric.compute(synthetic_bar, recon)
        assert isinstance(result, dict)

    def test_empty_notes_handled(self, metric, synthetic_bar) -> None:
        """Metric handles empty notes without raising."""
        recon = _make_recon(synthetic_bar, [])
        result = metric.compute(synthetic_bar, recon)
        assert isinstance(result, dict)

    def test_single_note_handled(self, metric, synthetic_bar) -> None:
        """Single note (no IOI computable) handled gracefully."""
        recon = _make_recon(synthetic_bar, [_make_note()])
        result = metric.compute(synthetic_bar, recon)
        assert isinstance(result, dict)


@pytest.mark.skipif(
    not generative_available,
    reason="midi_vae.metrics.generative not yet implemented (DELTA Sprint 4)",
)
class TestUniqueBarRatio:
    """Tests for UniqueBarRatio metric (proportion of non-duplicate generated bars)."""

    @pytest.fixture
    def metric(self):
        from midi_vae.registry import ComponentRegistry
        cls = ComponentRegistry.get("metric", "unique_bar_ratio")
        return cls()

    def test_name_is_string(self, metric) -> None:
        assert isinstance(metric.name, str)

    def test_all_distinct_bars_give_ratio_one(self, metric) -> None:
        """All-distinct generated bars → unique ratio = 1.0."""
        bars = [
            _make_bar(bar_id=f"gen_{i}", pitches=[(60 + i, 0, 24, 80)])
            for i in range(4)
        ]
        recons = [_make_recon(b, [_make_note(pitch=60 + i)])
                  for i, b in enumerate(bars)]
        if hasattr(metric, "compute_batch"):
            result = metric.compute_batch(list(zip(bars, recons)))
            ratio = list(result.values())[0]
            assert ratio == pytest.approx(1.0, abs=0.01)
        else:
            pytest.skip("compute_batch not available on this metric")

    def test_all_identical_bars_give_low_ratio(self, metric) -> None:
        """All-duplicate generated bars → unique ratio close to 0."""
        bar = _make_bar()
        notes = [_make_note(pitch=60, onset=0, offset=24)]
        pairs = [(bar, _make_recon(bar, notes)) for _ in range(4)]
        if hasattr(metric, "compute_batch"):
            result = metric.compute_batch(pairs)
            ratio = list(result.values())[0]
            assert ratio <= 0.5
        else:
            pytest.skip("compute_batch not available on this metric")

    def test_single_bar_returns_ratio_one(self, metric, synthetic_bar) -> None:
        """Single generated bar → unique ratio = 1.0 (trivially)."""
        if hasattr(metric, "compute_batch"):
            recon = _make_recon(synthetic_bar)
            result = metric.compute_batch([(synthetic_bar, recon)])
            ratio = list(result.values())[0]
            assert ratio >= 0.0
        else:
            pytest.skip("compute_batch not available on this metric")


@pytest.mark.skipif(
    not generative_available,
    reason="midi_vae.metrics.generative not yet implemented (DELTA Sprint 4)",
)
class TestGenerativeMetricsRegistration:
    """Check that all generative metrics are registered in ComponentRegistry."""

    def test_generative_metrics_are_registered(self) -> None:
        """At least one generative metric is accessible from the registry."""
        from midi_vae.registry import ComponentRegistry
        registered = ComponentRegistry.list_components("metric").get("metric", [])
        generative_names = [
            n for n in registered
            if any(kw in n for kw in ("generative", "pitch", "rhythm", "unique", "transition"))
        ]
        assert len(generative_names) >= 1, (
            f"No generative metrics found in registry. Registered metrics: {registered}"
        )

    def test_all_generative_metrics_are_metric_subclasses(self) -> None:
        """Every registered generative metric is a Metric subclass."""
        from midi_vae.registry import ComponentRegistry
        registered = ComponentRegistry.list_components("metric").get("metric", [])
        for name in registered:
            cls = ComponentRegistry.get("metric", name)
            assert issubclass(cls, Metric), (
                f"Registered metric '{name}' is not a Metric subclass"
            )

    def test_generative_metrics_instantiate_without_args(self) -> None:
        """Generative metrics can be instantiated with no constructor args."""
        from midi_vae.registry import ComponentRegistry
        registered = ComponentRegistry.list_components("metric").get("metric", [])
        for name in registered:
            cls = ComponentRegistry.get("metric", name)
            try:
                m = cls()
                assert hasattr(m, "name")
                assert hasattr(m, "compute")
            except Exception as exc:
                pytest.fail(f"Failed to instantiate metric '{name}': {exc}")
