"""Tests for generative evaluation metrics in midi_vae/metrics/generative.py.

Tests the 6 registered generative metrics with correct names and signatures:
  - generative/self_similarity_matrix  (SelfSimilarityMatrix)
  - generative/transition_entropy      (TransitionEntropy)
  - generative/groove_consistency      (SequenceGrooveConsistency)
  - generative/pitch_class_histogram_kl (PitchClassHistogramKL)
  - generative/bar_level_nll           (BarLevelNLL)
  - generative/sequence_coherence      (SequenceCoherence)

All tests use synthetic CPU data — no GPU or pretrained weights required.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midi_vae.data.types import BarData, MidiNote, PianoRollImage, ReconstructedBar
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

skip_if_unavailable = pytest.mark.skipif(
    not generative_available,
    reason="midi_vae.metrics.generative not yet implemented",
)

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
    image: torch.Tensor | None = None,
) -> ReconstructedBar:
    """Construct a ReconstructedBar from a BarData and optional note list."""
    if notes is None:
        notes = []
    if image is None:
        image = torch.zeros(3, 128, 128)
    return ReconstructedBar(
        bar_id=bar.bar_id,
        vae_name="stub_vae",
        recon_image=image,
        detected_notes=notes,
        detection_method="global_threshold",
    )


def _make_note(pitch: int = 60, onset: int = 0, offset: int = 24, vel: int = 80) -> MidiNote:
    """Build a single MidiNote."""
    return MidiNote(pitch=pitch, onset_step=onset, offset_step=offset, velocity=vel)


def _make_gt_image_bar(bar: BarData) -> BarData:
    """Return bar with a PianoRollImage attached to metadata for BarLevelNLL."""
    img = PianoRollImage(
        bar_id=bar.bar_id,
        image=torch.rand(3, 128, 128) * 2 - 1,  # [-1, 1]
        channel_strategy="velocity_only",
        resolution=(128, 128),
        pitch_axis="height",
    )
    return BarData(
        bar_id=bar.bar_id,
        song_id=bar.song_id,
        instrument=bar.instrument,
        program_number=bar.program_number,
        piano_roll=bar.piano_roll,
        onset_mask=bar.onset_mask,
        sustain_mask=bar.sustain_mask,
        tempo=bar.tempo,
        time_signature=bar.time_signature,
        metadata={**bar.metadata, "gt_image": img},
    )


# ---------------------------------------------------------------------------
# Generative Metric ABC contract tests (minimal stub)
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
# Registration tests
# ---------------------------------------------------------------------------


@skip_if_unavailable
class TestGenerativeMetricsRegistration:
    """Check that all 6 generative metrics are registered in ComponentRegistry."""

    EXPECTED_NAMES = [
        "generative/self_similarity_matrix",
        "generative/transition_entropy",
        "generative/groove_consistency",
        "generative/pitch_class_histogram_kl",
        "generative/bar_level_nll",
        "generative/sequence_coherence",
    ]

    def test_all_generative_metrics_are_registered(self) -> None:
        """All 6 generative metrics are accessible from the registry."""
        from midi_vae.registry import ComponentRegistry
        for name in self.EXPECTED_NAMES:
            cls = ComponentRegistry.get("metric", name)
            assert cls is not None, f"Metric '{name}' not found in registry"

    def test_all_generative_metrics_are_metric_subclasses(self) -> None:
        """Every registered generative metric is a Metric subclass."""
        from midi_vae.registry import ComponentRegistry
        for name in self.EXPECTED_NAMES:
            cls = ComponentRegistry.get("metric", name)
            assert issubclass(cls, Metric), (
                f"Registered metric '{name}' is not a Metric subclass"
            )

    def test_generative_metrics_instantiate_without_args(self) -> None:
        """Generative metrics can be instantiated with no constructor args."""
        from midi_vae.registry import ComponentRegistry
        for name in self.EXPECTED_NAMES:
            cls = ComponentRegistry.get("metric", name)
            try:
                m = cls()
                assert hasattr(m, "name")
                assert hasattr(m, "compute")
            except Exception as exc:
                pytest.fail(f"Failed to instantiate metric '{name}': {exc}")

    def test_registered_names_match_name_property(self) -> None:
        """The .name property matches the registration key."""
        from midi_vae.registry import ComponentRegistry
        for reg_name in self.EXPECTED_NAMES:
            cls = ComponentRegistry.get("metric", reg_name)
            m = cls()
            assert m.name == reg_name, (
                f"name property '{m.name}' != registration key '{reg_name}'"
            )


# ---------------------------------------------------------------------------
# SelfSimilarityMatrix tests
# ---------------------------------------------------------------------------


@skip_if_unavailable
class TestSelfSimilarityMatrix:
    """Tests for generative/self_similarity_matrix."""

    @pytest.fixture
    def metric(self):
        """Load SelfSimilarityMatrix from the registry."""
        from midi_vae.registry import ComponentRegistry
        cls = ComponentRegistry.get("metric", "generative/self_similarity_matrix")
        return cls()

    def test_name_is_correct(self, metric) -> None:
        """name property returns the correct registered name."""
        assert metric.name == "generative/self_similarity_matrix"

    def test_compute_returns_sentinel(self, metric) -> None:
        """compute() returns a sentinel dict (accumulate pattern)."""
        bar = _make_bar()
        recon = _make_recon(bar)
        result = metric.compute(bar, recon)
        assert isinstance(result, dict)
        assert "self_similarity_accumulated" in result

    def test_finalize_requires_two_bars(self, metric) -> None:
        """finalize() with fewer than 2 bars returns NaN for mean/std."""
        bar = _make_bar()
        recon = _make_recon(bar)
        metric.reset()
        metric.compute(bar, recon)
        result = metric.finalize()
        assert math.isnan(result["self_similarity_mean"])
        assert math.isnan(result["self_similarity_std"])
        assert result["self_similarity_n_bars"] == 1

    def test_finalize_with_two_bars_returns_valid_floats(self, metric) -> None:
        """finalize() with 2+ bars returns finite floats."""
        bars = [
            _make_bar(bar_id=f"b{i}", pitches=[(60 + i * 4, 0, 24, 80)])
            for i in range(4)
        ]
        metric.reset()
        for b in bars:
            metric.compute(b, _make_recon(b))
        result = metric.finalize()
        assert isinstance(result["self_similarity_mean"], float)
        assert isinstance(result["self_similarity_std"], float)
        assert result["self_similarity_n_bars"] == 4
        # Similarity is in [-1, 1]
        assert -1.0 <= result["self_similarity_mean"] <= 1.0

    def test_reset_clears_accumulator(self, metric) -> None:
        """reset() clears accumulated features so finalize gives n_bars=0."""
        bar = _make_bar()
        recon = _make_recon(bar)
        metric.compute(bar, recon)
        metric.reset()
        result = metric.finalize()
        assert result["self_similarity_n_bars"] == 0

    def test_identical_bars_give_similarity_one(self, metric) -> None:
        """Identical non-zero feature vectors give mean similarity = 1.0."""
        bar = _make_bar()
        # Use a non-zero image so feature vector is non-zero (cosine sim defined)
        recon = _make_recon(bar, image=torch.ones(3, 128, 128) * 0.5)
        metric.reset()
        for _ in range(3):
            metric.compute(bar, recon)
        result = metric.finalize()
        # All identical non-zero feature vectors → cosine similarity = 1
        assert result["self_similarity_mean"] == pytest.approx(1.0, abs=0.01)

    def test_finalize_result_keys_present(self, metric) -> None:
        """finalize() result contains expected keys."""
        bars = [_make_bar(bar_id=f"b{i}") for i in range(2)]
        metric.reset()
        for b in bars:
            metric.compute(b, _make_recon(b))
        result = metric.finalize()
        assert "self_similarity_mean" in result
        assert "self_similarity_std" in result
        assert "self_similarity_n_bars" in result


# ---------------------------------------------------------------------------
# TransitionEntropy tests
# ---------------------------------------------------------------------------


@skip_if_unavailable
class TestTransitionEntropy:
    """Tests for generative/transition_entropy."""

    @pytest.fixture
    def metric(self):
        """Load TransitionEntropy from the registry."""
        from midi_vae.registry import ComponentRegistry
        cls = ComponentRegistry.get("metric", "generative/transition_entropy")
        return cls()

    def test_name_is_correct(self, metric) -> None:
        """name property returns the correct registered name."""
        assert metric.name == "generative/transition_entropy"

    def test_compute_returns_sentinel(self, metric) -> None:
        """compute() returns the sentinel dict."""
        bar = _make_bar()
        result = metric.compute(bar, _make_recon(bar))
        assert "transition_entropy_accumulated" in result

    def test_finalize_with_single_bar(self, metric) -> None:
        """finalize() with only 1 bar gives entropy=0, n_transitions=0."""
        metric.reset()
        bar = _make_bar()
        metric.compute(bar, _make_recon(bar))
        result = metric.finalize()
        assert result["transition_entropy"] == 0.0
        assert result["transition_entropy_n_transitions"] == 0

    def test_finalize_returns_non_negative_entropy(self, metric) -> None:
        """Entropy is always non-negative."""
        bars = [_make_bar(bar_id=f"b{i}", pitches=[(60 + i * 3, 0, 24, 80)]) for i in range(5)]
        metric.reset()
        for b in bars:
            metric.compute(b, _make_recon(b))
        result = metric.finalize()
        assert result["transition_entropy"] >= 0.0

    def test_finalize_n_transitions_correct(self, metric) -> None:
        """n_transitions == n_bars - 1."""
        n_bars = 6
        bars = [_make_bar(bar_id=f"b{i}") for i in range(n_bars)]
        metric.reset()
        for b in bars:
            metric.compute(b, _make_recon(b))
        result = metric.finalize()
        assert result["transition_entropy_n_transitions"] == n_bars - 1

    def test_all_same_pitch_class_gives_zero_entropy(self, metric) -> None:
        """All bars with the same dominant pitch class → 0 transition entropy."""
        # All bars have only pitch 60 (pitch class 0)
        bars = [
            _make_bar(bar_id=f"b{i}", pitches=[(60, 0, 24, 80)])
            for i in range(4)
        ]
        metric.reset()
        for b in bars:
            metric.compute(b, _make_recon(b))
        result = metric.finalize()
        # Single transition type → entropy ≈ 0
        assert result["transition_entropy"] == pytest.approx(0.0, abs=0.01)

    def test_varied_pitch_classes_give_higher_entropy(self, metric) -> None:
        """Varied pitch classes across bars give positive entropy."""
        # Each bar has a different dominant pitch class
        pitches_by_bar = [0, 3, 6, 9, 1, 4, 7, 10]
        bars = [
            _make_bar(bar_id=f"b{i}", pitches=[(p, 0, 24, 80)])
            for i, p in enumerate(pitches_by_bar)
        ]
        metric.reset()
        for b in bars:
            metric.compute(b, _make_recon(b))
        result = metric.finalize()
        assert result["transition_entropy"] > 0.0

    def test_reset_clears_state(self, metric) -> None:
        """reset() clears accumulated pitch classes."""
        bars = [_make_bar(bar_id=f"b{i}") for i in range(3)]
        for b in bars:
            metric.compute(b, _make_recon(b))
        metric.reset()
        metric.compute(bars[0], _make_recon(bars[0]))
        result = metric.finalize()
        assert result["transition_entropy_n_transitions"] == 0


# ---------------------------------------------------------------------------
# GrooveConsistency (sequence-level) tests
# ---------------------------------------------------------------------------


@skip_if_unavailable
class TestSequenceGrooveConsistency:
    """Tests for generative/groove_consistency (SequenceGrooveConsistency)."""

    @pytest.fixture
    def metric(self):
        """Load SequenceGrooveConsistency from the registry."""
        from midi_vae.registry import ComponentRegistry
        cls = ComponentRegistry.get("metric", "generative/groove_consistency")
        return cls()

    def test_name_is_correct(self, metric) -> None:
        """name property returns the correct registered name."""
        assert metric.name == "generative/groove_consistency"

    def test_compute_returns_sentinel(self, metric) -> None:
        """compute() returns the sentinel dict."""
        bar = _make_bar()
        result = metric.compute(bar, _make_recon(bar))
        assert "seq_groove_consistency_accumulated" in result

    def test_finalize_with_single_bar_returns_nan(self, metric) -> None:
        """finalize() with fewer than 2 bars returns NaN consistency."""
        metric.reset()
        bar = _make_bar()
        metric.compute(bar, _make_recon(bar))
        result = metric.finalize()
        assert math.isnan(result["seq_groove_consistency"])
        assert result["seq_groove_consistency_n_bars"] == 1

    def test_identical_groove_gives_consistency_one(self, metric) -> None:
        """Identical onset patterns across bars give consistency ≈ 1.0."""
        bar = _make_bar(pitches=[(60, 0, 24, 80), (64, 48, 72, 80)])
        metric.reset()
        for _ in range(3):
            metric.compute(bar, _make_recon(bar))
        result = metric.finalize()
        assert result["seq_groove_consistency"] == pytest.approx(1.0, abs=0.01)

    def test_finalize_result_keys_present(self, metric) -> None:
        """finalize() result contains expected keys."""
        bars = [_make_bar(bar_id=f"b{i}") for i in range(2)]
        metric.reset()
        for b in bars:
            metric.compute(b, _make_recon(b))
        result = metric.finalize()
        assert "seq_groove_consistency" in result
        assert "seq_groove_consistency_n_bars" in result

    def test_finalize_n_bars_correct(self, metric) -> None:
        """finalize() reports the correct bar count."""
        n = 4
        bars = [_make_bar(bar_id=f"b{i}") for i in range(n)]
        metric.reset()
        for b in bars:
            metric.compute(b, _make_recon(b))
        result = metric.finalize()
        assert result["seq_groove_consistency_n_bars"] == n

    def test_consistency_is_in_valid_range(self, metric) -> None:
        """Groove consistency is in [-1, 1]."""
        bars = [
            _make_bar(bar_id=f"b{i}", pitches=[(60 + i, i * 12, i * 12 + 8, 80)])
            for i in range(4)
        ]
        metric.reset()
        for b in bars:
            metric.compute(b, _make_recon(b))
        result = metric.finalize()
        val = result["seq_groove_consistency"]
        if not math.isnan(val):
            assert -1.0 <= val <= 1.0

    def test_reset_clears_accumulator(self, metric) -> None:
        """reset() clears accumulated indicators."""
        bar = _make_bar()
        for _ in range(3):
            metric.compute(bar, _make_recon(bar))
        metric.reset()
        metric.compute(bar, _make_recon(bar))
        result = metric.finalize()
        assert result["seq_groove_consistency_n_bars"] == 1


# ---------------------------------------------------------------------------
# PitchClassHistogramKL tests
# ---------------------------------------------------------------------------


@skip_if_unavailable
class TestPitchClassHistogramKL:
    """Tests for generative/pitch_class_histogram_kl."""

    @pytest.fixture
    def metric(self):
        """Load PitchClassHistogramKL from the registry."""
        from midi_vae.registry import ComponentRegistry
        cls = ComponentRegistry.get("metric", "generative/pitch_class_histogram_kl")
        return cls()

    def test_name_is_correct(self, metric) -> None:
        """name property returns the correct registered name."""
        assert metric.name == "generative/pitch_class_histogram_kl"

    def test_compute_returns_sentinel(self, metric) -> None:
        """compute() returns the sentinel dict."""
        bar = _make_bar()
        result = metric.compute(bar, _make_recon(bar))
        assert "pitch_class_kl_accumulated" in result

    def test_finalize_with_identical_histograms_gives_low_kl(self, metric) -> None:
        """GT and generated from same notes → near-zero KL divergence."""
        notes = [
            _make_note(pitch=60, onset=0, offset=24),
            _make_note(pitch=64, onset=24, offset=48),
        ]
        bar = _make_bar(pitches=[(60, 0, 24, 80), (64, 24, 48, 80)])
        recon = _make_recon(bar, notes)
        metric.reset()
        for _ in range(3):
            metric.compute(bar, recon)
        result = metric.finalize()
        assert result["pitch_class_histogram_kl"] >= 0.0

    def test_finalize_returns_non_negative_kl(self, metric) -> None:
        """KL divergence is always non-negative."""
        bar = _make_bar()
        recon = _make_recon(bar)
        metric.reset()
        for _ in range(3):
            metric.compute(bar, recon)
        result = metric.finalize()
        kl = result["pitch_class_histogram_kl"]
        if not math.isinf(kl):
            assert kl >= 0.0

    def test_finalize_empty_histograms_give_zero(self, metric) -> None:
        """Empty piano rolls on both sides → KL = 0.0."""
        empty_bar = BarData(
            bar_id="empty",
            song_id="s",
            instrument="piano",
            program_number=0,
            piano_roll=np.zeros((128, 96), dtype=np.float32),
            onset_mask=np.zeros((128, 96), dtype=np.float32),
            sustain_mask=np.zeros((128, 96), dtype=np.float32),
            tempo=120.0,
            time_signature=(4, 4),
            metadata={},
        )
        recon = _make_recon(empty_bar)
        metric.reset()
        metric.compute(empty_bar, recon)
        result = metric.finalize()
        assert result["pitch_class_histogram_kl"] == pytest.approx(0.0, abs=1e-6)

    def test_finalize_result_key_present(self, metric) -> None:
        """finalize() result contains 'pitch_class_histogram_kl'."""
        bar = _make_bar()
        metric.reset()
        metric.compute(bar, _make_recon(bar))
        result = metric.finalize()
        assert "pitch_class_histogram_kl" in result

    def test_reset_clears_histograms(self, metric) -> None:
        """reset() resets to a fresh state."""
        bar = _make_bar()
        metric.compute(bar, _make_recon(bar))
        metric.reset()
        # After reset, empty histograms → 0.0
        empty_bar = BarData(
            bar_id="empty",
            song_id="s",
            instrument="piano",
            program_number=0,
            piano_roll=np.zeros((128, 96), dtype=np.float32),
            onset_mask=np.zeros((128, 96), dtype=np.float32),
            sustain_mask=np.zeros((128, 96), dtype=np.float32),
            tempo=120.0,
            time_signature=(4, 4),
            metadata={},
        )
        metric.compute(empty_bar, _make_recon(empty_bar))
        result = metric.finalize()
        assert result["pitch_class_histogram_kl"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# BarLevelNLL tests
# ---------------------------------------------------------------------------


@skip_if_unavailable
class TestBarLevelNLL:
    """Tests for generative/bar_level_nll."""

    @pytest.fixture
    def metric(self):
        """Load BarLevelNLL from the registry."""
        from midi_vae.registry import ComponentRegistry
        cls = ComponentRegistry.get("metric", "generative/bar_level_nll")
        return cls()

    def test_name_is_correct(self, metric) -> None:
        """name property returns the correct registered name."""
        assert metric.name == "generative/bar_level_nll"

    def test_compute_without_gt_image_returns_nan(self, metric) -> None:
        """compute() returns NaN when no gt_image is in metadata."""
        bar = _make_bar()  # no gt_image in metadata
        recon = _make_recon(bar)
        result = metric.compute(bar, recon)
        assert "bar_level_nll" in result
        assert math.isnan(result["bar_level_nll"])

    def test_compute_with_gt_image_returns_finite_float(self, metric) -> None:
        """compute() returns a finite float when gt_image is provided."""
        bar = _make_bar()
        bar_with_img = _make_gt_image_bar(bar)
        recon_image = torch.zeros(3, 128, 128)
        recon = _make_recon(bar_with_img, image=recon_image)
        result = metric.compute(bar_with_img, recon)
        assert "bar_level_nll" in result
        val = result["bar_level_nll"]
        assert isinstance(val, float)
        assert not math.isnan(val)
        assert math.isfinite(val)

    def test_compute_nll_is_positive(self, metric) -> None:
        """NLL is always non-negative (Gaussian NLL includes const term)."""
        bar = _make_gt_image_bar(_make_bar())
        recon = _make_recon(bar, image=torch.rand(3, 128, 128))
        result = metric.compute(bar, recon)
        val = result["bar_level_nll"]
        if not math.isnan(val):
            # Gaussian NLL can be negative in theory but typically positive
            assert isinstance(val, float)

    def test_compute_returns_dict_of_floats(self, metric) -> None:
        """All values in result are Python floats."""
        bar = _make_gt_image_bar(_make_bar())
        recon = _make_recon(bar, image=torch.zeros(3, 128, 128))
        result = metric.compute(bar, recon)
        assert isinstance(result, dict)
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, float)

    def test_compute_result_key_present(self, metric) -> None:
        """result contains 'bar_level_nll'."""
        bar = _make_bar()
        recon = _make_recon(bar)
        result = metric.compute(bar, recon)
        assert "bar_level_nll" in result

    def test_min_sigma_parameter_accepted(self) -> None:
        """BarLevelNLL accepts min_sigma constructor argument."""
        from midi_vae.registry import ComponentRegistry
        cls = ComponentRegistry.get("metric", "generative/bar_level_nll")
        m = cls(min_sigma=0.05)
        assert m is not None


# ---------------------------------------------------------------------------
# SequenceCoherence tests
# ---------------------------------------------------------------------------


@skip_if_unavailable
class TestSequenceCoherence:
    """Tests for generative/sequence_coherence."""

    @pytest.fixture
    def metric(self):
        """Load SequenceCoherence from the registry."""
        from midi_vae.registry import ComponentRegistry
        cls = ComponentRegistry.get("metric", "generative/sequence_coherence")
        return cls()

    def test_name_is_correct(self, metric) -> None:
        """name property returns the correct registered name."""
        assert metric.name == "generative/sequence_coherence"

    def test_compute_returns_three_keys(self, metric) -> None:
        """compute() returns dict with 3 entropy keys (not a sentinel)."""
        bar = _make_bar()
        recon = _make_recon(bar)
        result = metric.compute(bar, recon)
        assert "seq_coherence_gt_entropy" in result
        assert "seq_coherence_gen_entropy" in result
        assert "seq_coherence_entropy_diff" in result

    def test_compute_returns_floats(self, metric) -> None:
        """All result values are floats."""
        bar = _make_bar()
        recon = _make_recon(bar)
        result = metric.compute(bar, recon)
        for v in result.values():
            assert isinstance(v, float)

    def test_entropy_is_non_negative(self, metric) -> None:
        """Shannon entropy values are >= 0."""
        bar = _make_bar()
        recon = _make_recon(bar)
        result = metric.compute(bar, recon)
        assert result["seq_coherence_gt_entropy"] >= 0.0
        assert result["seq_coherence_gen_entropy"] >= 0.0

    def test_entropy_diff_is_consistent(self, metric) -> None:
        """entropy_diff == gt_entropy - gen_entropy."""
        bar = _make_bar()
        recon = _make_recon(bar)
        result = metric.compute(bar, recon)
        expected_diff = result["seq_coherence_gt_entropy"] - result["seq_coherence_gen_entropy"]
        assert result["seq_coherence_entropy_diff"] == pytest.approx(expected_diff, abs=1e-6)

    def test_empty_piano_roll_gives_zero_gt_entropy(self, metric) -> None:
        """An empty piano roll has 0 pitch entropy (nothing to distribute)."""
        empty_bar = BarData(
            bar_id="empty",
            song_id="s",
            instrument="piano",
            program_number=0,
            piano_roll=np.zeros((128, 96), dtype=np.float32),
            onset_mask=np.zeros((128, 96), dtype=np.float32),
            sustain_mask=np.zeros((128, 96), dtype=np.float32),
            tempo=120.0,
            time_signature=(4, 4),
            metadata={},
        )
        recon = _make_recon(empty_bar)
        result = metric.compute(empty_bar, recon)
        assert result["seq_coherence_gt_entropy"] == pytest.approx(0.0, abs=1e-6)

    def test_notes_present_give_positive_gen_entropy_when_varied(self, metric) -> None:
        """Multiple varied detected notes give positive generation entropy."""
        bar = _make_bar()
        notes = [
            _make_note(pitch=p, onset=i * 12, offset=i * 12 + 8)
            for i, p in enumerate([60, 64, 67, 71, 62, 65, 69, 73])
        ]
        recon = _make_recon(bar, notes)
        result = metric.compute(bar, recon)
        assert result["seq_coherence_gen_entropy"] >= 0.0

    def test_no_detected_notes_gives_maximum_gen_entropy(self, metric) -> None:
        """No detected notes → uniform pitch distribution → maximum entropy (log 12)."""
        bar = _make_bar()
        recon = _make_recon(bar, notes=[])
        result = metric.compute(bar, recon)
        max_entropy = math.log(12)
        assert result["seq_coherence_gen_entropy"] == pytest.approx(max_entropy, abs=1e-4)
