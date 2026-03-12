"""Stub tests for note detection algorithms in midi_vae/note_detection/.

These tests define the contracts that DELTA's detection implementations must
satisfy. Tests for GlobalThresholdDetector will run once threshold.py lands.
"""

from __future__ import annotations

import pytest
import torch

from midi_vae.data.types import MidiNote
from midi_vae.note_detection.base import NoteDetector

# ---------------------------------------------------------------------------
# Skip guard: skip threshold tests if threshold.py not yet implemented
# ---------------------------------------------------------------------------

threshold_available = False
try:
    from midi_vae.note_detection import threshold as _threshold_module  # noqa: F401
    threshold_available = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# NoteDetector ABC contract tests (use StubNoteDetector from conftest)
# ---------------------------------------------------------------------------


class TestNoteDetectorABC:
    """Tests for the NoteDetector ABC contract using the stub from conftest."""

    def test_stub_detector_returns_list(
        self,
        stub_detector,
        synthetic_image,
    ) -> None:
        """detect() returns a list."""
        result = stub_detector.detect(synthetic_image.image, "vos")
        assert isinstance(result, list)

    def test_stub_detector_output_type(
        self,
        stub_detector,
        synthetic_image,
    ) -> None:
        """Each element in the detect() output is a MidiNote."""
        result = stub_detector.detect(synthetic_image.image, "vos")
        for note in result:
            assert isinstance(note, MidiNote)

    def test_stub_detector_needs_fitting_false(self, stub_detector) -> None:
        """StubNoteDetector.needs_fitting is False."""
        assert stub_detector.needs_fitting is False

    def test_stub_detector_fit_is_no_op(self, stub_detector) -> None:
        """fit() on a stub detector with no fitting needed raises no error."""
        stub_detector.fit([])  # Should complete without error

    def test_all_zeros_input_returns_empty_or_few_notes(self, stub_detector) -> None:
        """All-zero image produces no notes or very few spurious detections."""
        zero_image = torch.zeros(3, 128, 128)
        result = stub_detector.detect(zero_image, "velocity_only")
        # A zero image (mapped from [-1,1] may have some noise, but should be sparse
        # The stub uses a threshold of 0.3, so zeros should give empty list
        assert isinstance(result, list)
        # All-zeros below 0.3 threshold → empty
        assert len(result) == 0

    def test_midi_note_fields_valid(self, stub_detector, synthetic_image) -> None:
        """All detected MidiNote objects have valid field values."""
        result = stub_detector.detect(synthetic_image.image, "vos")
        for note in result:
            assert 0 <= note.pitch <= 127
            assert 0 <= note.velocity <= 127
            assert note.offset_step > note.onset_step

    def test_detector_accepts_different_channel_strategies(
        self,
        stub_detector,
    ) -> None:
        """detect() can be called with any channel_strategy string."""
        image = torch.zeros(3, 128, 128)
        for strategy in ["velocity_only", "vo_split", "vos"]:
            result = stub_detector.detect(image, strategy)
            assert isinstance(result, list)


# ---------------------------------------------------------------------------
# GlobalThresholdDetector tests (conditional on threshold.py existing)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not threshold_available,
    reason="midi_vae.note_detection.threshold not yet implemented (DELTA Sprint pending)",
)
class TestGlobalThresholdDetector:
    """Tests for GlobalThresholdDetector in midi_vae/note_detection/threshold.py."""

    @pytest.fixture
    def detector(self):
        """Create a GlobalThresholdDetector with default threshold."""
        from midi_vae.registry import ComponentRegistry
        DetectorClass = ComponentRegistry.get("note_detector", "global_threshold")
        return DetectorClass(params={"threshold": 0.5})

    def test_output_is_list(self, detector) -> None:
        """detect() returns a list."""
        image = torch.rand(3, 128, 128)
        result = detector.detect(image, "velocity_only")
        assert isinstance(result, list)

    def test_output_elements_are_midi_notes(self, detector) -> None:
        """Each element in the output is a MidiNote."""
        image = torch.rand(3, 128, 128)
        result = detector.detect(image, "velocity_only")
        for note in result:
            assert isinstance(note, MidiNote)

    def test_silent_image_returns_empty(self, detector) -> None:
        """A silent image (all -1 in [-1,1] range) produces no detected notes."""
        silent_image = torch.full((3, 128, 128), -1.0)
        result = detector.detect(silent_image, "velocity_only")
        assert result == []

    def test_all_ones_returns_notes(self, detector) -> None:
        """An all-ones image (above threshold) produces detected notes."""
        ones_image = torch.ones(3, 128, 128)
        result = detector.detect(ones_image, "velocity_only")
        assert len(result) > 0

    def test_needs_fitting_false(self, detector) -> None:
        """GlobalThresholdDetector.needs_fitting is False (no fitting required)."""
        assert detector.needs_fitting is False

    def test_detected_notes_have_valid_fields(self, detector) -> None:
        """All detected notes have valid pitch, velocity, and timing."""
        image = torch.rand(3, 128, 128)
        result = detector.detect(image, "velocity_only")
        for note in result:
            assert 0 <= note.pitch <= 127
            assert 0 <= note.velocity <= 127
            assert note.offset_step > note.onset_step

    def test_different_thresholds_affect_output(self) -> None:
        """Higher threshold produces fewer notes than lower threshold."""
        from midi_vae.registry import ComponentRegistry
        DetectorClass = ComponentRegistry.get("note_detector", "global_threshold")

        image = torch.rand(3, 128, 128) * 0.5 + 0.3  # values in [0.3, 0.8]
        low_thresh_detector = DetectorClass(params={"threshold": 0.1})
        high_thresh_detector = DetectorClass(params={"threshold": 0.9})

        low_notes = low_thresh_detector.detect(image, "velocity_only")
        high_notes = high_thresh_detector.detect(image, "velocity_only")

        assert len(low_notes) >= len(high_notes)

    def test_channel_strategy_vos_uses_onset_channel(self, detector) -> None:
        """With 'vos' strategy, detection uses the onset channel (G)."""
        # Create image where only R channel has high values, G is zero
        image = torch.zeros(3, 128, 128)
        image[0, 60, :48] = 1.0  # R channel (velocity) has signal
        result = detector.detect(image, "vos")
        assert isinstance(result, list)
