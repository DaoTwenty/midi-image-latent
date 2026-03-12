"""Note detection algorithms for converting reconstructed images to MIDI notes."""

from midi_vae.note_detection.base import NoteDetector
from midi_vae.note_detection.threshold import GlobalThresholdDetector

__all__ = [
    "NoteDetector",
    "GlobalThresholdDetector",
]
