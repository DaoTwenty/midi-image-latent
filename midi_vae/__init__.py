"""MIDI Image VAE — Pretrained Image VAEs for MIDI Encoding via Piano-Roll Representations."""

__version__ = "0.1.0"

from midi_vae.config import ExperimentConfig, load_config
from midi_vae.registry import ComponentRegistry
from midi_vae.data.types import BarData, PianoRollImage, LatentEncoding, ReconstructedBar, MidiNote

__all__ = [
    "ExperimentConfig",
    "load_config",
    "ComponentRegistry",
    "BarData",
    "PianoRollImage",
    "LatentEncoding",
    "ReconstructedBar",
    "MidiNote",
]
