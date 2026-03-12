"""Data types, preprocessing, rendering, transforms, and dataset modules.

Importing this package registers all channel strategies and dataset
classes into the ComponentRegistry automatically.
"""

# Types must be imported first as they have no internal dependencies.
from midi_vae.data.types import BarData, PianoRollImage, LatentEncoding, ReconstructedBar, MidiNote

# Rendering imports types only — safe to import second.
# Importing this module triggers @ComponentRegistry.register decorators for
# channel strategies: velocity_only, vo_split, vos.
from midi_vae.data.rendering import (
    ChannelStrategy,
    VelocityOnlyStrategy,
    VOSplitStrategy,
    VOSStrategy,
    build_strategy,
)

# Transforms have no internal dependencies.
from midi_vae.data.transforms import (
    ResizeTransform,
    NormalizeTransform,
    PadTransform,
    PitchAxisFlip,
    Compose,
)

# Preprocessing imports types, registry, torch, numpy — safe.
# Importing triggers @ComponentRegistry.register decorator for midi_ingestor.
from midi_vae.data.preprocessing import MidiIngestor

# Datasets import rendering and preprocessing (already loaded above) — safe.
# Importing triggers @ComponentRegistry.register decorators for lpd5, pop909, maestro.
from midi_vae.data.datasets import LPD5Dataset, Pop909Dataset, MaestroDataset

__all__ = [
    # Types
    "BarData",
    "PianoRollImage",
    "LatentEncoding",
    "ReconstructedBar",
    "MidiNote",
    # Rendering
    "ChannelStrategy",
    "VelocityOnlyStrategy",
    "VOSplitStrategy",
    "VOSStrategy",
    "build_strategy",
    # Transforms
    "ResizeTransform",
    "NormalizeTransform",
    "PadTransform",
    "PitchAxisFlip",
    "Compose",
    # Preprocessing
    "MidiIngestor",
    # Datasets
    "LPD5Dataset",
    "Pop909Dataset",
    "MaestroDataset",
]
