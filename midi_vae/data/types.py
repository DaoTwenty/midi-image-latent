"""Core data type contracts for the MIDI Image VAE project.

All inter-stage data uses frozen dataclasses to ensure immutability and
type safety across pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(frozen=True)
class MidiNote:
    """A single MIDI note event with pitch, timing, and velocity."""

    pitch: int  # 0-127
    onset_step: int
    offset_step: int
    velocity: int  # 0-127

    def __post_init__(self) -> None:
        if not 0 <= self.pitch <= 127:
            raise ValueError(f"pitch must be 0-127, got {self.pitch}")
        if not 0 <= self.velocity <= 127:
            raise ValueError(f"velocity must be 0-127, got {self.velocity}")
        if self.offset_step <= self.onset_step:
            raise ValueError(
                f"offset_step ({self.offset_step}) must be > onset_step ({self.onset_step})"
            )


@dataclass(frozen=True)
class BarData:
    """A single bar of MIDI data extracted from a multi-track song.

    Contains the raw piano-roll matrix plus onset/sustain masks for
    multi-channel rendering strategies.
    """

    bar_id: str  # {song_id}_{track}_{bar_num}
    song_id: str
    instrument: str  # drums | bass | guitar | piano | strings
    program_number: int
    piano_roll: np.ndarray  # (128, T) velocity matrix
    onset_mask: np.ndarray  # (128, T) binary onset positions
    sustain_mask: np.ndarray  # (128, T) binary sustain positions
    tempo: float
    time_signature: tuple[int, int]
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PianoRollImage:
    """A rendered piano-roll image ready for VAE encoding.

    The image tensor is normalized to [-1, 1] with shape (3, H, W).
    """

    bar_id: str
    image: torch.Tensor  # (3, H, W) normalized to [-1, 1]
    channel_strategy: str
    resolution: tuple[int, int]
    pitch_axis: str  # height | width


@dataclass(frozen=True)
class LatentEncoding:
    """VAE latent encoding of a piano-roll image.

    Contains the mean (z_mu), standard deviation (z_sigma), and optionally
    a reparameterized sample (z_sample).
    """

    bar_id: str
    vae_name: str
    z_mu: torch.Tensor  # (C, H_lat, W_lat)
    z_sigma: torch.Tensor  # (C, H_lat, W_lat)
    z_sample: torch.Tensor | None = None


@dataclass(frozen=True)
class ReconstructedBar:
    """A reconstructed bar containing the decoded image and detected notes.

    The recon_image is continuous-valued (not thresholded).
    """

    bar_id: str
    vae_name: str
    recon_image: torch.Tensor  # (3, H, W) continuous-valued
    detected_notes: list[MidiNote]
    detection_method: str
