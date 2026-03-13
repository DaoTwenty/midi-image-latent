"""Shared test fixtures for the MIDI Image VAE project.

Provides synthetic data fixtures and stub implementations for testing
without requiring real MIDI files or pretrained models.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from midi_vae.data.types import BarData, PianoRollImage, LatentEncoding, ReconstructedBar, MidiNote
from midi_vae.config import ExperimentConfig, PathsConfig
from midi_vae.models.vae_wrapper import FrozenImageVAE, VAEConfig
from midi_vae.note_detection.base import NoteDetector


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_bar() -> BarData:
    """A 4-note bar with known values for deterministic testing."""
    T = 96
    piano_roll = np.zeros((128, T), dtype=np.float32)
    onset_mask = np.zeros((128, T), dtype=np.float32)
    sustain_mask = np.zeros((128, T), dtype=np.float32)

    # Note 1: C4 (pitch 60), steps 0-23, velocity 100
    piano_roll[60, 0:24] = 100
    onset_mask[60, 0] = 1.0
    sustain_mask[60, 1:24] = 1.0

    # Note 2: E4 (pitch 64), steps 24-47, velocity 80
    piano_roll[64, 24:48] = 80
    onset_mask[64, 24] = 1.0
    sustain_mask[64, 25:48] = 1.0

    # Note 3: G4 (pitch 67), steps 48-71, velocity 90
    piano_roll[67, 48:72] = 90
    onset_mask[67, 48] = 1.0
    sustain_mask[67, 49:72] = 1.0

    # Note 4: C5 (pitch 72), steps 72-95, velocity 110
    piano_roll[72, 72:96] = 110
    onset_mask[72, 72] = 1.0
    sustain_mask[72, 73:96] = 1.0

    return BarData(
        bar_id="test_song_piano_0",
        song_id="test_song",
        instrument="piano",
        program_number=0,
        piano_roll=piano_roll,
        onset_mask=onset_mask,
        sustain_mask=sustain_mask,
        tempo=120.0,
        time_signature=(4, 4),
        metadata={"test": True},
    )


@pytest.fixture
def synthetic_image() -> PianoRollImage:
    """A (3, 128, 128) image with known channel values."""
    image = torch.zeros(3, 128, 128)
    # Put some recognizable patterns
    image[0, 56:72, 0:32] = 0.8  # R channel: velocity
    image[1, 56:72, 0:2] = 1.0  # G channel: onset
    image[2, 56:72, 2:32] = 0.6  # B channel: sustain
    return PianoRollImage(
        bar_id="test_song_piano_0",
        image=image,
        channel_strategy="vos",
        resolution=(128, 128),
        pitch_axis="height",
    )


@pytest.fixture
def synthetic_latent() -> LatentEncoding:
    """A (4, 16, 16) latent encoding with known mu/sigma."""
    z_mu = torch.randn(4, 16, 16)
    z_sigma = torch.ones(4, 16, 16) * 0.1
    return LatentEncoding(
        bar_id="test_song_piano_0",
        vae_name="stub_vae",
        z_mu=z_mu,
        z_sigma=z_sigma,
        z_sample=z_mu,  # deterministic for testing
    )


@pytest.fixture
def synthetic_notes() -> list[MidiNote]:
    """Known MIDI notes matching synthetic_bar."""
    return [
        MidiNote(pitch=60, onset_step=0, offset_step=24, velocity=100),
        MidiNote(pitch=64, onset_step=24, offset_step=48, velocity=80),
        MidiNote(pitch=67, onset_step=48, offset_step=72, velocity=90),
        MidiNote(pitch=72, onset_step=72, offset_step=96, velocity=110),
    ]


# ---------------------------------------------------------------------------
# Stub implementations
# ---------------------------------------------------------------------------


class StubVAE(FrozenImageVAE):
    """Deterministic stub VAE for testing. Returns scaled-down input as latent."""

    def load_model(self) -> None:
        self._model = True  # Just mark as loaded

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = images.shape[0]
        # Downsample to 4x16x16 by avg pooling
        z_mu = torch.nn.functional.adaptive_avg_pool2d(images[:, :3], (16, 16))
        # Pad/truncate channels to 4
        if z_mu.shape[1] < 4:
            z_mu = torch.nn.functional.pad(z_mu, (0, 0, 0, 0, 0, 4 - z_mu.shape[1]))
        else:
            z_mu = z_mu[:, :4]
        z_sigma = torch.ones_like(z_mu) * 0.1
        return z_mu, z_sigma

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Upsample back to 3x128x128
        recon = torch.nn.functional.interpolate(z[:, :3], size=(128, 128), mode="bilinear")
        if recon.shape[1] < 3:
            recon = torch.nn.functional.pad(recon, (0, 0, 0, 0, 0, 3 - recon.shape[1]))
        return recon

    @property
    def latent_channels(self) -> int:
        return 4

    @property
    def latent_scale_factor(self) -> int:
        return 8


class StubNoteDetector(NoteDetector):
    """Stub detector that returns fixed notes for testing."""

    def detect(self, recon_image: torch.Tensor, channel_strategy: str) -> list[MidiNote]:
        # Simple threshold detection on velocity channel
        vel = recon_image[0]  # R channel
        notes = []
        for pitch in range(128):
            row = vel[pitch]
            in_note = False
            onset = 0
            for t in range(row.shape[0]):
                if row[t] > 0.3 and not in_note:
                    in_note = True
                    onset = t
                elif (row[t] <= 0.3 or t == row.shape[0] - 1) and in_note:
                    offset = t + 1 if row[t] > 0.3 else t
                    if offset > onset + 1:
                        notes.append(MidiNote(
                            pitch=pitch,
                            onset_step=onset,
                            offset_step=offset,
                            velocity=max(0, min(127, int(row[onset].item() * 127))),
                        ))
                    in_note = False
        return notes

    @property
    def needs_fitting(self) -> bool:
        return False


@pytest.fixture
def stub_vae() -> StubVAE:
    """Fast deterministic stub VAE."""
    config = VAEConfig(model_id="stub", name="stub_vae")
    vae = StubVAE(config=config, device="cpu")
    vae.load_model()
    return vae


@pytest.fixture
def stub_detector() -> StubNoteDetector:
    """Simple threshold-based stub detector."""
    return StubNoteDetector()


@pytest.fixture
def tmp_config(tmp_path: Path) -> ExperimentConfig:
    """A valid ExperimentConfig using temporary directories."""
    return ExperimentConfig(
        paths=PathsConfig(
            data_root=str(tmp_path / "data"),
            output_root=str(tmp_path / "outputs"),
            cache_dir=str(tmp_path / "cache"),
        ),
    )
