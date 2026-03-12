"""Tests for all data type contracts in midi_vae/data/types.py.

Verifies creation, immutability (frozen dataclasses), and field validation
for all inter-stage data types.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from midi_vae.data.types import (
    BarData,
    LatentEncoding,
    MidiNote,
    PianoRollImage,
    ReconstructedBar,
)


# ---------------------------------------------------------------------------
# MidiNote tests
# ---------------------------------------------------------------------------


class TestMidiNote:
    """Tests for MidiNote frozen dataclass."""

    def test_create_valid(self) -> None:
        """MidiNote with valid fields is created successfully."""
        note = MidiNote(pitch=60, onset_step=0, offset_step=24, velocity=100)
        assert note.pitch == 60
        assert note.onset_step == 0
        assert note.offset_step == 24
        assert note.velocity == 100

    def test_is_frozen(self) -> None:
        """MidiNote is immutable — assignment raises FrozenInstanceError."""
        note = MidiNote(pitch=60, onset_step=0, offset_step=24, velocity=100)
        with pytest.raises(Exception):  # FrozenInstanceError
            note.pitch = 61  # type: ignore[misc]

    def test_pitch_lower_bound(self) -> None:
        """Pitch 0 is valid."""
        note = MidiNote(pitch=0, onset_step=0, offset_step=1, velocity=64)
        assert note.pitch == 0

    def test_pitch_upper_bound(self) -> None:
        """Pitch 127 is valid."""
        note = MidiNote(pitch=127, onset_step=0, offset_step=1, velocity=64)
        assert note.pitch == 127

    def test_pitch_below_zero_raises(self) -> None:
        """Pitch below 0 raises ValueError."""
        with pytest.raises(ValueError, match="pitch must be 0-127"):
            MidiNote(pitch=-1, onset_step=0, offset_step=1, velocity=64)

    def test_pitch_above_127_raises(self) -> None:
        """Pitch above 127 raises ValueError."""
        with pytest.raises(ValueError, match="pitch must be 0-127"):
            MidiNote(pitch=128, onset_step=0, offset_step=1, velocity=64)

    def test_velocity_lower_bound(self) -> None:
        """Velocity 0 is valid."""
        note = MidiNote(pitch=60, onset_step=0, offset_step=1, velocity=0)
        assert note.velocity == 0

    def test_velocity_upper_bound(self) -> None:
        """Velocity 127 is valid."""
        note = MidiNote(pitch=60, onset_step=0, offset_step=1, velocity=127)
        assert note.velocity == 127

    def test_velocity_below_zero_raises(self) -> None:
        """Velocity below 0 raises ValueError."""
        with pytest.raises(ValueError, match="velocity must be 0-127"):
            MidiNote(pitch=60, onset_step=0, offset_step=1, velocity=-1)

    def test_velocity_above_127_raises(self) -> None:
        """Velocity above 127 raises ValueError."""
        with pytest.raises(ValueError, match="velocity must be 0-127"):
            MidiNote(pitch=60, onset_step=0, offset_step=1, velocity=128)

    def test_offset_must_be_greater_than_onset(self) -> None:
        """offset_step must be strictly greater than onset_step."""
        with pytest.raises(ValueError, match="offset_step"):
            MidiNote(pitch=60, onset_step=10, offset_step=10, velocity=100)

    def test_offset_before_onset_raises(self) -> None:
        """offset_step before onset_step raises ValueError."""
        with pytest.raises(ValueError, match="offset_step"):
            MidiNote(pitch=60, onset_step=10, offset_step=5, velocity=100)

    def test_equality(self) -> None:
        """Two notes with the same fields are equal."""
        note1 = MidiNote(pitch=60, onset_step=0, offset_step=24, velocity=100)
        note2 = MidiNote(pitch=60, onset_step=0, offset_step=24, velocity=100)
        assert note1 == note2

    def test_hashable(self) -> None:
        """Frozen dataclass is hashable (can be put in sets/dicts)."""
        note = MidiNote(pitch=60, onset_step=0, offset_step=24, velocity=100)
        s = {note}
        assert note in s


# ---------------------------------------------------------------------------
# BarData tests
# ---------------------------------------------------------------------------


class TestBarData:
    """Tests for BarData frozen dataclass."""

    def _make_bar(self, T: int = 96) -> BarData:
        """Create a minimal valid BarData."""
        return BarData(
            bar_id="song1_piano_0",
            song_id="song1",
            instrument="piano",
            program_number=0,
            piano_roll=np.zeros((128, T), dtype=np.float32),
            onset_mask=np.zeros((128, T), dtype=np.float32),
            sustain_mask=np.zeros((128, T), dtype=np.float32),
            tempo=120.0,
            time_signature=(4, 4),
        )

    def test_create_valid(self) -> None:
        """BarData with valid fields is created successfully."""
        bar = self._make_bar()
        assert bar.bar_id == "song1_piano_0"
        assert bar.song_id == "song1"
        assert bar.instrument == "piano"
        assert bar.program_number == 0
        assert bar.tempo == 120.0
        assert bar.time_signature == (4, 4)

    def test_is_frozen(self) -> None:
        """BarData is immutable."""
        bar = self._make_bar()
        with pytest.raises(Exception):  # FrozenInstanceError
            bar.bar_id = "other"  # type: ignore[misc]

    def test_piano_roll_shape(self) -> None:
        """piano_roll has (128, T) shape."""
        bar = self._make_bar(T=96)
        assert bar.piano_roll.shape == (128, 96)

    def test_onset_mask_shape(self) -> None:
        """onset_mask has (128, T) shape."""
        bar = self._make_bar(T=64)
        assert bar.onset_mask.shape == (128, 64)

    def test_sustain_mask_shape(self) -> None:
        """sustain_mask has (128, T) shape."""
        bar = self._make_bar(T=128)
        assert bar.sustain_mask.shape == (128, 128)

    def test_default_metadata_is_empty_dict(self) -> None:
        """metadata defaults to empty dict when not provided."""
        bar = self._make_bar()
        assert bar.metadata == {}

    def test_custom_metadata(self) -> None:
        """Custom metadata is stored correctly."""
        bar = BarData(
            bar_id="song1_piano_0",
            song_id="song1",
            instrument="piano",
            program_number=0,
            piano_roll=np.zeros((128, 96), dtype=np.float32),
            onset_mask=np.zeros((128, 96), dtype=np.float32),
            sustain_mask=np.zeros((128, 96), dtype=np.float32),
            tempo=120.0,
            time_signature=(4, 4),
            metadata={"source": "lpd5", "bar": 42},
        )
        assert bar.metadata["source"] == "lpd5"
        assert bar.metadata["bar"] == 42

    def test_bar_id_format(self) -> None:
        """bar_id follows {song_id}_{instrument}_{bar_num} convention."""
        bar = self._make_bar()
        assert "song1" in bar.bar_id
        assert "piano" in bar.bar_id

    def test_synthetic_bar_fixture(self, synthetic_bar: BarData) -> None:
        """synthetic_bar fixture provides a well-formed BarData."""
        assert synthetic_bar.piano_roll.shape == (128, 96)
        # Should have 4 notes set (C4, E4, G4, C5)
        assert np.any(synthetic_bar.piano_roll > 0)
        assert np.any(synthetic_bar.onset_mask > 0)
        assert np.any(synthetic_bar.sustain_mask > 0)


# ---------------------------------------------------------------------------
# PianoRollImage tests
# ---------------------------------------------------------------------------


class TestPianoRollImage:
    """Tests for PianoRollImage frozen dataclass."""

    def _make_image(self, H: int = 128, W: int = 128) -> PianoRollImage:
        """Create a minimal valid PianoRollImage."""
        return PianoRollImage(
            bar_id="song1_piano_0",
            image=torch.zeros(3, H, W),
            channel_strategy="velocity_only",
            resolution=(H, W),
            pitch_axis="height",
        )

    def test_create_valid(self) -> None:
        """PianoRollImage with valid fields is created successfully."""
        img = self._make_image()
        assert img.bar_id == "song1_piano_0"
        assert img.channel_strategy == "velocity_only"
        assert img.resolution == (128, 128)
        assert img.pitch_axis == "height"

    def test_is_frozen(self) -> None:
        """PianoRollImage is immutable."""
        img = self._make_image()
        with pytest.raises(Exception):
            img.bar_id = "other"  # type: ignore[misc]

    def test_image_tensor_shape(self) -> None:
        """image tensor has shape (3, H, W)."""
        img = self._make_image(H=128, W=128)
        assert img.image.shape == (3, 128, 128)

    def test_image_dtype_float(self) -> None:
        """image tensor is float."""
        img = self._make_image()
        assert img.image.dtype in (torch.float32, torch.float16, torch.bfloat16)

    def test_different_resolutions(self) -> None:
        """PianoRollImage supports non-square resolutions."""
        img = PianoRollImage(
            bar_id="test",
            image=torch.zeros(3, 64, 128),
            channel_strategy="vos",
            resolution=(64, 128),
            pitch_axis="height",
        )
        assert img.resolution == (64, 128)

    def test_synthetic_image_fixture(self, synthetic_image: PianoRollImage) -> None:
        """synthetic_image fixture provides a well-formed image."""
        assert synthetic_image.image.shape == (3, 128, 128)
        assert synthetic_image.channel_strategy == "vos"


# ---------------------------------------------------------------------------
# LatentEncoding tests
# ---------------------------------------------------------------------------


class TestLatentEncoding:
    """Tests for LatentEncoding frozen dataclass."""

    def test_create_valid(self) -> None:
        """LatentEncoding with valid fields is created successfully."""
        z_mu = torch.randn(4, 16, 16)
        z_sigma = torch.ones(4, 16, 16) * 0.1
        enc = LatentEncoding(
            bar_id="song1_piano_0",
            vae_name="sd_vae",
            z_mu=z_mu,
            z_sigma=z_sigma,
        )
        assert enc.bar_id == "song1_piano_0"
        assert enc.vae_name == "sd_vae"
        assert enc.z_sample is None

    def test_is_frozen(self) -> None:
        """LatentEncoding is immutable."""
        enc = LatentEncoding(
            bar_id="test",
            vae_name="sd_vae",
            z_mu=torch.zeros(4, 16, 16),
            z_sigma=torch.ones(4, 16, 16),
        )
        with pytest.raises(Exception):
            enc.bar_id = "other"  # type: ignore[misc]

    def test_z_mu_shape(self) -> None:
        """z_mu has (C, H_lat, W_lat) shape."""
        z_mu = torch.randn(4, 16, 16)
        enc = LatentEncoding(
            bar_id="test",
            vae_name="sd_vae",
            z_mu=z_mu,
            z_sigma=torch.ones(4, 16, 16),
        )
        assert enc.z_mu.shape == (4, 16, 16)

    def test_z_sigma_shape(self) -> None:
        """z_sigma has (C, H_lat, W_lat) shape."""
        z_sigma = torch.ones(4, 16, 16) * 0.5
        enc = LatentEncoding(
            bar_id="test",
            vae_name="sd_vae",
            z_mu=torch.zeros(4, 16, 16),
            z_sigma=z_sigma,
        )
        assert enc.z_sigma.shape == (4, 16, 16)

    def test_z_sample_optional(self) -> None:
        """z_sample is optional and defaults to None."""
        enc = LatentEncoding(
            bar_id="test",
            vae_name="sd_vae",
            z_mu=torch.zeros(4, 16, 16),
            z_sigma=torch.ones(4, 16, 16),
        )
        assert enc.z_sample is None

    def test_z_sample_provided(self) -> None:
        """z_sample can be provided."""
        z_mu = torch.randn(4, 16, 16)
        enc = LatentEncoding(
            bar_id="test",
            vae_name="sd_vae",
            z_mu=z_mu,
            z_sigma=torch.ones(4, 16, 16),
            z_sample=z_mu + 0.01,
        )
        assert enc.z_sample is not None
        assert enc.z_sample.shape == (4, 16, 16)

    def test_synthetic_latent_fixture(self, synthetic_latent: LatentEncoding) -> None:
        """synthetic_latent fixture provides a well-formed encoding."""
        assert synthetic_latent.z_mu.shape == (4, 16, 16)
        assert synthetic_latent.z_sigma.shape == (4, 16, 16)
        assert synthetic_latent.z_sample is not None


# ---------------------------------------------------------------------------
# ReconstructedBar tests
# ---------------------------------------------------------------------------


class TestReconstructedBar:
    """Tests for ReconstructedBar frozen dataclass."""

    def _make_recon(self) -> ReconstructedBar:
        """Create a minimal valid ReconstructedBar."""
        return ReconstructedBar(
            bar_id="song1_piano_0",
            vae_name="sd_vae",
            recon_image=torch.zeros(3, 128, 128),
            detected_notes=[],
            detection_method="global_threshold",
        )

    def test_create_valid(self) -> None:
        """ReconstructedBar with valid fields is created successfully."""
        recon = self._make_recon()
        assert recon.bar_id == "song1_piano_0"
        assert recon.vae_name == "sd_vae"
        assert recon.detected_notes == []
        assert recon.detection_method == "global_threshold"

    def test_is_frozen(self) -> None:
        """ReconstructedBar is immutable."""
        recon = self._make_recon()
        with pytest.raises(Exception):
            recon.bar_id = "other"  # type: ignore[misc]

    def test_recon_image_shape(self) -> None:
        """recon_image has shape (3, H, W)."""
        recon = self._make_recon()
        assert recon.recon_image.shape == (3, 128, 128)

    def test_detected_notes_list_of_midi_notes(self) -> None:
        """detected_notes is a list of MidiNote objects."""
        notes = [
            MidiNote(pitch=60, onset_step=0, offset_step=24, velocity=100),
            MidiNote(pitch=64, onset_step=24, offset_step=48, velocity=80),
        ]
        recon = ReconstructedBar(
            bar_id="test",
            vae_name="sd_vae",
            recon_image=torch.zeros(3, 128, 128),
            detected_notes=notes,
            detection_method="global_threshold",
        )
        assert len(recon.detected_notes) == 2
        assert all(isinstance(n, MidiNote) for n in recon.detected_notes)

    def test_empty_notes_list(self) -> None:
        """Empty notes list is valid (silence)."""
        recon = self._make_recon()
        assert recon.detected_notes == []
