"""Integration tests for the encode-decode pipeline.

Tests the full stub pipeline: render → encode → decode → detect → evaluate
using mocked/stub components so no GPU or network access is required.

Real-GPU stages are marked @pytest.mark.gpu and skipped by default.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from midi_vae.config import VAEConfig, SubLatentConfig
from midi_vae.data.types import (
    BarData,
    LatentEncoding,
    MidiNote,
    PianoRollImage,
    ReconstructedBar,
)
from midi_vae.models.vae_wrapper import FrozenImageVAE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def batch_bars() -> list[BarData]:
    """A small batch of 4 synthetic bars."""
    bars = []
    for i in range(4):
        T = 96
        piano_roll = np.zeros((128, T), dtype=np.float32)
        onset_mask = np.zeros((128, T), dtype=np.float32)
        sustain_mask = np.zeros((128, T), dtype=np.float32)
        pitch = 60 + i * 4
        piano_roll[pitch, 0:48] = 80 + i * 10
        onset_mask[pitch, 0] = 1.0
        sustain_mask[pitch, 1:48] = 1.0
        bars.append(BarData(
            bar_id=f"song_{i}_piano_0",
            song_id=f"song_{i}",
            instrument="piano",
            program_number=0,
            piano_roll=piano_roll,
            onset_mask=onset_mask,
            sustain_mask=sustain_mask,
            tempo=120.0,
            time_signature=(4, 4),
            metadata={"test": True},
        ))
    return bars


@pytest.fixture
def batch_images(batch_bars) -> list[PianoRollImage]:
    """Piano roll images for each bar in the batch."""
    images = []
    for bar in batch_bars:
        img = torch.zeros(3, 128, 128)
        images.append(PianoRollImage(
            bar_id=bar.bar_id,
            image=img,
            channel_strategy="vos",
            resolution=(128, 128),
            pitch_axis="height",
        ))
    return images


@pytest.fixture
def batch_latents(batch_images) -> list[LatentEncoding]:
    """Stub latent encodings for each image in the batch."""
    latents = []
    for img in batch_images:
        z = torch.randn(4, 16, 16)
        latents.append(LatentEncoding(
            bar_id=img.bar_id,
            vae_name="stub_vae",
            z_mu=z,
            z_sigma=torch.ones(4, 16, 16) * 0.1,
            z_sample=z,
        ))
    return latents


# ---------------------------------------------------------------------------
# Stage 1: Render → verify PianoRollImage contract
# ---------------------------------------------------------------------------


class TestRenderStageContract:
    """Tests for the render stage output contract."""

    def test_piano_roll_image_has_correct_shape(self, batch_images) -> None:
        """Rendered images have shape (3, H, W)."""
        for img in batch_images:
            assert img.image.shape == (3, 128, 128)

    def test_piano_roll_image_is_frozen_dataclass(self, batch_images) -> None:
        """PianoRollImage is a frozen dataclass (immutable)."""
        img = batch_images[0]
        with pytest.raises((AttributeError, TypeError)):
            img.bar_id = "modified"  # type: ignore[misc]

    def test_piano_roll_image_bar_id_is_string(self, batch_images) -> None:
        """Each image has a string bar_id."""
        for img in batch_images:
            assert isinstance(img.bar_id, str)
            assert len(img.bar_id) > 0

    def test_piano_roll_image_channel_strategy_stored(self, batch_images) -> None:
        """channel_strategy field is stored."""
        for img in batch_images:
            assert img.channel_strategy in ("velocity_only", "vo_split", "vos")


# ---------------------------------------------------------------------------
# Stage 2: Encode — stub VAE encode path
# ---------------------------------------------------------------------------


class TestEncodeStage:
    """Tests for the VAE encode stage using stub VAE."""

    def test_stub_vae_encode_returns_two_tensors(self, stub_vae, batch_images) -> None:
        """encode() returns exactly two tensors (z_mu, z_sigma)."""
        imgs_tensor = torch.stack([img.image for img in batch_images])
        result = stub_vae.encode(imgs_tensor)
        assert len(result) == 2

    def test_stub_vae_encode_shapes(self, stub_vae, batch_images) -> None:
        """encode() returns (B, C, H_lat, W_lat) tensors."""
        imgs_tensor = torch.stack([img.image for img in batch_images])
        z_mu, z_sigma = stub_vae.encode(imgs_tensor)
        B = len(batch_images)
        C = stub_vae.latent_channels
        assert z_mu.shape[0] == B
        assert z_mu.shape[1] == C
        assert z_sigma.shape == z_mu.shape

    def test_stub_vae_sigma_is_positive(self, stub_vae, batch_images) -> None:
        """z_sigma values are all positive."""
        imgs_tensor = torch.stack([img.image for img in batch_images])
        _, z_sigma = stub_vae.encode(imgs_tensor)
        assert (z_sigma > 0).all()

    def test_latent_encoding_dataclass_construction(self, batch_latents) -> None:
        """LatentEncoding dataclasses are constructed correctly."""
        for latent in batch_latents:
            assert isinstance(latent, LatentEncoding)
            assert latent.z_mu.ndim == 3  # (C, H_lat, W_lat)
            assert latent.z_sigma.ndim == 3

    def test_latent_encoding_bar_id_matches_image(self, batch_images, batch_latents) -> None:
        """LatentEncoding bar_id matches the source image."""
        for img, latent in zip(batch_images, batch_latents):
            assert latent.bar_id == img.bar_id


# ---------------------------------------------------------------------------
# Stage 3: Decode
# ---------------------------------------------------------------------------


class TestDecodeStage:
    """Tests for the VAE decode stage using stub VAE."""

    def test_stub_vae_decode_returns_3_channel_images(
        self, stub_vae, batch_latents
    ) -> None:
        """decode() returns (B, 3, H, W) images."""
        z_batch = torch.stack([lat.z_mu for lat in batch_latents])
        recon = stub_vae.decode(z_batch)
        assert recon.shape == (len(batch_latents), 3, 128, 128)

    def test_stub_vae_decode_is_deterministic(
        self, stub_vae, batch_latents
    ) -> None:
        """decode() is deterministic for the same input."""
        z_batch = torch.stack([lat.z_mu for lat in batch_latents])
        r1 = stub_vae.decode(z_batch)
        r2 = stub_vae.decode(z_batch)
        assert torch.allclose(r1, r2)

    def test_decode_output_is_float_tensor(
        self, stub_vae, batch_latents
    ) -> None:
        """decode() output is a float tensor."""
        z_batch = torch.stack([lat.z_mu for lat in batch_latents])
        recon = stub_vae.decode(z_batch)
        assert recon.is_floating_point()


# ---------------------------------------------------------------------------
# Stage 4: Detect
# ---------------------------------------------------------------------------


class TestDetectStage:
    """Tests for the note detection stage using stub detector."""

    def test_stub_detector_detects_from_recon(
        self, stub_detector, stub_vae, batch_latents
    ) -> None:
        """detect() returns a list from reconstructed image."""
        z_batch = torch.stack([lat.z_mu for lat in batch_latents])
        recon_batch = stub_vae.decode(z_batch)
        for i in range(recon_batch.shape[0]):
            notes = stub_detector.detect(recon_batch[i], "vos")
            assert isinstance(notes, list)
            for note in notes:
                assert isinstance(note, MidiNote)

    def test_detected_notes_have_valid_pitch_range(
        self, stub_detector, synthetic_image
    ) -> None:
        """Detected pitches are in [0, 127]."""
        notes = stub_detector.detect(synthetic_image.image, "vos")
        for note in notes:
            assert 0 <= note.pitch <= 127

    def test_detected_notes_have_valid_onset_offset(
        self, stub_detector, synthetic_image
    ) -> None:
        """onset_step < offset_step for all detected notes."""
        notes = stub_detector.detect(synthetic_image.image, "vos")
        for note in notes:
            assert note.onset_step < note.offset_step

    def test_reconstructed_bar_construction(
        self, stub_detector, synthetic_image
    ) -> None:
        """ReconstructedBar can be constructed from detect output."""
        notes = stub_detector.detect(synthetic_image.image, "vos")
        recon_bar = ReconstructedBar(
            bar_id=synthetic_image.bar_id,
            vae_name="stub_vae",
            recon_image=synthetic_image.image,
            detected_notes=notes,
            detection_method="stub",
        )
        assert isinstance(recon_bar, ReconstructedBar)
        assert isinstance(recon_bar.detected_notes, list)


# ---------------------------------------------------------------------------
# Full pipeline stub: render → encode → decode → detect
# ---------------------------------------------------------------------------


class TestFullPipelineStub:
    """End-to-end stub pipeline tests."""

    def test_pipeline_preserves_bar_id_throughout(
        self, stub_vae, stub_detector, batch_bars
    ) -> None:
        """bar_id is threaded through the full pipeline."""
        for bar in batch_bars:
            # Render (synthetic)
            img = PianoRollImage(
                bar_id=bar.bar_id,
                image=torch.zeros(3, 128, 128),
                channel_strategy="vos",
                resolution=(128, 128),
                pitch_axis="height",
            )
            # Encode
            z_mu, z_sigma = stub_vae.encode(img.image.unsqueeze(0))
            latent = LatentEncoding(
                bar_id=bar.bar_id,
                vae_name=stub_vae.name,
                z_mu=z_mu[0],
                z_sigma=z_sigma[0],
                z_sample=z_mu[0],
            )
            # Decode
            recon_img = stub_vae.decode(latent.z_mu.unsqueeze(0))[0]
            # Detect
            notes = stub_detector.detect(recon_img, "vos")
            # Assemble
            recon_bar = ReconstructedBar(
                bar_id=latent.bar_id,
                vae_name=latent.vae_name,
                recon_image=recon_img,
                detected_notes=notes,
                detection_method="stub",
            )
            # Check bar_id preserved
            assert recon_bar.bar_id == bar.bar_id

    def test_batch_encode_decode_shapes_consistent(
        self, stub_vae, batch_images
    ) -> None:
        """Batch encode-decode preserves batch dimension."""
        imgs = torch.stack([img.image for img in batch_images])
        z_mu, z_sigma = stub_vae.encode(imgs)
        recon = stub_vae.decode(z_mu)
        assert recon.shape[0] == imgs.shape[0]
        assert recon.shape[1] == 3  # 3 image channels

    def test_pipeline_runs_without_exceptions(
        self, stub_vae, stub_detector, synthetic_bar
    ) -> None:
        """Full stub pipeline runs end-to-end without raising exceptions."""
        # Render
        img_tensor = torch.zeros(3, 128, 128)
        # Mark some active pixels to give detector something to work with
        img_tensor[0, 56:72, 0:48] = 0.8

        image = PianoRollImage(
            bar_id=synthetic_bar.bar_id,
            image=img_tensor,
            channel_strategy="vos",
            resolution=(128, 128),
            pitch_axis="height",
        )

        # Encode
        z_mu, z_sigma = stub_vae.encode(image.image.unsqueeze(0))

        # Decode
        recon_img = stub_vae.decode(z_mu)[0]

        # Detect
        notes = stub_detector.detect(recon_img, "vos")

        # Assemble ReconstructedBar
        recon_bar = ReconstructedBar(
            bar_id=synthetic_bar.bar_id,
            vae_name=stub_vae.name,
            recon_image=recon_img,
            detected_notes=notes,
            detection_method="stub",
        )

        assert recon_bar is not None
        assert isinstance(recon_bar.detected_notes, list)


# ---------------------------------------------------------------------------
# Sub-latent integration stub
# ---------------------------------------------------------------------------


class TestSubLatentIntegration:
    """Test sub-latent encode-decode with PCA."""

    def test_pca_sublatent_encode_decode_shapes(self) -> None:
        """PCA sub-latent encode/decode preserves batch dimension."""
        from midi_vae.config import SubLatentConfig
        from midi_vae.models.sublatent.pca import PCASubLatent

        input_dim = 4 * 16 * 16  # 4 channels x 16x16 latent
        target_dim = 32
        cfg = SubLatentConfig(enabled=True, approach="pca", target_dim=target_dim)
        pca = PCASubLatent(config=cfg, input_dim=input_dim, device="cpu")

        torch.manual_seed(0)
        data = torch.randn(100, input_dim)
        pca.fit(data)

        # Encode batch
        x = torch.randn(8, input_dim)
        s = pca.encode(x)
        assert s.shape == (8, target_dim)

        # Decode batch
        z_hat = pca.decode(s)
        assert z_hat.shape == (8, input_dim)

    def test_pca_integrates_with_stub_vae_latents(self, stub_vae, batch_images) -> None:
        """PCA can be fit on and applied to real stub VAE outputs."""
        from midi_vae.config import SubLatentConfig
        from midi_vae.models.sublatent.pca import PCASubLatent

        imgs = torch.stack([img.image for img in batch_images])
        z_mu, _ = stub_vae.encode(imgs)

        # Flatten latents
        B, C, H, W = z_mu.shape
        flat = z_mu.view(B, -1)
        input_dim = flat.shape[1]

        # Need more samples than components for a real fit
        flat_expanded = flat.repeat(30, 1) + torch.randn(30 * B, input_dim) * 0.01
        target_dim = 8

        cfg = SubLatentConfig(enabled=True, approach="pca", target_dim=target_dim)
        pca = PCASubLatent(config=cfg, input_dim=input_dim, device="cpu")
        stats = pca.fit(flat_expanded)

        assert stats["n_samples"] == 30 * B
        s = pca.encode(flat)
        assert s.shape == (B, target_dim)


# ---------------------------------------------------------------------------
# GPU-deferred tests
# ---------------------------------------------------------------------------


def _make_synthetic_image_batch(
    batch_size: int = 1, height: int = 128, width: int = 128
) -> torch.Tensor:
    """Create a synthetic normalized image batch in [-1, 1] for GPU tests.

    Args:
        batch_size: Number of images.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        Float tensor of shape (B, 3, H, W) with values in [-1, 1].
    """
    torch.manual_seed(42)
    images = torch.rand(batch_size, 3, height, width) * 2.0 - 1.0
    return images


# VAE configs for parametrized tests.
# Format: (vae_name, dtype, expected_latent_channels, is_gated)
# expected_latent_channels=None means we trust whatever the model returns
_WORKING_VAES = [
    ("sd_vae_ft_mse", "float32", 4, False),
    ("sdxl_vae", "float32", 4, False),
    ("eq_vae_ema", "float32", 4, False),
    ("eq_sdxl_vae", "float32", 4, False),
    ("sd_v1_4", "float32", 4, False),
    ("playground_v25", "float32", 4, False),
    ("cogview4", "bfloat16", 16, False),
    ("flux2_tiny", "bfloat16", 128, False),  # 128ch, custom loader, scale_factor=16
]

_GATED_VAES = [
    ("sd3_medium", "float32", 16, True),
    ("flux1_dev", "bfloat16", 16, True),
    ("flux1_kontext", "bfloat16", 16, True),
    ("flux2_dev", "bfloat16", 32, True),
]


@pytest.mark.gpu
class TestRealVAEEncodeDecodeGPU:
    """Real GPU tests — require actual pretrained model weights and a CUDA device.

    Run with: pytest tests/test_integration_encode_decode.py -m gpu -v
    To also run gated models: pytest ... -m "gpu and gated"
    """

    def _run_encode_decode(
        self,
        vae_name: str,
        dtype: str,
        expected_latent_channels: int | None,
    ) -> None:
        """Helper: load VAE, encode a synthetic image, decode, assert invariants.

        Asserts:
        - Latent shape is (1, C, H_lat, W_lat) where C matches expected_latent_channels
          if provided (or any positive C otherwise).
        - No NaN values in z_mu or reconstruction.
        - Reconstruction shape is (1, 3, 128, 128).

        Args:
            vae_name: Registry name of the VAE to load.
            dtype: ``'float32'`` or ``'bfloat16'``.
            expected_latent_channels: Expected C, or None to skip channel check.
        """
        import gc

        import torch

        from midi_vae.config import VAEConfig
        from midi_vae.registry import ComponentRegistry

        assert torch.cuda.is_available(), "CUDA device required for GPU tests"
        device = "cuda"

        cfg = VAEConfig(
            model_id="",  # concrete wrapper overrides this via class attribute
            name=vae_name,
            dtype=dtype,
            batch_size=1,
        )

        vae_cls = ComponentRegistry.get("vae", vae_name)
        vae = vae_cls(config=cfg, device=device)
        vae.load_model()

        try:
            images = _make_synthetic_image_batch(batch_size=1).to(device)

            z_mu, z_sigma = vae.encode(images)

            # Shape assertions
            assert z_mu.ndim == 4, (
                f"{vae_name}: z_mu should be 4-D, got shape {z_mu.shape}"
            )
            assert z_mu.shape[0] == 1, (
                f"{vae_name}: batch dim mismatch, got {z_mu.shape[0]}"
            )
            if expected_latent_channels is not None:
                assert z_mu.shape[1] == expected_latent_channels, (
                    f"{vae_name}: expected {expected_latent_channels} latent channels, "
                    f"got {z_mu.shape[1]}"
                )
            else:
                assert z_mu.shape[1] > 0, (
                    f"{vae_name}: latent channels must be positive, got {z_mu.shape[1]}"
                )

            # NaN checks on latents (convert to float32 for isnan)
            assert not z_mu.float().isnan().any(), (
                f"{vae_name}: NaN in z_mu"
            )

            # Decode
            recon = vae.decode(z_mu)

            # Reconstruction shape: (1, 3, 128, 128)
            assert recon.shape == (1, 3, 128, 128), (
                f"{vae_name}: expected recon shape (1, 3, 128, 128), got {recon.shape}"
            )

            # NaN checks on reconstruction
            assert not recon.float().isnan().any(), (
                f"{vae_name}: NaN in reconstruction"
            )

        finally:
            # Free GPU memory regardless of test outcome
            del vae
            gc.collect()
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Working VAEs (non-gated)
    # ------------------------------------------------------------------

    def test_sd_vae_ft_mse_encode_decode(self) -> None:
        """sd_vae_ft_mse: 4-channel float32 encode/decode roundtrip on GPU."""
        self._run_encode_decode("sd_vae_ft_mse", "float32", 4)

    def test_sdxl_vae_encode_decode(self) -> None:
        """sdxl_vae: 4-channel float32 encode/decode roundtrip on GPU."""
        self._run_encode_decode("sdxl_vae", "float32", 4)

    def test_eq_vae_ema_encode_decode(self) -> None:
        """eq_vae_ema: 4-channel float32 encode/decode roundtrip on GPU."""
        self._run_encode_decode("eq_vae_ema", "float32", 4)

    def test_eq_sdxl_vae_encode_decode(self) -> None:
        """eq_sdxl_vae: 4-channel float32 encode/decode roundtrip on GPU."""
        self._run_encode_decode("eq_sdxl_vae", "float32", 4)

    def test_sd_v1_4_encode_decode(self) -> None:
        """sd_v1_4: 4-channel float32 encode/decode roundtrip on GPU."""
        self._run_encode_decode("sd_v1_4", "float32", 4)

    def test_playground_v25_encode_decode(self) -> None:
        """playground_v25: 4-channel float32 encode/decode roundtrip on GPU."""
        self._run_encode_decode("playground_v25", "float32", 4)

    def test_cogview4_encode_decode(self) -> None:
        """cogview4: 16-channel bfloat16 encode/decode roundtrip on GPU."""
        self._run_encode_decode("cogview4", "bfloat16", 16)

    def test_flux2_tiny_encode_decode(self) -> None:
        """flux2_tiny: bfloat16 custom AutoModel encode/decode roundtrip on GPU.

        The actual latent channel count is checked from model output (may be
        128 channels in practice despite registry declaring 16).
        """
        self._run_encode_decode("flux2_tiny", "bfloat16", 128)

    # ------------------------------------------------------------------
    # Gated VAEs (require HF license acceptance)
    # ------------------------------------------------------------------

    @pytest.mark.gated
    def test_sd3_medium_encode_decode(self) -> None:
        """sd3_medium: 16-channel float32 encode/decode roundtrip on GPU (gated)."""
        self._run_encode_decode("sd3_medium", "float32", 16)

    @pytest.mark.gated
    def test_flux1_dev_encode_decode(self) -> None:
        """flux1_dev: 16-channel bfloat16 encode/decode roundtrip on GPU (gated)."""
        self._run_encode_decode("flux1_dev", "bfloat16", 16)

    @pytest.mark.gated
    def test_flux1_kontext_encode_decode(self) -> None:
        """flux1_kontext: 16-channel bfloat16 encode/decode roundtrip on GPU (gated)."""
        self._run_encode_decode("flux1_kontext", "bfloat16", 16)

    @pytest.mark.gated
    def test_flux2_dev_encode_decode(self) -> None:
        """flux2_dev: 32-channel bfloat16 encode/decode roundtrip on GPU (gated)."""
        self._run_encode_decode("flux2_dev", "bfloat16", 32)
