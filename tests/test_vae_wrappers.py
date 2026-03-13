"""Tests for all 12 VAE wrappers in midi_vae/models/vae_registry.py.

Each wrapper is tested against the FrozenImageVAE ABC contract using mocked
diffusers models so no network access or GPU is required.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from midi_vae.config import VAEConfig
from midi_vae.models.vae_wrapper import FrozenImageVAE
from midi_vae.registry import ComponentRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_VAE_NAMES = [
    "sd_vae_ft_mse",
    "sdxl_vae",
    "eq_vae_ema",
    "eq_sdxl_vae",
    "sd_v1_4",
    "playground_v25",
    "sd3_medium",
    "flux1_dev",
    "flux1_kontext",
    "flux2_dev",
    "flux2_tiny",
    "cogview4",
]

# Latent channel expectations per VAE name
_EXPECTED_LATENT_CHANNELS: dict[str, int] = {
    "sd_vae_ft_mse": 4,
    "sdxl_vae": 4,
    "eq_vae_ema": 4,
    "eq_sdxl_vae": 4,
    "sd_v1_4": 4,
    "playground_v25": 4,
    "sd3_medium": 16,
    "flux1_dev": 16,
    "flux1_kontext": 16,
    "flux2_dev": 16,
    "flux2_tiny": 16,
    "cogview4": 16,
}


def _make_fake_latent_dist(channels: int, h: int = 16, w: int = 16) -> MagicMock:
    """Create a fake DiagonalGaussianDistribution-like mock."""
    dist = MagicMock()
    dist.mean = torch.zeros(1, channels, h, w)
    dist.std = torch.ones(1, channels, h, w) * 0.1
    return dist


def _make_fake_decoder_output(channels_in: int = 16, h: int = 128, w: int = 128) -> MagicMock:
    """Create a fake decoder output mock with a .sample attribute."""
    out = MagicMock()
    out.sample = torch.zeros(1, 3, h, w)
    return out


def _make_mock_autoencoderkl(latent_channels: int) -> MagicMock:
    """Build a mock AutoencoderKL model that returns correct-shaped tensors."""
    mock_model = MagicMock()
    mock_model.parameters.return_value = []

    # encode() returns an object with .latent_dist
    enc_result = MagicMock()
    enc_result.latent_dist = _make_fake_latent_dist(latent_channels)
    mock_model.encode.return_value = enc_result

    # decode() returns an object with .sample
    dec_result = MagicMock()
    dec_result.sample = torch.zeros(1, 3, 128, 128)
    mock_model.decode.return_value = dec_result

    # config attribute (for scaling_factor)
    mock_model.config = MagicMock()
    mock_model.config.scaling_factor = 0.18215

    return mock_model


def _build_vae(name: str) -> FrozenImageVAE:
    """Instantiate a VAE wrapper by registry name with a minimal config."""
    # Import vae_registry so decorators run and classes are registered
    import midi_vae.models.vae_registry  # noqa: F401

    cls = ComponentRegistry.get("vae", name)
    cfg = VAEConfig(model_id="stub/model", name=name, batch_size=4)
    return cls(config=cfg, device="cpu")


# ---------------------------------------------------------------------------
# Parametrised registration tests
# ---------------------------------------------------------------------------


class TestVAERegistration:
    """Verify every VAE is registered with the correct type key."""

    def test_all_vaes_registered(self) -> None:
        """All 12 VAE names are present in the component registry."""
        import midi_vae.models.vae_registry  # noqa: F401

        registered = ComponentRegistry.list_components("vae").get("vae", [])
        for name in ALL_VAE_NAMES:
            assert name in registered, f"VAE '{name}' not found in registry"

    @pytest.mark.parametrize("vae_name", ALL_VAE_NAMES)
    def test_registry_returns_frozen_image_vae_subclass(self, vae_name: str) -> None:
        """ComponentRegistry.get returns a FrozenImageVAE subclass."""
        import midi_vae.models.vae_registry  # noqa: F401

        cls = ComponentRegistry.get("vae", vae_name)
        assert issubclass(cls, FrozenImageVAE)


# ---------------------------------------------------------------------------
# Latent channel property tests
# ---------------------------------------------------------------------------


class TestLatentChannels:
    """Verify latent_channels property returns correct value for each VAE."""

    @pytest.mark.parametrize("vae_name,expected_channels", _EXPECTED_LATENT_CHANNELS.items())
    def test_latent_channels_property(self, vae_name: str, expected_channels: int) -> None:
        """latent_channels returns the expected integer."""
        vae = _build_vae(vae_name)
        assert vae.latent_channels == expected_channels

    @pytest.mark.parametrize("vae_name", ALL_VAE_NAMES)
    def test_latent_scale_factor_is_8(self, vae_name: str) -> None:
        """latent_scale_factor is 8 for all wrappers."""
        vae = _build_vae(vae_name)
        assert vae.latent_scale_factor == 8

    @pytest.mark.parametrize("vae_name", ALL_VAE_NAMES)
    def test_name_property_matches_config(self, vae_name: str) -> None:
        """name property returns config.name."""
        vae = _build_vae(vae_name)
        assert vae.name == vae_name


# ---------------------------------------------------------------------------
# DiffusersVAE encode/decode with mocked AutoencoderKL
# ---------------------------------------------------------------------------

# The 11 DiffusersVAE-based wrappers (all except flux2_tiny)
_DIFFUSERS_VAE_NAMES = [n for n in ALL_VAE_NAMES if n != "flux2_tiny"]


class TestDiffusersVAEEncodeDecodeShapes:
    """Test encode/decode tensor shapes for DiffusersVAE subclasses."""

    @pytest.mark.parametrize("vae_name", _DIFFUSERS_VAE_NAMES)
    def test_encode_returns_correct_shapes(self, vae_name: str) -> None:
        """encode() returns (z_mu, z_sigma) with shape (B, C, H_lat, W_lat)."""
        channels = _EXPECTED_LATENT_CHANNELS[vae_name]
        vae = _build_vae(vae_name)

        mock_model = _make_mock_autoencoderkl(channels)
        # Make encode return correct batch size
        enc_result = MagicMock()
        enc_result.latent_dist = _make_fake_latent_dist(channels, h=16, w=16)
        # Simulate a batch of 2 — concat results
        enc_result.latent_dist.mean = torch.zeros(2, channels, 16, 16)
        enc_result.latent_dist.std = torch.ones(2, channels, 16, 16) * 0.1
        mock_model.encode.return_value = enc_result
        vae._model = mock_model

        images = torch.zeros(2, 3, 128, 128)
        z_mu, z_sigma = vae.encode(images)

        assert z_mu.shape == (2, channels, 16, 16), f"z_mu shape mismatch for {vae_name}"
        assert z_sigma.shape == (2, channels, 16, 16), f"z_sigma shape mismatch for {vae_name}"

    @pytest.mark.parametrize("vae_name", _DIFFUSERS_VAE_NAMES)
    def test_decode_returns_correct_shape(self, vae_name: str) -> None:
        """decode() returns tensor of shape (B, 3, H, W)."""
        channels = _EXPECTED_LATENT_CHANNELS[vae_name]
        vae = _build_vae(vae_name)

        mock_model = _make_mock_autoencoderkl(channels)
        dec_result = MagicMock()
        dec_result.sample = torch.zeros(2, 3, 128, 128)
        mock_model.decode.return_value = dec_result
        vae._model = mock_model

        z = torch.zeros(2, channels, 16, 16)
        recon = vae.decode(z)

        assert recon.shape == (2, 3, 128, 128), f"recon shape mismatch for {vae_name}"

    @pytest.mark.parametrize("vae_name", _DIFFUSERS_VAE_NAMES)
    def test_encode_freezes_no_grad(self, vae_name: str) -> None:
        """encode() runs under no_grad context (model params should not require grad)."""
        channels = _EXPECTED_LATENT_CHANNELS[vae_name]
        vae = _build_vae(vae_name)
        mock_model = _make_mock_autoencoderkl(channels)
        enc_result = MagicMock()
        enc_result.latent_dist.mean = torch.zeros(1, channels, 16, 16)
        enc_result.latent_dist.std = torch.ones(1, channels, 16, 16) * 0.1
        mock_model.encode.return_value = enc_result
        vae._model = mock_model

        images = torch.zeros(1, 3, 128, 128)
        z_mu, z_sigma = vae.encode(images)

        # Results are plain tensors, not grad-tracked
        assert not z_mu.requires_grad

    @pytest.mark.parametrize("vae_name", _DIFFUSERS_VAE_NAMES)
    def test_encode_batching(self, vae_name: str) -> None:
        """encode() handles batches larger than batch_size correctly."""
        channels = _EXPECTED_LATENT_CHANNELS[vae_name]
        vae = _build_vae(vae_name)

        call_count = 0

        def fake_encode(chunk: torch.Tensor) -> MagicMock:
            nonlocal call_count
            call_count += 1
            b = chunk.shape[0]
            result = MagicMock()
            result.latent_dist.mean = torch.zeros(b, channels, 16, 16)
            result.latent_dist.std = torch.ones(b, channels, 16, 16) * 0.1
            return result

        mock_model = _make_mock_autoencoderkl(channels)
        mock_model.encode.side_effect = fake_encode
        vae._model = mock_model
        vae.config = VAEConfig(model_id="stub/model", name=vae_name, batch_size=2)

        images = torch.zeros(5, 3, 128, 128)  # 5 images, batch_size=2 → 3 calls
        z_mu, z_sigma = vae.encode(images)

        assert z_mu.shape[0] == 5
        assert call_count == 3  # ceil(5/2) == 3


# ---------------------------------------------------------------------------
# DiffusersVAE load_model (with mocked from_pretrained)
# ---------------------------------------------------------------------------


class TestDiffusersVAELoadModel:
    """Test load_model patches AutoencoderKL.from_pretrained correctly."""

    @pytest.mark.parametrize("vae_name", _DIFFUSERS_VAE_NAMES)
    def test_load_model_calls_from_pretrained(self, vae_name: str) -> None:
        """load_model calls AutoencoderKL.from_pretrained with correct args.

        DiffusersVAE.load_model does a local import of AutoencoderKL from
        diffusers, so we inject a mock diffusers module into sys.modules to
        avoid importing the real diffusers (which may have env issues).
        """
        channels = _EXPECTED_LATENT_CHANNELS[vae_name]
        vae = _build_vae(vae_name)

        mock_model = _make_mock_autoencoderkl(channels)
        mock_model.to.return_value = mock_model

        # Create a fake diffusers module with a mock AutoencoderKL
        mock_diffusers = MagicMock()
        mock_diffusers.AutoencoderKL.from_pretrained.return_value = mock_model
        saved = sys.modules.get("diffusers")
        sys.modules["diffusers"] = mock_diffusers
        try:
            vae.load_model()
        finally:
            if saved is not None:
                sys.modules["diffusers"] = saved
            else:
                sys.modules.pop("diffusers", None)

        assert vae._model is mock_model

    @pytest.mark.parametrize("vae_name", _DIFFUSERS_VAE_NAMES)
    def test_ensure_loaded_calls_load_model_once(self, vae_name: str) -> None:
        """ensure_loaded() triggers load_model exactly once."""
        channels = _EXPECTED_LATENT_CHANNELS[vae_name]
        vae = _build_vae(vae_name)

        load_called = []

        def fake_load() -> None:
            vae._model = _make_mock_autoencoderkl(channels)
            load_called.append(True)

        vae.load_model = fake_load  # type: ignore[method-assign]

        vae.ensure_loaded()
        vae.ensure_loaded()  # second call should be a no-op
        assert len(load_called) == 1


# ---------------------------------------------------------------------------
# flux2_tiny — separate wrapper (AutoModel, bfloat16)
# ---------------------------------------------------------------------------


class TestFlux2Tiny:
    """Tests for the flux2_tiny wrapper which uses AutoModel instead of AutoencoderKL."""

    def _build_flux2_tiny_with_mock_model(self) -> Any:
        """Return a Flux2Tiny instance with a mocked _model."""
        vae = _build_vae("flux2_tiny")
        mock_model = MagicMock()
        mock_model.parameters.return_value = []

        enc_result = MagicMock()
        enc_result.latent_dist.mean = torch.zeros(1, 16, 16, 16).to(torch.bfloat16)
        enc_result.latent_dist.std = torch.ones(1, 16, 16, 16).to(torch.bfloat16) * 0.1
        mock_model.encode.return_value = enc_result

        dec_result = MagicMock()
        dec_result.sample = torch.zeros(1, 3, 128, 128).to(torch.bfloat16)
        mock_model.decode.return_value = dec_result

        vae._model = mock_model
        return vae

    def test_latent_channels_is_16(self) -> None:
        """flux2_tiny latent_channels == 16."""
        vae = _build_vae("flux2_tiny")
        assert vae.latent_channels == 16

    def test_latent_scale_factor_is_8(self) -> None:
        """flux2_tiny latent_scale_factor == 8."""
        vae = _build_vae("flux2_tiny")
        assert vae.latent_scale_factor == 8

    def test_encode_returns_bfloat16_tensors(self) -> None:
        """flux2_tiny encode() accepts float input and returns tensors."""
        vae = self._build_flux2_tiny_with_mock_model()
        images = torch.zeros(1, 3, 128, 128)
        z_mu, z_sigma = vae.encode(images)
        assert z_mu.shape == (1, 16, 16, 16)

    def test_decode_returns_tensor(self) -> None:
        """flux2_tiny decode() returns a tensor with 3 image channels."""
        vae = self._build_flux2_tiny_with_mock_model()
        z = torch.zeros(1, 16, 16, 16)
        recon = vae.decode(z)
        assert recon.shape[1] == 3

    def test_load_model_uses_automodel(self) -> None:
        """flux2_tiny load_model uses transformers.AutoModel, not AutoencoderKL."""
        vae = _build_vae("flux2_tiny")
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.parameters.return_value = []

        with patch("transformers.AutoModel") as mock_automodel:
            mock_automodel.from_pretrained.return_value = mock_model
            vae.load_model()

        assert vae._model is mock_model


# ---------------------------------------------------------------------------
# DiffusersVAE dtype resolution
# ---------------------------------------------------------------------------


class TestDiffusersVAEDtypeResolution:
    """Test _resolve_dtype helper on DiffusersVAE subclasses."""

    def test_float32_dtype_resolves(self) -> None:
        """float32 config dtype resolves to torch.float32."""
        import midi_vae.models.vae_registry  # noqa: F401

        cls = ComponentRegistry.get("vae", "sd_vae_ft_mse")
        cfg = VAEConfig(model_id="stub", name="sd_vae_ft_mse", dtype="float32")
        vae = cls(config=cfg, device="cpu")
        assert vae._resolve_dtype() == torch.float32

    def test_bfloat16_dtype_resolves(self) -> None:
        """bfloat16 config dtype resolves to torch.bfloat16."""
        import midi_vae.models.vae_registry  # noqa: F401

        cls = ComponentRegistry.get("vae", "sd_vae_ft_mse")
        cfg = VAEConfig(model_id="stub", name="sd_vae_ft_mse", dtype="bfloat16")
        vae = cls(config=cfg, device="cpu")
        assert vae._resolve_dtype() == torch.bfloat16

    def test_invalid_dtype_raises_value_error(self) -> None:
        """Unsupported dtype string raises ValueError."""
        import midi_vae.models.vae_registry  # noqa: F401

        cls = ComponentRegistry.get("vae", "sd_vae_ft_mse")
        cfg = VAEConfig(model_id="stub", name="sd_vae_ft_mse", dtype="float16")
        vae = cls(config=cfg, device="cpu")
        with pytest.raises(ValueError, match="Unsupported dtype"):
            vae._resolve_dtype()


# ---------------------------------------------------------------------------
# Model ID spot checks
# ---------------------------------------------------------------------------


class TestModelIDs:
    """Spot-check model_id class attributes on DiffusersVAE subclasses."""

    def test_sd_vae_ft_mse_model_id(self) -> None:
        import midi_vae.models.vae_registry  # noqa: F401

        cls = ComponentRegistry.get("vae", "sd_vae_ft_mse")
        assert cls.model_id == "stabilityai/sd-vae-ft-mse"

    def test_flux1_dev_model_id(self) -> None:
        import midi_vae.models.vae_registry  # noqa: F401

        cls = ComponentRegistry.get("vae", "flux1_dev")
        assert cls.model_id == "black-forest-labs/FLUX.1-dev"

    def test_cogview4_model_id(self) -> None:
        import midi_vae.models.vae_registry  # noqa: F401

        cls = ComponentRegistry.get("vae", "cogview4")
        assert cls.model_id == "THUDM/CogView4-6B"

    def test_sd_v1_4_subfolder(self) -> None:
        import midi_vae.models.vae_registry  # noqa: F401

        cls = ComponentRegistry.get("vae", "sd_v1_4")
        assert cls.subfolder == "vae"

    def test_sd_vae_ft_mse_no_subfolder(self) -> None:
        import midi_vae.models.vae_registry  # noqa: F401

        cls = ComponentRegistry.get("vae", "sd_vae_ft_mse")
        assert cls.subfolder is None
