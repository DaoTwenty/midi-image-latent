"""Concrete VAE wrappers for all 12 HuggingFace models.

Each wrapper loads a pretrained AutoencoderKL (or compatible model) with all
parameters frozen, and exposes a uniform encode/decode interface through the
DiffusersVAE base class.

Loading patterns:
- Direct: AutoencoderKL.from_pretrained(model_id)
- Subfolder: AutoencoderKL.from_pretrained(model_id, subfolder='vae')
- Custom: flux2_tiny uses AutoModel with bfloat16
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from midi_vae.models.vae_wrapper import FrozenImageVAE
from midi_vae.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class DiffusersVAE(FrozenImageVAE):
    """Base class for HuggingFace diffusers AutoencoderKL wrappers.

    Subclasses declare class-level attributes for ``model_id``,
    ``subfolder``, ``_latent_channels``, and ``_latent_scale_factor``.
    Loading, freezing, dtype conversion, and the encode/decode logic all
    live here so concrete wrappers are purely declarative.

    Attributes:
        model_id: HuggingFace repo ID (must be set by subclass).
        subfolder: Repo subfolder that contains the VAE weights, or None.
        _latent_channels: Number of latent channels (4 or 16).
        _latent_scale_factor: Spatial downscale factor (typically 8).
    """

    model_id: str = ""
    subfolder: Optional[str] = None
    _latent_channels: int = 4
    _latent_scale_factor: int = 8

    # ---------------------------------------------------------------------------
    # Properties required by FrozenImageVAE ABC
    # ---------------------------------------------------------------------------

    @property
    def latent_channels(self) -> int:
        """Number of channels in the latent space."""
        return self._latent_channels

    @property
    def latent_scale_factor(self) -> int:
        """Spatial downscale factor from input image to latent map."""
        return self._latent_scale_factor

    # ---------------------------------------------------------------------------
    # Loading
    # ---------------------------------------------------------------------------

    def load_model(self) -> None:
        """Load AutoencoderKL from HuggingFace, freeze all parameters.

        Uses ``self.config.dtype`` to choose float32 or bfloat16.
        Moves the model to ``self.device`` and sets it to eval mode.

        Raises:
            ValueError: If the configured dtype is not supported.
        """
        from diffusers import AutoencoderKL  # local import — not all envs have diffusers

        dtype = self._resolve_dtype()

        load_kwargs: dict = {"torch_dtype": dtype}
        if self.subfolder is not None:
            load_kwargs["subfolder"] = self.subfolder

        logger.info(
            "Loading VAE %s (model_id=%s, subfolder=%s, dtype=%s, device=%s)",
            self.config.name,
            self.model_id,
            self.subfolder,
            self.config.dtype,
            self.device,
        )

        vae = AutoencoderKL.from_pretrained(self.model_id, **load_kwargs)
        vae = vae.to(self.device)
        vae.eval()

        # Freeze all parameters — VAE is never trained.
        for param in vae.parameters():
            param.requires_grad_(False)

        self._model = vae
        logger.info("VAE %s loaded and frozen.", self.config.name)

    # ---------------------------------------------------------------------------
    # Encode / Decode
    # ---------------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of images to latent mean and standard deviation.

        Processes the batch in chunks of ``config.batch_size`` to bound
        GPU memory usage.  The returned tensors are on the same device as
        the input model.

        Args:
            images: Float tensor, shape (B, 3, H, W), values in [-1, 1].

        Returns:
            Tuple ``(z_mu, z_sigma)`` each of shape (B, C, H_lat, W_lat).
        """
        self.ensure_loaded()

        dtype = self._resolve_dtype()
        images = images.to(device=self.device, dtype=dtype)

        batch_size = self.config.batch_size
        all_mu: list[torch.Tensor] = []
        all_sigma: list[torch.Tensor] = []

        for start in range(0, images.shape[0], batch_size):
            chunk = images[start : start + batch_size]
            dist = self._model.encode(chunk).latent_dist
            all_mu.append(dist.mean)
            all_sigma.append(dist.std)

        z_mu = torch.cat(all_mu, dim=0)
        z_sigma = torch.cat(all_sigma, dim=0)
        return z_mu, z_sigma

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors back to image space.

        Processes the batch in chunks of ``config.batch_size``.

        Args:
            z: Float tensor, shape (B, C, H_lat, W_lat).

        Returns:
            Reconstructed images, shape (B, 3, H, W).
        """
        self.ensure_loaded()

        dtype = self._resolve_dtype()
        z = z.to(device=self.device, dtype=dtype)

        batch_size = self.config.batch_size
        chunks: list[torch.Tensor] = []

        for start in range(0, z.shape[0], batch_size):
            chunk = z[start : start + batch_size]
            recon = self._model.decode(chunk).sample
            chunks.append(recon)

        return torch.cat(chunks, dim=0)

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    def _resolve_dtype(self) -> torch.dtype:
        """Map config dtype string to a torch.dtype.

        Returns:
            ``torch.float32`` or ``torch.bfloat16``.

        Raises:
            ValueError: If the dtype string is not recognised.
        """
        dtype_str = self.config.dtype
        if dtype_str == "float32":
            return torch.float32
        elif dtype_str == "bfloat16":
            return torch.bfloat16
        else:
            raise ValueError(
                f"Unsupported dtype '{dtype_str}'. Expected 'float32' or 'bfloat16'."
            )

    @property
    def scaling_factor(self) -> float:
        """VAE scaling factor read from the underlying model config.

        Falls back to the SD1.x default (0.18215) if the model is not yet
        loaded or the attribute is absent.
        """
        if self._model is not None and hasattr(self._model.config, "scaling_factor"):
            return float(self._model.config.scaling_factor)
        # Reasonable default — concrete wrappers may override this.
        return 0.18215


# =============================================================================
# Concrete VAE wrappers — Group 1: SD-family (4 latent channels, direct load)
# =============================================================================


@ComponentRegistry.register("vae", "sd_vae_ft_mse")
class SDVAEFtMse(DiffusersVAE):
    """Stable Diffusion VAE fine-tuned with MSE loss.

    HF model: stabilityai/sd-vae-ft-mse
    Latent channels: 4, loaded directly (no subfolder).
    """

    model_id = "stabilityai/sd-vae-ft-mse"
    subfolder = None
    _latent_channels = 4
    _latent_scale_factor = 8


@ComponentRegistry.register("vae", "sdxl_vae")
class SDXLVae(DiffusersVAE):
    """Stable Diffusion XL VAE.

    HF model: stabilityai/sdxl-vae
    Latent channels: 4, loaded directly (no subfolder).
    Scaling factor: 0.13025.
    """

    model_id = "stabilityai/sdxl-vae"
    subfolder = None
    _latent_channels = 4
    _latent_scale_factor = 8


@ComponentRegistry.register("vae", "eq_vae_ema")
class EqVaeEma(DiffusersVAE):
    """EQ-VAE with EMA weights (SD1.x compatible).

    HF model: zelaki/eq-vae-ema
    Latent channels: 4, loaded directly (no subfolder).
    """

    model_id = "zelaki/eq-vae-ema"
    subfolder = None
    _latent_channels = 4
    _latent_scale_factor = 8


@ComponentRegistry.register("vae", "eq_sdxl_vae")
class EqSdxlVae(DiffusersVAE):
    """EQ-VAE variant based on SDXL.

    HF model: KBlueLeaf/EQ-SDXL-VAE
    Latent channels: 4, loaded directly (no subfolder).
    """

    model_id = "KBlueLeaf/EQ-SDXL-VAE"
    subfolder = None
    _latent_channels = 4
    _latent_scale_factor = 8


# =============================================================================
# Concrete VAE wrappers — Group 2: SD-family (4 latent channels, subfolder)
# =============================================================================


@ComponentRegistry.register("vae", "sd_v1_4")
class SDV14(DiffusersVAE):
    """Stable Diffusion v1.4 VAE loaded from the 'vae' subfolder.

    HF model: CompVis/stable-diffusion-v1-4
    Latent channels: 4, loaded from subfolder='vae'.
    """

    model_id = "CompVis/stable-diffusion-v1-4"
    subfolder = "vae"
    _latent_channels = 4
    _latent_scale_factor = 8


@ComponentRegistry.register("vae", "playground_v25")
class PlaygroundV25(DiffusersVAE):
    """Playground v2.5 1024px aesthetic VAE.

    HF model: playgroundai/playground-v2.5-1024px-aesthetic
    Latent channels: 4, loaded from subfolder='vae'.
    """

    model_id = "playgroundai/playground-v2.5-1024px-aesthetic"
    subfolder = "vae"
    _latent_channels = 4
    _latent_scale_factor = 8
