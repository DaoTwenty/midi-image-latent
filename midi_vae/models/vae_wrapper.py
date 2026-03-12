"""Abstract base class for frozen pretrained image VAE wrappers.

All 12 VAE implementations inherit from FrozenImageVAE. The VAE parameters
are never trained — only used for encoding piano-roll images to latent space
and decoding back.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from midi_vae.config import VAEConfig


class FrozenImageVAE(ABC):
    """Abstract wrapper around a pretrained image VAE.

    Subclasses implement loading from HuggingFace and expose a uniform
    encode/decode interface regardless of the underlying VAE architecture.

    All parameters are frozen (no gradients).
    """

    def __init__(self, config: VAEConfig, device: str = "cpu") -> None:
        """Initialize the VAE wrapper.

        Args:
            config: VAE configuration with model_id, dtype, etc.
            device: Target device ('cpu', 'cuda', 'cuda:0', etc.).
        """
        self.config = config
        self.device = device
        self._model = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the pretrained VAE model and freeze all parameters.

        Must set self._model and move to self.device.
        """
        ...

    @abstractmethod
    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of images to latent distributions.

        Args:
            images: Batch of images, shape (B, 3, H, W), normalized to [-1, 1].

        Returns:
            Tuple of (z_mu, z_sigma), each shape (B, C, H_lat, W_lat).
        """
        ...

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors back to images.

        Args:
            z: Latent vectors, shape (B, C, H_lat, W_lat).

        Returns:
            Reconstructed images, shape (B, 3, H, W).
        """
        ...

    @property
    @abstractmethod
    def latent_channels(self) -> int:
        """Number of channels in the latent space (e.g., 4 for SD, 16 for FLUX)."""
        ...

    @property
    @abstractmethod
    def latent_scale_factor(self) -> int:
        """Spatial downscale factor from input to latent (e.g., 8)."""
        ...

    @property
    def name(self) -> str:
        """Short name for logging and tracking."""
        return self.config.name

    def ensure_loaded(self) -> None:
        """Load model if not already loaded."""
        if self._model is None:
            self.load_model()

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode images with automatic model loading and no-grad context.

        Args:
            images: Batch of images, shape (B, 3, H, W).

        Returns:
            Tuple of (z_mu, z_sigma).
        """
        self.ensure_loaded()
        return self.encode(images.to(self.device))

    @torch.no_grad()
    def decode_latents(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents with automatic model loading and no-grad context.

        Args:
            z: Latent vectors, shape (B, C, H_lat, W_lat).

        Returns:
            Reconstructed images.
        """
        self.ensure_loaded()
        return self.decode(z.to(self.device))
