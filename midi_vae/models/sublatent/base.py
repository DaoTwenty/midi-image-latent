"""Abstract base class for sub-latent dimensionality reduction models.

Sub-latent models project the high-dimensional VAE latent space (e.g., 4x16x16=1024)
into a compact representation (e.g., 64-dim) while preserving musically relevant structure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from midi_vae.config import SubLatentConfig


class SubLatentModel(ABC):
    """Abstract base for sub-latent projection models.

    Implementations include PCA, MLP autoencoder, sub-VAE, and conditioned variants.
    """

    def __init__(self, config: SubLatentConfig, input_dim: int, device: str = "cpu") -> None:
        """Initialize the sub-latent model.

        Args:
            config: Sub-latent configuration with target_dim, training params, etc.
            input_dim: Dimensionality of flattened VAE latent (C * H_lat * W_lat).
            device: Target device.
        """
        self.config = config
        self.input_dim = input_dim
        self.target_dim = config.target_dim
        self.device = device

    @abstractmethod
    def encode(self, z_mu: torch.Tensor) -> torch.Tensor:
        """Project VAE latent to compact sub-latent space.

        Args:
            z_mu: Flattened VAE latent means, shape (B, input_dim).

        Returns:
            Sub-latent codes, shape (B, target_dim).
        """
        ...

    @abstractmethod
    def decode(self, s: torch.Tensor) -> torch.Tensor:
        """Reconstruct VAE latent from sub-latent code.

        Args:
            s: Sub-latent codes, shape (B, target_dim).

        Returns:
            Reconstructed flattened VAE latents, shape (B, input_dim).
        """
        ...

    @abstractmethod
    def train_step(self, batch: torch.Tensor) -> dict[str, float]:
        """Perform one training step.

        Args:
            batch: Batch of flattened VAE latents, shape (B, input_dim).

        Returns:
            Dict of loss values (e.g., {'total_loss': 0.5, 'recon_loss': 0.4, 'kl_loss': 0.1}).
        """
        ...

    @abstractmethod
    def fit(self, data: torch.Tensor) -> dict[str, Any]:
        """Fit the model to a full dataset (for non-iterative methods like PCA).

        Args:
            data: Full dataset of flattened VAE latents, shape (N, input_dim).

        Returns:
            Dict of fitting statistics (e.g., explained variance ratios).
        """
        ...

    def save(self, path: str) -> None:
        """Save model state to disk.

        Args:
            path: File path to save to.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement save()")

    def load(self, path: str) -> None:
        """Load model state from disk.

        Args:
            path: File path to load from.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement load()")
