"""PCA-based sub-latent dimensionality reduction.

Projects high-dimensional VAE latents into a compact PCA subspace using
truncated SVD.  This is a non-iterative, closed-form method: call ``fit``
once on a large dataset, then use ``encode``/``decode`` for inference.

PCA cannot be trained with ``train_step``; calling it raises a clear error.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from midi_vae.models.sublatent.base import SubLatentModel
from midi_vae.registry import ComponentRegistry

logger = logging.getLogger(__name__)


@ComponentRegistry.register("sublatent", "pca")
class PCASubLatent(SubLatentModel):
    """PCA-based dimensionality reduction for VAE latents.

    Computes a truncated SVD on the centred data matrix to obtain the top-k
    principal components, then projects incoming latents onto those components
    for encoding and uses the pseudo-inverse for decoding.

    Attributes:
        _components: PCA basis vectors, shape (target_dim, input_dim).
        _mean: Per-feature mean used for centring, shape (input_dim,).
        _singular_values: Singular values for the retained components.
        _fitted: Whether ``fit`` has been called.
    """

    def __init__(self, config: Any, input_dim: int, device: str = "cpu") -> None:
        """Initialise the PCA sub-latent model.

        Args:
            config: Sub-latent configuration.  ``config.target_dim`` sets the
                number of principal components to retain.
            input_dim: Dimensionality of the flattened VAE latent (C*H*W).
            device: Target device for tensor operations.
        """
        super().__init__(config, input_dim, device)

        self._components: torch.Tensor | None = None      # (target_dim, input_dim)
        self._mean: torch.Tensor | None = None            # (input_dim,)
        self._singular_values: torch.Tensor | None = None  # (target_dim,)
        self._fitted: bool = False

    # ---------------------------------------------------------------------------
    # Fitting
    # ---------------------------------------------------------------------------

    def fit(self, data: torch.Tensor) -> dict[str, Any]:
        """Fit PCA on a dataset of flattened latent vectors.

        Centres the data, runs a full SVD, then retains the top
        ``target_dim`` components.  Results are stored as float32 tensors on
        ``self.device``.

        Args:
            data: Dataset of flattened VAE latents, shape (N, input_dim).
                N should be meaningfully larger than ``target_dim`` for
                stable results.

        Returns:
            Dict with:
            - ``"explained_variance_ratio"``: fraction of variance captured by
              each retained component, shape (target_dim,).
            - ``"total_explained_variance"``: sum of the above as a scalar float.
            - ``"n_samples"``: number of samples used for fitting.
            - ``"n_components"``: number of components retained.
        """
        data = data.to(device=self.device, dtype=torch.float32)
        n_samples = data.shape[0]

        if n_samples < self.target_dim:
            logger.warning(
                "PCA fit: n_samples=%d < target_dim=%d; adjusting target_dim.",
                n_samples,
                self.target_dim,
            )

        # Centre data
        mean = data.mean(dim=0)                             # (input_dim,)
        centred = data - mean.unsqueeze(0)                  # (N, input_dim)

        # Truncated SVD via torch.linalg.svd with full_matrices=False
        # U: (N, min(N,D)), S: (min(N,D),), Vh: (min(N,D), D)
        logger.info(
            "Fitting PCA: n_samples=%d, input_dim=%d, target_dim=%d",
            n_samples,
            self.input_dim,
            self.target_dim,
        )
        U, S, Vh = torch.linalg.svd(centred, full_matrices=False)

        k = min(self.target_dim, S.shape[0])
        components = Vh[:k]                                 # (k, input_dim)
        singular_values = S[:k]                             # (k,)

        # Explained variance per component
        total_variance = (S ** 2).sum()
        explained_variance = (singular_values ** 2) / (total_variance + 1e-10)

        self._mean = mean
        self._components = components
        self._singular_values = singular_values
        self._fitted = True

        stats = {
            "explained_variance_ratio": explained_variance.tolist(),
            "total_explained_variance": float(explained_variance.sum().item()),
            "n_samples": n_samples,
            "n_components": k,
        }
        logger.info(
            "PCA fit complete: total_explained_variance=%.4f",
            stats["total_explained_variance"],
        )
        return stats

    # ---------------------------------------------------------------------------
    # Encode / Decode
    # ---------------------------------------------------------------------------

    def encode(self, z_mu: torch.Tensor) -> torch.Tensor:
        """Project flattened VAE latents into PCA space.

        Args:
            z_mu: Flattened VAE latent means, shape (B, input_dim).

        Returns:
            PCA codes, shape (B, target_dim).

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._require_fitted()
        z_mu = z_mu.to(device=self.device, dtype=torch.float32)
        centred = z_mu - self._mean.unsqueeze(0)            # (B, input_dim)
        return centred @ self._components.T                 # (B, target_dim)

    def decode(self, s: torch.Tensor) -> torch.Tensor:
        """Reconstruct flattened VAE latents from PCA codes.

        Args:
            s: PCA codes, shape (B, target_dim).

        Returns:
            Reconstructed flattened VAE latents, shape (B, input_dim).

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        self._require_fitted()
        s = s.to(device=self.device, dtype=torch.float32)
        return s @ self._components + self._mean.unsqueeze(0)   # (B, input_dim)

    # ---------------------------------------------------------------------------
    # Training (not applicable for PCA)
    # ---------------------------------------------------------------------------

    def train_step(self, batch: torch.Tensor) -> dict[str, float]:
        """Not applicable for PCA — raises NotImplementedError.

        PCA is a closed-form method fitted once via ``fit``.  Use
        ``train_sublatent`` pipeline with approach='mlp' or 'sub_vae' for
        iterative training.

        Args:
            batch: Unused.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "PCASubLatent does not support iterative training. "
            "Call fit() on the full dataset instead."
        )

    # ---------------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save PCA components and mean to a .pt checkpoint file.

        Args:
            path: File path ending in ``.pt`` (or any extension).

        Raises:
            RuntimeError: If called before ``fit``.
        """
        self._require_fitted()
        checkpoint = {
            "components": self._components,
            "mean": self._mean,
            "singular_values": self._singular_values,
            "input_dim": self.input_dim,
            "target_dim": self.target_dim,
        }
        torch.save(checkpoint, path)
        logger.info("PCA checkpoint saved to %s", path)

    def load(self, path: str) -> None:
        """Load PCA components and mean from a .pt checkpoint file.

        Args:
            path: File path to load from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self._components = checkpoint["components"].to(self.device)
        self._mean = checkpoint["mean"].to(self.device)
        self._singular_values = checkpoint["singular_values"].to(self.device)
        self._fitted = True
        logger.info("PCA checkpoint loaded from %s", path)

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    def _require_fitted(self) -> None:
        """Raise a clear error if the model has not been fitted.

        Raises:
            RuntimeError: If ``fit`` has not been called.
        """
        if not self._fitted:
            raise RuntimeError(
                "PCASubLatent must be fitted before use. Call fit(data) first."
            )
