"""MLP-based sub-latent autoencoder.

Projects high-dimensional VAE latents through a configurable MLP bottleneck.
Encoder compresses to ``target_dim``; decoder reconstructs the original flat latent.
Fully iterative — train with ``train_step`` (or ``fit`` for a one-shot pass).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from midi_vae.config import SubLatentConfig
from midi_vae.models.sublatent.base import SubLatentModel
from midi_vae.registry import ComponentRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


def _build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: list[int],
    activation: str,
    dropout: float,
) -> nn.Sequential:
    """Build a fully-connected MLP.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
        hidden_dims: List of hidden layer widths.
        activation: Activation name (relu, gelu, silu, tanh, leaky_relu).
        dropout: Dropout probability applied after every activation.

    Returns:
        An ``nn.Sequential`` module.
    """
    act_cls = _ACTIVATIONS.get(activation.lower(), nn.ReLU)
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# MLPSubLatent
# ---------------------------------------------------------------------------


@ComponentRegistry.register("sublatent", "mlp")
class MLPSubLatent(SubLatentModel):
    """MLP autoencoder for sub-latent dimensionality reduction.

    Architecture::

        Encoder: input_dim → [hidden_dims] → target_dim
        Decoder: target_dim → [hidden_dims reversed] → input_dim

    All hyperparameters come from ``SubLatentConfig``; no hardcoded values.

    Attributes:
        encoder: MLP mapping (B, input_dim) → (B, target_dim).
        decoder: MLP mapping (B, target_dim) → (B, input_dim).
        _optimizer: Adam optimiser (created on first ``train_step`` call).
    """

    def __init__(
        self,
        config: SubLatentConfig,
        input_dim: int,
        device: str = "cpu",
    ) -> None:
        """Initialise MLP sub-latent model.

        Args:
            config: Sub-latent configuration.  Reads ``target_dim``,
                ``training.learning_rate``, ``training.weight_decay``,
                ``training.kl_weight``, and optionally ``hidden_dims``,
                ``activation``, ``dropout`` from ``config`` (falling back to
                sensible defaults when the config object does not carry them).
            input_dim: Flattened VAE latent dimension (C * H_lat * W_lat).
            device: Target device string.
        """
        super().__init__(config, input_dim, device)

        # Resolve optional architecture hyper-params stored in config
        # (not in the Pydantic schema, so we use getattr with defaults)
        hidden_dims: list[int] = getattr(config, "hidden_dims", [512, 256])
        activation: str = getattr(config, "activation", "relu")
        dropout: float = getattr(config, "dropout", 0.0)

        self.encoder = _build_mlp(
            in_dim=input_dim,
            out_dim=config.target_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )
        self.decoder = _build_mlp(
            in_dim=config.target_dim,
            out_dim=input_dim,
            hidden_dims=list(reversed(hidden_dims)),
            activation=activation,
            dropout=dropout,
        )

        # Move networks to target device
        self.encoder.to(device)
        self.decoder.to(device)

        self._optimizer: torch.optim.Optimizer | None = None

    # ---------------------------------------------------------------------------
    # Encode / Decode
    # ---------------------------------------------------------------------------

    def encode(self, z_mu: torch.Tensor) -> torch.Tensor:
        """Compress flattened VAE latents to sub-latent codes.

        Args:
            z_mu: Flattened VAE latent means, shape (B, input_dim).

        Returns:
            Sub-latent codes, shape (B, target_dim).
        """
        z_mu = z_mu.to(device=self.device, dtype=torch.float32)
        return self.encoder(z_mu)

    def decode(self, s: torch.Tensor) -> torch.Tensor:
        """Reconstruct flattened VAE latents from sub-latent codes.

        Args:
            s: Sub-latent codes, shape (B, target_dim).

        Returns:
            Reconstructed flattened VAE latents, shape (B, input_dim).
        """
        s = s.to(device=self.device, dtype=torch.float32)
        return self.decoder(s)

    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Lazily create the Adam optimiser."""
        if self._optimizer is None:
            params = list(self.encoder.parameters()) + list(self.decoder.parameters())
            self._optimizer = torch.optim.Adam(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        return self._optimizer

    def train_step(self, batch: torch.Tensor) -> dict[str, float]:
        """Perform one gradient step on a batch.

        Loss: ``pixel_weight * MSE(decode(encode(z)), z)``

        Args:
            batch: Flattened VAE latents, shape (B, input_dim).

        Returns:
            Dict with ``total_loss`` and ``recon_loss`` scalars.
        """
        self.encoder.train()
        self.decoder.train()

        batch = batch.to(device=self.device, dtype=torch.float32)
        opt = self._get_optimizer()
        opt.zero_grad()

        s = self.encoder(batch)
        z_hat = self.decoder(s)

        recon_loss = F.mse_loss(z_hat, batch)
        total_loss = self.config.training.pixel_weight * recon_loss

        total_loss.backward()
        opt.step()

        return {
            "total_loss": float(total_loss.item()),
            "recon_loss": float(recon_loss.item()),
        }

    # ---------------------------------------------------------------------------
    # Fit (one-shot iteration over a full dataset)
    # ---------------------------------------------------------------------------

    def fit(self, data: torch.Tensor) -> dict[str, Any]:
        """Train the MLP on a single pass over the full dataset.

        For proper multi-epoch training use the ``TrainSubLatentStage`` pipeline.
        This convenience wrapper performs one epoch and returns basic stats.

        Args:
            data: Full dataset of flattened VAE latents, shape (N, input_dim).

        Returns:
            Dict with ``total_loss``, ``recon_loss``, and ``n_samples``.
        """
        data = data.to(device=self.device, dtype=torch.float32)
        result = self.train_step(data)
        result["n_samples"] = data.shape[0]
        return result

    # ---------------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save encoder/decoder weights to a .pt checkpoint.

        Args:
            path: File path for the checkpoint.
        """
        checkpoint = {
            "encoder_state": self.encoder.state_dict(),
            "decoder_state": self.decoder.state_dict(),
            "input_dim": self.input_dim,
            "target_dim": self.target_dim,
        }
        torch.save(checkpoint, path)
        logger.info("MLPSubLatent checkpoint saved to %s", path)

    def load(self, path: str) -> None:
        """Load encoder/decoder weights from a .pt checkpoint.

        Args:
            path: File path to load from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder_state"])
        self.decoder.load_state_dict(checkpoint["decoder_state"])
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        logger.info("MLPSubLatent checkpoint loaded from %s", path)
