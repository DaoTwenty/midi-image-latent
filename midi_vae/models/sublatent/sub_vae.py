"""Variational MLP sub-latent model (sub-VAE).

Learns a probabilistic bottleneck over VAE latents.  The encoder predicts
``mu`` and ``log_var`` for a Gaussian; the decoder reconstructs from a
reparameterised sample.  Training uses MSE reconstruction loss plus a
KL-divergence regulariser.

Loss (per spec Section 9.3):
    total = pixel_weight * MSE(decode(s), z) + kl_weight * KL(q(s|z) || N(0,I))
    defaults: pixel_weight=1.0, kl_weight=0.001
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
# Helpers (shared with mlp.py — duplicated to avoid internal imports)
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
        in_dim: Input dimension.
        out_dim: Output dimension.
        hidden_dims: List of hidden widths.
        activation: Activation name.
        dropout: Dropout probability.

    Returns:
        ``nn.Sequential`` module.
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
# SubVAE
# ---------------------------------------------------------------------------


@ComponentRegistry.register("sublatent", "sub_vae")
class SubVAE(SubLatentModel):
    """Variational autoencoder over the VAE latent space.

    Architecture::

        Encoder: input_dim → [hidden_dims] → (mu, log_var)  both shape (B, target_dim)
        Decoder: target_dim → [hidden_dims reversed] → input_dim

    ``encode`` returns the deterministic mean ``mu`` (for inference).
    ``train_step`` uses the reparameterisation trick to sample ``s`` from
    ``q(s|z) = N(mu, exp(0.5 * log_var))``.

    Attributes:
        encoder_shared: Shared trunk from input to last hidden layer.
        mu_head: Linear head for the mean.
        logvar_head: Linear head for log-variance.
        decoder: MLP mapping target_dim → input_dim.
        _optimizer: Lazily created Adam optimiser.
    """

    def __init__(
        self,
        config: SubLatentConfig,
        input_dim: int,
        device: str = "cpu",
    ) -> None:
        """Initialise the sub-VAE.

        Args:
            config: Sub-latent configuration.  Reads ``target_dim``,
                ``training.pixel_weight``, ``training.kl_weight``,
                ``training.learning_rate``, ``training.weight_decay``.
                Optional fields: ``hidden_dims``, ``activation``, ``dropout``.
            input_dim: Flattened VAE latent dimension (C * H_lat * W_lat).
            device: Target device string.
        """
        super().__init__(config, input_dim, device)

        hidden_dims: list[int] = getattr(config, "hidden_dims", [512, 256])
        activation: str = getattr(config, "activation", "relu")
        dropout: float = getattr(config, "dropout", 0.0)

        # Shared trunk: input → last hidden layer
        act_cls = _ACTIVATIONS.get(activation.lower(), nn.ReLU)
        trunk_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            trunk_layers.append(nn.Linear(prev, h))
            trunk_layers.append(act_cls())
            if dropout > 0.0:
                trunk_layers.append(nn.Dropout(p=dropout))
            prev = h
        self.encoder_shared = nn.Sequential(*trunk_layers) if trunk_layers else nn.Identity()
        enc_out_dim = hidden_dims[-1] if hidden_dims else input_dim

        # Parallel mu / log_var heads
        self.mu_head = nn.Linear(enc_out_dim, config.target_dim)
        self.logvar_head = nn.Linear(enc_out_dim, config.target_dim)

        # Decoder
        self.decoder = _build_mlp(
            in_dim=config.target_dim,
            out_dim=input_dim,
            hidden_dims=list(reversed(hidden_dims)),
            activation=activation,
            dropout=dropout,
        )

        self.encoder_shared.to(device)
        self.mu_head.to(device)
        self.logvar_head.to(device)
        self.decoder.to(device)

        self._optimizer: torch.optim.Optimizer | None = None

    # ---------------------------------------------------------------------------
    # Reparameterisation
    # ---------------------------------------------------------------------------

    def _reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample from q(s|z) = N(mu, exp(0.5 * log_var)).

        During inference (eval mode or ``torch.no_grad``), returns ``mu``
        directly for deterministic behaviour.

        Args:
            mu: Mean tensor, shape (B, target_dim).
            log_var: Log-variance tensor, shape (B, target_dim).

        Returns:
            Sampled codes, shape (B, target_dim).
        """
        if not self.encoder_shared.training:
            return mu
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _encode_all(self, z_mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the encoder trunk and return (mu, log_var).

        Args:
            z_mu: Flattened latents, shape (B, input_dim).

        Returns:
            Tuple of (mu, log_var), each shape (B, target_dim).
        """
        h = self.encoder_shared(z_mu)
        return self.mu_head(h), self.logvar_head(h)

    # ---------------------------------------------------------------------------
    # Encode / Decode (SubLatentModel interface)
    # ---------------------------------------------------------------------------

    def encode(self, z_mu: torch.Tensor) -> torch.Tensor:
        """Return deterministic mean embedding for inference.

        Args:
            z_mu: Flattened VAE latent means, shape (B, input_dim).

        Returns:
            Sub-latent codes (mean), shape (B, target_dim).
        """
        z_mu = z_mu.to(device=self.device, dtype=torch.float32)
        mu, _ = self._encode_all(z_mu)
        return mu

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
            params = (
                list(self.encoder_shared.parameters())
                + list(self.mu_head.parameters())
                + list(self.logvar_head.parameters())
                + list(self.decoder.parameters())
            )
            self._optimizer = torch.optim.Adam(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        return self._optimizer

    def train_step(self, batch: torch.Tensor) -> dict[str, float]:
        """Perform one gradient step on a batch.

        Loss:
            recon_loss = pixel_weight * MSE(z_hat, z)
            kl_loss    = kl_weight   * mean(-0.5 * sum(1 + log_var - mu^2 - exp(log_var)))
            total_loss = recon_loss + kl_loss

        Args:
            batch: Flattened VAE latents, shape (B, input_dim).

        Returns:
            Dict with ``total_loss``, ``recon_loss``, and ``kl_loss`` scalars.
        """
        self.encoder_shared.train()
        self.mu_head.train()
        self.logvar_head.train()
        self.decoder.train()

        batch = batch.to(device=self.device, dtype=torch.float32)
        opt = self._get_optimizer()
        opt.zero_grad()

        mu, log_var = self._encode_all(batch)
        s = self._reparameterise(mu, log_var)
        z_hat = self.decoder(s)

        recon_loss = F.mse_loss(z_hat, batch)
        # KL divergence: KL(N(mu, sigma) || N(0,I))
        kl_loss = -0.5 * torch.mean(
            1.0 + log_var - mu.pow(2) - log_var.exp()
        )

        total_loss = (
            self.config.training.pixel_weight * recon_loss
            + self.config.training.kl_weight * kl_loss
        )

        total_loss.backward()
        opt.step()

        return {
            "total_loss": float(total_loss.item()),
            "recon_loss": float(recon_loss.item()),
            "kl_loss": float(kl_loss.item()),
        }

    # ---------------------------------------------------------------------------
    # Fit
    # ---------------------------------------------------------------------------

    def fit(self, data: torch.Tensor) -> dict[str, Any]:
        """Train on a single pass over the full dataset (one epoch).

        For multi-epoch training use ``TrainSubLatentStage``.

        Args:
            data: Full dataset of flattened VAE latents, shape (N, input_dim).

        Returns:
            Dict with ``total_loss``, ``recon_loss``, ``kl_loss``, ``n_samples``.
        """
        data = data.to(device=self.device, dtype=torch.float32)
        result = self.train_step(data)
        result["n_samples"] = data.shape[0]
        return result

    # ---------------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights to a .pt checkpoint.

        Args:
            path: Destination file path.
        """
        checkpoint = {
            "encoder_shared_state": self.encoder_shared.state_dict(),
            "mu_head_state": self.mu_head.state_dict(),
            "logvar_head_state": self.logvar_head.state_dict(),
            "decoder_state": self.decoder.state_dict(),
            "input_dim": self.input_dim,
            "target_dim": self.target_dim,
        }
        torch.save(checkpoint, path)
        logger.info("SubVAE checkpoint saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights from a .pt checkpoint.

        Args:
            path: File path to load from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder_shared.load_state_dict(checkpoint["encoder_shared_state"])
        self.mu_head.load_state_dict(checkpoint["mu_head_state"])
        self.logvar_head.load_state_dict(checkpoint["logvar_head_state"])
        self.decoder.load_state_dict(checkpoint["decoder_state"])
        self.encoder_shared.to(self.device)
        self.mu_head.to(self.device)
        self.logvar_head.to(self.device)
        self.decoder.to(self.device)
        logger.info("SubVAE checkpoint loaded from %s", path)
