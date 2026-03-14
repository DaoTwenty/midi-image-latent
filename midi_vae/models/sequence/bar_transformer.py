"""Autoregressive bar-level Transformer for latent sequence modelling.

Models temporal dependencies between consecutive bar latents.  During training
the model receives a teacher-forced input sequence and predicts the next bar's
latent vector at every position.  During generation it autoregressively extends
a prompt sequence one step at a time.

The latent vectors fed into the Transformer may be:

- Raw flattened VAE latents  (C * H_lat * W_lat)
- Compressed sub-latent codes (target_dim from a SubLatentModel)

The model is purely CPU-compatible; all tensors stay on whichever device
the inputs arrive on.

Registration::

    @ComponentRegistry.register('sequence_model', 'bar_transformer')
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from midi_vae.registry import ComponentRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------


class _SinusoidalPositionalEncoding(nn.Module):
    """Additive sinusoidal positional encoding (fixed, non-learnable).

    Attributes:
        pe: Pre-computed positional encoding buffer of shape (1, max_seq_len, d_model).
    """

    def __init__(self, d_model: int, max_seq_len: int, dropout: float) -> None:
        """Initialise the encoding table.

        Args:
            d_model: Model dimensionality; must be even.
            max_seq_len: Maximum sequence length to pre-compute.
            dropout: Dropout probability applied after adding the encoding.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build (max_seq_len, d_model) table
        position = torch.arange(max_seq_len).unsqueeze(1).float()  # (T, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the input.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Tensor of shape (B, T, d_model) with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1), :]  # type: ignore[index]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# BarTransformer
# ---------------------------------------------------------------------------


@ComponentRegistry.register("sequence_model", "bar_transformer")
class BarTransformer(nn.Module):
    """Autoregressive Transformer that models sequences of bar latents.

    Architecture overview::

        latent_in (B, T, latent_dim)
            → input_proj  (Linear latent_dim → d_model)
            → positional encoding
            → TransformerEncoder (n_layers × self-attention + FFN)
            → output_proj (Linear d_model → latent_dim)
            = predictions (B, T, latent_dim)

    The causal mask ensures that position *i* can only attend to positions
    ≤ *i*, so the model operates autoregressively even during the
    teacher-forced training forward pass.

    All hyperparameters are loaded from the config object (or from
    ``getattr`` fallbacks when the Pydantic schema does not include a field),
    so no values are hardcoded.

    Attributes:
        input_proj: Linear projection from latent space to d_model.
        pos_enc: Sinusoidal positional encoding module.
        transformer: ``nn.TransformerEncoder`` with causal masking.
        output_proj: Linear projection from d_model back to latent space.
        latent_dim: Dimensionality of input / output latent vectors.
        d_model: Internal Transformer dimensionality.
        max_seq_len: Maximum supported sequence length.
    """

    def __init__(self, config: Any) -> None:
        """Initialise the BarTransformer.

        Reads the following fields from *config* (with defaults)::

            config.latent_dim      int  – latent vector size     (default 64)
            config.d_model         int  – transformer width       (default 256)
            config.n_heads         int  – attention heads         (default 8)
            config.n_layers        int  – transformer layers      (default 4)
            config.d_ff            int  – feedforward dim         (default 1024)
            config.dropout         float – dropout probability    (default 0.1)
            config.max_seq_len     int  – max bars in a sequence  (default 256)
            config.learning_rate   float – Adam LR               (default 1e-4)
            config.weight_decay    float – Adam weight decay      (default 1e-5)

        When *config* is an ``ExperimentConfig``, the above fields are expected
        under a ``sequence`` sub-object (accessed via ``getattr``).  If
        ``config`` is a plain object / dict-like, fields are read directly.

        Args:
            config: Configuration object.  May be an ``ExperimentConfig``
                (reads ``config.sequence.*``) or any object whose attributes
                match the list above.
        """
        super().__init__()

        # Support both ExperimentConfig (reads .sequence sub-config) and
        # a plain object passed directly (for tests / standalone use).
        seq_cfg = getattr(config, "sequence", config)

        self.latent_dim: int = int(getattr(seq_cfg, "latent_dim", 64))
        self.d_model: int = int(getattr(seq_cfg, "d_model", 256))
        n_heads: int = int(getattr(seq_cfg, "n_heads", 8))
        n_layers: int = int(getattr(seq_cfg, "n_layers", 4))
        d_ff: int = int(getattr(seq_cfg, "d_ff", 1024))
        dropout: float = float(getattr(seq_cfg, "dropout", 0.1))
        self.max_seq_len: int = int(getattr(seq_cfg, "max_seq_len", 256))
        self._lr: float = float(getattr(seq_cfg, "learning_rate", 1e-4))
        self._weight_decay: float = float(getattr(seq_cfg, "weight_decay", 1e-5))

        # Ensure d_model is divisible by n_heads
        if self.d_model % n_heads != 0:
            adjusted = (self.d_model // n_heads) * n_heads
            logger.warning(
                "d_model=%d is not divisible by n_heads=%d; adjusting d_model to %d.",
                self.d_model,
                n_heads,
                adjusted,
            )
            self.d_model = adjusted

        # --- Layers ---
        self.input_proj = nn.Linear(self.latent_dim, self.d_model)

        self.pos_enc = _SinusoidalPositionalEncoding(
            d_model=self.d_model,
            max_seq_len=self.max_seq_len,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        self.output_proj = nn.Linear(self.d_model, self.latent_dim)

        self._optimizer: torch.optim.Optimizer | None = None

        logger.info(
            "BarTransformer: latent_dim=%d, d_model=%d, n_heads=%d, n_layers=%d, "
            "d_ff=%d, dropout=%.2f, max_seq_len=%d",
            self.latent_dim,
            self.d_model,
            n_heads,
            n_layers,
            d_ff,
            dropout,
            self.max_seq_len,
        )

    # ---------------------------------------------------------------------------
    # Core forward
    # ---------------------------------------------------------------------------

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build an upper-triangular causal attention mask.

        The mask has ``-inf`` in the upper triangle (future positions) and
        ``0`` on the diagonal and below (past positions).

        Args:
            seq_len: Sequence length T.
            device: Target device.

        Returns:
            Float mask of shape (T, T).
        """
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )
        return mask

    def forward(
        self,
        z_sequence: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Causal forward pass for teacher-forced training.

        Predicts ``z_sequence[:, i+1, :]`` from ``z_sequence[:, :i+1, :]``
        at each position *i*.  The output at position *i* is the model's
        prediction of the latent at position *i+1*.

        Args:
            z_sequence: Batch of latent sequences, shape ``(B, T, latent_dim)``.
            mask: Optional additional attention mask, shape ``(T, T)``.
                Added to the auto-generated causal mask when provided.

        Returns:
            Predicted latent sequences, shape ``(B, T, latent_dim)``.
        """
        B, T, _ = z_sequence.shape
        device = z_sequence.device

        # Project to d_model
        x = self.input_proj(z_sequence)  # (B, T, d_model)
        x = self.pos_enc(x)

        # Build causal mask
        causal = self._causal_mask(T, device)
        if mask is not None:
            causal = causal + mask

        # Transformer encoder (causally masked).
        # We pass only the explicit causal mask (not is_causal=True) to remain
        # compatible across PyTorch 2.x minor versions where combining both
        # can trigger an assertion in the attention kernel.
        x = self.transformer(x, mask=causal)  # (B, T, d_model)

        # Project back to latent space
        predictions = self.output_proj(x)  # (B, T, latent_dim)
        return predictions

    # ---------------------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_z: torch.Tensor,
        n_steps: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressively generate ``n_steps`` bars following a prompt.

        At each step the model's prediction for the last position is used as
        the next latent, optionally scaled by *temperature*.

        Args:
            prompt_z: Prompt bar latents, shape ``(B, P, latent_dim)`` or
                ``(P, latent_dim)`` (will be expanded to batch size 1).
            n_steps: Number of new bars to generate.
            temperature: Scaling applied to the predicted latent before
                appending.  Values > 1 increase variance; < 1 reduce it.

        Returns:
            Generated latents of shape ``(B, P + n_steps, latent_dim)``
            (includes the original prompt).
        """
        if prompt_z.dim() == 2:
            prompt_z = prompt_z.unsqueeze(0)  # (1, P, latent_dim)

        was_training = self.training
        self.eval()

        context = prompt_z.clone()

        for _ in range(n_steps):
            # Trim to max_seq_len to avoid exceeding positional encoding table
            context_in = context[:, -self.max_seq_len :, :]
            preds = self.forward(context_in)  # (B, T, latent_dim)
            # Take prediction at the last position
            next_z = preds[:, -1:, :] * temperature  # (B, 1, latent_dim)
            context = torch.cat([context, next_z], dim=1)

        if was_training:
            self.train()

        return context  # (B, P + n_steps, latent_dim)

    # ---------------------------------------------------------------------------
    # Training step
    # ---------------------------------------------------------------------------

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Lazily create the Adam optimiser."""
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self._lr,
                weight_decay=self._weight_decay,
            )
        return self._optimizer

    def training_step(self, batch: torch.Tensor) -> dict[str, float]:
        """Perform one teacher-forced autoregressive training step.

        The model receives ``batch[:, :-1, :]`` as input and must predict
        ``batch[:, 1:, :]`` at every position.

        Loss: ``MSE(predictions, targets)``

        Args:
            batch: Latent sequence batch, shape ``(B, T, latent_dim)``.
                Must have T >= 2.

        Returns:
            Dict with keys ``total_loss`` and ``seq_loss``.

        Raises:
            ValueError: If the sequence length is less than 2.
        """
        if batch.size(1) < 2:
            raise ValueError(
                f"Sequence length must be >= 2 for autoregressive training, got {batch.size(1)}."
            )

        self.train()
        opt = self._get_optimizer()
        opt.zero_grad()

        # Teacher forcing: shift by one position
        inputs = batch[:, :-1, :]   # (B, T-1, latent_dim)
        targets = batch[:, 1:, :]   # (B, T-1, latent_dim)

        predictions = self.forward(inputs)  # (B, T-1, latent_dim)
        loss = F.mse_loss(predictions, targets)

        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        opt.step()

        return {
            "total_loss": float(loss.item()),
            "seq_loss": float(loss.item()),
        }

    # ---------------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and architecture metadata to a .pt checkpoint.

        Args:
            path: Filesystem path to save to.
        """
        checkpoint = {
            "model_state": self.state_dict(),
            "latent_dim": self.latent_dim,
            "d_model": self.d_model,
        }
        torch.save(checkpoint, path)
        logger.info("BarTransformer checkpoint saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights from a .pt checkpoint.

        Args:
            path: Filesystem path to load from.
        """
        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["model_state"])
        logger.info("BarTransformer checkpoint loaded from %s", path)
