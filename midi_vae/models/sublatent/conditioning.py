"""Feature conditioning for sub-latent models.

Injects musical context (instrument, tempo, time signature) into the
sub-latent representation using one of three fusion strategies:

- ``concat``: Concatenate encoded features with the latent vector.
- ``film``:   Feature-wise Linear Modulation — learned scale + shift on latent.
- ``cross_attention``: Multi-head cross-attention with features as keys/values.

Registration name: ``conditioning``.

Spec reference: Section 9.4.
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
# Known instrument classes (fallback when vocab not passed explicitly)
# ---------------------------------------------------------------------------

_DEFAULT_INSTRUMENTS: list[str] = ["drums", "bass", "guitar", "piano", "strings"]

# ---------------------------------------------------------------------------
# Feature encoder
# ---------------------------------------------------------------------------


class FeatureEncoder(nn.Module):
    """Encodes raw conditioning features into a single embedding vector.

    Features:
        - instrument (str → index → learned embedding)
        - tempo       (scalar float → normalised scalar)
        - time_sig    (tuple[int,int] → one-hot over common signatures)

    Attributes:
        instrument_embed: Learnable embedding table.
        time_sig_map: Maps (numerator, denominator) to a one-hot index.
        embed_dim: Output dimensionality.
    """

    # Supported time signatures in one-hot order
    _TIME_SIGS: list[tuple[int, int]] = [
        (2, 4), (3, 4), (4, 4), (6, 8), (3, 8), (5, 4), (7, 8),
    ]
    _MAX_TEMPO: float = 240.0  # normalisation ceiling

    def __init__(
        self,
        embed_dim: int,
        instrument_vocab: list[str],
    ) -> None:
        """Initialise the feature encoder.

        Args:
            embed_dim: Dimension of the produced feature vector.
            instrument_vocab: Ordered list of instrument names.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.instrument_vocab: dict[str, int] = {
            name: i for i, name in enumerate(instrument_vocab)
        }

        n_instruments = len(instrument_vocab)
        n_time_sigs = len(self._TIME_SIGS)

        # Raw feature dimension: instrument_embed_dim + 1 (tempo) + n_time_sigs
        # We project to embed_dim at the end.
        inst_embed_dim = max(4, embed_dim // 4)
        raw_dim = inst_embed_dim + 1 + n_time_sigs

        self.instrument_embed = nn.Embedding(n_instruments + 1, inst_embed_dim)
        self.time_sig_map = {ts: i for i, ts in enumerate(self._TIME_SIGS)}
        self.proj = nn.Linear(raw_dim, embed_dim)

    def _encode_instrument(self, instrument: str | int) -> int:
        """Map instrument name/index to an embedding table index.

        Args:
            instrument: Instrument name string or pre-computed integer index.

        Returns:
            Integer index into the embedding table.
        """
        if isinstance(instrument, int):
            return min(instrument, len(self.instrument_vocab))
        return self.instrument_vocab.get(instrument, len(self.instrument_vocab))  # OOV → last slot

    def _encode_time_sig(self, time_sig: tuple[int, int]) -> torch.Tensor:
        """One-hot encode a time signature.

        Args:
            time_sig: (numerator, denominator) tuple.

        Returns:
            Float tensor of shape (n_time_sigs,).
        """
        n = len(self._TIME_SIGS)
        idx = self.time_sig_map.get(time_sig, 0)  # default to first bucket if unknown
        one_hot = torch.zeros(n)
        one_hot[idx] = 1.0
        return one_hot

    def forward(
        self,
        instruments: list[str | int],
        tempos: list[float],
        time_sigs: list[tuple[int, int]],
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Encode a batch of conditioning features into embedding vectors.

        Args:
            instruments: Batch of instrument names or integer indices, length B.
            tempos: Batch of tempo values in BPM, length B.
            time_sigs: Batch of (numerator, denominator) tuples, length B.
            device: Target device for output tensor.

        Returns:
            Feature embeddings, shape (B, embed_dim).
        """
        B = len(instruments)

        # Instrument embeddings
        inst_indices = torch.tensor(
            [self._encode_instrument(i) for i in instruments],
            dtype=torch.long,
            device=device,
        )
        inst_emb = self.instrument_embed(inst_indices)  # (B, inst_embed_dim)

        # Normalised tempo
        tempo_t = torch.tensor(tempos, dtype=torch.float32, device=device).unsqueeze(1)
        tempo_t = (tempo_t / self._MAX_TEMPO).clamp(0.0, 1.0)  # (B, 1)

        # One-hot time signatures
        ts_list = [self._encode_time_sig(ts) for ts in time_sigs]
        ts_t = torch.stack(ts_list, dim=0).to(device=device)  # (B, n_time_sigs)

        # Concatenate and project
        raw = torch.cat([inst_emb, tempo_t, ts_t], dim=-1)  # (B, raw_dim)
        return self.proj(raw)  # (B, embed_dim)


# ---------------------------------------------------------------------------
# FiLM module
# ---------------------------------------------------------------------------


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer.

    Computes scale ``gamma`` and shift ``beta`` from a condition vector,
    then applies: ``gamma * x + beta``.

    Args:
        cond_dim: Condition vector dimension.
        feat_dim: Feature dimension to modulate.
    """

    def __init__(self, cond_dim: int, feat_dim: int) -> None:
        super().__init__()
        self.scale_head = nn.Linear(cond_dim, feat_dim)
        self.shift_head = nn.Linear(cond_dim, feat_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning.

        Args:
            x: Input features, shape (B, feat_dim).
            cond: Condition vector, shape (B, cond_dim).

        Returns:
            Modulated features, shape (B, feat_dim).
        """
        gamma = self.scale_head(cond)   # (B, feat_dim)
        beta = self.shift_head(cond)    # (B, feat_dim)
        return gamma * x + beta


# ---------------------------------------------------------------------------
# Cross-attention conditioning
# ---------------------------------------------------------------------------


class CrossAttentionConditioner(nn.Module):
    """Single-layer multi-head cross-attention conditioner.

    Query: sub-latent vector (reshaped as a single token).
    Key/Value: condition features (also a single token).

    Args:
        latent_dim: Dimension of the latent query vector.
        cond_dim: Dimension of the condition key/value vector.
        num_heads: Number of attention heads.
    """

    def __init__(self, latent_dim: int, cond_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        # Project condition to match latent_dim for attention
        self.cond_proj = nn.Linear(cond_dim, latent_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention conditioning.

        Args:
            x: Latent query, shape (B, latent_dim).
            cond: Condition vector, shape (B, cond_dim).

        Returns:
            Conditioned latent, shape (B, latent_dim).
        """
        # Reshape to sequence length 1 for MHA: (B, 1, dim)
        query = x.unsqueeze(1)
        kv = self.cond_proj(cond).unsqueeze(1)  # (B, 1, latent_dim)

        attn_out, _ = self.attn(query, kv, kv)   # (B, 1, latent_dim)
        out = self.out_proj(attn_out.squeeze(1))  # (B, latent_dim)
        return x + out  # residual connection


# ---------------------------------------------------------------------------
# FeatureConditioner  (SubLatentModel wrapper)
# ---------------------------------------------------------------------------


@ComponentRegistry.register("sublatent", "conditioning")
class FeatureConditioner(SubLatentModel):
    """Wraps a SubLatentModel to inject musical conditioning features.

    This class is itself a ``SubLatentModel`` that delegates encode/decode
    to an inner model while fusing conditioning features at the bottleneck.

    Three fusion modes are supported (set via ``config.conditioning.fusion``):
        - ``concat``: Append the feature embedding to the latent, then
          project back to ``target_dim``.
        - ``film``: Scale and shift the latent via FiLM.
        - ``cross_attention``: Single-head cross-attention conditioning.

    Feature types (set via ``config.conditioning.features``):
        - ``instrument``: Learned embedding over 5 instrument classes.
        - ``tempo``: Normalised BPM scalar.
        - ``time_sig``: One-hot over 7 common time signatures.

    Attributes:
        inner: The wrapped SubLatentModel.
        feature_encoder: Encodes raw features → embed_dim vector.
        fusion: One of ``concat``, ``film``, ``cross_attention``.
        film_layer: FiLM conditioning layer (only for ``film`` fusion).
        cross_attn: Cross-attention layer (only for ``cross_attention`` fusion).
        concat_proj: Projection layer (only for ``concat`` fusion).
    """

    def __init__(
        self,
        config: SubLatentConfig,
        input_dim: int,
        device: str = "cpu",
        inner: SubLatentModel | None = None,
        instrument_vocab: list[str] | None = None,
    ) -> None:
        """Initialise the feature conditioner.

        Args:
            config: Sub-latent configuration. Uses ``conditioning.embed_dim``
                and ``conditioning.fusion``.
            input_dim: Flattened VAE latent dimension.
            device: Target device string.
            inner: Optional pre-built inner SubLatentModel. If not provided
                the conditioner operates as a pass-through identity.
            instrument_vocab: Ordered list of instrument names. Defaults to
                the 5 standard instruments.
        """
        super().__init__(config, input_dim, device)

        self.inner = inner
        self.fusion: str = "concat"
        self.embed_dim: int = 32

        if config.conditioning is not None:
            self.fusion = config.conditioning.fusion
            self.embed_dim = config.conditioning.embed_dim

        vocab = instrument_vocab or _DEFAULT_INSTRUMENTS
        self.feature_encoder = FeatureEncoder(
            embed_dim=self.embed_dim,
            instrument_vocab=vocab,
        ).to(device)

        target_dim = config.target_dim

        if self.fusion == "concat":
            # Concatenate condition to target_dim → project back to target_dim
            self.concat_proj = nn.Linear(target_dim + self.embed_dim, target_dim).to(device)
        elif self.fusion == "film":
            self.film_layer = FiLMLayer(
                cond_dim=self.embed_dim,
                feat_dim=target_dim,
            ).to(device)
        elif self.fusion == "cross_attention":
            n_heads = max(1, target_dim // 64)
            # Ensure target_dim is divisible by num_heads
            while n_heads > 1 and target_dim % n_heads != 0:
                n_heads -= 1
            self.cross_attn = CrossAttentionConditioner(
                latent_dim=target_dim,
                cond_dim=self.embed_dim,
                num_heads=n_heads,
            ).to(device)
        else:
            logger.warning(
                "FeatureConditioner: unknown fusion mode '%s'; defaulting to 'concat'.",
                self.fusion,
            )
            self.fusion = "concat"
            self.concat_proj = nn.Linear(target_dim + self.embed_dim, target_dim).to(device)

    # ---------------------------------------------------------------------------
    # Conditioning application
    # ---------------------------------------------------------------------------

    def apply_conditioning(
        self,
        s: torch.Tensor,
        instruments: list[str | int],
        tempos: list[float],
        time_sigs: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Apply the configured fusion strategy to a sub-latent tensor.

        Args:
            s: Sub-latent codes, shape (B, target_dim).
            instruments: Batch of instrument names or indices.
            tempos: Batch of tempo values in BPM.
            time_sigs: Batch of (numerator, denominator) pairs.

        Returns:
            Conditioned sub-latent codes, shape (B, target_dim).
        """
        cond = self.feature_encoder(instruments, tempos, time_sigs, device=self.device)

        if self.fusion == "concat":
            combined = torch.cat([s, cond], dim=-1)  # (B, target_dim + embed_dim)
            return self.concat_proj(combined)          # (B, target_dim)
        elif self.fusion == "film":
            return self.film_layer(s, cond)
        elif self.fusion == "cross_attention":
            return self.cross_attn(s, cond)
        else:
            return s  # no-op fallback

    # ---------------------------------------------------------------------------
    # SubLatentModel interface
    # ---------------------------------------------------------------------------

    def encode(self, z_mu: torch.Tensor) -> torch.Tensor:
        """Encode VAE latent to sub-latent code via the inner model.

        Args:
            z_mu: Flattened VAE latent means, shape (B, input_dim).

        Returns:
            Sub-latent codes, shape (B, target_dim).
        """
        if self.inner is not None:
            return self.inner.encode(z_mu)
        # Identity fallback (primarily useful for direct conditioning use)
        z_mu = z_mu.to(device=self.device, dtype=torch.float32)
        return z_mu[:, : self.target_dim]

    def decode(self, s: torch.Tensor) -> torch.Tensor:
        """Decode sub-latent code to VAE latent via the inner model.

        Args:
            s: Sub-latent codes, shape (B, target_dim).

        Returns:
            Reconstructed flattened VAE latents, shape (B, input_dim).
        """
        if self.inner is not None:
            return self.inner.decode(s)
        # Identity fallback
        s = s.to(device=self.device, dtype=torch.float32)
        out = torch.zeros(s.shape[0], self.input_dim, device=self.device, dtype=torch.float32)
        out[:, : self.target_dim] = s
        return out

    def train_step(self, batch: torch.Tensor) -> dict[str, float]:
        """Delegate training to the inner model.

        Note: conditioning features are not passed here. For conditioned
        training call ``apply_conditioning`` on the encoded sub-latent
        before decoding within a custom training loop.

        Args:
            batch: Flattened VAE latents, shape (B, input_dim).

        Returns:
            Loss dict from the inner model, or empty dict if no inner model.
        """
        if self.inner is not None:
            return self.inner.train_step(batch)
        return {"total_loss": 0.0}

    def fit(self, data: torch.Tensor) -> dict[str, Any]:
        """Fit the inner model on the full dataset.

        Args:
            data: Full dataset of flattened VAE latents, shape (N, input_dim).

        Returns:
            Fit stats from the inner model, or empty dict.
        """
        if self.inner is not None:
            return self.inner.fit(data)
        return {"n_samples": data.shape[0]}

    # ---------------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save conditioning layers (and inner model if present).

        Args:
            path: Destination file path.
        """
        state: dict[str, Any] = {
            "feature_encoder": self.feature_encoder.state_dict(),
            "fusion": self.fusion,
            "embed_dim": self.embed_dim,
        }
        if self.fusion == "concat":
            state["concat_proj"] = self.concat_proj.state_dict()
        elif self.fusion == "film":
            state["film_layer"] = self.film_layer.state_dict()
        elif self.fusion == "cross_attention":
            state["cross_attn"] = self.cross_attn.state_dict()

        if self.inner is not None:
            inner_path = path.replace(".pt", "_inner.pt")
            self.inner.save(inner_path)
            state["inner_path"] = inner_path

        torch.save(state, path)
        logger.info("FeatureConditioner saved to %s", path)

    def load(self, path: str) -> None:
        """Load conditioning layers from a checkpoint.

        Args:
            path: File path to load from.
        """
        state = torch.load(path, map_location=self.device)
        self.feature_encoder.load_state_dict(state["feature_encoder"])
        if self.fusion == "concat" and "concat_proj" in state:
            self.concat_proj.load_state_dict(state["concat_proj"])
        elif self.fusion == "film" and "film_layer" in state:
            self.film_layer.load_state_dict(state["film_layer"])
        elif self.fusion == "cross_attention" and "cross_attn" in state:
            self.cross_attn.load_state_dict(state["cross_attn"])
        logger.info("FeatureConditioner loaded from %s", path)
