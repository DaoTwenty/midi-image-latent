"""TrainSequenceStage: pipeline stage for training the BarTransformer.

Reads ``LatentEncoding`` objects (or sub-latent codes) from the pipeline
context, groups them by ``song_id`` to form per-song sequences, batches those
sequences, and trains the autoregressive BarTransformer.

Context inputs (one of the two is used, sub_latent preferred when available):
    latent_encodings: list[LatentEncoding]  — raw VAE latent encodings.
    sublatent_sequences: dict[str, torch.Tensor]  — pre-computed sub-latent
        codes keyed by song_id, shape (T, target_dim) per entry.

Context outputs:
    sequence_model_path: str  — path to the saved best-checkpoint .pt file.
    sequence_train_stats: dict  — final epoch losses and metadata.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from midi_vae.config import ExperimentConfig
from midi_vae.data.types import LatentEncoding
from midi_vae.models.sequence.bar_transformer import BarTransformer
from midi_vae.pipelines.base import PipelineStage, StageIO, compute_hash
from midi_vae.registry import ComponentRegistry
from midi_vae.utils.device import get_device

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------


class _SequenceDataset(Dataset):
    """Dataset of fixed-length subsequences drawn from variable-length songs.

    Long songs are segmented into non-overlapping windows of ``seq_len`` bars.
    Songs shorter than ``min_len`` are discarded.

    Attributes:
        windows: List of (T, latent_dim) tensors; each is a sequence window.
    """

    def __init__(
        self,
        sequences: list[torch.Tensor],
        seq_len: int,
        min_len: int = 2,
    ) -> None:
        """Build the windowed dataset.

        Args:
            sequences: List of (T_i, latent_dim) tensors.
            seq_len: Window length; sequences shorter than this are used as-is.
            min_len: Discard sequences shorter than this many bars.
        """
        self.windows: list[torch.Tensor] = []
        for seq in sequences:
            T = seq.size(0)
            if T < min_len:
                continue
            if T <= seq_len:
                self.windows.append(seq)
            else:
                # Slide non-overlapping windows
                for start in range(0, T - seq_len + 1, seq_len):
                    window = seq[start : start + seq_len]
                    if window.size(0) >= min_len:
                        self.windows.append(window)

        logger.info(
            "_SequenceDataset: %d windows from %d songs (seq_len=%d).",
            len(self.windows),
            len(sequences),
            seq_len,
        )

    def __len__(self) -> int:
        """Return the number of sequence windows."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a single sequence window.

        Args:
            idx: Window index.

        Returns:
            Tensor of shape (T, latent_dim).
        """
        return self.windows[idx]


def _collate_pad(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate variable-length sequences into a padded batch with a mask.

    Pads all sequences to the length of the longest one in the batch.

    Args:
        batch: List of (T_i, latent_dim) tensors.

    Returns:
        Tuple of:
            - padded: (B, T_max, latent_dim) float32 tensor.
            - lengths: (B,) int64 tensor of original sequence lengths.
    """
    lengths = torch.tensor([s.size(0) for s in batch], dtype=torch.long)
    T_max = int(lengths.max().item())
    latent_dim = batch[0].size(1)
    padded = torch.zeros(len(batch), T_max, latent_dim, dtype=torch.float32)
    for i, seq in enumerate(batch):
        padded[i, : seq.size(0), :] = seq.to(dtype=torch.float32)
    return padded, lengths


# ---------------------------------------------------------------------------
# Pipeline stage
# ---------------------------------------------------------------------------


@ComponentRegistry.register("pipeline_stage", "train_sequence")
class TrainSequenceStage(PipelineStage):
    """Pipeline stage that trains a BarTransformer on bar-level latent sequences.

    Workflow:
    1. Pull sub-latent codes (preferred) or raw VAE latents from context.
    2. Group bars by ``song_id`` to form per-song sequences.
    3. Build a windowed sequence dataset and 90/10 train/val split.
    4. Instantiate a ``BarTransformer`` via the registry.
    5. Run the training loop with early stopping.
    6. Save the best checkpoint; emit path and stats to context.

    The model is configured from ``config.sequence`` (a sub-object) or from
    ``getattr`` fallbacks when that sub-object does not exist.

    Attributes:
        _model: The instantiated BarTransformer (set during ``run``).
        _checkpoint_dir: Directory for saving checkpoints.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        checkpoint_dir: str | None = None,
    ) -> None:
        """Initialise the training stage.

        Args:
            config: Full experiment configuration.  Sequence hyper-parameters
                are read from ``config.sequence`` (with ``getattr`` fallbacks).
            checkpoint_dir: Override for checkpoint save location.  Defaults
                to ``config.paths.output_root/sequence_checkpoints``.
        """
        super().__init__(config)
        self._model: BarTransformer | None = None

        if checkpoint_dir is not None:
            self._checkpoint_dir = Path(checkpoint_dir)
        else:
            self._checkpoint_dir = (
                Path(config.paths.output_root) / "sequence_checkpoints"
            )

    # ---------------------------------------------------------------------------
    # PipelineStage interface
    # ---------------------------------------------------------------------------

    def io(self) -> StageIO:
        """Declare inputs and outputs.

        Returns:
            StageIO with ``latent_encodings`` as input and
            ``sequence_model_path`` / ``sequence_train_stats`` as outputs.
        """
        return StageIO(
            inputs=("latent_encodings",),
            outputs=("sequence_model_path", "sequence_train_stats"),
        )

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute BarTransformer training.

        Args:
            context: Pipeline context.  Must contain ``latent_encodings``
                (list[LatentEncoding]) or optionally
                ``sublatent_sequences`` (dict[str, Tensor]).

        Returns:
            Dict with ``sequence_model_path`` and ``sequence_train_stats``.
        """
        device = get_device(self.config.device)

        # ------------------------------------------------------------------ #
        # 1. Build per-song latent sequences
        # ------------------------------------------------------------------ #
        sequences, latent_dim = self._build_sequences(context, device)

        if not sequences:
            logger.warning("TrainSequenceStage: no sequences; skipping training.")
            return {
                "sequence_model_path": "",
                "sequence_train_stats": {"skipped": True},
            }

        logger.info(
            "TrainSequenceStage: %d songs, latent_dim=%d.", len(sequences), latent_dim
        )

        # ------------------------------------------------------------------ #
        # 2. Sequence dataset + train/val split
        # ------------------------------------------------------------------ #
        seq_cfg = getattr(self.config, "sequence", self.config)
        seq_len: int = int(getattr(seq_cfg, "seq_len", 16))
        batch_size: int = int(getattr(seq_cfg, "batch_size", 32))
        epochs: int = int(getattr(seq_cfg, "epochs", 100))
        patience: int = int(getattr(seq_cfg, "patience", 10))

        dataset = _SequenceDataset(sequences, seq_len=seq_len, min_len=2)

        if len(dataset) == 0:
            logger.warning("TrainSequenceStage: dataset is empty after windowing; skipping.")
            return {
                "sequence_model_path": "",
                "sequence_train_stats": {"skipped": True, "reason": "empty_dataset"},
            }

        n_val = max(1, int(len(dataset) * 0.1))
        n_train = len(dataset) - n_val

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val]
        )

        train_loader: DataLoader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_pad,
            drop_last=False,
        )
        val_loader: DataLoader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_pad,
            drop_last=False,
        )

        # ------------------------------------------------------------------ #
        # 3. Instantiate BarTransformer
        # ------------------------------------------------------------------ #
        # Build a lightweight config proxy that exposes latent_dim alongside
        # all other sequence config fields.
        transformer_cfg = _SequenceConfigProxy(self.config, latent_dim)

        model: BarTransformer = ComponentRegistry.create(
            "sequence_model",
            "bar_transformer",
            transformer_cfg,
        )
        model.to(device)
        self._model = model

        logger.info(
            "TrainSequenceStage: BarTransformer on device=%s. "
            "train_windows=%d, val_windows=%d.",
            device,
            n_train,
            n_val,
        )

        # ------------------------------------------------------------------ #
        # 4. Training loop with early stopping
        # ------------------------------------------------------------------ #
        best_val_loss = math.inf
        patience_counter = 0
        best_checkpoint_path = ""
        history: list[dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            # --- Train ---
            train_losses = self._train_epoch(model, train_loader, device)
            avg_train = _average_losses(train_losses)

            # --- Validate ---
            val_losses = self._val_epoch(model, val_loader, device)
            avg_val = _average_losses(val_losses)

            val_loss = avg_val.get("total_loss", 0.0)

            epoch_stats = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in avg_train.items()},
                **{f"val_{k}": v for k, v in avg_val.items()},
            }
            history.append(epoch_stats)

            log_every = max(1, epochs // 10)
            if epoch % log_every == 0 or epoch == 1:
                logger.info(
                    "Epoch %d/%d — train_loss=%.4f, val_loss=%.4f",
                    epoch,
                    epochs,
                    avg_train.get("total_loss", 0.0),
                    val_loss,
                )

            # --- Checkpoint ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_checkpoint_path = self._save_checkpoint(model, epoch)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(
                    "TrainSequenceStage: early stopping at epoch %d (patience=%d).",
                    epoch,
                    patience,
                )
                break

        logger.info(
            "TrainSequenceStage: done. best_val_loss=%.4f, checkpoint=%s",
            best_val_loss,
            best_checkpoint_path,
        )

        train_stats = {
            "latent_dim": latent_dim,
            "n_songs": len(sequences),
            "n_windows_train": n_train,
            "n_windows_val": n_val,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(history),
            "final_epoch": history[-1] if history else {},
        }

        return {
            "sequence_model_path": best_checkpoint_path,
            "sequence_train_stats": train_stats,
        }

    # ---------------------------------------------------------------------------
    # Helpers — data preparation
    # ---------------------------------------------------------------------------

    def _build_sequences(
        self,
        context: dict[str, Any],
        device: torch.device,
    ) -> tuple[list[torch.Tensor], int]:
        """Build per-song latent sequences from the pipeline context.

        Prefers ``sublatent_sequences`` when available; falls back to grouping
        raw ``latent_encodings`` by ``song_id``.

        Args:
            context: Pipeline context.
            device: Target device.

        Returns:
            Tuple of (list of (T, latent_dim) tensors, latent_dim).
        """
        # --- Prefer pre-computed sub-latent codes ---
        sublatent_seqs: dict[str, torch.Tensor] | None = context.get(
            "sublatent_sequences"
        )
        if sublatent_seqs:
            seqs = [
                v.to(dtype=torch.float32)
                for v in sublatent_seqs.values()
                if v.dim() == 2 and v.size(0) >= 1
            ]
            if seqs:
                latent_dim = seqs[0].size(1)
                return seqs, latent_dim

        # --- Fall back to raw VAE latent encodings grouped by song ---
        encodings: list[LatentEncoding] = context.get("latent_encodings", [])
        if not encodings:
            return [], 0

        # Group by song_id preserving insertion order (proxy for bar order)
        song_bars: dict[str, list[torch.Tensor]] = {}
        for enc in encodings:
            # bar_id format: {song_id}_{track}_{bar_num}
            song_id = enc.bar_id.rsplit("_", 2)[0] if "_" in enc.bar_id else enc.bar_id
            flat = enc.z_mu.to(dtype=torch.float32).flatten()
            song_bars.setdefault(song_id, []).append(flat)

        seqs = []
        for bars in song_bars.values():
            seq = torch.stack(bars, dim=0)  # (T, input_dim)
            seqs.append(seq)

        if not seqs:
            return [], 0

        latent_dim = seqs[0].size(1)
        return seqs, latent_dim

    # ---------------------------------------------------------------------------
    # Helpers — training / validation epochs
    # ---------------------------------------------------------------------------

    @staticmethod
    def _train_epoch(
        model: BarTransformer,
        loader: DataLoader,
        device: torch.device,
    ) -> list[dict[str, float]]:
        """Run one training epoch.

        Args:
            model: The BarTransformer.
            loader: Training DataLoader yielding (padded, lengths) tuples.
            device: Compute device.

        Returns:
            List of per-batch loss dicts.
        """
        losses: list[dict[str, float]] = []
        for padded, _lengths in loader:
            padded = padded.to(device=device, dtype=torch.float32)
            if padded.size(1) < 2:
                continue
            loss_dict = model.training_step(padded)
            losses.append(loss_dict)
        return losses

    @staticmethod
    @torch.no_grad()
    def _val_epoch(
        model: BarTransformer,
        loader: DataLoader,
        device: torch.device,
    ) -> list[dict[str, float]]:
        """Run one validation epoch (no gradients).

        Args:
            model: The BarTransformer.
            loader: Validation DataLoader yielding (padded, lengths) tuples.
            device: Compute device.

        Returns:
            List of per-batch loss dicts.
        """
        model.eval()
        losses: list[dict[str, float]] = []
        for padded, _lengths in loader:
            padded = padded.to(device=device, dtype=torch.float32)
            if padded.size(1) < 2:
                continue
            inputs = padded[:, :-1, :]
            targets = padded[:, 1:, :]
            preds = model(inputs)
            val_loss = float(torch.nn.functional.mse_loss(preds, targets).item())
            losses.append({"total_loss": val_loss, "seq_loss": val_loss})
        model.train()
        return losses

    # ---------------------------------------------------------------------------
    # Helpers — checkpoint
    # ---------------------------------------------------------------------------

    def _save_checkpoint(self, model: BarTransformer, epoch: int) -> str:
        """Save a BarTransformer checkpoint.

        Args:
            model: The BarTransformer to save.
            epoch: Current epoch number (used in filename).

        Returns:
            Absolute path string to the saved checkpoint.
        """
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self._checkpoint_dir / f"bar_transformer_epoch{epoch:04d}.pt"
        model.save(str(ckpt_path))
        return str(ckpt_path)

    # ---------------------------------------------------------------------------
    # Cache key
    # ---------------------------------------------------------------------------

    def cache_key(self, context: dict[str, Any]) -> str | None:
        """Compute a cache key from sequence config and input size.

        Args:
            context: Pipeline context.

        Returns:
            Hex digest or None if no encodings are present.
        """
        encodings = context.get("latent_encodings", [])
        sublatent_seqs = context.get("sublatent_sequences", {})

        n_items = len(encodings) + len(sublatent_seqs)
        if n_items == 0:
            return None

        seq_cfg = getattr(self.config, "sequence", self.config)
        return compute_hash(
            getattr(seq_cfg, "latent_dim", 64),
            getattr(seq_cfg, "d_model", 256),
            getattr(seq_cfg, "n_heads", 8),
            getattr(seq_cfg, "n_layers", 4),
            getattr(seq_cfg, "epochs", 100),
            getattr(seq_cfg, "learning_rate", 1e-4),
            getattr(seq_cfg, "batch_size", 32),
            n_items,
        )


# ---------------------------------------------------------------------------
# Internal config proxy
# ---------------------------------------------------------------------------


class _SequenceConfigProxy:
    """Lightweight proxy that merges ExperimentConfig.sequence with latent_dim override.

    Allows ``BarTransformer`` to receive a single config object while keeping
    ``latent_dim`` consistent with the actual data (which may differ from what
    is written in YAML when using raw vs. sub-latent representations).

    Attributes:
        latent_dim: Overridden latent dimensionality.
    """

    def __init__(self, experiment_config: ExperimentConfig, latent_dim: int) -> None:
        """Wrap an ExperimentConfig with an overridden latent_dim.

        Args:
            experiment_config: Full experiment configuration.
            latent_dim: Actual latent dimensionality inferred from the data.
        """
        seq_cfg = getattr(experiment_config, "sequence", None)
        self._seq_cfg = seq_cfg
        # Override latent_dim so BarTransformer uses the data-inferred dimension
        self.latent_dim = latent_dim
        # Expose a ``sequence`` attribute that points to self (for the
        # BarTransformer's ``getattr(config, 'sequence', config)`` pattern)
        self.sequence = self

    def __getattr__(self, name: str) -> Any:
        """Fall through to the underlying sequence config or default values.

        Args:
            name: Attribute name.

        Returns:
            Value from the sequence sub-config, or a sensible default.
        """
        _defaults: dict[str, Any] = {
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 4,
            "d_ff": 1024,
            "dropout": 0.1,
            "max_seq_len": 256,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "seq_len": 16,
            "batch_size": 32,
            "epochs": 100,
            "patience": 10,
        }
        if self._seq_cfg is not None:
            try:
                return getattr(self._seq_cfg, name)
            except AttributeError:
                pass
        if name in _defaults:
            return _defaults[name]
        raise AttributeError(f"_SequenceConfigProxy has no attribute '{name}'")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _average_losses(loss_dicts: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of per-batch loss dicts.

    Args:
        loss_dicts: List of loss dicts.

    Returns:
        Dict with averaged values.  Empty dict if input is empty.
    """
    if not loss_dicts:
        return {}
    keys = loss_dicts[0].keys()
    return {k: sum(d[k] for d in loss_dicts) / len(loss_dicts) for k in keys}
