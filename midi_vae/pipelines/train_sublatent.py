"""TrainSubLatentStage: pipeline stage for training sub-latent models.

Reads ``LatentEncoding`` objects from the pipeline context, stacks their
``z_mu`` tensors into a flat dataset, instantiates the configured
``SubLatentModel``, runs a training loop, and saves the best checkpoint.

Context inputs:
    latent_encodings: list[LatentEncoding]  — produced by an upstream encode stage.

Context outputs:
    sublatent_model_path: str  — path to the saved best-checkpoint .pt file.
    sublatent_train_stats: dict  — final epoch losses and metadata.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from midi_vae.config import ExperimentConfig
from midi_vae.data.types import LatentEncoding
from midi_vae.models.sublatent.base import SubLatentModel
from midi_vae.pipelines.base import PipelineStage, StageIO, compute_hash
from midi_vae.registry import ComponentRegistry
from midi_vae.utils.device import get_device

logger = logging.getLogger(__name__)


@ComponentRegistry.register("pipeline_stage", "train_sublatent")
class TrainSubLatentStage(PipelineStage):
    """Pipeline stage that trains a sub-latent dimensionality reduction model.

    Workflow:
    1. Pull ``latent_encodings`` from context.
    2. Stack ``z_mu`` tensors → flat (N, input_dim) tensor.
    3. Split 90/10 into train / validation sets.
    4. Instantiate the configured ``SubLatentModel`` via the registry.
    5. Run training loop with early stopping.
    6. Save best checkpoint; emit path and stats to context.

    Attributes:
        _model: The instantiated SubLatentModel (set during ``run``).
        _checkpoint_dir: Directory for saving checkpoints.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        checkpoint_dir: str | None = None,
    ) -> None:
        """Initialise the training stage.

        Args:
            config: Full experiment configuration.  Training hyper-parameters
                are read from ``config.sublatent.training``.
            checkpoint_dir: Override for checkpoint save location. Defaults to
                ``config.paths.output_root/sublatent_checkpoints``.
        """
        super().__init__(config)
        self._model: SubLatentModel | None = None

        if checkpoint_dir is not None:
            self._checkpoint_dir = Path(checkpoint_dir)
        else:
            self._checkpoint_dir = (
                Path(config.paths.output_root) / "sublatent_checkpoints"
            )

    # ---------------------------------------------------------------------------
    # PipelineStage interface
    # ---------------------------------------------------------------------------

    def io(self) -> StageIO:
        """Declare inputs and outputs.

        Returns:
            StageIO with ``latent_encodings`` as input and
            ``sublatent_model_path`` / ``sublatent_train_stats`` as outputs.
        """
        return StageIO(
            inputs=("latent_encodings",),
            outputs=("sublatent_model_path", "sublatent_train_stats"),
        )

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute sub-latent model training.

        Args:
            context: Pipeline context containing ``latent_encodings``.

        Returns:
            Dict with ``sublatent_model_path`` and ``sublatent_train_stats``.
        """
        sublatent_cfg = self.config.sublatent
        train_cfg = sublatent_cfg.training
        device = get_device(self.config.device)
        device_str = str(device)

        # ------------------------------------------------------------------ #
        # 1. Build dataset from latent encodings
        # ------------------------------------------------------------------ #
        encodings: list[LatentEncoding] = context.get("latent_encodings", [])
        if not encodings:
            logger.warning("TrainSubLatentStage: no latent_encodings in context; skipping.")
            return {
                "sublatent_model_path": "",
                "sublatent_train_stats": {"skipped": True},
            }

        logger.info(
            "TrainSubLatentStage: building dataset from %d latent encodings.", len(encodings)
        )

        flat_tensors: list[torch.Tensor] = []
        for enc in encodings:
            z = enc.z_mu.to(dtype=torch.float32)
            flat_tensors.append(z.flatten())  # (input_dim,)

        data: torch.Tensor = torch.stack(flat_tensors, dim=0)  # (N, input_dim)
        input_dim = data.shape[1]
        n_samples = data.shape[0]

        logger.info(
            "TrainSubLatentStage: N=%d, input_dim=%d, target_dim=%d",
            n_samples, input_dim, sublatent_cfg.target_dim,
        )

        # ------------------------------------------------------------------ #
        # 2. Train / val split (90/10)
        # ------------------------------------------------------------------ #
        n_val = max(1, int(n_samples * 0.1))
        n_train = n_samples - n_val
        perm = torch.randperm(n_samples)
        train_data = data[perm[:n_train]]
        val_data = data[perm[n_train:]]

        train_loader = DataLoader(
            TensorDataset(train_data),
            batch_size=train_cfg.batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            TensorDataset(val_data),
            batch_size=train_cfg.batch_size,
            shuffle=False,
            drop_last=False,
        )

        # ------------------------------------------------------------------ #
        # 3. Instantiate sub-latent model
        # ------------------------------------------------------------------ #
        approach = sublatent_cfg.approach
        logger.info("TrainSubLatentStage: instantiating '%s' sub-latent model.", approach)

        model: SubLatentModel = ComponentRegistry.create(
            "sublatent",
            approach,
            sublatent_cfg,
            input_dim,
            device_str,
        )
        self._model = model

        # PCA / UMAP are non-iterative — delegate to fit() and exit
        if approach in ("pca", "umap"):
            logger.info(
                "TrainSubLatentStage: approach='%s' is non-iterative; calling fit().", approach
            )
            stats = model.fit(train_data.to(device))
            checkpoint_path = self._save_checkpoint(model, approach, epoch=0)
            return {
                "sublatent_model_path": checkpoint_path,
                "sublatent_train_stats": stats,
            }

        # ------------------------------------------------------------------ #
        # 4. Iterative training loop
        # ------------------------------------------------------------------ #
        best_val_loss = math.inf
        patience_counter = 0
        best_checkpoint_path = ""
        history: list[dict[str, float]] = []

        for epoch in range(1, train_cfg.epochs + 1):
            # --- Train ---
            epoch_train_losses: list[dict[str, float]] = []
            for (batch,) in train_loader:
                loss_dict = model.train_step(batch.to(device))
                epoch_train_losses.append(loss_dict)

            avg_train = self._average_losses(epoch_train_losses)

            # --- Validate ---
            val_losses = self._evaluate(model, val_loader, device)
            avg_val = val_losses

            epoch_stats = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in avg_train.items()},
                **{f"val_{k}": v for k, v in avg_val.items()},
            }
            history.append(epoch_stats)

            val_loss = avg_val.get("total_loss", avg_val.get("recon_loss", 0.0))

            if epoch % max(1, train_cfg.epochs // 10) == 0 or epoch == 1:
                logger.info(
                    "Epoch %d/%d — train_loss=%.4f, val_loss=%.4f",
                    epoch,
                    train_cfg.epochs,
                    avg_train.get("total_loss", 0.0),
                    val_loss,
                )

            # --- Checkpoint best model ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_checkpoint_path = self._save_checkpoint(model, approach, epoch)
            else:
                patience_counter += 1

            # --- Early stopping ---
            if patience_counter >= train_cfg.patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d).",
                    epoch,
                    train_cfg.patience,
                )
                break

        logger.info(
            "TrainSubLatentStage: training complete. best_val_loss=%.4f, checkpoint=%s",
            best_val_loss,
            best_checkpoint_path,
        )

        train_stats = {
            "approach": approach,
            "n_samples": n_samples,
            "input_dim": input_dim,
            "target_dim": sublatent_cfg.target_dim,
            "best_val_loss": best_val_loss,
            "epochs_trained": len(history),
            "final_epoch": history[-1] if history else {},
        }

        return {
            "sublatent_model_path": best_checkpoint_path,
            "sublatent_train_stats": train_stats,
        }

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(
        self,
        model: SubLatentModel,
        loader: DataLoader,
        device: torch.device,
    ) -> dict[str, float]:
        """Compute average validation loss without gradient tracking.

        Args:
            model: The sub-latent model.
            loader: DataLoader over validation data.
            device: Compute device.

        Returns:
            Dict of averaged loss keys.
        """
        all_losses: list[dict[str, float]] = []

        # Put pytorch submodules in eval mode if available
        for attr in ("encoder", "decoder", "encoder_shared", "mu_head", "logvar_head"):
            module = getattr(model, attr, None)
            if isinstance(module, torch.nn.Module):
                module.eval()

        for (batch,) in loader:
            # Re-run forward without gradient; train_step uses .backward()
            # so we replicate the forward pass here using the same loss.
            batch = batch.to(device=str(device), dtype=torch.float32)
            s = model.encode(batch)
            z_hat = model.decode(s)
            recon_loss = float(torch.nn.functional.mse_loss(z_hat, batch).item())
            loss_dict: dict[str, float] = {
                "recon_loss": recon_loss,
                "total_loss": recon_loss,
            }
            all_losses.append(loss_dict)

        # Restore training mode
        for attr in ("encoder", "decoder", "encoder_shared", "mu_head", "logvar_head"):
            module = getattr(model, attr, None)
            if isinstance(module, torch.nn.Module):
                module.train()

        return self._average_losses(all_losses)

    @staticmethod
    def _average_losses(loss_dicts: list[dict[str, float]]) -> dict[str, float]:
        """Average a list of per-step loss dicts.

        Args:
            loss_dicts: List of loss dicts from ``train_step``.

        Returns:
            Dict with averaged values.
        """
        if not loss_dicts:
            return {}
        keys = loss_dicts[0].keys()
        return {k: sum(d[k] for d in loss_dicts) / len(loss_dicts) for k in keys}

    def _save_checkpoint(
        self, model: SubLatentModel, approach: str, epoch: int
    ) -> str:
        """Save a model checkpoint and return the path.

        Args:
            model: The sub-latent model to save.
            approach: Sub-latent approach name for the filename.
            epoch: Current training epoch for the filename.

        Returns:
            Absolute path string of the saved checkpoint.
        """
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self._checkpoint_dir / f"{approach}_epoch{epoch:04d}.pt"
        model.save(str(ckpt_path))
        return str(ckpt_path)

    def cache_key(self, context: dict[str, Any]) -> str | None:
        """Compute a cache key from sub-latent config and encoding count.

        Args:
            context: Pipeline context.

        Returns:
            Hex digest or None if no encodings present.
        """
        encodings = context.get("latent_encodings", [])
        if not encodings:
            return None

        sl_cfg = self.config.sublatent
        return compute_hash(
            sl_cfg.approach,
            sl_cfg.target_dim,
            sl_cfg.training.epochs,
            sl_cfg.training.learning_rate,
            sl_cfg.training.batch_size,
            sl_cfg.training.patience,
            len(encodings),
        )
