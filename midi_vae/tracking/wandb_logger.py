"""Weights & Biases integration with online and offline mode support.

Supports two usage patterns:
    1. **Online** (``wandb_mode="online"``): Logs directly to wandb cloud.
       Requires internet access. Use on login nodes or machines with network.
    2. **Offline** (``wandb_mode="offline"``): Logs to a local directory.
       No internet required. Sync later from a login node with::

           wandb sync <run_dir>

       Or use the helper script::

           bash scripts/wandb_sync.sh

The logger wraps the wandb Python API and gracefully degrades if wandb
is not installed or if initialization fails.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from midi_vae.config import ExperimentConfig, TrackingConfig

logger = logging.getLogger(__name__)


class WandbLogger:
    """Thin wrapper around wandb for experiment tracking.

    Handles initialization, metric logging, artifact saving, and
    graceful fallback when wandb is unavailable or disabled.

    The ``mode`` is determined by ``TrackingConfig.wandb_mode``:
        - ``"online"``: real-time cloud logging (needs internet).
        - ``"offline"``: local logging, sync later with ``wandb sync``.
        - ``"disabled"``: no wandb at all (equivalent to wandb_enabled=False).

    Args:
        config: Full experiment configuration.
        experiment_id: Unique experiment ID from ExperimentTracker.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        experiment_id: str,
    ) -> None:
        self.config = config
        self.tracking = config.tracking
        self.experiment_id = experiment_id
        self._run: Any = None  # wandb.Run or None
        self._enabled = False

        if not self.tracking.wandb_enabled:
            logger.info("WandbLogger: disabled via config (wandb_enabled=false)")
            return

        mode = self.tracking.wandb_mode.lower()
        if mode == "disabled":
            logger.info("WandbLogger: disabled via wandb_mode='disabled'")
            return

        self._init_wandb(mode)

    def _init_wandb(self, mode: str) -> None:
        """Initialize a wandb run in the given mode.

        Args:
            mode: One of 'online' or 'offline'.
        """
        try:
            import wandb
        except ImportError:
            logger.warning(
                "WandbLogger: wandb not installed. "
                "Install with: pip install wandb"
            )
            return

        # Validate API key for online mode (offline can log without it)
        api_key = os.environ.get("WANDB_API_KEY", "")
        if mode == "online" and (not api_key or api_key == "YOUR_KEY_HERE"):
            logger.warning(
                "WandbLogger: WANDB_API_KEY not set or is placeholder. "
                "Falling back to offline mode. Set your key in .env or run: wandb login"
            )
            mode = "offline"

        # Set mode via environment variable (wandb respects this)
        os.environ["WANDB_MODE"] = mode

        # Set offline directory if specified
        wandb_dir = self.tracking.wandb_dir
        if wandb_dir:
            wandb_dir_path = Path(wandb_dir)
            wandb_dir_path.mkdir(parents=True, exist_ok=True)
            os.environ["WANDB_DIR"] = str(wandb_dir_path)

        # Resolve entity: config > env var > None (wandb default)
        entity = self.tracking.wandb_entity or os.environ.get("WANDB_ENTITY") or None

        try:
            # Build config dict for wandb (flatten the Pydantic model)
            wandb_config = self.config.model_dump()

            run = wandb.init(
                project=self.tracking.wandb_project,
                entity=entity,
                name=self.experiment_id,
                config=wandb_config,
                tags=self.tracking.wandb_tags or [],
                dir=wandb_dir,
                resume="allow",  # allows resuming interrupted runs
            )

            self._run = run
            self._enabled = True

            logger.info(
                "WandbLogger: initialized (mode=%s, project=%s, run=%s)",
                mode,
                self.tracking.wandb_project,
                run.id if run else "unknown",
            )

            if mode == "offline":
                run_dir = getattr(run, "dir", "unknown")
                logger.info(
                    "WandbLogger: offline run data at %s — "
                    "sync later with: wandb sync %s",
                    run_dir,
                    run_dir,
                )

        except Exception as exc:
            logger.warning(
                "WandbLogger: failed to initialize wandb: %s. "
                "Continuing without wandb logging.",
                exc,
            )
            self._run = None
            self._enabled = False

    @property
    def enabled(self) -> bool:
        """Whether wandb logging is active."""
        return self._enabled

    @property
    def run(self) -> Any:
        """The underlying wandb.Run object, or None."""
        return self._run

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        """Log scalar metrics to wandb.

        Args:
            metrics: Dict of metric name -> value.
            step: Optional global step number.
            commit: Whether to commit this log entry immediately.
        """
        if not self._enabled or self._run is None:
            return

        try:
            log_kwargs: dict[str, Any] = {}
            if step is not None:
                log_kwargs["step"] = step
            log_kwargs["commit"] = commit
            self._run.log(metrics, **log_kwargs)
        except Exception as exc:
            logger.warning("WandbLogger: failed to log metrics: %s", exc)

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Update the run summary (final metrics shown in the dashboard).

        Args:
            summary: Dict of summary key -> value.
        """
        if not self._enabled or self._run is None:
            return

        try:
            for key, value in summary.items():
                self._run.summary[key] = value
        except Exception as exc:
            logger.warning("WandbLogger: failed to update summary: %s", exc)

    def log_artifact(
        self,
        path: str | Path,
        name: str,
        artifact_type: str = "result",
    ) -> None:
        """Log a file or directory as a wandb artifact.

        Args:
            path: Path to the file or directory to log.
            name: Artifact name.
            artifact_type: Artifact type (e.g., 'result', 'model', 'dataset').
        """
        if not self._enabled or self._run is None:
            return

        try:
            import wandb

            artifact = wandb.Artifact(name=name, type=artifact_type)
            path = Path(path)
            if path.is_dir():
                artifact.add_dir(str(path))
            else:
                artifact.add_file(str(path))
            self._run.log_artifact(artifact)
            logger.info("WandbLogger: logged artifact '%s' (%s)", name, artifact_type)
        except Exception as exc:
            logger.warning("WandbLogger: failed to log artifact: %s", exc)

    def log_table(
        self,
        key: str,
        columns: list[str],
        data: list[list[Any]],
    ) -> None:
        """Log a table to wandb (useful for per-condition metric comparison).

        Args:
            key: Table key in the wandb dashboard.
            columns: Column header names.
            data: List of rows, each a list of values.
        """
        if not self._enabled or self._run is None:
            return

        try:
            import wandb

            table = wandb.Table(columns=columns, data=data)
            self._run.log({key: table})
        except Exception as exc:
            logger.warning("WandbLogger: failed to log table: %s", exc)

    def finish(self) -> None:
        """Finalize the wandb run.

        Call this when the experiment is complete. For offline mode,
        prints the sync command to use from a login node.
        """
        if not self._enabled or self._run is None:
            return

        try:
            mode = self.tracking.wandb_mode.lower()
            run_dir = getattr(self._run, "dir", None)

            self._run.finish()

            if mode == "offline" and run_dir:
                # Print sync instructions for the user
                sync_path = Path(run_dir).parent
                logger.info(
                    "WandbLogger: offline run complete. Sync with:\n"
                    "  wandb sync %s",
                    sync_path,
                )
                print(
                    f"\n[wandb] Offline run saved. Sync from login node with:\n"
                    f"  wandb sync {sync_path}\n"
                )
        except Exception as exc:
            logger.warning("WandbLogger: error during finish: %s", exc)
        finally:
            self._run = None
            self._enabled = False
