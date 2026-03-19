"""Background thread that watches a relay directory for metric JSON files
from distributed ranks and forwards them to a single wandb run.

In a multi-node SLURM sweep, each rank runs its own SweepExecutor and writes
small JSON relay files to a shared NFS directory.  Only rank 0 initialises
wandb; this watcher (running on rank 0) polls the relay directory and logs
every new file to the single wandb run, giving real-time visibility across
all ranks from a single dashboard entry.

Usage (rank 0 only)::

    from midi_vae.tracking.wandb_relay import WandbRelayWatcher
    watcher = WandbRelayWatcher(relay_dir, wandb_logger, poll_interval=5.0)
    watcher.start()
    # ... run the sweep ...
    watcher.stop()   # final drain + join
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class WandbRelayWatcher:
    """Watches a shared relay directory and logs metrics to wandb.

    Other ranks write small JSON files to the relay directory.
    This watcher (running on rank 0) picks them up and logs to wandb.

    The watcher runs in a daemon background thread so it never blocks the
    main sweep loop.  Call :meth:`stop` when the sweep is complete to
    perform a final drain of any files written near the end of execution.

    Args:
        relay_dir: Shared directory (NFS) where ranks write metric JSONs.
        wandb_logger: The :class:`~midi_vae.tracking.wandb_logger.WandbLogger`
            instance owned by rank 0.
        poll_interval: Seconds between directory polls (default: 5.0).
    """

    def __init__(
        self,
        relay_dir: Path,
        wandb_logger: Any,
        poll_interval: float = 5.0,
    ) -> None:
        self.relay_dir = Path(relay_dir)
        self.relay_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_logger = wandb_logger
        self.poll_interval = poll_interval
        self._seen: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background watcher thread.

        Idempotent — calling start() a second time has no effect.
        """
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._watch_loop,
            daemon=True,
            name="wandb-relay-watcher",
        )
        self._thread.start()
        logger.info("WandbRelayWatcher: started watching %s", self.relay_dir)

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the watcher and do a final drain of any remaining files.

        Args:
            timeout: Maximum seconds to wait for the background thread to
                finish after signalling it to stop.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        # Final drain — pick up any files written after the last poll.
        self._drain()
        logger.info(
            "WandbRelayWatcher: stopped (processed %d relay files total)",
            len(self._seen),
        )

    def _watch_loop(self) -> None:
        """Poll loop running in the background thread."""
        while not self._stop.is_set():
            self._drain()
            self._stop.wait(self.poll_interval)

    def _drain(self) -> None:
        """Process all new ``cond_*.json`` files in the relay directory."""
        try:
            for path in sorted(self.relay_dir.glob("cond_*.json")):
                fname = path.name
                if fname in self._seen:
                    continue
                self._seen.add(fname)
                try:
                    with path.open() as fh:
                        data = json.load(fh)
                    self._log_relay(data)
                except Exception as exc:
                    logger.warning(
                        "WandbRelayWatcher: failed to process %s: %s", fname, exc
                    )
        except Exception as exc:
            logger.warning(
                "WandbRelayWatcher: error scanning relay dir: %s", exc
            )

    def _log_relay(self, data: dict[str, Any]) -> None:
        """Forward a single relay file's metrics to wandb.

        Metrics are prefixed with ``<condition_label>/<vae_or_key>/`` so that
        each condition appears as a separate group in the wandb dashboard.

        Args:
            data: Parsed relay JSON dictionary (see :func:`SweepExecutor._write_relay`).
        """
        label = data.get("condition_label", "unknown")
        rank = data.get("rank", -1)
        summary = data.get("metrics_summary", {})

        if not summary:
            logger.debug(
                "WandbRelayWatcher: empty metrics for %s (rank %d)", label, rank
            )
            return

        # Flatten nested {vae: {metric: value}} structure for wandb.
        flat_metrics: dict[str, Any] = {}
        for vae_or_key, values in summary.items():
            if isinstance(values, dict):
                for metric_name, value in values.items():
                    flat_metrics[f"{label}/{vae_or_key}/{metric_name}"] = value
            else:
                flat_metrics[f"{label}/{vae_or_key}"] = values

        if flat_metrics:
            self.wandb_logger.log_metrics(flat_metrics, commit=True)
            logger.info(
                "WandbRelayWatcher: logged %d metrics for '%s' (rank %d)",
                len(flat_metrics),
                label,
                rank,
            )
