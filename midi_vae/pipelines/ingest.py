"""IngestStage: pipeline stage for loading raw MIDI files into BarData objects.

This stage is the first in the experiment pipeline.  It reads all MIDI
files from data_root, invokes MidiIngestor, and produces a list of
BarData objects in the pipeline context under the key "bars".

Context outputs:
    bars: list[BarData]  — all extracted bars across all files/instruments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from midi_vae.config import ExperimentConfig
from midi_vae.data.preprocessing import MidiIngestor
from midi_vae.data.types import BarData
from midi_vae.pipelines.base import PipelineStage, StageIO, compute_hash

logger = logging.getLogger(__name__)

# Map dataset name to glob pattern for the raw files
_DATASET_GLOB: dict[str, str] = {
    "lpd5": "**/*.npz",
    "pop909": "**/*.mid",
    "maestro": "**/*.midi",
}


class IngestStage(PipelineStage):
    """Pipeline stage that loads raw MIDI files and extracts BarData objects.

    Reads MIDI files from config.paths.data_root using the configured
    dataset type (lpd5 | pop909 | maestro).  All valid bars are collected
    and stored under the context key 'bars'.

    Args:
        config: Full experiment configuration.
        max_files: Optional override to limit the number of files loaded.
                   Useful for debugging.  If None, all files are loaded.
    """

    def __init__(self, config: ExperimentConfig, max_files: int | None = None) -> None:
        super().__init__(config)
        self._max_files = max_files

    def io(self) -> StageIO:
        """Declare that this stage produces 'bars' with no upstream inputs.

        Returns:
            StageIO with empty inputs and ("bars",) as output.
        """
        return StageIO(inputs=(), outputs=("bars",))

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Load all MIDI files and return extracted bars.

        Args:
            context: Pipeline context (not used; this is the first stage).

        Returns:
            Dict with key "bars" mapping to a list of BarData objects.
        """
        data_cfg = self.config.data
        data_root = Path(self.config.paths.data_root)
        dataset_name = data_cfg.dataset.lower()

        glob_pattern = _DATASET_GLOB.get(dataset_name, "**/*.mid")
        if dataset_name not in _DATASET_GLOB:
            logger.warning(
                "Unknown dataset '%s'; defaulting glob to '**/*.mid'", dataset_name
            )

        ingestor = MidiIngestor(
            time_steps=data_cfg.time_steps,
            min_notes_per_bar=data_cfg.min_notes_per_bar,
            instruments=data_cfg.instruments,
            bars_per_instrument=data_cfg.bars_per_instrument,
            seed=self.config.seed,
        )

        if not data_root.exists():
            logger.warning("data_root does not exist: %s", data_root)
            return {"bars": []}

        all_files = sorted(data_root.glob(glob_pattern))
        if self._max_files is not None:
            all_files = all_files[: self._max_files]

        logger.info(
            "IngestStage: loading %d %s files from %s",
            len(all_files),
            dataset_name,
            data_root,
        )

        bars: list[BarData] = []
        for file_path in all_files:
            file_bars = ingestor.ingest_file(file_path)
            bars.extend(file_bars)

        logger.info("IngestStage: extracted %d bars total", len(bars))
        return {"bars": bars}

    def cache_key(self, context: dict[str, Any]) -> str | None:
        """Compute a cache key based on data config and data_root contents.

        Args:
            context: Current pipeline context (unused here).

        Returns:
            A hex digest string for caching, or None if data_root is missing.
        """
        data_root = Path(self.config.paths.data_root)
        if not data_root.exists():
            return None

        return compute_hash(
            str(data_root),
            self.config.data.dataset,
            self.config.data.instruments,
            self.config.data.min_notes_per_bar,
            self.config.data.time_steps,
            self.config.data.bars_per_instrument,
            self.config.seed,
        )
