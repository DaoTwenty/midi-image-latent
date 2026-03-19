"""IngestStage: pipeline stage for loading raw MIDI files into BarData objects.

This stage is the first in the experiment pipeline.  It reads all MIDI
files from data_root, invokes MidiIngestor, and produces a list of
BarData objects in the pipeline context under the key "bars".

For large datasets the streaming mode should be preferred: call
:meth:`IngestStage.resolve_files` to obtain the file list and then
iterate over file batches with :func:`iter_file_batches`, feeding each
batch's bars to the downstream pipeline stages.  This avoids loading
all bars into memory simultaneously.

Context outputs:
    bars: list[BarData]  — all extracted bars across all files/instruments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Generator

from midi_vae.config import ExperimentConfig
from midi_vae.data.preprocessing import MidiIngestor
from midi_vae.data.subset import apply_subset
from midi_vae.data.types import BarData
from midi_vae.pipelines.base import PipelineStage, StageIO, compute_hash

logger = logging.getLogger(__name__)

# Default number of MIDI files to ingest per batch in streaming mode.
# Chosen to keep peak memory for BarData at ~1-2 GB per batch
# (~300 bars/file × 72 KB/bar × 100 files ≈ 2.1 GB).
DEFAULT_FILE_BATCH_SIZE: int = 100


def iter_chunks(
    bars: list[Any],
    chunk_size: int,
    max_bars: int | None = None,
) -> "Generator[list[Any], None, None]":
    """Yield successive chunks from *bars*, stopping at *max_bars* total.

    Args:
        bars: Full list of BarData (or any items) to iterate over.
        chunk_size: Maximum number of items per yielded chunk.
        max_bars: Hard cap on the total number of items yielded across all
            chunks.  If None, all items are yielded.  When the cap is reached
            mid-chunk the chunk is truncated and iteration stops.

    Yields:
        Successive sub-lists of *bars* each of length <= *chunk_size*.
    """
    total_yielded = 0
    for start in range(0, len(bars), chunk_size):
        if max_bars is not None and total_yielded >= max_bars:
            return
        chunk = bars[start : start + chunk_size]
        if max_bars is not None:
            remaining = max_bars - total_yielded
            chunk = chunk[:remaining]
        total_yielded += len(chunk)
        yield chunk


def iter_file_batches(
    files: list[Path],
    batch_size: int = DEFAULT_FILE_BATCH_SIZE,
) -> Generator[list[Path], None, None]:
    """Yield successive batches of file paths.

    Args:
        files: Full sorted list of file paths.
        batch_size: Maximum number of files per batch.

    Yields:
        Sub-lists of *files* each of length <= *batch_size*.
    """
    for start in range(0, len(files), batch_size):
        yield files[start : start + batch_size]


# Map dataset name to glob pattern for the raw files
_DATASET_GLOB: dict[str, str] = {
    "lakh": "**/*.mid",
    "pop909": "**/*.mid",
    "maestro": "**/*.midi",
}


class IngestStage(PipelineStage):
    """Pipeline stage that loads raw MIDI files and extracts BarData objects.

    Reads MIDI files from config.paths.data_root using the configured
    dataset type (lakh | pop909 | maestro).  All valid bars are collected
    and stored under the context key 'bars'.

    Args:
        config: Full experiment configuration.
        max_files: Optional override to limit the number of files loaded.
                   Useful for debugging.  If None, all files are loaded.
    """

    def __init__(self, config: ExperimentConfig, max_files: int | None = None) -> None:
        super().__init__(config)
        # Explicit max_files overrides config; otherwise read from config.data.max_files
        self._max_files = max_files if max_files is not None else config.data.max_files

    def io(self) -> StageIO:
        """Declare that this stage produces 'bars' with no upstream inputs.

        Returns:
            StageIO with empty inputs and ("bars",) as output.
        """
        return StageIO(inputs=(), outputs=("bars",))

    def resolve_files(self) -> list[Path]:
        """Discover and filter MIDI file paths without loading any data.

        Applies dataset glob, subset filtering, and max_files cap — the
        same logic as :meth:`run` but returns the file list instead of
        loading bars.  Use this with :func:`iter_file_batches` for
        streaming ingestion that avoids materialising all bars at once.

        Returns:
            Sorted list of Path objects to MIDI files.
        """
        data_cfg = self.config.data
        data_root = Path(self.config.paths.data_root)
        dataset_name = data_cfg.dataset.lower()

        if data_cfg.subset and data_cfg.subset.glob_pattern:
            glob_pattern = data_cfg.subset.glob_pattern
        else:
            glob_pattern = _DATASET_GLOB.get(dataset_name, "**/*.mid")
            if dataset_name not in _DATASET_GLOB:
                logger.warning(
                    "Unknown dataset '%s'; defaulting glob to '**/*.mid'", dataset_name
                )

        if not data_root.exists():
            logger.warning("data_root does not exist: %s", data_root)
            return []

        all_files = sorted(data_root.glob(glob_pattern))
        if not all_files and dataset_name == "maestro":
            all_files = sorted(data_root.glob("**/*.mid"))

        if data_cfg.subset is not None:
            all_files = apply_subset(
                all_files, data_cfg.subset, data_root, self.config.seed
            )

        if self._max_files is not None:
            all_files = all_files[: self._max_files]

        return all_files

    def make_ingestor(self) -> MidiIngestor:
        """Create a MidiIngestor configured from this stage's config.

        Returns:
            A MidiIngestor instance ready to ingest files.
        """
        data_cfg = self.config.data
        return MidiIngestor(
            time_steps=data_cfg.time_steps,
            min_notes_per_bar=data_cfg.min_notes_per_bar,
            instruments=data_cfg.instruments,
            bars_per_instrument=data_cfg.bars_per_instrument,
            seed=self.config.seed,
        )

    def ingest_batch(
        self,
        file_batch: list[Path],
        ingestor: MidiIngestor | None = None,
        num_workers: int = 1,
    ) -> list[BarData]:
        """Ingest a batch of files and return their bars.

        This is the building block for streaming ingestion: call
        :meth:`resolve_files`, split with :func:`iter_file_batches`,
        and pass each batch here.

        Args:
            file_batch: File paths to ingest in this batch.
            ingestor: Reusable MidiIngestor (created via :meth:`make_ingestor`).
                If None, a new one is created.
            num_workers: Number of parallel workers for this batch.

        Returns:
            List of BarData objects from the given files.
        """
        if ingestor is None:
            ingestor = self.make_ingestor()

        if num_workers > 1 and len(file_batch) > 1:
            return ingestor.ingest_files_parallel(
                file_batch, max_workers=num_workers
            )
        else:
            bars: list[BarData] = []
            for fp in file_batch:
                bars.extend(ingestor.ingest_file(fp))
            return bars

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

        # When a custom glob_pattern is specified in SubsetConfig, use it for
        # initial file discovery instead of the default dataset glob.
        if data_cfg.subset and data_cfg.subset.glob_pattern:
            glob_pattern = data_cfg.subset.glob_pattern
        else:
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
        # MAESTRO uses both .midi and .mid — if primary glob found nothing, try the other
        if not all_files and dataset_name == "maestro":
            all_files = sorted(data_root.glob("**/*.mid"))

        # Apply subset filtering (subdirectory, glob_pattern intersection, file_list, fraction)
        # before the hard max_files cap so the cap acts on the already-filtered set.
        if data_cfg.subset is not None:
            all_files = apply_subset(
                all_files, data_cfg.subset, data_root, self.config.seed
            )

        if self._max_files is not None:
            all_files = all_files[: self._max_files]

        num_workers: int = getattr(self.config, "num_workers", 1)

        logger.info(
            "IngestStage: loading %d %s files from %s (num_workers=%d)",
            len(all_files),
            dataset_name,
            data_root,
            num_workers,
        )

        logger.warning(
            "IngestStage.run(): loading ALL bars into memory at once. "
            "For large datasets, use the streaming path in SweepExecutor "
            "(set pipeline_chunk_size > 0) to avoid OOM."
        )

        if num_workers > 1:
            bars: list[BarData] = ingestor.ingest_files_parallel(
                all_files, max_workers=num_workers
            )
        else:
            bars = []
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

        subset_key = ""
        if self.config.data.subset:
            s = self.config.data.subset
            subset_key = (
                f"{s.subdirectory}|{s.glob_pattern}|{s.file_list}"
                f"|{s.fraction}|{s.fraction_seed}"
            )

        return compute_hash(
            str(data_root),
            self.config.data.dataset,
            self.config.data.instruments,
            self.config.data.min_notes_per_bar,
            self.config.data.time_steps,
            self.config.data.bars_per_instrument,
            self.config.seed,
            subset_key,
            self._max_files,
        )
