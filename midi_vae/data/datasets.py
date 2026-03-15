"""PyTorch Dataset implementations for MIDI piano-roll data.

Each dataset class wraps a preprocessed set of BarData objects and
renders them into PianoRollImage tensors on the fly.

Available datasets:
    LakhDataset    — Lakh MIDI Dataset (LMD full, raw .mid files).
    Pop909Dataset  — POP909 dataset (.mid files).
    MaestroDataset — MAESTRO classical piano dataset (.midi files).

All datasets are registered in the ComponentRegistry and can be
retrieved by name.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from midi_vae.config import DataConfig, RenderConfig
from midi_vae.data.preprocessing import MidiIngestor
from midi_vae.data.rendering import build_strategy, ChannelStrategy
from midi_vae.data.transforms import ResizeTransform
from midi_vae.data.types import BarData, PianoRollImage
from midi_vae.registry import ComponentRegistry

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────


class MidiDataset(Dataset):
    """Base class for MIDI piano-roll datasets.

    Subclasses implement _load_bars() to provide a list of BarData
    objects.  The base class handles rendering and optional transforms.

    Args:
        data_config: DataConfig controlling which instruments and bars are loaded.
        render_config: RenderConfig controlling channel strategy and normalization.
        data_root: Filesystem root for the raw dataset.
        seed: Random seed for reproducible bar sampling.
    """

    def __init__(
        self,
        data_config: DataConfig,
        render_config: RenderConfig,
        data_root: str | Path,
        seed: int = 42,
    ) -> None:
        self.data_config = data_config
        self.render_config = render_config
        self.data_root = Path(data_root)
        self.seed = seed

        self._strategy: ChannelStrategy = build_strategy(
            name=render_config.channel_strategy,
            pitch_axis=render_config.pitch_axis,
            normalize_low=render_config.normalize_range[0],
            normalize_high=render_config.normalize_range[1],
        )

        self._resize = ResizeTransform(
            target_resolution=tuple(data_config.target_resolution),
            method=render_config.resize_method,
        )

        self._bars: list[BarData] = self._load_bars()
        logger.info(
            "%s loaded %d bars from %s",
            self.__class__.__name__,
            len(self._bars),
            self.data_root,
        )

    # ── Abstract ──────────────────────────────────────────────────────────────

    def _load_bars(self) -> list[BarData]:
        """Load and preprocess all bars for this dataset.

        Returns:
            Flat list of BarData objects.
        """
        raise NotImplementedError

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Return the number of bars in this dataset."""
        return len(self._bars)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a rendered bar as a dict containing a PianoRollImage tensor.

        Args:
            idx: Integer index into the dataset.

        Returns:
            Dict with keys:
                "image"   : torch.Tensor of shape (3, H, W) in normalize_range.
                "bar_id"  : str identifier.
                "channel_strategy": str name.
                "resolution": tuple (H, W).
                "pitch_axis": str.
        """
        bar = self._bars[idx]
        tensor = self._render(bar)

        image_obj = PianoRollImage(
            bar_id=bar.bar_id,
            image=tensor,
            channel_strategy=self.render_config.channel_strategy,
            resolution=tuple(tensor.shape[1:]),
            pitch_axis=self.render_config.pitch_axis,
        )

        return {
            "image": image_obj.image,
            "bar_id": image_obj.bar_id,
            "channel_strategy": image_obj.channel_strategy,
            "resolution": image_obj.resolution,
            "pitch_axis": image_obj.pitch_axis,
        }

    def get_piano_roll_image(self, idx: int) -> PianoRollImage:
        """Return a fully typed PianoRollImage for bar at index idx.

        Args:
            idx: Integer index into the dataset.

        Returns:
            PianoRollImage dataclass instance.
        """
        bar = self._bars[idx]
        tensor = self._render(bar)
        return PianoRollImage(
            bar_id=bar.bar_id,
            image=tensor,
            channel_strategy=self.render_config.channel_strategy,
            resolution=tuple(tensor.shape[1:]),
            pitch_axis=self.render_config.pitch_axis,
        )

    # ── Internal rendering ────────────────────────────────────────────────────

    def _render(self, bar: BarData) -> torch.Tensor:
        """Render a single bar to a (3, H, W) tensor.

        The bar is first rendered by the channel strategy, then resized
        to target_resolution.

        Args:
            bar: BarData object.

        Returns:
            Tensor of shape (3, H, W).
        """
        tensor = self._strategy.render(bar)
        tensor = self._resize(tensor)
        return tensor

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_ingestor(self) -> MidiIngestor:
        """Build a MidiIngestor configured from data_config and seed.

        Returns:
            Configured MidiIngestor instance.
        """
        return MidiIngestor(
            time_steps=self.data_config.time_steps,
            min_notes_per_bar=self.data_config.min_notes_per_bar,
            instruments=self.data_config.instruments,
            bars_per_instrument=self.data_config.bars_per_instrument,
            seed=self.seed,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Lakh MIDI Dataset
# ──────────────────────────────────────────────────────────────────────────────


@ComponentRegistry.register("dataset", "lakh")
class LakhDataset(MidiDataset):
    """Dataset for the Lakh MIDI Dataset (LMD full version, raw .mid files).

    Expects data_root to contain .mid files in a flat or nested directory
    structure.  Files are loaded recursively via ``**/*.mid``.

    Download the dataset with::

        bash scripts/download_data.sh lakh

    which will place files at ``data/lakh/``.  Expected structure::

        data_root/
            <hash_prefix>/
                <song_id>.mid
            ...

    or any nested structure — all ``.mid`` files are discovered recursively.

    Args:
        data_config: DataConfig controlling instruments and bar sampling.
        render_config: RenderConfig controlling rendering strategy.
        data_root: Path to the root of the Lakh MIDI dataset.
        seed: Random seed for reproducible bar sampling.
        max_files: If set, limit the number of .mid files loaded.  Useful
                   for quick debugging runs.
    """

    def __init__(
        self,
        data_config: DataConfig,
        render_config: RenderConfig,
        data_root: str | Path,
        seed: int = 42,
        max_files: int | None = None,
    ) -> None:
        self._max_files = max_files
        super().__init__(data_config, render_config, data_root, seed)

    def _load_bars(self) -> list[BarData]:
        """Recursively load all .mid files from data_root.

        Returns:
            List of BarData objects from all valid tracks/files.
        """
        if not self.data_root.exists():
            logger.warning("LakhDataset data_root does not exist: %s", self.data_root)
            return []

        ingestor = self._build_ingestor()
        bars: list[BarData] = []

        files = sorted(self.data_root.glob("**/*.mid"))
        if self._max_files is not None:
            files = files[: self._max_files]

        logger.info("LakhDataset: scanning %d .mid files in %s", len(files), self.data_root)

        for file_path in files:
            file_bars = ingestor.ingest_file(file_path)
            bars.extend(file_bars)

        return bars


# ──────────────────────────────────────────────────────────────────────────────
# Pop909
# ──────────────────────────────────────────────────────────────────────────────


@ComponentRegistry.register("dataset", "pop909")
class Pop909Dataset(MidiDataset):
    """Dataset for the POP909 multi-instrument piano dataset (.mid files).

    Expects data_root to contain numbered song directories each with a
    .mid file::

        data_root/
            001/
                001.mid
            002/
                002.mid

    Args:
        data_config: DataConfig controlling instruments and bar sampling.
        render_config: RenderConfig controlling rendering strategy.
        data_root: Path to the root of the POP909 dataset.
        seed: Random seed for reproducible bar sampling.
        max_files: If set, limit the number of .mid files loaded.
    """

    def __init__(
        self,
        data_config: DataConfig,
        render_config: RenderConfig,
        data_root: str | Path,
        seed: int = 42,
        max_files: int | None = None,
    ) -> None:
        self._max_files = max_files
        super().__init__(data_config, render_config, data_root, seed)

    def _load_bars(self) -> list[BarData]:
        """Recursively load all .mid files from data_root.

        Returns:
            List of BarData objects from all valid tracks/files.
        """
        if not self.data_root.exists():
            logger.warning("Pop909Dataset data_root does not exist: %s", self.data_root)
            return []

        ingestor = self._build_ingestor()
        bars: list[BarData] = []

        files = sorted(self.data_root.glob("**/*.mid"))
        if self._max_files is not None:
            files = files[: self._max_files]

        logger.info("Pop909Dataset: scanning %d .mid files in %s", len(files), self.data_root)

        for file_path in files:
            file_bars = ingestor.ingest_file(file_path)
            bars.extend(file_bars)

        return bars


# ──────────────────────────────────────────────────────────────────────────────
# Maestro
# ──────────────────────────────────────────────────────────────────────────────


@ComponentRegistry.register("dataset", "maestro")
class MaestroDataset(MidiDataset):
    """Dataset for the MAESTRO classical piano dataset (.midi files).

    MAESTRO contains recordings of classical piano performances in MIDI
    format, organized by year::

        data_root/
            2004/
                MIDI-Unprocessed_01_R1_2004_01_ORIG_MID--AUDIO_01_R1_2004_01_Track01_wav.midi
            2006/
                ...

    Args:
        data_config: DataConfig controlling instruments and bar sampling.
        render_config: RenderConfig controlling rendering strategy.
        data_root: Path to the root of the MAESTRO dataset.
        seed: Random seed for reproducible bar sampling.
        max_files: If set, limit the number of .midi files loaded.
    """

    def __init__(
        self,
        data_config: DataConfig,
        render_config: RenderConfig,
        data_root: str | Path,
        seed: int = 42,
        max_files: int | None = None,
    ) -> None:
        self._max_files = max_files
        super().__init__(data_config, render_config, data_root, seed)

    def _load_bars(self) -> list[BarData]:
        """Recursively load all .midi files from data_root.

        Returns:
            List of BarData objects from all valid tracks.
        """
        if not self.data_root.exists():
            logger.warning("MaestroDataset data_root does not exist: %s", self.data_root)
            return []

        ingestor = self._build_ingestor()
        bars: list[BarData] = []

        # MAESTRO uses both .midi and .mid extensions
        files = sorted(
            list(self.data_root.glob("**/*.midi"))
            + list(self.data_root.glob("**/*.mid"))
        )
        if self._max_files is not None:
            files = files[: self._max_files]

        logger.info(
            "MaestroDataset: scanning %d MIDI files in %s", len(files), self.data_root
        )

        for file_path in files:
            file_bars = ingestor.ingest_file(file_path)
            bars.extend(file_bars)

        return bars
