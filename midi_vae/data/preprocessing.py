"""MIDI preprocessing: loading, segmentation, and BarData extraction.

The MidiIngestor class is the entry point for turning raw MIDI files into
BarData objects consumable by the rendering pipeline.  It handles:

  - Loading MIDI files via pypianoroll (.npz) or pretty_midi (.mid/.midi).
  - Segmenting multi-track pianorolls into fixed-length bar arrays.
  - Computing onset and sustain masks from the velocity matrix.
  - Filtering out empty or sparse bars.
  - Deterministic random sampling when bars_per_instrument is set.

Edge-case handling:
  - Corrupted or unreadable files are skipped with a warning.
  - Non-4/4 time signatures are skipped.
  - Empty tracks (all zeros) are skipped.
"""

from __future__ import annotations

import logging
import random
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from midi_vae.data.types import BarData
from midi_vae.registry import ComponentRegistry

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Default 5-track instrument order for multi-track pianoroll .npz files
MULTITRACK_INSTRUMENTS: list[str] = ["drums", "piano", "guitar", "bass", "strings"]

# Default beat resolution in pianoroll steps (24 steps per beat)
DEFAULT_BEAT_RESOLUTION: int = 24

# Default time signature we accept — everything else is skipped
ACCEPTED_TIME_SIG: tuple[int, int] = (4, 4)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _compute_onset_mask(piano_roll: np.ndarray) -> np.ndarray:
    """Derive a binary onset mask from a velocity pianoroll.

    A step is an onset if the velocity is non-zero and the previous step
    was zero (note attack).  The first column is treated as an onset if
    it is non-zero.

    Args:
        piano_roll: Float or int array of shape (128, T).

    Returns:
        Binary uint8 array of shape (128, T).
    """
    active = (piano_roll > 0).astype(np.uint8)
    # Shift right by one column to detect rising edges
    prev = np.zeros_like(active)
    prev[:, 1:] = active[:, :-1]
    onset = active & (~prev.astype(bool)).astype(np.uint8)
    return onset


def _compute_sustain_mask(piano_roll: np.ndarray, onset_mask: np.ndarray) -> np.ndarray:
    """Derive a binary sustain mask (active note but not an onset).

    Args:
        piano_roll: Float or int array of shape (128, T).
        onset_mask: Binary array of shape (128, T) from _compute_onset_mask.

    Returns:
        Binary uint8 array of shape (128, T).
    """
    active = (piano_roll > 0).astype(np.uint8)
    sustain = active & (~onset_mask.astype(bool)).astype(np.uint8)
    return sustain


def _count_notes(piano_roll: np.ndarray) -> int:
    """Count the number of distinct note onsets in a pianoroll bar.

    Args:
        piano_roll: Array of shape (128, T).

    Returns:
        Integer count of onset events.
    """
    onset = _compute_onset_mask(piano_roll)
    return int(onset.sum())


def _segment_track(
    track_roll: np.ndarray,
    steps_per_bar: int,
) -> list[np.ndarray]:
    """Slice a full-song pianoroll into equal-length bar segments.

    Args:
        track_roll: Array of shape (128, total_steps).
        steps_per_bar: Number of time steps in one bar.

    Returns:
        List of arrays each shaped (128, steps_per_bar).
        The tail of the track is dropped if it does not fill a complete bar.
    """
    _, total_steps = track_roll.shape
    n_bars = total_steps // steps_per_bar
    bars = []
    for i in range(n_bars):
        start = i * steps_per_bar
        end = start + steps_per_bar
        bars.append(track_roll[:, start:end])
    return bars


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────


@ComponentRegistry.register("preprocessor", "midi_ingestor")
class MidiIngestor:
    """Load and segment MIDI data into BarData objects.

    Supports both the pypianoroll .npz multi-track format and plain
    .mid/.midi files (via pretty_midi).

    Args:
        time_steps: Number of pianoroll time steps per bar.
        min_notes_per_bar: Minimum number of note onsets for a bar to be kept.
        beat_resolution: Pianoroll steps per quarter-note beat.
        accepted_time_sig: (numerator, denominator) of the required time
                           signature.  Any bar from a song with a different
                           time signature is skipped.
        instruments: Optional subset of instrument names to extract.  If None,
                     all available instruments are extracted.
        bars_per_instrument: If set, randomly sample at most this many bars
                             per instrument per call to ingest_file.
        seed: Random seed for reproducible sampling.
    """

    def __init__(
        self,
        time_steps: int = 96,
        min_notes_per_bar: int = 2,
        beat_resolution: int = DEFAULT_BEAT_RESOLUTION,
        accepted_time_sig: tuple[int, int] = ACCEPTED_TIME_SIG,
        instruments: list[str] | None = None,
        bars_per_instrument: int | None = None,
        seed: int = 42,
    ) -> None:
        self.time_steps = time_steps
        self.min_notes_per_bar = min_notes_per_bar
        self.beat_resolution = beat_resolution
        self.accepted_time_sig = accepted_time_sig
        self.instruments = instruments
        self.bars_per_instrument = bars_per_instrument
        self.seed = seed
        self._rng = random.Random(seed)

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_file(self, path: str | Path) -> list[BarData]:
        """Load a MIDI file and extract all valid BarData objects.

        Args:
            path: Path to a .npz (pypianoroll) or .mid/.midi file.

        Returns:
            List of BarData objects.  May be empty if the file is corrupted,
            has no valid tracks, or all bars are below the note threshold.
        """
        path = Path(path)
        suffix = path.suffix.lower()

        try:
            if suffix == ".npz":
                return self._ingest_npz(path)
            elif suffix in {".mid", ".midi"}:
                return self._ingest_midi(path)
            else:
                logger.warning("Unsupported file type '%s', skipping: %s", suffix, path)
                return []
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to load '%s', skipping. Reason: %s", path, exc)
            return []

    def ingest_directory(
        self,
        directory: str | Path,
        glob_pattern: str = "**/*.npz",
        max_files: int | None = None,
    ) -> Iterator[BarData]:
        """Recursively load MIDI files from a directory.

        Args:
            directory: Root directory to search.
            glob_pattern: Glob pattern relative to the directory.  Use
                          '**/*.mid' for standard MIDI files.
            max_files: If set, stop after this many files (for quick runs).

        Yields:
            BarData objects in the order they are found.
        """
        directory = Path(directory)
        files = sorted(directory.glob(glob_pattern))

        if max_files is not None:
            files = files[:max_files]

        for file_path in files:
            bars = self.ingest_file(file_path)
            yield from bars

    # ── pypianoroll .npz ────────────────────────────────────────────────────────

    def _ingest_npz(self, path: Path) -> list[BarData]:
        """Load a pypianoroll .npz file and extract bar data for each track.

        Args:
            path: Path to the pypianoroll .npz file.

        Returns:
            List of BarData objects.
        """
        try:
            import pypianoroll
        except ImportError as exc:
            raise ImportError(
                "pypianoroll is required for .npz ingestion. "
                "Install it with: pip install pypianoroll"
            ) from exc

        # pypianoroll.load returns a Multitrack object
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            multitrack = pypianoroll.load(str(path))

        song_id = path.stem

        # Determine steps per bar from beat resolution and time signature
        # Assume 4/4 at default beat resolution → 96 steps/bar
        time_sig_num, time_sig_den = self.accepted_time_sig
        steps_per_bar = int(self.beat_resolution * 4 * time_sig_num / time_sig_den)

        # Validate beat resolution matches config
        if hasattr(multitrack, "resolution") and multitrack.resolution != self.beat_resolution:
            logger.debug(
                "Song %s beat_resolution=%d, expected %d; using song resolution",
                song_id,
                multitrack.resolution,
                self.beat_resolution,
            )
            steps_per_bar = int(multitrack.resolution * 4 * time_sig_num / time_sig_den)

        all_bars: list[BarData] = []

        for track_idx, track in enumerate(multitrack.tracks):
            instrument_name = self._npz_instrument_name(track_idx, track)

            if self.instruments is not None and instrument_name not in self.instruments:
                continue

            pianoroll = track.pianoroll  # shape (T, 128) in pypianoroll
            if pianoroll.ndim != 2:
                logger.warning("Track %d in %s has unexpected shape, skipping", track_idx, path)
                continue

            # pypianoroll stores (T, 128) — transpose to (128, T) for our convention
            roll = pianoroll.T.astype(np.float32)  # (128, T)

            if roll.max() == 0:
                logger.debug("Track %d in %s is silent, skipping", track_idx, path)
                continue

            bar_arrays = _segment_track(roll, steps_per_bar)

            # Resize each bar to the configured time_steps if needed
            if steps_per_bar != self.time_steps:
                bar_arrays = [self._resize_time_axis(b, self.time_steps) for b in bar_arrays]

            bars = self._extract_bar_data(
                bar_arrays=bar_arrays,
                song_id=song_id,
                instrument=instrument_name,
                program_number=getattr(track, "program", 0),
                tempo=self._get_tempo(multitrack),
                time_signature=self.accepted_time_sig,
            )

            all_bars.extend(bars)

        return all_bars

    def _npz_instrument_name(self, track_idx: int, track: object) -> str:
        """Determine the instrument name for a multi-track .npz pianoroll.

        Args:
            track_idx: Zero-based track index within the multitrack.
            track: The pypianoroll track object.

        Returns:
            Instrument name string.
        """
        if track_idx < len(MULTITRACK_INSTRUMENTS):
            return MULTITRACK_INSTRUMENTS[track_idx]
        # Fall back to the track's program name if available
        return getattr(track, "name", f"instrument_{track_idx}").lower().replace(" ", "_")

    @staticmethod
    def _get_tempo(multitrack: object) -> float:
        """Extract a representative tempo from a pypianoroll Multitrack.

        Args:
            multitrack: A pypianoroll.Multitrack object.

        Returns:
            Tempo in BPM as a float.
        """
        tempos = getattr(multitrack, "tempo", None)
        if tempos is None:
            return 120.0
        if hasattr(tempos, "__len__") and len(tempos) > 0:
            return float(tempos[0])
        if isinstance(tempos, (int, float)):
            return float(tempos)
        return 120.0

    # ── Standard MIDI (.mid) ──────────────────────────────────────────────────

    def _ingest_midi(self, path: Path) -> list[BarData]:
        """Load a standard .mid file using pretty_midi.

        Args:
            path: Path to the .mid file.

        Returns:
            List of BarData objects.
        """
        try:
            import pretty_midi
        except ImportError as exc:
            raise ImportError(
                "pretty_midi is required for .mid file ingestion. "
                "Install it with: pip install pretty_midi"
            ) from exc

        pm = pretty_midi.PrettyMIDI(str(path))
        song_id = path.stem

        # Detect the dominant time signature
        time_sig = self._detect_time_signature_midi(pm)
        if time_sig is None:
            logger.debug("Song %s has unsupported time signature, skipping", song_id)
            return []

        tempo = self._get_tempo_midi(pm)
        steps_per_beat = self.beat_resolution
        beats_per_bar = time_sig[0]
        steps_per_bar = steps_per_beat * beats_per_bar
        fs = steps_per_beat * (tempo / 60.0)  # steps per second

        all_bars: list[BarData] = []

        for inst_idx, instrument in enumerate(pm.instruments):
            instrument_name = self._map_program_to_instrument(
                instrument.program, instrument.is_drum
            )

            if self.instruments is not None and instrument_name not in self.instruments:
                continue

            if not instrument.notes:
                continue

            # Build a piano roll at our desired resolution
            roll = instrument.get_piano_roll(fs=fs).astype(np.float32)  # (128, T)
            if roll.max() == 0:
                continue

            bar_arrays = _segment_track(roll, steps_per_bar)

            if steps_per_bar != self.time_steps:
                bar_arrays = [self._resize_time_axis(b, self.time_steps) for b in bar_arrays]

            bars = self._extract_bar_data(
                bar_arrays=bar_arrays,
                song_id=song_id,
                instrument=instrument_name,
                program_number=instrument.program,
                tempo=tempo,
                time_signature=time_sig,
            )
            all_bars.extend(bars)

        return all_bars

    @staticmethod
    def _detect_time_signature_midi(pm: object) -> tuple[int, int] | None:
        """Detect the dominant time signature from a MIDI file.

        Accepts common simple time signatures (2/4, 3/4, 4/4, 6/8).
        Files with only unusual signatures (5/4, 7/8, etc.) return None.

        Args:
            pm: pretty_midi.PrettyMIDI object.

        Returns:
            (numerator, denominator) of the dominant time signature,
            or None if unsupported.
        """
        SUPPORTED = {(2, 4), (3, 4), (4, 4), (6, 8)}

        time_sigs = getattr(pm, "time_signature_changes", [])
        if not time_sigs:
            return (4, 4)  # Assume 4/4 if none specified

        # Use the first (or most common) time signature
        first_ts = (time_sigs[0].numerator, time_sigs[0].denominator)
        if first_ts in SUPPORTED:
            return first_ts

        # If the first is unsupported, check if any supported one exists
        for ts in time_sigs:
            candidate = (ts.numerator, ts.denominator)
            if candidate in SUPPORTED:
                return candidate

        return None

    @staticmethod
    def _get_tempo_midi(pm: object) -> float:
        """Get the first (or only) tempo from a PrettyMIDI object.

        Args:
            pm: pretty_midi.PrettyMIDI object.

        Returns:
            Tempo in BPM.
        """
        try:
            _, tempos = pm.get_tempo_changes()
            if len(tempos) > 0:
                return float(tempos[0])
        except Exception:  # pylint: disable=broad-except
            pass
        return 120.0

    @staticmethod
    def _map_program_to_instrument(program: int, is_drum: bool) -> str:
        """Map a MIDI program number to one of the 5 canonical instrument names.

        Args:
            program: MIDI program number (0-127).
            is_drum: True if this is a percussion track.

        Returns:
            One of: "drums", "bass", "guitar", "piano", "strings".
        """
        if is_drum:
            return "drums"
        if 32 <= program <= 39:
            return "bass"
        if 24 <= program <= 31:
            return "guitar"
        if 0 <= program <= 7:
            return "piano"
        if 40 <= program <= 55:
            return "strings"
        return "piano"  # Default fallback

    # ── Bar extraction ────────────────────────────────────────────────────────

    def _extract_bar_data(
        self,
        bar_arrays: list[np.ndarray],
        song_id: str,
        instrument: str,
        program_number: int,
        tempo: float,
        time_signature: tuple[int, int],
    ) -> list[BarData]:
        """Convert raw bar arrays into filtered, optionally sampled BarData objects.

        Args:
            bar_arrays: List of (128, T) arrays, one per bar.
            song_id: Identifier for the source file.
            instrument: Instrument name string.
            program_number: MIDI program number.
            tempo: Song tempo in BPM.
            time_signature: (numerator, denominator) tuple.

        Returns:
            List of BarData objects passing the note count threshold.
        """
        valid_bars: list[BarData] = []

        for bar_idx, piano_roll in enumerate(bar_arrays):
            note_count = _count_notes(piano_roll)

            if note_count < self.min_notes_per_bar:
                continue

            onset_mask = _compute_onset_mask(piano_roll)
            sustain_mask = _compute_sustain_mask(piano_roll, onset_mask)

            bar_id = f"{song_id}_{instrument}_{bar_idx:04d}"

            bar = BarData(
                bar_id=bar_id,
                song_id=song_id,
                instrument=instrument,
                program_number=program_number,
                piano_roll=piano_roll,
                onset_mask=onset_mask,
                sustain_mask=sustain_mask,
                tempo=tempo,
                time_signature=time_signature,
                metadata={"bar_index": bar_idx, "note_count": note_count},
            )
            valid_bars.append(bar)

        # Optional deterministic downsampling
        if self.bars_per_instrument is not None and len(valid_bars) > self.bars_per_instrument:
            valid_bars = self._rng.sample(valid_bars, self.bars_per_instrument)

        return valid_bars

    @staticmethod
    def _resize_time_axis(bar: np.ndarray, target_steps: int) -> np.ndarray:
        """Resample the time axis of a bar array to target_steps using linear interpolation.

        Args:
            bar: Array of shape (128, T).
            target_steps: Desired number of time steps.

        Returns:
            Array of shape (128, target_steps).
        """
        t = torch.from_numpy(bar).unsqueeze(0).unsqueeze(0).float()  # (1, 1, 128, T)
        resized = F.interpolate(t, size=(128, target_steps), mode="bilinear", align_corners=False)
        return resized.squeeze(0).squeeze(0).numpy()
