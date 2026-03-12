"""Harmony and note-level evaluation metrics.

Includes OnsetF1, OnsetPrecision, OnsetRecall (onset detection quality),
NoteDensityPearson (correlation of per-step note counts),
PitchClassHistogramCorrelation, and IntervalHistogramCorrelation.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

from midi_vae.data.types import BarData, ReconstructedBar, MidiNote
from midi_vae.metrics.base import Metric
from midi_vae.registry import ComponentRegistry


def _onset_positions_from_notes(notes: list[MidiNote], num_steps: int) -> np.ndarray:
    """Build a binary onset indicator array from a list of MidiNote objects.

    Args:
        notes: List of detected or ground-truth MidiNote events.
        num_steps: Length of the time axis.

    Returns:
        Boolean array of shape (num_steps,) with True at each onset step.
    """
    arr = np.zeros(num_steps, dtype=bool)
    for note in notes:
        if 0 <= note.onset_step < num_steps:
            arr[note.onset_step] = True
    return arr


def _onset_positions_from_mask(onset_mask: np.ndarray) -> np.ndarray:
    """Collapse a (128, T) onset mask to a (T,) presence indicator.

    A time step is considered an onset if any pitch has an onset there.

    Args:
        onset_mask: Binary array of shape (128, T).

    Returns:
        Boolean array of shape (T,).
    """
    return onset_mask.any(axis=0)


def _match_onsets(
    gt_onsets: np.ndarray,
    pred_onsets: np.ndarray,
    tolerance: int,
) -> tuple[int, int, int]:
    """Match predicted onset steps to ground-truth onset steps with a tolerance window.

    Each ground-truth onset can be matched at most once.
    Each predicted onset can be matched at most once.

    Args:
        gt_onsets: Sorted array of ground-truth onset step indices.
        pred_onsets: Sorted array of predicted onset step indices.
        tolerance: Maximum allowed step difference for a match.

    Returns:
        Tuple of (true_positives, false_positives, false_negatives).
    """
    gt_matched = np.zeros(len(gt_onsets), dtype=bool)
    pred_matched = np.zeros(len(pred_onsets), dtype=bool)

    # Greedy matching: for each predicted onset, find the nearest unmatched GT onset
    for p_idx, p_step in enumerate(pred_onsets):
        best_gt_idx = -1
        best_dist = tolerance + 1
        for g_idx, g_step in enumerate(gt_onsets):
            if gt_matched[g_idx]:
                continue
            dist = abs(int(p_step) - int(g_step))
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_gt_idx = g_idx
        if best_gt_idx >= 0:
            gt_matched[best_gt_idx] = True
            pred_matched[p_idx] = True

    tp = int(pred_matched.sum())
    fp = int((~pred_matched).sum())
    fn = int((~gt_matched).sum())
    return tp, fp, fn


def _compute_onset_metrics(
    gt: BarData,
    recon: ReconstructedBar,
    tolerance: int,
) -> dict[str, float]:
    """Shared computation for onset precision, recall, and F1.

    Ground-truth onsets are taken from ``gt.onset_mask``.
    Predicted onsets are taken from the onset_step of each detected note.

    Args:
        gt: Ground-truth bar.
        recon: Reconstructed bar with detected notes.
        tolerance: Allowed step tolerance for matching.

    Returns:
        Dict with keys 'onset_precision', 'onset_recall', 'onset_f1'.
    """
    num_steps = gt.piano_roll.shape[1]  # T

    # Ground-truth onset positions from the mask
    gt_onset_arr = _onset_positions_from_mask(gt.onset_mask)
    gt_onsets = np.where(gt_onset_arr)[0]

    # Predicted onset positions from detected notes
    pred_onset_arr = _onset_positions_from_notes(recon.detected_notes, num_steps)
    pred_onsets = np.where(pred_onset_arr)[0]

    if len(gt_onsets) == 0 and len(pred_onsets) == 0:
        return {'onset_precision': 1.0, 'onset_recall': 1.0, 'onset_f1': 1.0}

    if len(gt_onsets) == 0:
        return {'onset_precision': 0.0, 'onset_recall': 1.0, 'onset_f1': 0.0}

    if len(pred_onsets) == 0:
        return {'onset_precision': 1.0, 'onset_recall': 0.0, 'onset_f1': 0.0}

    tp, fp, fn = _match_onsets(gt_onsets, pred_onsets, tolerance)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        'onset_precision': precision,
        'onset_recall': recall,
        'onset_f1': f1,
    }


@ComponentRegistry.register('metric', 'onset_f1')
class OnsetF1(Metric):
    """F1 score for onset detection accuracy.

    Compares onset positions in detected notes against the ground-truth
    onset_mask with a configurable tolerance window.

    Parameters (constructor args):
        tolerance (int): Number of time steps within which a predicted onset
            counts as a true positive. Default 1.

    Returns:
        {'onset_f1': float, 'onset_precision': float, 'onset_recall': float}
    """

    def __init__(self, tolerance: int = 1) -> None:
        """Initialise OnsetF1 metric.

        Args:
            tolerance: Onset tolerance window in time steps (default 1).
        """
        self._tolerance = tolerance

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'onset_f1'

    @property
    def requires_notes(self) -> bool:
        """This metric requires detected notes."""
        return True

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Compute onset F1, precision, and recall.

        Args:
            gt: Ground-truth bar with onset_mask of shape (128, T).
            recon: Reconstructed bar with detected notes.

        Returns:
            Dict with keys 'onset_f1', 'onset_precision', 'onset_recall'.
        """
        return _compute_onset_metrics(gt, recon, self._tolerance)


@ComponentRegistry.register('metric', 'onset_precision')
class OnsetPrecision(Metric):
    """Precision for onset detection.

    Fraction of predicted onsets that match a ground-truth onset
    within the tolerance window.

    Parameters (constructor args):
        tolerance (int): Default 1 step.

    Returns:
        {'onset_precision': float}
    """

    def __init__(self, tolerance: int = 1) -> None:
        """Initialise OnsetPrecision metric.

        Args:
            tolerance: Onset tolerance window in time steps (default 1).
        """
        self._tolerance = tolerance

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'onset_precision'

    @property
    def requires_notes(self) -> bool:
        """This metric requires detected notes."""
        return True

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Compute onset precision.

        Args:
            gt: Ground-truth bar.
            recon: Reconstructed bar with detected notes.

        Returns:
            Dict with key 'onset_precision'.
        """
        result = _compute_onset_metrics(gt, recon, self._tolerance)
        return {'onset_precision': result['onset_precision']}


@ComponentRegistry.register('metric', 'onset_recall')
class OnsetRecall(Metric):
    """Recall for onset detection.

    Fraction of ground-truth onsets that are matched by a predicted onset
    within the tolerance window.

    Parameters (constructor args):
        tolerance (int): Default 1 step.

    Returns:
        {'onset_recall': float}
    """

    def __init__(self, tolerance: int = 1) -> None:
        """Initialise OnsetRecall metric.

        Args:
            tolerance: Onset tolerance window in time steps (default 1).
        """
        self._tolerance = tolerance

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'onset_recall'

    @property
    def requires_notes(self) -> bool:
        """This metric requires detected notes."""
        return True

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Compute onset recall.

        Args:
            gt: Ground-truth bar.
            recon: Reconstructed bar with detected notes.

        Returns:
            Dict with key 'onset_recall'.
        """
        result = _compute_onset_metrics(gt, recon, self._tolerance)
        return {'onset_recall': result['onset_recall']}


def _note_density_per_step(notes: list[MidiNote], num_steps: int) -> np.ndarray:
    """Count the number of notes active at each time step.

    A note is active at step t if onset_step <= t < offset_step.

    Args:
        notes: List of MidiNote events.
        num_steps: Total number of time steps.

    Returns:
        Integer array of shape (num_steps,) with the polyphony count per step.
    """
    density = np.zeros(num_steps, dtype=np.int32)
    for note in notes:
        start = max(0, note.onset_step)
        end = min(num_steps, note.offset_step)
        if end > start:
            density[start:end] += 1
    return density


def _gt_note_density_from_piano_roll(piano_roll: np.ndarray) -> np.ndarray:
    """Compute per-step note density from a piano-roll velocity matrix.

    Args:
        piano_roll: Velocity matrix of shape (128, T). Nonzero = active note.

    Returns:
        Integer array of shape (T,) counting active pitches per step.
    """
    return (piano_roll > 0).sum(axis=0).astype(np.int32)


@ComponentRegistry.register('metric', 'note_density_pearson')
class NoteDensityPearson(Metric):
    """Pearson correlation of note density (polyphony) per time step.

    Computes per-step note counts for both ground truth and reconstruction,
    then returns the Pearson r between the two sequences.

    Ground-truth density comes from the piano_roll field of BarData.
    Reconstructed density comes from detected_notes in ReconstructedBar.

    Returns:
        {'note_density_pearson': float}  — in [-1, 1].
        Returns 0.0 if one or both sequences are constant (Pearson undefined).
    """

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'note_density_pearson'

    @property
    def requires_notes(self) -> bool:
        """This metric requires detected notes."""
        return True

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Compute Pearson r of note density per time step.

        Args:
            gt: Ground-truth bar with piano_roll of shape (128, T).
            recon: Reconstructed bar with detected_notes.

        Returns:
            Dict with key 'note_density_pearson'.
        """
        num_steps = gt.piano_roll.shape[1]

        gt_density = _gt_note_density_from_piano_roll(gt.piano_roll).astype(float)
        pred_density = _note_density_per_step(recon.detected_notes, num_steps).astype(float)

        # Pearson r requires variance in both signals
        if gt_density.std() < 1e-8 or pred_density.std() < 1e-8:
            return {'note_density_pearson': 0.0}

        r, _ = pearsonr(gt_density, pred_density)
        # Guard against NaN from degenerate inputs
        if np.isnan(r):
            r = 0.0
        return {'note_density_pearson': float(r)}


def _pitch_class_histogram(notes: list[MidiNote]) -> np.ndarray:
    """Build a 12-bin pitch class (chroma) histogram from a list of notes.

    Each note contributes 1 count to its pitch class (pitch % 12).

    Args:
        notes: List of MidiNote events.

    Returns:
        Float array of shape (12,) summing to 1 (or all-zeros if empty).
    """
    hist = np.zeros(12, dtype=np.float64)
    for note in notes:
        hist[note.pitch % 12] += 1.0
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _gt_pitch_class_histogram(piano_roll: np.ndarray) -> np.ndarray:
    """Build a pitch class histogram from a (128, T) piano-roll matrix.

    Each (pitch, time) cell with nonzero velocity contributes to pitch % 12.

    Args:
        piano_roll: Velocity matrix of shape (128, T).

    Returns:
        Float array of shape (12,) normalised to sum to 1.
    """
    hist = np.zeros(12, dtype=np.float64)
    for pitch in range(128):
        active = (piano_roll[pitch, :] > 0).sum()
        if active > 0:
            hist[pitch % 12] += float(active)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


@ComponentRegistry.register('metric', 'pitch_class_histogram_correlation')
class PitchClassHistogramCorrelation(Metric):
    """Pearson correlation between ground-truth and predicted pitch class histograms.

    Computes 12-bin chroma histograms from the ground-truth piano_roll and
    the detected notes, then computes the Pearson r between the two histograms.

    Returns:
        {'pitch_class_histogram_correlation': float}  in [-1, 1].
        Returns 0.0 when histograms are flat (no variance).
    """

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'pitch_class_histogram_correlation'

    @property
    def requires_notes(self) -> bool:
        """This metric requires detected notes."""
        return True

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Compute pitch class histogram correlation.

        Args:
            gt: Ground-truth bar with piano_roll of shape (128, T).
            recon: Reconstructed bar with detected_notes.

        Returns:
            Dict with key 'pitch_class_histogram_correlation'.
        """
        gt_hist = _gt_pitch_class_histogram(gt.piano_roll)
        pred_hist = _pitch_class_histogram(recon.detected_notes)

        if gt_hist.std() < 1e-8 or pred_hist.std() < 1e-8:
            # Degenerate case: one histogram is flat
            return {'pitch_class_histogram_correlation': 0.0}

        r, _ = pearsonr(gt_hist, pred_hist)
        if np.isnan(r):
            r = 0.0
        return {'pitch_class_histogram_correlation': float(r)}


def _interval_histogram(notes: list[MidiNote], num_bins: int = 25) -> np.ndarray:
    """Build an interval histogram from a list of notes.

    Intervals are computed between all simultaneously active notes within
    each time step (i.e., polyphonic intervals). Interval values range
    from 0 (unison) to 127 semitones; bins cover [0, num_bins).

    Args:
        notes: List of MidiNote events.
        num_bins: Number of histogram bins (default 25 to cover 0-24 semitones).

    Returns:
        Float array of shape (num_bins,) normalised to sum to 1.
    """
    hist = np.zeros(num_bins, dtype=np.float64)
    if not notes:
        return hist

    # Find the time span
    max_step = max(n.offset_step for n in notes)
    if max_step == 0:
        return hist

    # Collect active pitches per step
    from collections import defaultdict
    active: dict[int, list[int]] = defaultdict(list)
    for note in notes:
        for t in range(note.onset_step, note.offset_step):
            active[t].append(note.pitch)

    count = 0.0
    for t, pitches in active.items():
        pitches_sorted = sorted(set(pitches))
        for i in range(len(pitches_sorted)):
            for j in range(i + 1, len(pitches_sorted)):
                interval = pitches_sorted[j] - pitches_sorted[i]
                bin_idx = min(interval, num_bins - 1)
                hist[bin_idx] += 1.0
                count += 1.0

    if count > 0:
        hist /= count
    return hist


def _gt_interval_histogram(piano_roll: np.ndarray, num_bins: int = 25) -> np.ndarray:
    """Build an interval histogram from a (128, T) piano-roll matrix.

    At each time step, collects all active pitches and accumulates their
    pairwise intervals.

    Args:
        piano_roll: Velocity matrix of shape (128, T).
        num_bins: Number of histogram bins.

    Returns:
        Float array of shape (num_bins,) normalised to sum to 1.
    """
    hist = np.zeros(num_bins, dtype=np.float64)
    T = piano_roll.shape[1]
    count = 0.0

    for t in range(T):
        active_pitches = np.where(piano_roll[:, t] > 0)[0].tolist()
        for i in range(len(active_pitches)):
            for j in range(i + 1, len(active_pitches)):
                interval = active_pitches[j] - active_pitches[i]
                bin_idx = min(interval, num_bins - 1)
                hist[bin_idx] += 1.0
                count += 1.0

    if count > 0:
        hist /= count
    return hist


@ComponentRegistry.register('metric', 'interval_histogram_correlation')
class IntervalHistogramCorrelation(Metric):
    """Pearson correlation between ground-truth and predicted interval histograms.

    Computes pairwise interval distributions from all simultaneously active
    notes in the ground truth (from piano_roll) and the detected notes, then
    returns the Pearson r between the two 25-bin histograms.

    Returns:
        {'interval_histogram_correlation': float}  in [-1, 1].
        Returns 0.0 for flat histograms (no polyphony or all unison).
    """

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'interval_histogram_correlation'

    @property
    def requires_notes(self) -> bool:
        """This metric requires detected notes."""
        return True

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Compute interval histogram correlation.

        Args:
            gt: Ground-truth bar with piano_roll of shape (128, T).
            recon: Reconstructed bar with detected_notes.

        Returns:
            Dict with key 'interval_histogram_correlation'.
        """
        gt_hist = _gt_interval_histogram(gt.piano_roll)
        pred_hist = _interval_histogram(recon.detected_notes)

        if gt_hist.std() < 1e-8 or pred_hist.std() < 1e-8:
            return {'interval_histogram_correlation': 0.0}

        r, _ = pearsonr(gt_hist, pred_hist)
        if np.isnan(r):
            r = 0.0
        return {'interval_histogram_correlation': float(r)}
