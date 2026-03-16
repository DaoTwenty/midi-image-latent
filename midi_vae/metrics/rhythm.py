"""Rhythm evaluation metrics: IOI distribution and groove consistency.

Compares rhythmic structure between ground-truth MIDI notes and detected
notes using inter-onset-interval (IOI) distributions and autocorrelation-
based groove consistency.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

from midi_vae.data.types import BarData, ReconstructedBar, MidiNote
from midi_vae.metrics.base import Metric
from midi_vae.registry import ComponentRegistry

_MAX_IOI_BINS: int = 32  # Max IOI bin count for histogram


def _onset_times(notes: list[MidiNote]) -> np.ndarray:
    """Extract sorted unique onset step indices from a list of notes.

    Args:
        notes: List of MidiNote events.

    Returns:
        Sorted integer array of onset steps. Empty if no notes.
    """
    if not notes:
        return np.array([], dtype=np.int64)
    return np.sort(np.unique([n.onset_step for n in notes]).astype(np.int64))


def _gt_onset_times(onset_mask: np.ndarray) -> np.ndarray:
    """Extract onset step indices from a (128, T) onset mask.

    Any time step with at least one pitch onset is included.

    Args:
        onset_mask: Binary array of shape (128, T).

    Returns:
        Sorted integer array of onset steps.
    """
    any_onset = onset_mask.any(axis=0)  # (T,)
    return np.where(any_onset)[0].astype(np.int64)


def _ioi_histogram(
    onset_steps: np.ndarray,
    num_bins: int = _MAX_IOI_BINS,
    max_ioi: int | None = None,
) -> np.ndarray:
    """Compute a normalised IOI histogram from onset step indices.

    Args:
        onset_steps: Sorted array of onset step indices.
        num_bins: Number of histogram bins. Default 32.
        max_ioi: Maximum IOI value to include; bins cover [1, max_ioi].
            If None, uses the observed maximum (or 1 if fewer than 2 onsets).

    Returns:
        Float array of shape (num_bins,) normalised to sum to 1.
        All-zero if fewer than 2 onset steps.
    """
    hist = np.zeros(num_bins, dtype=np.float64)
    if len(onset_steps) < 2:
        return hist

    iois = np.diff(onset_steps).astype(np.float64)
    iois = iois[iois > 0]  # Discard zero-length gaps

    if len(iois) == 0:
        return hist

    if max_ioi is None:
        max_ioi = int(iois.max())

    max_ioi = max(max_ioi, 1)
    raw_hist, _ = np.histogram(iois, bins=num_bins, range=(1, max_ioi + 1))
    hist = raw_hist.astype(np.float64)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Compute KL divergence D_KL(p || q).

    Args:
        p: Reference distribution, shape (N,).
        q: Approximation distribution, shape (N,).
        eps: Epsilon for numerical stability.

    Returns:
        Non-negative float.
    """
    p_safe = np.clip(p, eps, None)
    q_safe = np.clip(q, eps, None)
    p_safe /= p_safe.sum()
    q_safe /= q_safe.sum()
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


@ComponentRegistry.register('metric', 'ioi_distribution_kl')
class IOIDistributionKL(Metric):
    """KL divergence between ground-truth and predicted IOI distributions.

    Computes inter-onset-interval (IOI) histograms from the ground-truth
    onset_mask and from detected note onset times, then returns
    D_KL(gt_ioi || pred_ioi).

    Returns:
        {'ioi_distribution_kl': float}  — non-negative, lower is better.
    """

    def __init__(self, num_bins: int = _MAX_IOI_BINS) -> None:
        """Initialise IOIDistributionKL.

        Args:
            num_bins: Number of histogram bins for IOI distributions. Default 32.
        """
        self._num_bins = num_bins

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'ioi_distribution_kl'

    @property
    def requires_notes(self) -> bool:
        """This metric requires detected notes."""
        return True

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
        gt_image: "PianoRollImage | None" = None,
    ) -> dict[str, float]:
        """Compute KL divergence of IOI distributions.

        Args:
            gt: Ground-truth bar with onset_mask of shape (128, T).
            recon: Reconstructed bar with detected_notes.

        Returns:
            Dict with key 'ioi_distribution_kl'.
        """
        gt_onsets = _gt_onset_times(gt.onset_mask)
        pred_onsets = _onset_times(recon.detected_notes)

        # Compute max IOI from both for a shared range
        all_iois: list[float] = []
        if len(gt_onsets) >= 2:
            all_iois.extend(np.diff(gt_onsets).tolist())
        if len(pred_onsets) >= 2:
            all_iois.extend(np.diff(pred_onsets).tolist())

        if not all_iois:
            return {'ioi_distribution_kl': 0.0}

        max_ioi = int(max(all_iois)) if all_iois else 1

        gt_hist = _ioi_histogram(gt_onsets, self._num_bins, max_ioi)
        pred_hist = _ioi_histogram(pred_onsets, self._num_bins, max_ioi)

        if gt_hist.sum() < 1e-10:
            return {'ioi_distribution_kl': 0.0}

        if pred_hist.sum() < 1e-10:
            return {'ioi_distribution_kl': float('inf')}

        return {'ioi_distribution_kl': _kl_divergence(gt_hist, pred_hist)}


def _groove_autocorrelation(
    onset_indicator: np.ndarray,
    max_lag: int | None = None,
) -> np.ndarray:
    """Compute normalised autocorrelation of a binary onset indicator.

    The autocorrelation captures the periodic structure of the rhythmic
    pattern. Peaks at lags corresponding to beat subdivisions indicate
    strong groove.

    Args:
        onset_indicator: 1-D binary array of shape (T,) with 1 at onset steps.
        max_lag: Maximum lag to compute. Defaults to T // 2.

    Returns:
        Float array of autocorrelation coefficients, shape (max_lag,).
        Normalised so that lag-0 = 1.0.
    """
    T = len(onset_indicator)
    if T == 0:
        return np.zeros(0)

    if max_lag is None:
        max_lag = max(T // 2, 1)

    x = onset_indicator.astype(np.float64)
    x_mean = x.mean()
    x_centered = x - x_mean
    var = np.dot(x_centered, x_centered)

    if var < 1e-10:
        return np.zeros(max_lag)

    acorr = np.zeros(max_lag, dtype=np.float64)
    for lag in range(max_lag):
        if lag == 0:
            acorr[0] = 1.0
        else:
            acorr[lag] = np.dot(x_centered[:T - lag], x_centered[lag:]) / var

    return acorr


@ComponentRegistry.register('metric', 'groove_consistency')
class GrooveConsistency(Metric):
    """Rhythmic groove consistency between ground truth and reconstruction.

    Computes the Pearson correlation between the autocorrelation functions
    of the ground-truth onset pattern and the predicted onset pattern.
    High correlation indicates the reconstruction preserves the rhythmic
    periodicity (groove) of the original.

    Returns:
        {'groove_consistency': float}  in [-1, 1], higher is better.
        Returns 0.0 when one or both autocorrelations are flat.
    """

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'groove_consistency'

    @property
    def requires_notes(self) -> bool:
        """This metric requires detected notes."""
        return True

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
        gt_image: "PianoRollImage | None" = None,
    ) -> dict[str, float]:
        """Compute groove consistency via autocorrelation correlation.

        Args:
            gt: Ground-truth bar with onset_mask of shape (128, T).
            recon: Reconstructed bar with detected_notes.

        Returns:
            Dict with key 'groove_consistency'.
        """
        T = gt.onset_mask.shape[1]

        gt_indicator = gt.onset_mask.any(axis=0).astype(np.float64)

        pred_indicator = np.zeros(T, dtype=np.float64)
        for note in recon.detected_notes:
            if 0 <= note.onset_step < T:
                pred_indicator[note.onset_step] = 1.0

        max_lag = max(T // 2, 1)
        gt_acorr = _groove_autocorrelation(gt_indicator, max_lag)
        pred_acorr = _groove_autocorrelation(pred_indicator, max_lag)

        # Skip lag-0 (always 1.0) to get meaningful correlation
        gt_tail = gt_acorr[1:]
        pred_tail = pred_acorr[1:]

        if len(gt_tail) == 0:
            return {'groove_consistency': 0.0}

        if gt_tail.std() < 1e-8 or pred_tail.std() < 1e-8:
            return {'groove_consistency': 0.0}

        r, _ = pearsonr(gt_tail, pred_tail)
        if np.isnan(r):
            r = 0.0
        return {'groove_consistency': float(r)}
