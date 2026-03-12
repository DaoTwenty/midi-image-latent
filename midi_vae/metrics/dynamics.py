"""Dynamics evaluation metrics: velocity distribution comparisons.

Compares velocity profiles between ground-truth MIDI notes and detected
notes to assess how well the reconstruction preserves dynamics.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

from midi_vae.data.types import BarData, ReconstructedBar, MidiNote
from midi_vae.metrics.base import Metric
from midi_vae.registry import ComponentRegistry

_NUM_VEL_BINS: int = 32  # Histogram resolution for velocity distributions


def _velocity_profile(notes: list[MidiNote], num_steps: int) -> np.ndarray:
    """Build a per-step mean velocity profile from a list of notes.

    At each time step the mean velocity of all active notes is computed.
    Steps with no active notes have velocity 0.

    Args:
        notes: List of MidiNote events.
        num_steps: Total number of time steps in the bar.

    Returns:
        Float array of shape (num_steps,) with mean velocity per step.
    """
    vel_sum = np.zeros(num_steps, dtype=np.float64)
    count = np.zeros(num_steps, dtype=np.float64)
    for note in notes:
        start = max(0, note.onset_step)
        end = min(num_steps, note.offset_step)
        if end > start:
            vel_sum[start:end] += note.velocity
            count[start:end] += 1.0
    nonzero = count > 0
    result = np.zeros(num_steps, dtype=np.float64)
    result[nonzero] = vel_sum[nonzero] / count[nonzero]
    return result


def _gt_velocity_profile(piano_roll: np.ndarray) -> np.ndarray:
    """Build a per-step mean velocity profile from a (128, T) piano-roll matrix.

    Args:
        piano_roll: Velocity matrix of shape (128, T). Zero = inactive.

    Returns:
        Float array of shape (T,) with mean velocity per step (0 if silent).
    """
    T = piano_roll.shape[1]
    active = piano_roll > 0  # (128, T)
    vel_sum = piano_roll.astype(np.float64).sum(axis=0)  # (T,)
    count = active.sum(axis=0).astype(np.float64)        # (T,)
    result = np.zeros(T, dtype=np.float64)
    nonzero = count > 0
    result[nonzero] = vel_sum[nonzero] / count[nonzero]
    return result


def _velocity_histogram(
    notes: list[MidiNote],
    num_bins: int = _NUM_VEL_BINS,
) -> np.ndarray:
    """Build a normalised velocity histogram from a list of notes.

    Args:
        notes: List of MidiNote events.
        num_bins: Number of histogram bins covering [0, 127]. Default 32.

    Returns:
        Float array of shape (num_bins,) summing to 1 (or all zeros if empty).
    """
    if not notes:
        return np.zeros(num_bins, dtype=np.float64)
    velocities = np.array([n.velocity for n in notes], dtype=np.float64)
    hist, _ = np.histogram(velocities, bins=num_bins, range=(0, 128))
    hist = hist.astype(np.float64)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def _gt_velocity_histogram(
    piano_roll: np.ndarray,
    num_bins: int = _NUM_VEL_BINS,
) -> np.ndarray:
    """Build a normalised velocity histogram from a (128, T) piano-roll.

    Args:
        piano_roll: Velocity matrix of shape (128, T).
        num_bins: Number of histogram bins. Default 32.

    Returns:
        Float array of shape (num_bins,) summing to 1.
    """
    velocities = piano_roll[piano_roll > 0].astype(np.float64)
    if len(velocities) == 0:
        return np.zeros(num_bins, dtype=np.float64)
    hist, _ = np.histogram(velocities, bins=num_bins, range=(0, 128))
    hist = hist.astype(np.float64)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Compute KL divergence D_KL(p || q) for discrete distributions.

    Args:
        p: Reference distribution, shape (N,). Should sum to 1.
        q: Approximation distribution, shape (N,). Should sum to 1.
        eps: Small constant to avoid log(0). Default 1e-10.

    Returns:
        Non-negative float representing KL divergence in nats.
    """
    p_safe = np.clip(p, eps, None)
    q_safe = np.clip(q, eps, None)
    # Re-normalise after clipping to avoid slight distortions
    p_safe = p_safe / p_safe.sum()
    q_safe = q_safe / q_safe.sum()
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


@ComponentRegistry.register('metric', 'velocity_mse')
class VelocityMSE(Metric):
    """Mean squared error between ground-truth and predicted velocity profiles.

    Computes per-step mean velocity for both ground truth (from piano_roll)
    and detected notes, then returns their MSE across time steps.

    Returns:
        {'velocity_mse': float}  — non-negative, lower is better.
    """

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'velocity_mse'

    @property
    def requires_notes(self) -> bool:
        """This metric requires detected notes."""
        return True

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Compute MSE of velocity profiles.

        Args:
            gt: Ground-truth bar with piano_roll of shape (128, T).
            recon: Reconstructed bar with detected_notes.

        Returns:
            Dict with key 'velocity_mse'.
        """
        num_steps = gt.piano_roll.shape[1]
        gt_prof = _gt_velocity_profile(gt.piano_roll)
        pred_prof = _velocity_profile(recon.detected_notes, num_steps)
        mse = float(np.mean((gt_prof - pred_prof) ** 2))
        return {'velocity_mse': mse}


@ComponentRegistry.register('metric', 'velocity_correlation')
class VelocityCorrelation(Metric):
    """Pearson correlation between ground-truth and predicted velocity profiles.

    Computes per-step mean velocity for both ground truth and detected notes,
    then returns the Pearson r of the two time series.

    Returns:
        {'velocity_correlation': float}  in [-1, 1].
        Returns 0.0 when one or both profiles are constant.
    """

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'velocity_correlation'

    @property
    def requires_notes(self) -> bool:
        """This metric requires detected notes."""
        return True

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Compute Pearson correlation of velocity profiles.

        Args:
            gt: Ground-truth bar with piano_roll of shape (128, T).
            recon: Reconstructed bar with detected_notes.

        Returns:
            Dict with key 'velocity_correlation'.
        """
        num_steps = gt.piano_roll.shape[1]
        gt_prof = _gt_velocity_profile(gt.piano_roll)
        pred_prof = _velocity_profile(recon.detected_notes, num_steps)

        if gt_prof.std() < 1e-8 or pred_prof.std() < 1e-8:
            return {'velocity_correlation': 0.0}

        r, _ = pearsonr(gt_prof, pred_prof)
        if np.isnan(r):
            r = 0.0
        return {'velocity_correlation': float(r)}


@ComponentRegistry.register('metric', 'velocity_histogram_kl')
class VelocityHistogramKL(Metric):
    """KL divergence between ground-truth and predicted velocity histograms.

    Computes 32-bin velocity histograms for both ground truth and detected
    notes, then returns D_KL(gt_hist || pred_hist).

    Returns:
        {'velocity_histogram_kl': float}  — non-negative, lower is better.
        Returns 0.0 when both histograms are identical.
    """

    def __init__(self, num_bins: int = _NUM_VEL_BINS) -> None:
        """Initialise VelocityHistogramKL.

        Args:
            num_bins: Number of histogram bins covering velocity [0, 128).
                Default 32.
        """
        self._num_bins = num_bins

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'velocity_histogram_kl'

    @property
    def requires_notes(self) -> bool:
        """This metric requires detected notes."""
        return True

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Compute KL divergence of velocity histograms.

        Args:
            gt: Ground-truth bar with piano_roll of shape (128, T).
            recon: Reconstructed bar with detected_notes.

        Returns:
            Dict with key 'velocity_histogram_kl'.
        """
        gt_hist = _gt_velocity_histogram(gt.piano_roll, self._num_bins)
        pred_hist = _velocity_histogram(recon.detected_notes, self._num_bins)

        # If ground truth has no notes, KL is undefined; return 0
        if gt_hist.sum() < 1e-10:
            return {'velocity_histogram_kl': 0.0}

        # If prediction has no notes but GT does, return large value
        if pred_hist.sum() < 1e-10:
            return {'velocity_histogram_kl': float('inf')}

        kl = _kl_divergence(gt_hist, pred_hist)
        return {'velocity_histogram_kl': kl}
