"""Information-theoretic metrics for reconstruction quality.

Measures information content, entropy differences, mutual information,
KL divergence, and sparsity of reconstructed piano-roll images.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from midi_vae.data.types import BarData, ReconstructedBar
from midi_vae.metrics.base import Metric
from midi_vae.registry import ComponentRegistry


def _pitch_distribution_from_piano_roll(piano_roll: np.ndarray) -> np.ndarray:
    """Compute a normalised pitch probability distribution from a piano-roll matrix.

    Sums velocity activations over time for each pitch, then normalises.

    Args:
        piano_roll: Velocity matrix of shape (128, T). Zero = inactive.

    Returns:
        Float array of shape (128,) summing to 1. All-zeros if silent.
    """
    pitch_counts = (piano_roll > 0).sum(axis=1).astype(np.float64)  # (128,)
    total = pitch_counts.sum()
    if total > 0:
        pitch_counts /= total
    return pitch_counts


def _pitch_distribution_from_image(image: torch.Tensor) -> np.ndarray:
    """Compute a normalised pitch probability distribution from a reconstructed image.

    Uses the velocity channel (index 0), remapped to [0, 1]. Sums across time
    for each pitch row to get a per-pitch activation magnitude.

    Args:
        image: Reconstructed image tensor of shape (3, H, W) in [-1, 1].

    Returns:
        Float array of shape (H,) normalised to sum to 1.
    """
    vel = image[0].cpu().float()
    vel = ((vel + 1.0) / 2.0).clamp(0.0, 1.0)  # (H, W) in [0, 1]
    pitch_sums = vel.sum(dim=1).numpy()  # (H,)
    total = pitch_sums.sum()
    if total > 0:
        pitch_sums = pitch_sums / total
    return pitch_sums


def _shannon_entropy(distribution: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Shannon entropy H = -sum(p * log(p)) for a discrete distribution.

    Args:
        distribution: Non-negative array summing to (approximately) 1.
        eps: Small constant to avoid log(0).

    Returns:
        Entropy in nats as a non-negative float.
    """
    p = np.clip(distribution, eps, None)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


@ComponentRegistry.register('metric', 'note_entropy_diff')
class NoteEntropyDiff(Metric):
    """Difference in Shannon entropy of pitch distributions.

    Computes H_orig - H_recon where H is Shannon entropy of the pitch
    probability distribution. A value near zero means the reconstruction
    preserves the same pitch diversity as the original.

    The original distribution is derived from gt.piano_roll (128, T).
    The reconstructed distribution is derived from the velocity channel of
    recon.recon_image.

    Returns:
        {'note_entropy_diff': float}  — signed difference in nats.
        {'note_entropy_orig': float}  — entropy of original.
        {'note_entropy_recon': float} — entropy of reconstruction.
    """

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'note_entropy_diff'

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Compute Shannon entropy difference between original and reconstructed pitch distributions.

        Args:
            gt: Ground-truth bar with piano_roll of shape (128, T).
            recon: Reconstructed bar with recon_image of shape (3, H, W).

        Returns:
            Dict with keys 'note_entropy_diff', 'note_entropy_orig', 'note_entropy_recon'.
        """
        orig_dist = _pitch_distribution_from_piano_roll(gt.piano_roll)
        recon_dist = _pitch_distribution_from_image(recon.recon_image)

        h_orig = _shannon_entropy(orig_dist)
        h_recon = _shannon_entropy(recon_dist)

        return {
            'note_entropy_diff': h_orig - h_recon,
            'note_entropy_orig': h_orig,
            'note_entropy_recon': h_recon,
        }


def _marginal_pitch(piano_roll: np.ndarray, num_pitch_bins: int = 32) -> np.ndarray:
    """Compute a coarse marginal pitch distribution (binned).

    Args:
        piano_roll: Velocity matrix of shape (128, T).
        num_pitch_bins: Number of pitch bins. Default 32 (4 pitches per bin).

    Returns:
        Normalised float array of shape (num_pitch_bins,).
    """
    full = _pitch_distribution_from_piano_roll(piano_roll)  # (128,)
    binned = np.zeros(num_pitch_bins, dtype=np.float64)
    pitches_per_bin = 128 // num_pitch_bins
    for i in range(num_pitch_bins):
        binned[i] = full[i * pitches_per_bin:(i + 1) * pitches_per_bin].sum()
    total = binned.sum()
    if total > 0:
        binned /= total
    return binned


def _marginal_time(piano_roll: np.ndarray, num_time_bins: int = 32) -> np.ndarray:
    """Compute a coarse marginal time distribution (binned).

    Args:
        piano_roll: Velocity matrix of shape (128, T).
        num_time_bins: Number of time bins. Default 32.

    Returns:
        Normalised float array of shape (num_time_bins,).
    """
    T = piano_roll.shape[1]
    active_per_step = (piano_roll > 0).sum(axis=0).astype(np.float64)  # (T,)

    binned = np.zeros(num_time_bins, dtype=np.float64)
    steps_per_bin = max(1, T // num_time_bins)
    for i in range(num_time_bins):
        binned[i] = active_per_step[i * steps_per_bin:(i + 1) * steps_per_bin].sum()
    total = binned.sum()
    if total > 0:
        binned /= total
    return binned


def _joint_pitch_time(piano_roll: np.ndarray, num_pitch_bins: int = 16, num_time_bins: int = 16) -> np.ndarray:
    """Compute joint pitch-time distribution from a piano-roll matrix.

    Args:
        piano_roll: Velocity matrix of shape (128, T).
        num_pitch_bins: Number of pitch bins. Default 16.
        num_time_bins: Number of time bins. Default 16.

    Returns:
        Normalised float array of shape (num_pitch_bins, num_time_bins).
    """
    T = piano_roll.shape[1]
    pitches_per_bin = max(1, 128 // num_pitch_bins)
    steps_per_bin = max(1, T // num_time_bins)

    joint = np.zeros((num_pitch_bins, num_time_bins), dtype=np.float64)
    for pi in range(num_pitch_bins):
        p_start = pi * pitches_per_bin
        p_end = min(128, p_start + pitches_per_bin)
        for ti in range(num_time_bins):
            t_start = ti * steps_per_bin
            t_end = min(T, t_start + steps_per_bin)
            joint[pi, ti] = (piano_roll[p_start:p_end, t_start:t_end] > 0).sum()

    total = joint.sum()
    if total > 0:
        joint /= total
    return joint


@ComponentRegistry.register('metric', 'mutual_information_pitch_time')
class MutualInformationPitchTime(Metric):
    """Mutual information between pitch and time-step distributions in the piano roll.

    Computes I(Pitch; Time) = H(Pitch) + H(Time) - H(Pitch, Time) using
    binned marginal and joint distributions from the ground-truth piano roll.
    Also computes the same for the reconstruction and returns both.

    Returns:
        {'mi_pitch_time_orig': float}  — MI of original in nats.
        {'mi_pitch_time_recon': float} — MI of reconstruction in nats.
        {'mi_pitch_time_diff': float}  — orig minus recon.
    """

    def __init__(self, num_pitch_bins: int = 16, num_time_bins: int = 16) -> None:
        """Initialise MutualInformationPitchTime.

        Args:
            num_pitch_bins: Number of coarse pitch bins. Default 16.
            num_time_bins: Number of coarse time bins. Default 16.
        """
        self._num_pitch_bins = num_pitch_bins
        self._num_time_bins = num_time_bins

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'mutual_information_pitch_time'

    @staticmethod
    def _mi_from_piano_roll(
        piano_roll: np.ndarray,
        num_pitch_bins: int,
        num_time_bins: int,
    ) -> float:
        """Compute MI(Pitch; Time) from a piano-roll matrix.

        Args:
            piano_roll: Velocity matrix of shape (128, T).
            num_pitch_bins: Number of pitch bins for marginals.
            num_time_bins: Number of time bins for marginals.

        Returns:
            Mutual information in nats.
        """
        eps = 1e-12
        joint = _joint_pitch_time(piano_roll, num_pitch_bins, num_time_bins)

        if joint.sum() < eps:
            return 0.0

        # Marginals
        p_pitch = joint.sum(axis=1)  # (num_pitch_bins,)
        p_time = joint.sum(axis=0)   # (num_time_bins,)

        mi = 0.0
        for pi in range(num_pitch_bins):
            for ti in range(num_time_bins):
                p_ij = joint[pi, ti]
                if p_ij < eps:
                    continue
                denom = p_pitch[pi] * p_time[ti]
                if denom < eps:
                    continue
                mi += p_ij * np.log(p_ij / denom)
        return float(max(0.0, mi))

    @staticmethod
    def _mi_from_image(
        image: torch.Tensor,
        num_pitch_bins: int,
        num_time_bins: int,
    ) -> float:
        """Compute MI(Pitch; Time) from a reconstructed image.

        Args:
            image: Reconstructed image tensor of shape (3, H, W) in [-1, 1].
            num_pitch_bins: Number of pitch bins for marginals.
            num_time_bins: Number of time bins for marginals.

        Returns:
            Mutual information in nats.
        """
        eps = 1e-12
        vel = image[0].cpu().float()
        vel = ((vel + 1.0) / 2.0).clamp(0.0, 1.0).numpy()  # (H, W)
        H, W = vel.shape

        h_per_bin = max(1, H // num_pitch_bins)
        w_per_bin = max(1, W // num_time_bins)

        joint = np.zeros((num_pitch_bins, num_time_bins), dtype=np.float64)
        for pi in range(num_pitch_bins):
            r_start = pi * h_per_bin
            r_end = min(H, r_start + h_per_bin)
            for ti in range(num_time_bins):
                c_start = ti * w_per_bin
                c_end = min(W, c_start + w_per_bin)
                joint[pi, ti] = vel[r_start:r_end, c_start:c_end].sum()

        total = joint.sum()
        if total < eps:
            return 0.0
        joint /= total

        p_pitch = joint.sum(axis=1)
        p_time = joint.sum(axis=0)

        mi = 0.0
        for pi in range(num_pitch_bins):
            for ti in range(num_time_bins):
                p_ij = joint[pi, ti]
                if p_ij < eps:
                    continue
                denom = p_pitch[pi] * p_time[ti]
                if denom < eps:
                    continue
                mi += p_ij * np.log(p_ij / denom)
        return float(max(0.0, mi))

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Compute mutual information for both original and reconstructed piano rolls.

        Args:
            gt: Ground-truth bar with piano_roll of shape (128, T).
            recon: Reconstructed bar with recon_image of shape (3, H, W).

        Returns:
            Dict with keys 'mi_pitch_time_orig', 'mi_pitch_time_recon', 'mi_pitch_time_diff'.
        """
        mi_orig = self._mi_from_piano_roll(
            gt.piano_roll, self._num_pitch_bins, self._num_time_bins
        )
        mi_recon = self._mi_from_image(
            recon.recon_image, self._num_pitch_bins, self._num_time_bins
        )
        return {
            'mi_pitch_time_orig': mi_orig,
            'mi_pitch_time_recon': mi_recon,
            'mi_pitch_time_diff': mi_orig - mi_recon,
        }


@ComponentRegistry.register('metric', 'kl_divergence_pitch_dist')
class KLDivergencePitchDist(Metric):
    """KL divergence between original and reconstructed pitch distributions.

    Computes KL(P_orig || P_recon) over pitch histograms derived from
    the ground-truth piano roll (128 bins) and the reconstructed image
    velocity channel (H bins, linearly interpolated to 128 if needed).

    Lower values mean the reconstruction preserves the original pitch
    distribution more faithfully.

    Returns:
        {'kl_pitch_dist': float}  — non-negative, lower is better.
    """

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'kl_divergence_pitch_dist'

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Compute KL(P_orig || P_recon) over pitch distributions.

        Args:
            gt: Ground-truth bar with piano_roll of shape (128, T).
            recon: Reconstructed bar with recon_image of shape (3, H, W).

        Returns:
            Dict with key 'kl_pitch_dist'.
        """
        eps = 1e-10

        orig_dist = _pitch_distribution_from_piano_roll(gt.piano_roll)  # (128,)

        # Get reconstruction pitch distribution (H bins)
        raw_recon = _pitch_distribution_from_image(recon.recon_image)  # (H,)

        # Resample reconstruction distribution to 128 bins if needed
        H = len(raw_recon)
        if H != 128:
            recon_tensor = torch.tensor(raw_recon, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            recon_tensor = F.interpolate(
                recon_tensor, size=128, mode='linear', align_corners=False
            ).squeeze()
            recon_dist = recon_tensor.numpy().astype(np.float64)
            total = recon_dist.sum()
            if total > 0:
                recon_dist /= total
        else:
            recon_dist = raw_recon

        # If both are silent, KL = 0
        if orig_dist.sum() < eps:
            return {'kl_pitch_dist': 0.0}

        # If reconstruction is silent but original is not, large divergence
        if recon_dist.sum() < eps:
            return {'kl_pitch_dist': float('inf')}

        p = np.clip(orig_dist, eps, None)
        q = np.clip(recon_dist, eps, None)
        p = p / p.sum()
        q = q / q.sum()

        kl = float(np.sum(p * np.log(p / q)))
        return {'kl_pitch_dist': max(0.0, kl)}


@ComponentRegistry.register('metric', 'activation_sparsity')
class ActivationSparsity(Metric):
    """Fraction of near-zero activations in the reconstructed image.

    Measures how sparse the reconstruction output is by counting the fraction
    of pixels in the velocity channel whose value (remapped to [0, 1]) is
    below a near-zero threshold. Higher sparsity means fewer predicted notes.

    Also computes the ground-truth sparsity from the piano roll for comparison.

    Parameters (constructor args):
        near_zero_threshold (float): Pixels below this value (in [0, 1] space)
            count as inactive. Default 0.05.

    Returns:
        {'activation_sparsity_recon': float}  — fraction in [0, 1].
        {'activation_sparsity_gt': float}     — ground-truth sparsity.
        {'activation_sparsity_diff': float}   — recon minus gt.
    """

    def __init__(self, near_zero_threshold: float = 0.05) -> None:
        """Initialise ActivationSparsity.

        Args:
            near_zero_threshold: Activation below this value is considered inactive.
        """
        self._threshold = near_zero_threshold

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'activation_sparsity'

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Compute activation sparsity of the reconstructed image.

        Args:
            gt: Ground-truth bar with piano_roll of shape (128, T).
            recon: Reconstructed bar with recon_image of shape (3, H, W).

        Returns:
            Dict with keys 'activation_sparsity_recon', 'activation_sparsity_gt',
            'activation_sparsity_diff'.
        """
        # Reconstruction sparsity: fraction of velocity-channel pixels near zero
        vel = recon.recon_image[0].cpu().float()
        vel_01 = ((vel + 1.0) / 2.0).clamp(0.0, 1.0)
        total_pixels = vel_01.numel()
        inactive = int((vel_01 < self._threshold).sum().item())
        sparsity_recon = inactive / total_pixels if total_pixels > 0 else 0.0

        # Ground-truth sparsity: fraction of piano-roll cells that are zero
        total_cells = gt.piano_roll.size
        inactive_gt = int((gt.piano_roll == 0).sum())
        sparsity_gt = inactive_gt / total_cells if total_cells > 0 else 0.0

        return {
            'activation_sparsity_recon': sparsity_recon,
            'activation_sparsity_gt': sparsity_gt,
            'activation_sparsity_diff': sparsity_recon - sparsity_gt,
        }
