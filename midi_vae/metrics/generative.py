"""Generative quality metrics for evaluating sequence Transformer outputs.

These metrics evaluate properties of generated (not reconstructed) bar
sequences, measuring structural patterns, transition diversity, rhythmic
stability, and pitch class fidelity.

Metrics in this module operate primarily on sequences of BarData and
ReconstructedBar objects. Many are best used at the sequence/batch level;
per-bar ``compute()`` calls accumulate state that is finalised via
``finalize()``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from midi_vae.data.types import BarData, LatentEncoding, ReconstructedBar, MidiNote
from midi_vae.metrics.base import Metric
from midi_vae.registry import ComponentRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1]. Returns 0.0 for zero vectors.
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _pitch_class_histogram_from_notes(notes: list[MidiNote]) -> np.ndarray:
    """Build a 12-bin pitch class histogram from a list of MidiNote events.

    Each note contributes 1 count to its pitch class (pitch % 12).

    Args:
        notes: List of MidiNote events.

    Returns:
        Float array of shape (12,) normalised to sum to 1.
        All-zeros if the list is empty.
    """
    hist = np.zeros(12, dtype=np.float64)
    for note in notes:
        hist[note.pitch % 12] += 1.0
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _pitch_class_histogram_from_piano_roll(piano_roll: np.ndarray) -> np.ndarray:
    """Build a 12-bin pitch class histogram from a (128, T) piano-roll matrix.

    Each (pitch, step) cell with nonzero velocity contributes to pitch % 12.

    Args:
        piano_roll: Velocity matrix of shape (128, T).

    Returns:
        Float array of shape (12,) normalised to sum to 1.
    """
    hist = np.zeros(12, dtype=np.float64)
    for pitch in range(128):
        active = float((piano_roll[pitch, :] > 0).sum())
        if active > 0:
            hist[pitch % 12] += active
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Compute KL divergence D_KL(p || q).

    Args:
        p: Reference distribution of shape (N,). Must sum to ~1.
        q: Approximation distribution of shape (N,). Must sum to ~1.
        eps: Epsilon for numerical stability.

    Returns:
        Non-negative float. Returns 0.0 when both are empty.
    """
    p_safe = np.clip(p, eps, None)
    q_safe = np.clip(q, eps, None)
    p_safe = p_safe / p_safe.sum()
    q_safe = q_safe / q_safe.sum()
    return float(max(0.0, np.sum(p_safe * np.log(p_safe / q_safe))))


def _shannon_entropy(distribution: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Shannon entropy H = -sum(p * log(p)).

    Args:
        distribution: Non-negative array. Will be normalised internally.
        eps: Small constant to avoid log(0).

    Returns:
        Entropy in nats, non-negative float. Returns 0.0 if distribution sums to 0.
    """
    total = distribution.sum()
    if total < eps:
        return 0.0
    p = np.clip(distribution / total, eps, None)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def _onset_indicator(notes: list[MidiNote], num_steps: int) -> np.ndarray:
    """Build a binary onset indicator array of shape (num_steps,).

    Args:
        notes: List of MidiNote events.
        num_steps: Length of the time axis.

    Returns:
        Boolean array of shape (num_steps,) with True at each onset step.
    """
    arr = np.zeros(num_steps, dtype=bool)
    for note in notes:
        if 0 <= note.onset_step < num_steps:
            arr[note.onset_step] = True
    return arr


def _bar_feature_vector(gt: BarData, recon: ReconstructedBar) -> np.ndarray:
    """Compute a compact feature vector for a bar from its latent or image.

    Uses ``gt.metadata['latent'].z_mu`` if available; falls back to
    flattening the first channel of ``recon.recon_image``.

    Args:
        gt: Ground-truth bar (may contain 'latent' in metadata).
        recon: Reconstructed bar with recon_image of shape (3, H, W).

    Returns:
        1-D float64 numpy array.
    """
    latent = gt.metadata.get('latent', None)
    if latent is not None and hasattr(latent, 'z_mu'):
        return latent.z_mu.cpu().float().numpy().flatten()
    # Fallback: mean across spatial dims of each channel -> (C,)
    img = recon.recon_image.cpu().float()
    return img.mean(dim=[1, 2]).numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# SelfSimilarityMatrix
# ---------------------------------------------------------------------------


@ComponentRegistry.register('metric', 'generative/self_similarity_matrix')
class SelfSimilarityMatrix(Metric):
    """Self-similarity statistics of generated bar sequences.

    Measures structural repetition and variety by computing pairwise cosine
    similarity between bar feature vectors in a sequence. Reports mean and
    standard deviation of all off-diagonal similarities.

    High mean similarity indicates repetitive, structurally coherent
    sequences. Low mean with high std indicates varied sequences. The
    ideal balance depends on the musical context.

    Accumulates (feature_vector,) pairs via ``compute()``; call
    ``finalize()`` after all bars in the sequence are processed.

    Returns from ``compute()``:
        {'self_similarity_accumulated': 1.0}  — sentinel.

    Returns from ``finalize()``:
        dict with keys:
          - 'self_similarity_mean': float  — mean off-diagonal cosine similarity.
          - 'self_similarity_std': float   — std of off-diagonal cosine similarity.
          - 'self_similarity_n_bars': int  — number of bars accumulated.
    """

    def __init__(self) -> None:
        """Initialise SelfSimilarityMatrix."""
        self._features: list[np.ndarray] = []

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'generative/self_similarity_matrix'

    def reset(self) -> None:
        """Reset the accumulator for a new sequence."""
        self._features = []

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Accumulate one bar's feature vector.

        Args:
            gt: Ground-truth bar. May contain 'latent' in metadata.
            recon: Reconstructed bar with recon_image.

        Returns:
            Sentinel dict. Call finalize() for the true metric.
        """
        feat = _bar_feature_vector(gt, recon)
        self._features.append(feat)
        return {'self_similarity_accumulated': 1.0}

    def finalize(self) -> dict[str, float]:
        """Compute the self-similarity statistics across the accumulated sequence.

        Returns:
            Dict with keys 'self_similarity_mean', 'self_similarity_std',
            'self_similarity_n_bars'.
        """
        n = len(self._features)
        if n < 2:
            return {
                'self_similarity_mean': float('nan'),
                'self_similarity_std': float('nan'),
                'self_similarity_n_bars': n,
            }

        sims: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                sims.append(_cosine_similarity(self._features[i], self._features[j]))

        arr = np.array(sims, dtype=np.float64)
        return {
            'self_similarity_mean': float(arr.mean()),
            'self_similarity_std': float(arr.std()),
            'self_similarity_n_bars': n,
        }


# ---------------------------------------------------------------------------
# TransitionEntropy
# ---------------------------------------------------------------------------


def _dominant_pitch_class(notes: list[MidiNote]) -> int:
    """Return the most common pitch class (0-11) in a list of notes.

    Returns 0 for empty lists.

    Args:
        notes: List of MidiNote events.

    Returns:
        Integer in [0, 11].
    """
    if not notes:
        return 0
    hist = np.zeros(12, dtype=np.int32)
    for note in notes:
        hist[note.pitch % 12] += 1
    return int(hist.argmax())


def _dominant_pitch_class_from_piano_roll(piano_roll: np.ndarray) -> int:
    """Return the most common pitch class (0-11) from a (128, T) piano roll.

    Args:
        piano_roll: Velocity matrix of shape (128, T).

    Returns:
        Integer in [0, 11].
    """
    hist = np.zeros(12, dtype=np.float64)
    for pitch in range(128):
        hist[pitch % 12] += float((piano_roll[pitch, :] > 0).sum())
    return int(hist.argmax())


@ComponentRegistry.register('metric', 'generative/transition_entropy')
class TransitionEntropy(Metric):
    """Shannon entropy of pitch-class transition distribution across generated bars.

    Builds a 12x12 transition matrix where entry (i, j) counts how often the
    dominant pitch class transitions from i in bar t to j in bar t+1. Entropy
    of this transition matrix (flattened and normalised) measures harmonic
    diversity.

    Low entropy indicates predictable, repetitive harmonic motion.
    High entropy indicates varied but potentially incoherent harmonic motion.

    Accumulates dominant pitch classes via ``compute()``; call ``finalize()``
    for the transition entropy.

    Returns from ``compute()``:
        {'transition_entropy_accumulated': 1.0}  — sentinel.

    Returns from ``finalize()``:
        dict with keys:
          - 'transition_entropy': float  — Shannon entropy in nats.
          - 'transition_entropy_n_transitions': int  — number of transitions.
    """

    def __init__(self) -> None:
        """Initialise TransitionEntropy."""
        self._pitch_classes: list[int] = []

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'generative/transition_entropy'

    def reset(self) -> None:
        """Reset the accumulator for a new sequence."""
        self._pitch_classes = []

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Accumulate the dominant pitch class of this bar.

        Uses detected notes if available; falls back to gt.piano_roll.

        Args:
            gt: Ground-truth bar with piano_roll.
            recon: Reconstructed bar with detected_notes.

        Returns:
            Sentinel dict.
        """
        if recon.detected_notes:
            pc = _dominant_pitch_class(recon.detected_notes)
        else:
            pc = _dominant_pitch_class_from_piano_roll(gt.piano_roll)
        self._pitch_classes.append(pc)
        return {'transition_entropy_accumulated': 1.0}

    def finalize(self) -> dict[str, float]:
        """Compute the transition entropy across the accumulated sequence.

        Returns:
            Dict with keys 'transition_entropy' and
            'transition_entropy_n_transitions'.
        """
        n = len(self._pitch_classes)
        if n < 2:
            return {
                'transition_entropy': 0.0,
                'transition_entropy_n_transitions': 0,
            }

        transition_matrix = np.zeros((12, 12), dtype=np.float64)
        for i in range(n - 1):
            pc_from = self._pitch_classes[i]
            pc_to = self._pitch_classes[i + 1]
            transition_matrix[pc_from, pc_to] += 1.0

        # Compute entropy of the flat normalised distribution
        flat = transition_matrix.flatten()
        entropy = _shannon_entropy(flat)
        n_transitions = int(n - 1)

        return {
            'transition_entropy': entropy,
            'transition_entropy_n_transitions': n_transitions,
        }


# ---------------------------------------------------------------------------
# GrooveConsistency (sequence-level version)
# ---------------------------------------------------------------------------


@ComponentRegistry.register('metric', 'generative/groove_consistency')
class SequenceGrooveConsistency(Metric):
    """Rhythmic groove consistency across a generated bar sequence.

    Measures how consistent the rhythmic patterns are across bars by computing
    the mean pairwise cosine similarity of per-bar onset indicator vectors.
    A value near 1 means very consistent groove; near 0 means erratic rhythms.

    This complements ``midi_vae.metrics.rhythm.GrooveConsistency`` (which
    compares gt vs reconstruction for a single bar). This metric compares bars
    within a generated sequence against each other.

    Accumulates onset indicator vectors via ``compute()``; call ``finalize()``
    for the consistency score.

    Returns from ``compute()``:
        {'seq_groove_consistency_accumulated': 1.0}  — sentinel.

    Returns from ``finalize()``:
        dict with keys:
          - 'seq_groove_consistency': float  — mean pairwise cosine similarity in [-1, 1].
          - 'seq_groove_consistency_n_bars': int  — number of bars.
    """

    def __init__(self, num_steps: int = 96) -> None:
        """Initialise SequenceGrooveConsistency.

        Args:
            num_steps: Expected number of time steps per bar. Used when
                inferring the indicator length from notes (default 96).
        """
        self._num_steps = num_steps
        self._indicators: list[np.ndarray] = []

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'generative/groove_consistency'

    def reset(self) -> None:
        """Reset the accumulator for a new sequence."""
        self._indicators = []

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Accumulate the onset indicator for one bar.

        Uses detected notes if available; falls back to gt.onset_mask.

        Args:
            gt: Ground-truth bar with onset_mask of shape (128, T).
            recon: Reconstructed bar with detected_notes.

        Returns:
            Sentinel dict.
        """
        T = gt.onset_mask.shape[1]
        if recon.detected_notes:
            indicator = _onset_indicator(recon.detected_notes, T).astype(np.float64)
        else:
            indicator = gt.onset_mask.any(axis=0).astype(np.float64)
        self._indicators.append(indicator)
        return {'seq_groove_consistency_accumulated': 1.0}

    def finalize(self) -> dict[str, float]:
        """Compute mean pairwise groove consistency.

        Returns:
            Dict with keys 'seq_groove_consistency' and
            'seq_groove_consistency_n_bars'.
        """
        n = len(self._indicators)
        if n < 2:
            return {
                'seq_groove_consistency': float('nan'),
                'seq_groove_consistency_n_bars': n,
            }

        sims: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                sims.append(_cosine_similarity(self._indicators[i], self._indicators[j]))

        arr = np.array(sims, dtype=np.float64)
        return {
            'seq_groove_consistency': float(arr.mean()),
            'seq_groove_consistency_n_bars': n,
        }


# ---------------------------------------------------------------------------
# PitchClassHistogramKL
# ---------------------------------------------------------------------------


@ComponentRegistry.register('metric', 'generative/pitch_class_histogram_kl')
class PitchClassHistogramKL(Metric):
    """KL divergence between aggregate pitch class distributions of generated vs real bars.

    Accumulates pitch class histograms from ground-truth bars (``gt.piano_roll``)
    and from generated/reconstructed bars (``recon.detected_notes`` or
    ``recon.recon_image``). After calling ``finalize()``, returns
    D_KL(real || generated).

    Lower values mean the generated bars match the real pitch class
    distribution more closely.

    Returns from ``compute()``:
        {'pitch_class_kl_accumulated': 1.0}  — sentinel.

    Returns from ``finalize()``:
        dict with keys:
          - 'pitch_class_histogram_kl': float  — non-negative KL divergence.
    """

    def __init__(self) -> None:
        """Initialise PitchClassHistogramKL."""
        self._gt_hist: np.ndarray = np.zeros(12, dtype=np.float64)
        self._gen_hist: np.ndarray = np.zeros(12, dtype=np.float64)

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'generative/pitch_class_histogram_kl'

    def reset(self) -> None:
        """Reset the accumulated histograms."""
        self._gt_hist = np.zeros(12, dtype=np.float64)
        self._gen_hist = np.zeros(12, dtype=np.float64)

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Accumulate pitch class counts for one bar pair.

        Ground-truth counts come from gt.piano_roll.
        Generated counts come from recon.detected_notes if non-empty;
        otherwise the velocity channel of recon.recon_image is used.

        Args:
            gt: Ground-truth bar with piano_roll of shape (128, T).
            recon: Reconstructed bar with detected_notes and recon_image.

        Returns:
            Sentinel dict.
        """
        gt_bar_hist = _pitch_class_histogram_from_piano_roll(gt.piano_roll)
        self._gt_hist += gt_bar_hist

        if recon.detected_notes:
            gen_bar_hist = _pitch_class_histogram_from_notes(recon.detected_notes)
        else:
            # Derive from image velocity channel
            vel = recon.recon_image[0].cpu().float()
            vel_01 = ((vel + 1.0) / 2.0).clamp(0.0, 1.0)
            H = vel_01.shape[0]
            # Map H rows to 128 pitches then to 12 pitch classes
            # Use per-row mean activation as pitch strength
            row_means = vel_01.mean(dim=1).numpy()  # (H,)
            gen_bar_hist = np.zeros(12, dtype=np.float64)
            for h in range(H):
                pitch_approx = int(round(h * 127.0 / max(H - 1, 1)))
                gen_bar_hist[pitch_approx % 12] += float(row_means[h])
            total = gen_bar_hist.sum()
            if total > 0:
                gen_bar_hist /= total

        self._gen_hist += gen_bar_hist
        return {'pitch_class_kl_accumulated': 1.0}

    def finalize(self) -> dict[str, float]:
        """Compute KL(real || generated) on the accumulated histograms.

        Returns:
            Dict with key 'pitch_class_histogram_kl'.
        """
        gt_total = self._gt_hist.sum()
        gen_total = self._gen_hist.sum()

        if gt_total < 1e-10 and gen_total < 1e-10:
            return {'pitch_class_histogram_kl': 0.0}
        if gen_total < 1e-10:
            return {'pitch_class_histogram_kl': float('inf')}
        if gt_total < 1e-10:
            return {'pitch_class_histogram_kl': 0.0}

        p = self._gt_hist / gt_total
        q = self._gen_hist / gen_total

        return {'pitch_class_histogram_kl': _kl_divergence(p, q)}


# ---------------------------------------------------------------------------
# BarLevelNLL
# ---------------------------------------------------------------------------


@ComponentRegistry.register('metric', 'generative/bar_level_nll')
class BarLevelNLL(Metric):
    """Negative log-likelihood of generated bars under a Gaussian image model.

    Estimates the NLL of the reconstructed image under the assumption that
    each pixel follows a Gaussian distribution N(mu, sigma^2) derived from
    the ground-truth image. Lower NLL indicates the reconstruction is more
    plausible under the data distribution.

    As a per-bar metric, the reference mean and variance are taken from the
    ground-truth piano roll image channel (channel 0). This is a proxy for
    the true sequence Transformer NLL and does not require the model.

    Returns:
        {'bar_level_nll': float}  — mean negative log-likelihood per pixel.
    """

    def __init__(self, min_sigma: float = 0.1) -> None:
        """Initialise BarLevelNLL.

        Args:
            min_sigma: Minimum pixel standard deviation to avoid division by
                zero or degenerate Gaussians. Default 0.1.
        """
        self._min_sigma = min_sigma

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'generative/bar_level_nll'

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Compute per-pixel Gaussian NLL of the reconstructed image.

        Uses the ground-truth image stored in ``gt.metadata['gt_image']`` if
        available. Falls back to deriving per-pixel statistics from the
        ground-truth piano roll projected to image space.

        Args:
            gt: Ground-truth bar. May contain 'gt_image' (PianoRollImage) in metadata.
            recon: Reconstructed bar with recon_image of shape (3, H, W).

        Returns:
            Dict with key 'bar_level_nll'. Returns NaN if no reference available.
        """
        gt_image_obj = gt.metadata.get('gt_image', None)
        if gt_image_obj is not None and hasattr(gt_image_obj, 'image'):
            gt_pixels = gt_image_obj.image[0].cpu().float()  # (H, W) velocity channel
        else:
            # Fallback: cannot compute NLL without a reference image
            return {'bar_level_nll': float('nan')}

        pred_pixels = recon.recon_image[0].cpu().float()  # (H, W)

        if gt_pixels.shape != pred_pixels.shape:
            pred_pixels = F.interpolate(
                pred_pixels.unsqueeze(0).unsqueeze(0),
                size=gt_pixels.shape,
                mode='bilinear',
                align_corners=False,
            ).squeeze()

        # Remap both to [0, 1]
        gt_01 = ((gt_pixels + 1.0) / 2.0).clamp(0.0, 1.0)
        pred_01 = ((pred_pixels + 1.0) / 2.0).clamp(0.0, 1.0)

        # Estimate per-pixel mean and variance from gt (batch mean/std over spatial dims)
        mu = gt_01.mean()
        sigma = max(float(gt_01.std().item()), self._min_sigma)

        # Gaussian NLL: 0.5 * log(2*pi*sigma^2) + 0.5 * (x-mu)^2 / sigma^2
        diff_sq = ((pred_01 - mu) ** 2).mean().item()
        nll = 0.5 * math.log(2.0 * math.pi * sigma ** 2) + 0.5 * diff_sq / (sigma ** 2)
        return {'bar_level_nll': float(nll)}


# ---------------------------------------------------------------------------
# SequenceCoherence  (bonus utility metric)
# ---------------------------------------------------------------------------


@ComponentRegistry.register('metric', 'generative/sequence_coherence')
class SequenceCoherence(Metric):
    """Coherence of a generated sequence measured by pitch-class entropy per bar.

    Computes the average Shannon entropy of per-bar pitch class histograms.
    A coherent (tonally focused) bar has low entropy; a chromatic/atonal bar
    has high entropy. This is reported both for ground-truth and generated bars.

    Returns from ``compute()``:
        dict with keys:
          - 'seq_coherence_gt_entropy': float    — entropy of gt pitch classes.
          - 'seq_coherence_gen_entropy': float   — entropy of generated pitch classes.
          - 'seq_coherence_entropy_diff': float  — gt minus generated.
    """

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'generative/sequence_coherence'

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Compute per-bar pitch class entropy for gt and generated bars.

        Args:
            gt: Ground-truth bar with piano_roll of shape (128, T).
            recon: Reconstructed bar with detected_notes.

        Returns:
            Dict with 'seq_coherence_gt_entropy', 'seq_coherence_gen_entropy',
            'seq_coherence_entropy_diff'.
        """
        gt_hist = _pitch_class_histogram_from_piano_roll(gt.piano_roll)
        gt_entropy = _shannon_entropy(gt_hist)

        if recon.detected_notes:
            gen_hist = _pitch_class_histogram_from_notes(recon.detected_notes)
        else:
            gen_hist = np.ones(12, dtype=np.float64) / 12.0  # Uniform (maximum entropy)
        gen_entropy = _shannon_entropy(gen_hist)

        return {
            'seq_coherence_gt_entropy': gt_entropy,
            'seq_coherence_gen_entropy': gen_entropy,
            'seq_coherence_entropy_diff': gt_entropy - gen_entropy,
        }
