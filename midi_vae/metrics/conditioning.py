"""Conditioning quality metrics for evaluating conditioned generation (Experiment 4D).

These metrics measure how faithfully generated outputs honour conditioning
signals (instrument, key, tempo, etc.) and how smoothly the latent space
interpolates between conditions.

Like the generative metrics, many operate at the batch/sequence level and
follow the accumulate-then-finalize pattern.
"""

from __future__ import annotations

import warnings

import numpy as np

from midi_vae.data.types import BarData, LatentEncoding, ReconstructedBar, MidiNote
from midi_vae.metrics.base import Metric
from midi_vae.registry import ComponentRegistry

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_latent_or_image(gt: BarData, recon: ReconstructedBar) -> np.ndarray:
    """Return a flat feature vector from latent encoding or reconstructed image.

    Prefers ``gt.metadata['latent'].z_mu``; falls back to mean-pooled channels
    of ``recon.recon_image``.

    Args:
        gt: Ground-truth bar (may contain 'latent' in metadata).
        recon: Reconstructed bar with recon_image of shape (3, H, W).

    Returns:
        1-D float64 numpy array.
    """
    latent = gt.metadata.get('latent', None)
    if latent is not None and hasattr(latent, 'z_mu'):
        return latent.z_mu.cpu().float().numpy().flatten().astype(np.float64)
    # Fallback: mean across spatial dims of each channel
    img = recon.recon_image.cpu().float()
    return img.mean(dim=[1, 2]).numpy().astype(np.float64)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Float in [-1, 1]. Returns 0.0 for zero vectors.
    """
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _dominant_pitch_class_from_piano_roll(piano_roll: np.ndarray) -> int:
    """Return dominant pitch class (0-11) from a (128, T) piano roll.

    Args:
        piano_roll: Velocity matrix of shape (128, T).

    Returns:
        Integer in [0, 11].
    """
    hist = np.zeros(12, dtype=np.float64)
    for pitch in range(128):
        hist[pitch % 12] += float((piano_roll[pitch, :] > 0).sum())
    return int(hist.argmax())


def _dominant_pitch_class_from_notes(notes: list[MidiNote]) -> int:
    """Return dominant pitch class (0-11) from a list of notes.

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


# ---------------------------------------------------------------------------
# ConditioningFidelity
# ---------------------------------------------------------------------------


@ComponentRegistry.register('metric', 'conditioning/fidelity')
class ConditioningFidelity(Metric):
    """Measures how well generated outputs match the conditioning instrument label.

    Uses an accumulate-then-finalize pattern. Each call to ``compute()``
    accumulates a (feature_vector, intended_instrument) pair. Calling
    ``finalize()`` fits a logistic regression on the accumulated data and
    returns the classification accuracy on a held-out split.

    High accuracy means that the conditioning signal is being faithfully
    respected and that the latent representation carries enough information
    to distinguish instruments.

    Requires ``gt.instrument`` as the intended conditioning label.
    Reads feature vectors from ``gt.metadata['latent'].z_mu`` if available;
    falls back to mean-pooled channels of ``recon.recon_image``.

    Returns from ``compute()``:
        {'conditioning_fidelity_accumulated': 1.0}  — sentinel.

    Returns from ``finalize()``:
        float  — classification accuracy in [0, 1], or NaN if not computable.
    """

    def __init__(self, max_iter: int = 200) -> None:
        """Initialise ConditioningFidelity.

        Args:
            max_iter: Maximum iterations for logistic regression.
        """
        self._max_iter = max_iter
        self._features: list[np.ndarray] = []
        self._labels: list[str] = []

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'conditioning/fidelity'

    def reset(self) -> None:
        """Reset the accumulator for a new evaluation run."""
        self._features = []
        self._labels = []

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Accumulate one (feature, instrument_label) pair.

        Args:
            gt: Ground-truth bar with instrument field.
            recon: Reconstructed bar with recon_image.

        Returns:
            Sentinel dict.
        """
        feat = _flatten_latent_or_image(gt, recon)
        self._features.append(feat)
        self._labels.append(gt.instrument)
        return {'conditioning_fidelity_accumulated': 1.0}

    def finalize(self, random_state: int = 42) -> float:
        """Fit a linear classifier and return held-out accuracy.

        Args:
            random_state: Random seed for train/test split.

        Returns:
            Accuracy in [0, 1], or NaN if not computable.
        """
        if not _SKLEARN_AVAILABLE:
            return float('nan')
        if len(self._features) < 4:
            return float('nan')

        X = np.stack(self._features, axis=0)
        le = LabelEncoder()
        y = le.fit_transform(self._labels)

        if len(le.classes_) < 2:
            return float('nan')

        rng = np.random.default_rng(random_state)
        indices = np.arange(len(X))
        rng.shuffle(indices)
        split = max(1, int(0.8 * len(X)))
        train_idx = indices[:split]
        test_idx = indices[split:]

        if len(test_idx) == 0:
            return float('nan')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                clf = LogisticRegression(
                    max_iter=self._max_iter,
                    random_state=random_state,
                    solver='lbfgs',
                    multi_class='auto',
                )
            except TypeError:
                # sklearn >= 1.8 removed multi_class parameter
                clf = LogisticRegression(
                    max_iter=self._max_iter,
                    random_state=random_state,
                    solver='lbfgs',
                )
            clf.fit(X[train_idx], y[train_idx])
            acc = float(clf.score(X[test_idx], y[test_idx]))

        return acc


# ---------------------------------------------------------------------------
# AttributeAccuracy
# ---------------------------------------------------------------------------


@ComponentRegistry.register('metric', 'conditioning/attribute_accuracy')
class AttributeAccuracy(Metric):
    """Classification accuracy of an intended attribute in generated outputs.

    A flexible version of ConditioningFidelity that works with any string
    attribute stored in ``gt.metadata['condition_label']``. If no such key
    exists, falls back to ``gt.instrument``.

    Accumulates (feature_vector, label) pairs and fits a logistic regression
    at ``finalize()`` time.

    Returns from ``compute()``:
        {'attribute_accuracy_accumulated': 1.0}  — sentinel.

    Returns from ``finalize()``:
        float  — accuracy in [0, 1], or NaN if not computable.
    """

    def __init__(self, max_iter: int = 200) -> None:
        """Initialise AttributeAccuracy.

        Args:
            max_iter: Maximum iterations for logistic regression.
        """
        self._max_iter = max_iter
        self._features: list[np.ndarray] = []
        self._labels: list[str] = []

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'conditioning/attribute_accuracy'

    def reset(self) -> None:
        """Reset the accumulator."""
        self._features = []
        self._labels = []

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Accumulate one (feature, label) pair.

        Reads the conditioning label from ``gt.metadata['condition_label']``
        if present; otherwise uses ``gt.instrument``.

        Args:
            gt: Ground-truth bar.
            recon: Reconstructed bar.

        Returns:
            Sentinel dict.
        """
        feat = _flatten_latent_or_image(gt, recon)
        label = str(gt.metadata.get('condition_label', gt.instrument))
        self._features.append(feat)
        self._labels.append(label)
        return {'attribute_accuracy_accumulated': 1.0}

    def finalize(self, random_state: int = 42) -> float:
        """Fit a linear classifier and return held-out accuracy.

        Args:
            random_state: Random seed for train/test split.

        Returns:
            Accuracy in [0, 1], or NaN if not computable.
        """
        if not _SKLEARN_AVAILABLE:
            return float('nan')
        if len(self._features) < 4:
            return float('nan')

        X = np.stack(self._features, axis=0)
        le = LabelEncoder()
        y = le.fit_transform(self._labels)

        if len(le.classes_) < 2:
            return float('nan')

        rng = np.random.default_rng(random_state)
        indices = np.arange(len(X))
        rng.shuffle(indices)
        split = max(1, int(0.8 * len(X)))
        train_idx = indices[:split]
        test_idx = indices[split:]

        if len(test_idx) == 0:
            return float('nan')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                clf = LogisticRegression(
                    max_iter=self._max_iter,
                    random_state=random_state,
                    solver='lbfgs',
                    multi_class='auto',
                )
            except TypeError:
                # sklearn >= 1.8 removed multi_class parameter
                clf = LogisticRegression(
                    max_iter=self._max_iter,
                    random_state=random_state,
                    solver='lbfgs',
                )
            clf.fit(X[train_idx], y[train_idx])
            return float(clf.score(X[test_idx], y[test_idx]))


# ---------------------------------------------------------------------------
# InterpolationSmoothness
# ---------------------------------------------------------------------------


@ComponentRegistry.register('metric', 'conditioning/interpolation_smoothness')
class InterpolationSmoothness(Metric):
    """Measures smoothness of latent interpolation paths.

    Accumulates consecutive bar feature vectors (representing steps along an
    interpolation path) and computes the mean second-order finite-difference
    magnitude (curvature proxy). Lower curvature means smoother interpolation.

    Also computes the mean step size (first-order finite difference) as a
    complementary measure of interpolation stride.

    Accumulates feature vectors via ``compute()``; call ``finalize()`` for
    smoothness statistics.

    Returns from ``compute()``:
        {'interpolation_smoothness_accumulated': 1.0}  — sentinel.

    Returns from ``finalize()``:
        dict with keys:
          - 'interpolation_smoothness': float   — mean curvature (lower = smoother).
          - 'interpolation_step_size': float    — mean consecutive L2 distance.
          - 'interpolation_n_steps': int        — number of interpolation steps.
    """

    def __init__(self) -> None:
        """Initialise InterpolationSmoothness."""
        self._features: list[np.ndarray] = []

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'conditioning/interpolation_smoothness'

    def reset(self) -> None:
        """Reset the accumulator."""
        self._features = []

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Accumulate one step's feature vector.

        Args:
            gt: Ground-truth bar (may contain 'latent' in metadata).
            recon: Reconstructed bar with recon_image.

        Returns:
            Sentinel dict.
        """
        feat = _flatten_latent_or_image(gt, recon)
        self._features.append(feat)
        return {'interpolation_smoothness_accumulated': 1.0}

    def finalize(self) -> dict[str, float]:
        """Compute smoothness and step size statistics.

        Returns:
            Dict with 'interpolation_smoothness', 'interpolation_step_size',
            'interpolation_n_steps'.
        """
        n = len(self._features)
        if n < 2:
            return {
                'interpolation_smoothness': float('nan'),
                'interpolation_step_size': float('nan'),
                'interpolation_n_steps': n,
            }

        vecs = np.stack(self._features, axis=0)  # (N, D)

        # First-order differences (step sizes)
        deltas = np.diff(vecs, axis=0)  # (N-1, D)
        step_sizes = np.linalg.norm(deltas, axis=1)  # (N-1,)
        mean_step = float(step_sizes.mean())

        if n < 3:
            # Not enough points for second-order difference
            return {
                'interpolation_smoothness': float('nan'),
                'interpolation_step_size': mean_step,
                'interpolation_n_steps': n,
            }

        # Second-order differences (curvature proxy)
        second_diffs = np.diff(deltas, axis=0)  # (N-2, D)
        curvatures = np.linalg.norm(second_diffs, axis=1)  # (N-2,)
        mean_curvature = float(curvatures.mean())

        return {
            'interpolation_smoothness': mean_curvature,
            'interpolation_step_size': mean_step,
            'interpolation_n_steps': n,
        }


# ---------------------------------------------------------------------------
# ConditionalPitchAlignment
# ---------------------------------------------------------------------------


@ComponentRegistry.register('metric', 'conditioning/pitch_alignment')
class ConditionalPitchAlignment(Metric):
    """Measures pitch-class alignment between intended and generated content.

    For each bar, computes the cosine similarity between the 12-bin pitch class
    histogram of the ground truth (intended pitch content) and the generated bar
    (detected notes or image). Provides a per-bar estimate of how well the
    generation matches the harmonic intent of the conditioning.

    Returns:
        {'conditioning_pitch_alignment': float}  — cosine similarity in [-1, 1].
        Higher is better (1.0 = perfect pitch class match).
    """

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'conditioning/pitch_alignment'

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Compute pitch class alignment for one bar.

        Args:
            gt: Ground-truth bar with piano_roll of shape (128, T).
            recon: Reconstructed bar with detected_notes.

        Returns:
            Dict with key 'conditioning_pitch_alignment'.
        """
        gt_hist = np.zeros(12, dtype=np.float64)
        for pitch in range(128):
            gt_hist[pitch % 12] += float((gt.piano_roll[pitch, :] > 0).sum())
        gt_total = gt_hist.sum()
        if gt_total > 0:
            gt_hist /= gt_total

        gen_hist = np.zeros(12, dtype=np.float64)
        if recon.detected_notes:
            for note in recon.detected_notes:
                gen_hist[note.pitch % 12] += 1.0
        else:
            # Derive from velocity channel image
            vel = recon.recon_image[0].cpu().float()
            vel_01 = ((vel + 1.0) / 2.0).clamp(0.0, 1.0)
            H = vel_01.shape[0]
            row_means = vel_01.mean(dim=1).numpy()
            for h in range(H):
                pitch_approx = int(round(h * 127.0 / max(H - 1, 1)))
                gen_hist[pitch_approx % 12] += float(row_means[h])

        gen_total = gen_hist.sum()
        if gen_total > 0:
            gen_hist /= gen_total

        sim = _cosine_sim(gt_hist, gen_hist)
        return {'conditioning_pitch_alignment': sim}


# ---------------------------------------------------------------------------
# DisentanglementScore
# ---------------------------------------------------------------------------


@ComponentRegistry.register('metric', 'conditioning/disentanglement_score')
class DisentanglementScore(Metric):
    """Measures disentanglement by comparing intra-class vs inter-class latent distances.

    A high disentanglement score means bars from the same instrument class are
    closer in latent space than bars from different classes — indicating that
    the conditioning attribute is encoded in a separable region of the space.

    Score = (mean inter-class distance - mean intra-class distance) / mean inter-class distance

    Values near 1 indicate strong disentanglement; near 0 indicates no separation.

    Accumulates (feature_vector, instrument_label) pairs via ``compute()``;
    call ``finalize()`` for the score.

    Returns from ``compute()``:
        {'disentanglement_accumulated': 1.0}  — sentinel.

    Returns from ``finalize()``:
        dict with keys:
          - 'disentanglement_score': float        — in (-inf, 1], higher is better.
          - 'intra_class_distance': float         — mean intra-class L2.
          - 'inter_class_distance': float         — mean inter-class L2.
    """

    def __init__(self, max_pairs: int = 2000) -> None:
        """Initialise DisentanglementScore.

        Args:
            max_pairs: Maximum number of pairs to sample per class combination
                to keep computation tractable. Default 2000.
        """
        self._max_pairs = max_pairs
        self._features: list[np.ndarray] = []
        self._labels: list[str] = []

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'conditioning/disentanglement_score'

    def reset(self) -> None:
        """Reset the accumulator."""
        self._features = []
        self._labels = []

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Accumulate one (feature, label) pair.

        Args:
            gt: Ground-truth bar with instrument field.
            recon: Reconstructed bar with recon_image.

        Returns:
            Sentinel dict.
        """
        feat = _flatten_latent_or_image(gt, recon)
        self._features.append(feat)
        self._labels.append(gt.instrument)
        return {'disentanglement_accumulated': 1.0}

    def finalize(self, random_state: int = 42) -> dict[str, float]:
        """Compute the disentanglement score.

        Args:
            random_state: Random seed for pair subsampling.

        Returns:
            Dict with 'disentanglement_score', 'intra_class_distance',
            'inter_class_distance'.
        """
        nan_result: dict[str, float] = {
            'disentanglement_score': float('nan'),
            'intra_class_distance': float('nan'),
            'inter_class_distance': float('nan'),
        }

        if len(self._features) < 4:
            return nan_result

        X = np.stack(self._features, axis=0)  # (N, D)
        labels = np.array(self._labels)
        unique_labels = np.unique(labels)

        if len(unique_labels) < 2:
            return nan_result

        rng = np.random.default_rng(random_state)

        intra_dists: list[float] = []
        inter_dists: list[float] = []

        # Intra-class distances
        for lbl in unique_labels:
            idx = np.where(labels == lbl)[0]
            if len(idx) < 2:
                continue
            # Subsample pairs
            n_pairs = min(self._max_pairs, len(idx) * (len(idx) - 1) // 2)
            sampled_i = rng.choice(idx, size=n_pairs, replace=True)
            sampled_j = rng.choice(idx, size=n_pairs, replace=True)
            mask = sampled_i != sampled_j
            sampled_i = sampled_i[mask]
            sampled_j = sampled_j[mask]
            if len(sampled_i) == 0:
                continue
            diffs = X[sampled_i] - X[sampled_j]
            intra_dists.extend(np.linalg.norm(diffs, axis=1).tolist())

        # Inter-class distances
        for li in range(len(unique_labels)):
            for lj in range(li + 1, len(unique_labels)):
                idx_i = np.where(labels == unique_labels[li])[0]
                idx_j = np.where(labels == unique_labels[lj])[0]
                n_pairs = min(self._max_pairs, len(idx_i) * len(idx_j))
                sampled_i = rng.choice(idx_i, size=n_pairs, replace=True)
                sampled_j = rng.choice(idx_j, size=n_pairs, replace=True)
                diffs = X[sampled_i] - X[sampled_j]
                inter_dists.extend(np.linalg.norm(diffs, axis=1).tolist())

        if not intra_dists or not inter_dists:
            return nan_result

        mean_intra = float(np.mean(intra_dists))
        mean_inter = float(np.mean(inter_dists))

        if mean_inter < 1e-10:
            return nan_result

        score = (mean_inter - mean_intra) / mean_inter

        return {
            'disentanglement_score': score,
            'intra_class_distance': mean_intra,
            'inter_class_distance': mean_inter,
        }
