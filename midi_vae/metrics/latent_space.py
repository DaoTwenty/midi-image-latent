"""Latent space quality metrics.

Metrics that analyse the structure and quality of the VAE latent space,
including variance utilisation, channel correlation, and classification
accuracy via linear probes.

For linear probe and silhouette metrics that require batches of samples,
implement an accumulator pattern: call ``compute()`` for each sample to
accumulate data, then call ``finalize()`` to obtain the final metric value.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from midi_vae.data.types import BarData, LatentEncoding, ReconstructedBar
from midi_vae.metrics.base import Metric
from midi_vae.registry import ComponentRegistry

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import silhouette_score as _sklearn_silhouette
    from sklearn.preprocessing import LabelEncoder
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def _flatten_latent(z: torch.Tensor) -> np.ndarray:
    """Flatten a latent tensor to a 1-D numpy array.

    Args:
        z: Latent tensor of arbitrary shape (C, H_lat, W_lat) or (D,).

    Returns:
        1-D float64 numpy array.
    """
    return z.cpu().float().numpy().flatten()


@ComponentRegistry.register('metric', 'latent_variance')
class LatentVariance(Metric):
    """Average variance across latent dimensions.

    Measures how much of the latent capacity is being utilised. A latent space
    with very low variance across dimensions is "collapsed" — all points cluster
    near the same location and the VAE is not encoding information effectively.

    Requires a ``LatentEncoding`` passed via the ``gt.metadata['latent']`` key.
    Falls back to computing variance from the reconstructed image if no latent
    is available.

    Returns:
        {'latent_variance_mean': float}  — average variance across dims.
        {'latent_variance_min': float}   — minimum per-dim variance.
        {'latent_variance_max': float}   — maximum per-dim variance.

    Note:
        This metric is designed for batch-level use. When called on a single
        sample, it returns variance across spatial/channel dimensions of a
        single z_mu tensor. For true cross-sample variance, use
        ``LatentVarianceBatch`` or accumulate results across many samples.
    """

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'latent_variance'

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Compute variance statistics of the latent encoding.

        Reads ``LatentEncoding`` from ``gt.metadata['latent']`` if available.
        Falls back to computing pixel-level variance of the recon image.

        Args:
            gt: Ground-truth bar. May contain 'latent' in metadata.
            recon: Reconstructed bar with recon_image.

        Returns:
            Dict with keys 'latent_variance_mean', 'latent_variance_min',
            'latent_variance_max'.
        """
        latent: LatentEncoding | None = gt.metadata.get('latent', None)

        if latent is not None:
            z = latent.z_mu.cpu().float()
            # z shape: (C, H_lat, W_lat) — compute per-channel variance over spatial dims
            z_flat = z.reshape(z.shape[0], -1)  # (C, H*W)
            per_channel_var = z_flat.var(dim=1).numpy()  # (C,)
        else:
            # Fallback: use recon image channels
            img = recon.recon_image.cpu().float()  # (3, H, W)
            img_flat = img.reshape(img.shape[0], -1)  # (3, H*W)
            per_channel_var = img_flat.var(dim=1).numpy()  # (3,)

        return {
            'latent_variance_mean': float(per_channel_var.mean()),
            'latent_variance_min': float(per_channel_var.min()),
            'latent_variance_max': float(per_channel_var.max()),
        }


@ComponentRegistry.register('metric', 'latent_channel_correlation')
class LatentChannelCorrelation(Metric):
    """Mean pairwise absolute correlation between latent channels.

    Lower values indicate more independent (disentangled) latent dimensions,
    which is generally desirable for a well-structured latent space.

    Reads ``LatentEncoding`` from ``gt.metadata['latent']``.
    Falls back to image channels if no latent is provided.

    Returns:
        {'latent_channel_correlation_mean': float}  — mean |r| across pairs, in [0, 1].
        {'latent_channel_correlation_max': float}   — maximum |r|.
    """

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'latent_channel_correlation'

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Compute mean pairwise correlation across latent channels.

        Args:
            gt: Ground-truth bar. May contain 'latent' in metadata.
            recon: Reconstructed bar with recon_image.

        Returns:
            Dict with keys 'latent_channel_correlation_mean',
            'latent_channel_correlation_max'.
        """
        latent: LatentEncoding | None = gt.metadata.get('latent', None)

        if latent is not None:
            z = latent.z_mu.cpu().float()
            z_flat = z.reshape(z.shape[0], -1).numpy()  # (C, H*W)
        else:
            img = recon.recon_image.cpu().float()
            z_flat = img.reshape(img.shape[0], -1).numpy()  # (3, H*W)

        C = z_flat.shape[0]
        if C < 2:
            return {
                'latent_channel_correlation_mean': 0.0,
                'latent_channel_correlation_max': 0.0,
            }

        # Compute correlation matrix
        corr_matrix = np.corrcoef(z_flat)  # (C, C)

        # Collect off-diagonal absolute values
        abs_corrs = []
        for i in range(C):
            for j in range(i + 1, C):
                val = corr_matrix[i, j]
                if not np.isnan(val):
                    abs_corrs.append(abs(val))

        if not abs_corrs:
            return {
                'latent_channel_correlation_mean': 0.0,
                'latent_channel_correlation_max': 0.0,
            }

        return {
            'latent_channel_correlation_mean': float(np.mean(abs_corrs)),
            'latent_channel_correlation_max': float(np.max(abs_corrs)),
        }


class _LinearProbeAccumulator:
    """Internal accumulator for batch-level linear probe metrics.

    Accumulates (feature_vector, label) pairs across ``compute()`` calls
    and fits a linear classifier in ``finalize()``.
    """

    def __init__(self) -> None:
        self._features: list[np.ndarray] = []
        self._labels: list[str] = []

    def add(self, feature: np.ndarray, label: str) -> None:
        """Add a single (feature, label) pair.

        Args:
            feature: 1-D float array.
            label: String label.
        """
        self._features.append(feature)
        self._labels.append(label)

    def finalize(self, max_iter: int = 200, random_state: int = 42) -> float:
        """Fit a logistic regression and return held-out accuracy.

        Uses a stratified 80/20 train-test split. Returns NaN if sklearn is
        not available or if there are too few samples / label classes.

        Args:
            max_iter: Max iterations for logistic regression solver.
            random_state: Random seed for reproducibility.

        Returns:
            Classification accuracy in [0, 1], or NaN if not computable.
        """
        if not _SKLEARN_AVAILABLE:
            return float('nan')

        if len(self._features) < 4:
            return float('nan')

        X = np.stack(self._features, axis=0)  # (N, D)
        le = LabelEncoder()
        y = le.fit_transform(self._labels)  # (N,)

        n_classes = len(le.classes_)
        if n_classes < 2:
            return float('nan')

        # Stratified 80/20 split
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
            clf = LogisticRegression(
                max_iter=max_iter,
                random_state=random_state,
                solver='lbfgs',
                multi_class='auto',
            )
            clf.fit(X[train_idx], y[train_idx])
            acc = float(clf.score(X[test_idx], y[test_idx]))

        return acc


@ComponentRegistry.register('metric', 'linear_probe_pitch_accuracy')
class LinearProbePitchAccuracy(Metric):
    """Train a linear classifier on latent vectors to predict dominant pitch class.

    This is a batch-level metric: individual ``compute()`` calls accumulate
    (latent_vector, pitch_class_label) pairs. Call ``finalize()`` after
    processing all samples to get the final accuracy.

    The dominant pitch class (0-11) is determined from ``gt.piano_roll`` as
    the pitch class with the highest total activation count.

    Reads latent vector from ``gt.metadata['latent']`` (z_mu flattened).
    Falls back to flattened recon_image if no latent is available.

    Returns from ``finalize()``:
        float — accuracy in [0, 1], or NaN if sklearn unavailable.

    Returns from ``compute()``:
        {'linear_probe_pitch_accuracy_accumulated': 1.0}  — sentinel indicating
        data was added. Call ``finalize()`` for the true metric.
    """

    def __init__(self, max_iter: int = 200) -> None:
        """Initialise LinearProbePitchAccuracy.

        Args:
            max_iter: Maximum iterations for logistic regression.
        """
        self._max_iter = max_iter
        self._accumulator = _LinearProbeAccumulator()

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'linear_probe_pitch_accuracy'

    def reset(self) -> None:
        """Reset the accumulator for a new evaluation run."""
        self._accumulator = _LinearProbeAccumulator()

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Accumulate one (latent_vector, pitch_class) pair.

        Args:
            gt: Ground-truth bar. May contain 'latent' in metadata.
            recon: Reconstructed bar with recon_image.

        Returns:
            Sentinel dict indicating data was accumulated. Call finalize() for
            true accuracy.
        """
        latent: LatentEncoding | None = gt.metadata.get('latent', None)

        if latent is not None:
            feature = _flatten_latent(latent.z_mu)
        else:
            feature = recon.recon_image.cpu().float().numpy().flatten()

        # Dominant pitch class from piano roll
        pitch_counts = (gt.piano_roll > 0).sum(axis=1)  # (128,)
        dominant_pitch = int(pitch_counts.argmax())
        pitch_class = str(dominant_pitch % 12)

        self._accumulator.add(feature, pitch_class)
        return {'linear_probe_pitch_accuracy_accumulated': 1.0}

    def finalize(self) -> float:
        """Compute and return the final linear probe accuracy.

        Returns:
            Accuracy in [0, 1], or NaN if not computable.
        """
        return self._accumulator.finalize(max_iter=self._max_iter)


@ComponentRegistry.register('metric', 'linear_probe_instrument_accuracy')
class LinearProbeInstrumentAccuracy(Metric):
    """Train a linear classifier on latent vectors to predict instrument label.

    Same accumulator pattern as ``LinearProbePitchAccuracy``. The instrument
    label comes from ``gt.instrument``.

    Returns from ``finalize()``:
        float — accuracy in [0, 1], or NaN if sklearn unavailable.

    Returns from ``compute()``:
        {'linear_probe_instrument_accuracy_accumulated': 1.0} — sentinel.
    """

    def __init__(self, max_iter: int = 200) -> None:
        """Initialise LinearProbeInstrumentAccuracy.

        Args:
            max_iter: Maximum iterations for logistic regression.
        """
        self._max_iter = max_iter
        self._accumulator = _LinearProbeAccumulator()

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'linear_probe_instrument_accuracy'

    def reset(self) -> None:
        """Reset the accumulator for a new evaluation run."""
        self._accumulator = _LinearProbeAccumulator()

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Accumulate one (latent_vector, instrument_label) pair.

        Args:
            gt: Ground-truth bar with instrument field.
            recon: Reconstructed bar with recon_image.

        Returns:
            Sentinel dict indicating data was accumulated.
        """
        latent: LatentEncoding | None = gt.metadata.get('latent', None)

        if latent is not None:
            feature = _flatten_latent(latent.z_mu)
        else:
            feature = recon.recon_image.cpu().float().numpy().flatten()

        self._accumulator.add(feature, gt.instrument)
        return {'linear_probe_instrument_accuracy_accumulated': 1.0}

    def finalize(self) -> float:
        """Compute and return the final linear probe accuracy.

        Returns:
            Accuracy in [0, 1], or NaN if not computable.
        """
        return self._accumulator.finalize(max_iter=self._max_iter)


@ComponentRegistry.register('metric', 'silhouette_score')
class SilhouetteScore(Metric):
    """Silhouette coefficient of latent vectors clustered by instrument.

    Measures how well-separated latent representations are by instrument class.
    A high silhouette score (near 1) indicates clear instrument clusters;
    near 0 means overlapping clusters; negative means incorrect clustering.

    Accumulates (latent_vector, instrument_label) pairs across ``compute()``
    calls. Call ``finalize()`` to compute the sklearn silhouette score.

    Returns from ``finalize()``:
        float — silhouette coefficient in [-1, 1], or NaN if not computable.

    Returns from ``compute()``:
        {'silhouette_score_accumulated': 1.0} — sentinel.
    """

    def __init__(self, sample_size: int = 1000) -> None:
        """Initialise SilhouetteScore.

        Args:
            sample_size: Maximum number of samples to use for silhouette
                computation (subsample if more are accumulated). Default 1000.
        """
        self._sample_size = sample_size
        self._features: list[np.ndarray] = []
        self._labels: list[str] = []

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'silhouette_score'

    def reset(self) -> None:
        """Reset the accumulator for a new evaluation run."""
        self._features = []
        self._labels = []

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Accumulate one (latent_vector, instrument_label) pair.

        Args:
            gt: Ground-truth bar with instrument field.
            recon: Reconstructed bar with recon_image.

        Returns:
            Sentinel dict indicating data was accumulated.
        """
        latent: LatentEncoding | None = gt.metadata.get('latent', None)

        if latent is not None:
            feature = _flatten_latent(latent.z_mu)
        else:
            feature = recon.recon_image.cpu().float().numpy().flatten()

        self._features.append(feature)
        self._labels.append(gt.instrument)
        return {'silhouette_score_accumulated': 1.0}

    def finalize(self, random_state: int = 42) -> float:
        """Compute and return the silhouette score.

        Args:
            random_state: Random seed for subsampling if needed.

        Returns:
            Silhouette coefficient in [-1, 1], or NaN if not computable.
        """
        if not _SKLEARN_AVAILABLE:
            return float('nan')

        if len(self._features) < 4:
            return float('nan')

        X = np.stack(self._features, axis=0)  # (N, D)
        le = LabelEncoder()
        y = le.fit_transform(self._labels)

        if len(le.classes_) < 2:
            return float('nan')

        # Subsample if too large (silhouette is O(N^2))
        N = len(X)
        if N > self._sample_size:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(N, size=self._sample_size, replace=False)
            X = X[idx]
            y = y[idx]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                score = float(_sklearn_silhouette(X, y))
            except Exception:
                score = float('nan')

        return score
