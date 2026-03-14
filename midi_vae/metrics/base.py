"""Abstract base class for metrics and the MetricsEngine orchestrator.

Metrics evaluate reconstruction quality by comparing ground-truth BarData
with ReconstructedBar outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from midi_vae.data.types import BarData, ReconstructedBar
from midi_vae.registry import ComponentRegistry


class Metric(ABC):
    """Abstract base for evaluation metrics.

    Each metric computes one or more named scalar values comparing
    a ground-truth bar to its reconstruction.
    """

    @abstractmethod
    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Compute metric value(s) for a single bar.

        Args:
            gt: Ground-truth bar data with original piano roll and notes.
            recon: Reconstructed bar with decoded image and detected notes.

        Returns:
            Dict mapping metric names to float values.
            Keys should be namespaced (e.g., 'onset_f1/precision', 'pixel_mse/value').
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique metric identifier used in results tables."""
        ...

    @property
    def requires_notes(self) -> bool:
        """Whether this metric requires detected notes (vs just images).

        Override to return True for note-level metrics (F1, pitch accuracy, etc.).
        Default is False (image-level metrics).
        """
        return False


class MetricsEngine:
    """Orchestrates evaluation across multiple metrics.

    Loads metrics from the registry based on config, runs them all
    on each (ground_truth, reconstruction) pair, and collects results.
    """

    def __init__(self, metric_names: list[str]) -> None:
        """Initialize the metrics engine.

        Args:
            metric_names: List of metric names to evaluate. Use ['all'] to
                load all registered metrics.
        """
        self.metrics: list[Metric] = []
        self._load_metrics(metric_names)

    # Category -> individual metric name mapping
    _CATEGORIES: dict[str, list[str]] = {
        "reconstruction": ["pixel_mse", "ssim", "psnr"],
        "harmony": [
            "onset_f1", "onset_precision", "onset_recall",
            "note_density_pearson", "pitch_class_histogram_correlation",
            "interval_histogram_correlation",
        ],
        "rhythm": ["ioi_distribution_kl", "groove_consistency"],
        "dynamics": ["velocity_mse", "velocity_correlation", "velocity_histogram_kl"],
        "information": [
            "note_entropy_diff", "mutual_information_pitch_time",
            "kl_divergence_pitch_dist", "activation_sparsity",
        ],
        "latent_space": [
            "latent_variance", "latent_channel_correlation",
            "linear_probe_pitch_accuracy", "linear_probe_instrument_accuracy",
            "silhouette_score",
        ],
        "generative": [
            "generative/self_similarity_matrix", "generative/transition_entropy",
            "generative/groove_consistency", "generative/pitch_class_histogram_kl",
            "generative/bar_level_nll", "generative/sequence_coherence",
        ],
        "conditioning": [
            "conditioning/fidelity", "conditioning/attribute_accuracy",
            "conditioning/interpolation_smoothness", "conditioning/pitch_alignment",
            "conditioning/disentanglement_score",
        ],
    }

    def _load_metrics(self, metric_names: list[str]) -> None:
        """Load metric instances from the registry.

        Accepts individual metric names, category names (e.g., 'reconstruction',
        'harmony'), or ['all'] to load everything.

        Args:
            metric_names: List of metric names, category names, or ['all'].
        """
        if metric_names == ["all"]:
            available = ComponentRegistry.list_components("metric")
            names = available.get("metric", [])
        else:
            # Expand category names to individual metrics
            names: list[str] = []
            for entry in metric_names:
                if entry in self._CATEGORIES:
                    names.extend(self._CATEGORIES[entry])
                else:
                    names.append(entry)

        for name in names:
            try:
                metric_cls = ComponentRegistry.get("metric", name)
                self.metrics.append(metric_cls())
            except KeyError:
                import logging
                logging.getLogger(__name__).warning(
                    "Metric '%s' not registered — skipping", name
                )

    def evaluate(
        self,
        gt: BarData,
        recon: ReconstructedBar,
    ) -> dict[str, float]:
        """Run all loaded metrics on a single (gt, recon) pair.

        Args:
            gt: Ground-truth bar data.
            recon: Reconstructed bar.

        Returns:
            Combined dict of all metric results.
        """
        results: dict[str, float] = {}
        for metric in self.metrics:
            if metric.requires_notes and not recon.detected_notes:
                continue
            metric_results = metric.compute(gt, recon)
            results.update(metric_results)
        return results

    def evaluate_batch(
        self,
        pairs: list[tuple[BarData, ReconstructedBar]],
    ) -> list[dict[str, float]]:
        """Run all metrics on a batch of (gt, recon) pairs.

        Args:
            pairs: List of (ground_truth, reconstruction) tuples.

        Returns:
            List of result dicts, one per pair.
        """
        return [self.evaluate(gt, recon) for gt, recon in pairs]
