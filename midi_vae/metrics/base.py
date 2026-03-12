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

    def _load_metrics(self, metric_names: list[str]) -> None:
        """Load metric instances from the registry.

        Args:
            metric_names: List of metric names or ['all'].
        """
        if metric_names == ["all"]:
            available = ComponentRegistry.list_components("metric")
            names = available.get("metric", [])
        else:
            names = metric_names

        for name in names:
            metric_cls = ComponentRegistry.get("metric", name)
            self.metrics.append(metric_cls())

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
