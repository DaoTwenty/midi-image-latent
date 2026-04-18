"""Abstract base class for metrics and the MetricsEngine orchestrator.

Metrics evaluate reconstruction quality by comparing ground-truth BarData
with ReconstructedBar outputs.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from midi_vae.data.types import BarData, PianoRollImage, ReconstructedBar
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
        gt_image: PianoRollImage | None = None,
    ) -> dict[str, float]:
        """Compute metric value(s) for a single bar.

        Args:
            gt: Ground-truth bar data with original piano roll and notes.
            recon: Reconstructed bar with decoded image and detected notes.
            gt_image: Optional rendered ground-truth PianoRollImage. Required
                by image-level metrics (PixelMSE, SSIM, PSNR). Metrics that
                do not need the rendered image may ignore this parameter.

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

    @property
    def requires_image(self) -> bool:
        """Whether this metric requires a rendered PianoRollImage (gt_image).

        Override to return True for pixel-level image metrics (PixelMSE, SSIM,
        PSNR, etc.). When True, MetricsEngine will skip this metric if no
        images lookup dict is available, and will pass the matching gt_image
        to compute().  Default is False.
        """
        return False


class MetricsEngine:
    """Orchestrates evaluation across multiple metrics.

    Loads metrics from the registry based on config, runs them all
    on each (ground_truth, reconstruction) pair, and collects results.
    """

    def __init__(
        self,
        metric_names: list[str],
        exclude: list[str] | None = None,
    ) -> None:
        """Initialize the metrics engine.

        Args:
            metric_names: List of metric/category names to evaluate.
                Supports three entry types mixed freely:
                  - ``"all"``          — every registered metric
                  - category name      — e.g. ``"reconstruction"``, ``"harmony"``
                  - individual name    — e.g. ``"pixel_mse"``, ``"ssim"``
                Prefix an entry with ``-`` to subtract it after expansion:
                  ``["harmony", "-interval_histogram_correlation"]``
            exclude: Optional explicit exclude list (applied after expansion).
                Equivalent to prefixing names with ``-`` in *metric_names*.
        """
        self.metrics: list[Metric] = []
        self._load_metrics(metric_names, exclude=exclude)

    # Category -> individual metric name mapping
    # NOTE: ssim and mutual_information_pitch_time are registered but excluded
    # from default categories due to high per-bar cost (3.7ms and 1.2ms
    # respectively — together 61% of total metric time).  They can still be
    # requested explicitly by name when needed.
    # onset_precision / onset_recall removed — onset_f1 already returns all
    # three values (precision, recall, f1) in a single call.
    # velocity_histogram_kl removed — redundant with velocity_mse +
    # velocity_correlation which together capture the same signal.
    _CATEGORIES: dict[str, list[str]] = {
        "reconstruction": ["pixel_mse", "psnr"],
        "harmony": [
            "onset_f1",
            "note_density_pearson", "pitch_class_histogram_correlation",
            "interval_histogram_correlation",
        ],
        "rhythm": ["ioi_distribution_kl", "groove_consistency"],
        "dynamics": ["velocity_mse", "velocity_correlation"],
        "information": [
            "note_entropy_diff",
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

    def _load_metrics(
        self,
        metric_names: list[str],
        exclude: list[str] | None = None,
    ) -> None:
        """Load metric instances from the registry.

        Supports individual names, category names, ``"all"``, inline
        negations (``"-ssim"``), and an explicit *exclude* list.

        Resolution order:
        1. Expand ``"all"`` or category names into individual metric names.
        2. Collect inline negations (entries starting with ``-``).
        3. Merge inline negations with *exclude* list.
        4. Remove excluded names from the final set.

        Args:
            metric_names: List of metric names, category names, ``"all"``,
                or negated names (prefixed with ``-``).
            exclude: Optional explicit list of metric names to exclude.
        """
        # Separate additions from inline negations
        additions: list[str] = []
        negations: set[str] = set()
        for entry in metric_names:
            stripped = entry.strip()
            if stripped.startswith("-"):
                negations.add(stripped.lstrip("-").strip())
            elif stripped == "all":
                available = ComponentRegistry.list_components("metric")
                additions.extend(available.get("metric", []))
            elif stripped in self._CATEGORIES:
                additions.extend(self._CATEGORIES[stripped])
            else:
                additions.append(stripped)

        # Merge inline negations with explicit exclude
        if exclude:
            negations.update(exclude)

        # De-duplicate while preserving order, then remove excluded
        seen: set[str] = set()
        names: list[str] = []
        for name in additions:
            if name not in seen and name not in negations:
                seen.add(name)
                names.append(name)

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
        images_by_id: dict[str, PianoRollImage] | None = None,
    ) -> dict[str, float]:
        """Run all loaded metrics on a single (gt, recon) pair.

        Args:
            gt: Ground-truth bar data.
            recon: Reconstructed bar.
            images_by_id: Optional mapping from bar_id to PianoRollImage.
                When provided, image-requiring metrics receive the matching
                gt_image. When absent, metrics with requires_image=True are
                skipped (they would return NaN anyway).

        Returns:
            Combined dict of all metric results.
        """
        gt_image: PianoRollImage | None = None
        if images_by_id is not None:
            gt_image = images_by_id.get(recon.bar_id)

        results: dict[str, float] = {}
        for metric in self.metrics:
            if metric.requires_notes and not recon.detected_notes:
                continue
            if metric.requires_image and images_by_id is None:
                continue
            metric_results = metric.compute(gt, recon, gt_image=gt_image)
            results.update(metric_results)
        return results

    def evaluate_batch(
        self,
        pairs: list[tuple[BarData, ReconstructedBar]],
        images_by_id: dict[str, PianoRollImage] | None = None,
    ) -> list[dict[str, float]]:
        """Run all metrics on a batch of (gt, recon) pairs.

        Args:
            pairs: List of (ground_truth, reconstruction) tuples.
            images_by_id: Optional mapping from bar_id to PianoRollImage,
                forwarded to each evaluate() call.

        Returns:
            List of result dicts, one per pair.
        """
        return [self.evaluate(gt, recon, images_by_id) for gt, recon in pairs]

    def evaluate_batch_parallel(
        self,
        pairs: list[tuple[BarData, ReconstructedBar]],
        images_by_id: dict[str, PianoRollImage] | None = None,
        max_workers: int | None = None,
    ) -> list[dict[str, float]]:
        """Run metrics on pairs using ThreadPoolExecutor for parallel evaluation.

        Each (gt, recon) pair is evaluated independently in a thread.  Because
        the metric computations are numpy/torch-heavy and release the GIL for
        significant portions of their runtime, threading provides a meaningful
        speedup on multi-core machines.

        Results are returned in the same order as the input pairs.

        Args:
            pairs: List of (ground_truth, reconstruction) tuples.
            images_by_id: Optional mapping from bar_id to PianoRollImage,
                forwarded to each evaluate() call.
            max_workers: Maximum number of worker threads.  Defaults to
                ``min(4, os.cpu_count() or 1)``.  Set to 1 to effectively
                run sequentially (identical to :meth:`evaluate_batch`).

        Returns:
            List of result dicts, one per pair, in input order.
        """
        if not pairs:
            return []

        if max_workers is None:
            max_workers = min(4, os.cpu_count() or 1)

        # With a single worker, avoid thread overhead
        if max_workers <= 1:
            return self.evaluate_batch(pairs, images_by_id)

        results: list[dict[str, float]] = [{}] * len(pairs)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map future -> original index so order can be restored
            future_to_idx = {
                executor.submit(self.evaluate, gt, recon, images_by_id): idx
                for idx, (gt, recon) in enumerate(pairs)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return results
