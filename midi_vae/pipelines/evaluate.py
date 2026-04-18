"""EvaluateStage: pipeline stage that computes metrics over reconstructed bars.

This stage reads "bars" (original BarData) and "reconstructed_bars" from the
pipeline context.  It runs all configured metrics via MetricsEngine and outputs
the aggregated results as per-bar dicts and a per-VAE summary of mean values.

Context inputs:
    bars: list[BarData]
    reconstructed_bars: dict[str, list[ReconstructedBar]]

Context outputs:
    metrics: dict[str, list[dict[str, float]]]
        Keys are VAE short names.  Each value is a list of per-bar metric dicts.
    metrics_summary: dict[str, dict[str, float]]
        Per-VAE mean of all scalar metrics.
"""

from __future__ import annotations

import logging
from typing import Any

from midi_vae.config import ExperimentConfig
from midi_vae.data.types import BarData, PianoRollImage, ReconstructedBar
from midi_vae.metrics.base import MetricsEngine
import midi_vae.metrics  # noqa: F401 — trigger registration of all metrics
from midi_vae.pipelines.base import PipelineStage, StageIO, compute_hash
from midi_vae.registry import ComponentRegistry

logger = logging.getLogger(__name__)


@ComponentRegistry.register("pipeline_stage", "evaluate")
class EvaluateStage(PipelineStage):
    """Pipeline stage that evaluates reconstruction quality via MetricsEngine.

    For each VAE present in "reconstructed_bars", the stage pairs each
    ReconstructedBar with its corresponding BarData by bar_id, then runs
    all configured metrics.

    Args:
        config: Full experiment configuration.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)
        self._engine: MetricsEngine | None = None

    def _get_engine(self) -> MetricsEngine:
        """Lazily build the MetricsEngine.

        Returns:
            MetricsEngine loaded with the metrics listed in config.metrics.
        """
        if self._engine is None:
            self._engine = MetricsEngine(
                list(self.config.metrics),
                exclude=list(self.config.metrics_exclude),
            )
        return self._engine

    def io(self) -> StageIO:
        """Declare inputs and outputs.

        Returns:
            StageIO with inputs=("bars", "reconstructed_bars", "images") and
            outputs=("metrics", "metrics_summary").

        Note: "images" is optional — the stage works without it, but
        image-requiring metrics (PixelMSE, SSIM, PSNR) will be skipped
        if it is absent.
        """
        return StageIO(
            inputs=("bars", "reconstructed_bars", "images"),
            outputs=("metrics", "metrics_summary"),
        )

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Compute metrics for all reconstructed bars.

        Args:
            context: Pipeline context with "bars", "reconstructed_bars", and
                optionally "images" (list[PianoRollImage] from RenderStage).

        Returns:
            Dict with keys "metrics" and "metrics_summary".
        """
        bars: list[BarData] = context.get("bars", [])
        reconstructed_bars: dict[str, list[ReconstructedBar]] = context.get(
            "reconstructed_bars", {}
        )
        images: list[PianoRollImage] = context.get("images", [])

        if not reconstructed_bars:
            logger.warning("EvaluateStage received no reconstructed_bars")
            return {"metrics": {}, "metrics_summary": {}}

        if not bars:
            logger.warning("EvaluateStage received no ground-truth bars")
            return {"metrics": {}, "metrics_summary": {}}

        # Build bar_id -> BarData lookup
        gt_by_id: dict[str, BarData] = {bar.bar_id: bar for bar in bars}

        # Build bar_id -> PianoRollImage lookup (None when images not available)
        images_by_id: dict[str, PianoRollImage] | None = None
        if images:
            images_by_id = {img.bar_id: img for img in images}
            logger.info(
                "EvaluateStage: image lookup built for %d bars", len(images_by_id)
            )
        else:
            logger.warning(
                "EvaluateStage: no 'images' in context — image-level metrics "
                "(PixelMSE, SSIM, PSNR) will be skipped"
            )

        engine = self._get_engine()
        metrics: dict[str, list[dict[str, float]]] = {}
        metrics_summary: dict[str, dict[str, float]] = {}

        for vae_name, recon_list in reconstructed_bars.items():
            logger.info(
                "EvaluateStage: evaluating %d bars for VAE '%s'",
                len(recon_list),
                vae_name,
            )

            pairs: list[tuple[BarData, ReconstructedBar]] = []
            skipped = 0
            for recon in recon_list:
                gt = gt_by_id.get(recon.bar_id)
                if gt is None:
                    skipped += 1
                    continue
                pairs.append((gt, recon))

            if skipped:
                logger.warning(
                    "EvaluateStage: %d bars had no matching ground truth (VAE='%s')",
                    skipped,
                    vae_name,
                )

            if not pairs:
                metrics[vae_name] = []
                metrics_summary[vae_name] = {}
                continue

            num_workers = getattr(self.config, 'num_workers', 1)
            if num_workers > 1:
                logger.info(
                    "EvaluateStage: using parallel evaluation with %d workers",
                    num_workers,
                )
                per_bar_results = engine.evaluate_batch_parallel(
                    pairs,
                    images_by_id=images_by_id,
                    max_workers=num_workers,
                )
            else:
                per_bar_results = engine.evaluate_batch(pairs, images_by_id=images_by_id)
            metrics[vae_name] = per_bar_results

            # Compute mean per metric key
            if per_bar_results:
                summary: dict[str, float] = {}
                all_keys: set[str] = set().union(  # type: ignore[arg-type]
                    *[d.keys() for d in per_bar_results]
                )
                for key in all_keys:
                    values = [d[key] for d in per_bar_results if key in d]
                    summary[key] = sum(values) / len(values) if values else float("nan")
                metrics_summary[vae_name] = summary
            else:
                metrics_summary[vae_name] = {}

            logger.info(
                "EvaluateStage: completed evaluation for VAE '%s' (%d pairs)",
                vae_name,
                len(pairs),
            )

        return {"metrics": metrics, "metrics_summary": metrics_summary}

    def cache_key(self, context: dict[str, Any]) -> str | None:
        """Compute a cache key from reconstructed bar IDs and metrics config.

        Args:
            context: Current pipeline context.

        Returns:
            Hex digest string, or None if reconstructed_bars are not available.
        """
        reconstructed_bars: dict[str, list[ReconstructedBar]] | None = context.get(
            "reconstructed_bars"
        )
        if not reconstructed_bars:
            return None

        all_ids = tuple(
            recon.bar_id
            for vae_name in sorted(reconstructed_bars.keys())
            for recon in reconstructed_bars[vae_name]
        )
        vae_names = tuple(sorted(reconstructed_bars.keys()))

        return compute_hash(
            all_ids,
            vae_names,
            tuple(sorted(self.config.metrics)),
        )
