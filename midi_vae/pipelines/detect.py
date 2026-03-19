"""DetectStage: pipeline stage that runs NoteDetector on reconstructed images.

This stage reads "recon_images" and "images" from the pipeline context,
runs the configured NoteDetector on each reconstructed image, and produces
a list of ReconstructedBar objects under the "reconstructed_bars" key.

Context inputs:
    recon_images: dict[str, list[tuple[str, torch.Tensor]]]
    images: list[PianoRollImage]   (used to recover channel_strategy per bar)

Context outputs:
    reconstructed_bars: dict[str, list[ReconstructedBar]]
        Keys are VAE short names.
"""

from __future__ import annotations

import logging
from typing import Any

from midi_vae.config import ExperimentConfig
from midi_vae.data.types import MidiNote, PianoRollImage, ReconstructedBar
from midi_vae.note_detection.base import NoteDetector
import midi_vae.note_detection  # noqa: F401 — trigger registration of all detectors
from midi_vae.pipelines.base import PipelineStage, StageIO, compute_hash
from midi_vae.registry import ComponentRegistry

logger = logging.getLogger(__name__)


@ComponentRegistry.register("pipeline_stage", "detect")
class DetectStage(PipelineStage):
    """Pipeline stage that converts reconstructed images into ReconstructedBar objects.

    Runs the note detector specified in config.note_detection.method on each
    decoded image.  The channel strategy for each bar is recovered from the
    "images" context entry (the original rendered images).

    If the chosen detector requires fitting (needs_fitting == True), it is
    fit on the first 20 % of the bars before running detection on all bars.

    Args:
        config: Full experiment configuration.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)
        self._detector: NoteDetector | None = None

    def _get_detector(self) -> NoteDetector:
        """Lazily build the note detector from the registry.

        Returns:
            A NoteDetector instance configured with params from config.

        Raises:
            KeyError: If the detection method is not registered.
        """
        if self._detector is None:
            method = self.config.note_detection.method
            params = dict(self.config.note_detection.params)
            detector_cls = ComponentRegistry.get("note_detector", method)
            self._detector = detector_cls(params=params)
        return self._detector

    def io(self) -> StageIO:
        """Declare inputs and outputs.

        Returns:
            StageIO with inputs=("recon_images", "images") and
            outputs=("reconstructed_bars",).
        """
        return StageIO(
            inputs=("recon_images", "images"),
            outputs=("reconstructed_bars",),
        )

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Run note detection on all reconstructed images.

        Args:
            context: Pipeline context with "recon_images" and "images".

        Returns:
            Dict with key "reconstructed_bars" mapping to
            dict[str, list[ReconstructedBar]].
        """
        recon_images: dict[str, list[tuple[str, Any]]] = context.get(
            "recon_images", {}
        )
        images: list[PianoRollImage] = context.get("images", [])

        if not recon_images:
            logger.warning("DetectStage received no recon_images")
            return {"reconstructed_bars": {}}

        # Build bar_id -> channel_strategy lookup from original images
        channel_strategy_by_id: dict[str, str] = {
            img.bar_id: img.channel_strategy for img in images
        }
        fallback_strategy = self.config.render.channel_strategy

        detector = self._get_detector()
        detection_method = self.config.note_detection.method

        reconstructed_bars: dict[str, list[ReconstructedBar]] = {}

        # Check once whether the detector supports batched detection
        has_detect_batch = callable(getattr(detector, 'detect_batch', None))

        for vae_name, pairs in recon_images.items():
            logger.info(
                "DetectStage: detecting notes in %d bars for VAE '%s' (method=%s%s)",
                len(pairs),
                vae_name,
                detection_method,
                ", batched" if has_detect_batch else "",
            )

            bars: list[ReconstructedBar] = []

            if has_detect_batch and pairs:
                # All pairs with the same channel strategy can be batched.
                # Group by strategy to preserve correctness, then call detect_batch.
                # Collect (bar_id, tensor, strategy) triples
                triples = [
                    (bar_id, recon_tensor, channel_strategy_by_id.get(bar_id, fallback_strategy))
                    for bar_id, recon_tensor in pairs
                ]

                # Process per-strategy groups
                # We want to preserve the original order, so group by strategy
                # but keep track of original positions.
                strategy_groups: dict[str, list[tuple[int, str, Any]]] = {}
                for orig_idx, (bar_id, recon_tensor, strategy) in enumerate(triples):
                    strategy_groups.setdefault(strategy, []).append(
                        (orig_idx, bar_id, recon_tensor)
                    )

                # Allocate result list by original index
                note_results: list[list[MidiNote] | None] = [None] * len(pairs)
                for strategy, group in strategy_groups.items():
                    orig_indices = [g[0] for g in group]
                    tensors = [g[2] for g in group]
                    try:
                        batch_notes = detector.detect_batch(tensors, strategy)
                    except Exception as exc:  # pylint: disable=broad-except
                        logger.warning(
                            "DetectStage: detect_batch failed for strategy '%s' — "
                            "falling back to per-image detection. %s",
                            strategy,
                            exc,
                        )
                        batch_notes = []
                        for tensor in tensors:
                            try:
                                batch_notes.append(detector.detect(tensor, strategy))
                            except Exception as exc2:  # pylint: disable=broad-except
                                logger.warning(
                                    "DetectStage: per-image fallback also failed. %s", exc2
                                )
                                batch_notes.append([])

                    for orig_idx, per_image_notes in zip(orig_indices, batch_notes):
                        note_results[orig_idx] = per_image_notes

                for orig_idx, (bar_id, recon_tensor) in enumerate(pairs):
                    notes = note_results[orig_idx] or []
                    bars.append(
                        ReconstructedBar(
                            bar_id=bar_id,
                            vae_name=vae_name,
                            recon_image=recon_tensor,
                            detected_notes=notes,
                            detection_method=detection_method,
                        )
                    )
            else:
                # Fallback: per-image detection
                for bar_id, recon_tensor in pairs:
                    channel_strategy = channel_strategy_by_id.get(
                        bar_id, fallback_strategy
                    )
                    try:
                        notes = detector.detect(recon_tensor, channel_strategy)
                    except Exception as exc:  # pylint: disable=broad-except
                        logger.warning(
                            "DetectStage: detection failed for bar '%s' — skipping. %s",
                            bar_id,
                            exc,
                        )
                        notes = []

                    bars.append(
                        ReconstructedBar(
                            bar_id=bar_id,
                            vae_name=vae_name,
                            recon_image=recon_tensor,
                            detected_notes=notes,
                            detection_method=detection_method,
                        )
                    )

            reconstructed_bars[vae_name] = bars
            logger.info(
                "DetectStage: produced %d ReconstructedBars for VAE '%s'",
                len(bars),
                vae_name,
            )

        return {"reconstructed_bars": reconstructed_bars}

    def cache_key(self, context: dict[str, Any]) -> str | None:
        """Compute a cache key from recon image IDs and detection config.

        Args:
            context: Current pipeline context.

        Returns:
            Hex digest string, or None if recon_images are not available.
        """
        recon_images: dict[str, list[tuple[str, Any]]] | None = context.get(
            "recon_images"
        )
        if not recon_images:
            return None

        all_ids = tuple(
            bar_id
            for vae_name in sorted(recon_images.keys())
            for bar_id, _ in recon_images[vae_name]
        )
        vae_names = tuple(sorted(recon_images.keys()))

        return compute_hash(
            all_ids,
            vae_names,
            self.config.note_detection.method,
            str(sorted(self.config.note_detection.params.items())),
        )
