"""RenderStage: pipeline stage that converts BarData into PianoRollImage tensors.

This stage sits immediately after IngestStage in the pipeline.  It reads
the "bars" context key produced by IngestStage, renders each bar using the
configured channel strategy, applies resize/normalize transforms, and
outputs "images" — a list of PianoRollImage objects.

Context inputs:
    bars: list[BarData]

Context outputs:
    images: list[PianoRollImage]
"""

from __future__ import annotations

import logging
from typing import Any

from midi_vae.config import ExperimentConfig
from midi_vae.data.rendering import build_strategy, ChannelStrategy
from midi_vae.data.transforms import ResizeTransform
from midi_vae.data.types import BarData, PianoRollImage
from midi_vae.pipelines.base import PipelineStage, StageIO, compute_hash

logger = logging.getLogger(__name__)


class RenderStage(PipelineStage):
    """Pipeline stage that renders BarData objects into (3, H, W) image tensors.

    Reads the "bars" list from context, applies the configured channel
    strategy and resize transform, and returns all rendered images under
    the context key "images".

    The rendering is deterministic: same config + same bars always produces
    the same tensors.

    Args:
        config: Full experiment configuration.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)

        render_cfg = config.render
        data_cfg = config.data

        self._strategy: ChannelStrategy = build_strategy(
            name=render_cfg.channel_strategy,
            pitch_axis=render_cfg.pitch_axis,
            normalize_low=render_cfg.normalize_range[0],
            normalize_high=render_cfg.normalize_range[1],
        )

        self._resize = ResizeTransform(
            target_resolution=tuple(data_cfg.target_resolution),
            method=render_cfg.resize_method,
        )

    def io(self) -> StageIO:
        """Declare that this stage reads 'bars' and produces 'images'.

        Returns:
            StageIO with inputs=("bars",) and outputs=("images",).
        """
        return StageIO(inputs=("bars",), outputs=("images",))

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Render all bars from context into PianoRollImage objects.

        Args:
            context: Pipeline context containing "bars": list[BarData].

        Returns:
            Dict with key "images" mapping to list[PianoRollImage].
        """
        bars: list[BarData] = context.get("bars", [])

        if not bars:
            logger.warning("RenderStage received no bars to render")
            return {"images": []}

        render_cfg = self.config.render
        data_cfg = self.config.data

        logger.info("RenderStage: rendering %d bars with strategy '%s'", len(bars), render_cfg.channel_strategy)

        images: list[PianoRollImage] = []
        for bar in bars:
            try:
                tensor = self._strategy.render(bar)
                tensor = self._resize(tensor)
                image = PianoRollImage(
                    bar_id=bar.bar_id,
                    image=tensor,
                    channel_strategy=render_cfg.channel_strategy,
                    resolution=tuple(tensor.shape[1:]),
                    pitch_axis=render_cfg.pitch_axis,
                )
                images.append(image)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "RenderStage failed to render bar '%s', skipping. Reason: %s",
                    bar.bar_id,
                    exc,
                )

        logger.info("RenderStage: produced %d images", len(images))
        return {"images": images}

    def cache_key(self, context: dict[str, Any]) -> str | None:
        """Compute a cache key based on render config and the bar IDs in context.

        Args:
            context: Current pipeline context.

        Returns:
            Hex digest string, or None if bars are not yet available.
        """
        bars: list[BarData] | None = context.get("bars")
        if not bars:
            return None

        bar_ids = tuple(b.bar_id for b in bars)

        return compute_hash(
            self.config.render.channel_strategy,
            self.config.render.pitch_axis,
            self.config.render.normalize_range,
            self.config.render.resize_method,
            self.config.data.target_resolution,
            bar_ids,
        )
