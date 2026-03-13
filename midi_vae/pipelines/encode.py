"""EncodeStage: pipeline stage that encodes PianoRollImages via a frozen VAE.

This stage reads the "images" context key produced by RenderStage, batches
the images, and encodes them through each configured VAE.  Outputs are
stored under the "latents" context key — a dict keyed by VAE name.

Context inputs:
    images: list[PianoRollImage]

Context outputs:
    latents: dict[str, list[LatentEncoding]]
        Keys are VAE short names.  Each list aligns positionally with 'images'.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from midi_vae.config import ExperimentConfig
from midi_vae.data.types import LatentEncoding, PianoRollImage
from midi_vae.models.vae_wrapper import FrozenImageVAE
from midi_vae.pipelines.base import PipelineStage, StageIO, compute_hash
from midi_vae.registry import ComponentRegistry

logger = logging.getLogger(__name__)


@ComponentRegistry.register("pipeline_stage", "encode")
class EncodeStage(PipelineStage):
    """Pipeline stage that encodes piano-roll images into VAE latent vectors.

    For each VAE configured in config.vaes, images are batched and passed
    through the frozen VAE encoder.  Outputs are collected into the "latents"
    context key.

    Since VAE inference requires GPU for real workloads, this stage is designed
    to be GPU-ready: it passes each image batch to the device managed by the
    VAE wrapper.  On CPU-only environments the stage will run but will be slow.

    Args:
        config: Full experiment configuration.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)

    def io(self) -> StageIO:
        """Declare that this stage reads 'images' and produces 'latents'.

        Returns:
            StageIO with inputs=("images",) and outputs=("latents",).
        """
        return StageIO(inputs=("images",), outputs=("latents",))

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Encode all images using each configured VAE.

        Args:
            context: Pipeline context containing "images": list[PianoRollImage].

        Returns:
            Dict with key "latents" mapping to dict[str, list[LatentEncoding]].
            Inner dict keys are VAE short names.
        """
        images: list[PianoRollImage] = context.get("images", [])

        if not images:
            logger.warning("EncodeStage received no images to encode")
            return {"latents": {}}

        vae_cfgs = self.config.vaes
        if not vae_cfgs:
            logger.warning("EncodeStage: no VAEs configured")
            return {"latents": {}}

        latents: dict[str, list[LatentEncoding]] = {}

        for vae_cfg in vae_cfgs:
            vae_name = vae_cfg.name
            logger.info(
                "EncodeStage: encoding %d images with VAE '%s'",
                len(images),
                vae_name,
            )

            try:
                vae_cls = ComponentRegistry.get("vae", vae_name)
                vae: FrozenImageVAE = vae_cls(vae_cfg, device=self.config.device)
            except KeyError:
                logger.error(
                    "EncodeStage: VAE '%s' not registered — skipping", vae_name
                )
                continue

            encodings: list[LatentEncoding] = self._encode_with_vae(
                vae, images, vae_cfg.batch_size, vae_cfg.latent_type
            )
            latents[vae_name] = encodings
            logger.info(
                "EncodeStage: produced %d encodings for VAE '%s'",
                len(encodings),
                vae_name,
            )

        return {"latents": latents}

    def _encode_with_vae(
        self,
        vae: FrozenImageVAE,
        images: list[PianoRollImage],
        batch_size: int,
        latent_type: str,
    ) -> list[LatentEncoding]:
        """Encode a list of images with a single VAE in batches.

        Args:
            vae: The frozen VAE wrapper to use.
            images: Images to encode.
            batch_size: Number of images per forward pass.
            latent_type: One of 'mean', 'sample', or 'both'.

        Returns:
            List of LatentEncoding objects aligned with the input images.
        """
        encodings: list[LatentEncoding] = []

        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            # Stack into (B, 3, H, W)
            image_tensors = torch.stack([img.image for img in batch_images])

            z_mu, z_sigma = vae.encode_images(image_tensors)

            # Optionally draw a reparameterised sample
            z_sample: torch.Tensor | None = None
            if latent_type in ("sample", "both"):
                eps = torch.randn_like(z_mu)
                z_sample = z_mu + eps * z_sigma

            # Unpack batch dimension
            for i, img in enumerate(batch_images):
                encoding = LatentEncoding(
                    bar_id=img.bar_id,
                    vae_name=vae.name,
                    z_mu=z_mu[i].cpu(),
                    z_sigma=z_sigma[i].cpu(),
                    z_sample=z_sample[i].cpu() if z_sample is not None else None,
                )
                encodings.append(encoding)

        return encodings

    def cache_key(self, context: dict[str, Any]) -> str | None:
        """Compute a cache key based on render config, VAE configs, and image IDs.

        Args:
            context: Current pipeline context.

        Returns:
            Hex digest string, or None if images are not yet available.
        """
        images: list[PianoRollImage] | None = context.get("images")
        if not images:
            return None

        bar_ids = tuple(img.bar_id for img in images)
        vae_names = tuple(v.name for v in self.config.vaes)
        vae_dtypes = tuple(v.dtype for v in self.config.vaes)

        return compute_hash(
            bar_ids,
            vae_names,
            vae_dtypes,
            self.config.device,
        )
