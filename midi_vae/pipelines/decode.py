"""DecodeStage: pipeline stage that decodes LatentEncodings back to images.

This stage reads the "latents" context key produced by EncodeStage,
decodes each encoding through the corresponding frozen VAE, and outputs
a dict of reconstructed image tensors under "recon_images".

Context inputs:
    latents: dict[str, list[LatentEncoding]]

Context outputs:
    recon_images: dict[str, list[tuple[str, torch.Tensor]]]
        Keys are VAE short names.  Each value is a list of (bar_id, tensor)
        tuples where tensor has shape (3, H, W).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from midi_vae.config import ExperimentConfig
from midi_vae.data.types import LatentEncoding
from midi_vae.models.vae_wrapper import FrozenImageVAE
import midi_vae.models.vae_registry  # noqa: F401 — trigger registration of all VAEs
from midi_vae.pipelines.base import PipelineStage, StageIO, compute_hash
from midi_vae.registry import ComponentRegistry

logger = logging.getLogger(__name__)


@ComponentRegistry.register("pipeline_stage", "decode")
class DecodeStage(PipelineStage):
    """Pipeline stage that decodes latent encodings back to image tensors.

    For each VAE that appears in the "latents" context dict, the corresponding
    VAE is loaded from the registry and used to decode the stored latent vectors.
    The decoding uses z_mu by default (deterministic).

    Args:
        config: Full experiment configuration.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)

    def io(self) -> StageIO:
        """Declare that this stage reads 'latents' and produces 'recon_images'.

        Returns:
            StageIO with inputs=("latents",) and outputs=("recon_images",).
        """
        return StageIO(inputs=("latents",), outputs=("recon_images",))

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Decode all latent encodings for each VAE.

        Args:
            context: Pipeline context containing
                "latents": dict[str, list[LatentEncoding]].

        Returns:
            Dict with key "recon_images" mapping to
            dict[str, list[tuple[str, torch.Tensor]]].
        """
        latents: dict[str, list[LatentEncoding]] = context.get("latents", {})

        if not latents:
            logger.warning("DecodeStage received no latents to decode")
            return {"recon_images": {}}

        recon_images: dict[str, list[tuple[str, torch.Tensor]]] = {}

        # Build a lookup from vae_name -> VAEConfig
        vae_cfg_by_name = {v.name: v for v in self.config.vaes}

        for vae_name, encodings in latents.items():
            if not encodings:
                continue

            vae_cfg = vae_cfg_by_name.get(vae_name)
            if vae_cfg is None:
                logger.warning(
                    "DecodeStage: no VAEConfig found for '%s' — skipping", vae_name
                )
                continue

            try:
                vae_cls = ComponentRegistry.get("vae", vae_name)
                vae: FrozenImageVAE = vae_cls(vae_cfg, device=self.config.device)
            except KeyError:
                logger.error(
                    "DecodeStage: VAE '%s' not registered — skipping", vae_name
                )
                continue

            logger.info(
                "DecodeStage: decoding %d latents with VAE '%s'",
                len(encodings),
                vae_name,
            )

            decoded = self._decode_with_vae(vae, encodings, vae_cfg.batch_size)
            recon_images[vae_name] = decoded
            logger.info(
                "DecodeStage: produced %d reconstructions for VAE '%s'",
                len(decoded),
                vae_name,
            )

        return {"recon_images": recon_images}

    def _decode_with_vae(
        self,
        vae: FrozenImageVAE,
        encodings: list[LatentEncoding],
        batch_size: int,
    ) -> list[tuple[str, torch.Tensor]]:
        """Decode a list of LatentEncoding objects with a single VAE in batches.

        Decoding uses z_mu (the distribution mean) for deterministic output.
        If z_sample is present it is used instead when latent_type == 'sample'.

        Args:
            vae: The frozen VAE wrapper.
            encodings: Latent encodings to decode.
            batch_size: Number of latents per forward pass.

        Returns:
            List of (bar_id, recon_image_tensor) tuples.  Each tensor has
            shape (3, H, W) and is moved to CPU.
        """
        results: list[tuple[str, torch.Tensor]] = []

        for start in range(0, len(encodings), batch_size):
            batch = encodings[start : start + batch_size]

            # Prefer z_sample if available, fall back to z_mu
            z_list = [
                enc.z_sample if enc.z_sample is not None else enc.z_mu
                for enc in batch
            ]
            z = torch.stack(z_list)  # (B, C, H_lat, W_lat)

            recon = vae.decode_latents(z)  # (B, 3, H, W)

            for i, enc in enumerate(batch):
                results.append((enc.bar_id, recon[i].cpu()))

        return results

    def cache_key(self, context: dict[str, Any]) -> str | None:
        """Compute a cache key from the latent encoding IDs.

        Args:
            context: Current pipeline context.

        Returns:
            Hex digest string, or None if latents are not available.
        """
        latents: dict[str, list[LatentEncoding]] | None = context.get("latents")
        if not latents:
            return None

        # Flatten all bar_ids across all VAEs for hashing
        all_ids = tuple(
            enc.bar_id
            for vae_name in sorted(latents.keys())
            for enc in latents[vae_name]
        )
        vae_names = tuple(sorted(latents.keys()))

        return compute_hash(all_ids, vae_names)
