"""Model wrappers for pretrained VAEs and sub-latent models."""

from midi_vae.models.vae_wrapper import FrozenImageVAE
from midi_vae.models.vae_registry import DiffusersVAE  # noqa: F401 — triggers registration

__all__ = [
    "FrozenImageVAE",
    "DiffusersVAE",
]
