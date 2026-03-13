"""Sub-latent dimensionality reduction models.

All concrete implementations are registered via ``@ComponentRegistry.register``
and exported here for convenient direct import.
"""

from midi_vae.models.sublatent.base import SubLatentModel
from midi_vae.models.sublatent.conditioning import FeatureConditioner
from midi_vae.models.sublatent.mlp import MLPSubLatent
from midi_vae.models.sublatent.pca import PCASubLatent
from midi_vae.models.sublatent.sub_vae import SubVAE

__all__ = [
    "SubLatentModel",
    "PCASubLatent",
    "MLPSubLatent",
    "SubVAE",
    "FeatureConditioner",
]
