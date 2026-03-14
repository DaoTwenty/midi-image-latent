"""Evaluation metrics for reconstruction quality assessment."""

from midi_vae.metrics.base import Metric, MetricsEngine
from midi_vae.metrics.reconstruction import PixelMSE, SSIM, PSNR
from midi_vae.metrics.harmony import (
    OnsetF1,
    OnsetPrecision,
    OnsetRecall,
    NoteDensityPearson,
    PitchClassHistogramCorrelation,
    IntervalHistogramCorrelation,
)
from midi_vae.metrics.dynamics import VelocityMSE, VelocityCorrelation, VelocityHistogramKL
from midi_vae.metrics.rhythm import IOIDistributionKL, GrooveConsistency
from midi_vae.metrics.information import (
    NoteEntropyDiff,
    MutualInformationPitchTime,
    KLDivergencePitchDist,
    ActivationSparsity,
)
from midi_vae.metrics.latent_space import (
    LatentVariance,
    LatentChannelCorrelation,
    LinearProbePitchAccuracy,
    LinearProbeInstrumentAccuracy,
    SilhouetteScore,
)
from midi_vae.metrics.generative import (
    SelfSimilarityMatrix,
    TransitionEntropy,
    SequenceGrooveConsistency,
    PitchClassHistogramKL,
    BarLevelNLL,
    SequenceCoherence,
)
from midi_vae.metrics.conditioning import (
    ConditioningFidelity,
    AttributeAccuracy,
    InterpolationSmoothness,
    ConditionalPitchAlignment,
    DisentanglementScore,
)

__all__ = [
    "Metric",
    "MetricsEngine",
    # reconstruction
    "PixelMSE",
    "SSIM",
    "PSNR",
    # harmony
    "OnsetF1",
    "OnsetPrecision",
    "OnsetRecall",
    "NoteDensityPearson",
    "PitchClassHistogramCorrelation",
    "IntervalHistogramCorrelation",
    # dynamics
    "VelocityMSE",
    "VelocityCorrelation",
    "VelocityHistogramKL",
    # rhythm
    "IOIDistributionKL",
    "GrooveConsistency",
    # information
    "NoteEntropyDiff",
    "MutualInformationPitchTime",
    "KLDivergencePitchDist",
    "ActivationSparsity",
    # latent space
    "LatentVariance",
    "LatentChannelCorrelation",
    "LinearProbePitchAccuracy",
    "LinearProbeInstrumentAccuracy",
    "SilhouetteScore",
    # generative
    "SelfSimilarityMatrix",
    "TransitionEntropy",
    "SequenceGrooveConsistency",
    "PitchClassHistogramKL",
    "BarLevelNLL",
    "SequenceCoherence",
    # conditioning
    "ConditioningFidelity",
    "AttributeAccuracy",
    "InterpolationSmoothness",
    "ConditionalPitchAlignment",
    "DisentanglementScore",
]
