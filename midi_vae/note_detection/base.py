"""Abstract base class for note detection algorithms.

Note detectors convert continuous-valued reconstructed images back into
discrete MIDI note events (pitch, onset, offset, velocity).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from midi_vae.data.types import MidiNote, BarData


class NoteDetector(ABC):
    """Abstract base for note detection from reconstructed piano-roll images.

    Detectors convert a continuous image tensor back to a list of MidiNote objects.
    Some detectors require fitting on validation data (e.g., adaptive thresholds, HMM).
    """

    def __init__(self, params: dict | None = None) -> None:
        """Initialize the note detector.

        Args:
            params: Detection-specific parameters from config.
        """
        self.params = params or {}

    @abstractmethod
    def detect(
        self,
        recon_image: torch.Tensor,
        channel_strategy: str,
    ) -> list[MidiNote]:
        """Detect MIDI notes from a reconstructed piano-roll image.

        Args:
            recon_image: Continuous-valued image, shape (3, H, W).
            channel_strategy: The channel strategy used for rendering
                ('velocity_only', 'vo_split', 'vos').

        Returns:
            List of detected MidiNote objects.
        """
        ...

    def fit(self, validation_bars: list[tuple[BarData, torch.Tensor]]) -> None:
        """Fit detector parameters on validation data.

        Override for detectors that need fitting (adaptive thresholds, HMM, GMM).
        Default implementation is a no-op.

        Args:
            validation_bars: List of (ground_truth_bar, reconstructed_image) pairs.
        """
        pass

    @property
    @abstractmethod
    def needs_fitting(self) -> bool:
        """Whether this detector requires fitting before use."""
        ...
