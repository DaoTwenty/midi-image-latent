"""Channel strategy implementations for piano-roll rendering.

Each strategy converts a BarData object into a (3, H, W) image tensor
normalized to [-1, 1]. Three strategies are provided:

  - velocity_only: All three channels carry the velocity matrix.
  - vo_split:      R=velocity, G=onset_mask, B=zeros.
  - vos:           R=velocity, G=onset_mask, B=sustain_mask.

All strategies are registered in the ComponentRegistry and can be
retrieved by name.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import torch

from midi_vae.data.types import BarData
from midi_vae.registry import ComponentRegistry

logger = logging.getLogger(__name__)

# Velocity values are integers in [0, 127].  We map them to [0, 1] first,
# then shift to [-1, 1] using the formula  x_norm = x / 127 * 2 - 1.
_VELOCITY_MAX: float = 127.0


def _velocity_to_unit(arr: np.ndarray) -> np.ndarray:
    """Map a velocity array from [0, 127] to [0.0, 1.0].

    Args:
        arr: Numpy array with integer velocity values in [0, 127].

    Returns:
        Float array in [0.0, 1.0].
    """
    return arr.astype(np.float32) / _VELOCITY_MAX


def _unit_to_range(arr: np.ndarray, low: float = -1.0, high: float = 1.0) -> np.ndarray:
    """Linearly rescale a [0, 1] float array to [low, high].

    Args:
        arr: Float array in [0.0, 1.0].
        low: Lower bound of the output range.
        high: Upper bound of the output range.

    Returns:
        Rescaled float array in [low, high].
    """
    return arr * (high - low) + low


def _mask_to_range(arr: np.ndarray, low: float = -1.0, high: float = 1.0) -> np.ndarray:
    """Map a binary {0, 1} mask to [low, high].

    Args:
        arr: Binary numpy array with values 0 or 1.
        low: Value assigned to 0.
        high: Value assigned to 1.

    Returns:
        Float array in {low, high}.
    """
    return arr.astype(np.float32) * (high - low) + low


def _piano_roll_to_tensor(
    channels: list[np.ndarray],
    pitch_axis: str = "height",
) -> torch.Tensor:
    """Stack channel arrays into a (3, H, W) tensor.

    Each channel array must have shape (128, T) where 128 is the pitch
    dimension and T is the time dimension.

    Args:
        channels: List of exactly 3 numpy arrays each shaped (128, T).
        pitch_axis: "height" places pitch along axis 0 (rows), "width"
                    transposes so pitch runs along columns.

    Returns:
        torch.Tensor of shape (3, H, W) where H and W depend on pitch_axis.
    """
    assert len(channels) == 3, f"Expected 3 channels, got {len(channels)}"

    stacked = np.stack(channels, axis=0)  # (3, 128, T)

    if pitch_axis == "width":
        # Transpose pitch and time axes: (3, 128, T) -> (3, T, 128)
        stacked = stacked.transpose(0, 2, 1)

    return torch.from_numpy(stacked)


class ChannelStrategy(ABC):
    """Abstract base class for channel rendering strategies.

    Subclasses convert a BarData into a normalized (3, H, W) image tensor.
    The output is always normalized to [-1, 1].
    """

    def __init__(
        self,
        pitch_axis: str = "height",
        normalize_low: float = -1.0,
        normalize_high: float = 1.0,
    ) -> None:
        """Initialize the channel strategy.

        Args:
            pitch_axis: "height" places pitch on the vertical axis (rows),
                        "width" places it on the horizontal axis (columns).
            normalize_low: Lower bound of the normalized output range.
            normalize_high: Upper bound of the normalized output range.
        """
        self.pitch_axis = pitch_axis
        self.normalize_low = normalize_low
        self.normalize_high = normalize_high

    @abstractmethod
    def render(self, bar: BarData) -> torch.Tensor:
        """Render a bar of MIDI data into a (3, H, W) image tensor.

        Args:
            bar: A BarData instance containing piano_roll, onset_mask,
                 and sustain_mask arrays of shape (128, T).

        Returns:
            torch.Tensor of shape (3, H, W) with values in [normalize_low,
            normalize_high].
        """
        ...

    def _finalize(self, channels: list[np.ndarray]) -> torch.Tensor:
        """Convert channel list to a tensor respecting the pitch_axis setting.

        Args:
            channels: List of 3 numpy arrays each shaped (128, T).

        Returns:
            torch.Tensor of shape (3, H, W).
        """
        return _piano_roll_to_tensor(channels, pitch_axis=self.pitch_axis)


@ComponentRegistry.register("channel_strategy", "velocity_only")
class VelocityOnlyStrategy(ChannelStrategy):
    """All three channels carry the velocity matrix.

    Channel mapping:
        R = velocity (normalized to [-1, 1])
        G = velocity (normalized to [-1, 1])
        B = velocity (normalized to [-1, 1])

    This strategy produces a grayscale-equivalent image where brightness
    encodes note velocity.
    """

    def render(self, bar: BarData) -> torch.Tensor:
        """Render bar using velocity-only channel strategy.

        Args:
            bar: BarData with piano_roll array of shape (128, T).

        Returns:
            torch.Tensor of shape (3, H, W) where all channels are
            the normalized velocity matrix.
        """
        vel = _unit_to_range(
            _velocity_to_unit(bar.piano_roll),
            low=self.normalize_low,
            high=self.normalize_high,
        )
        return self._finalize([vel, vel, vel])


@ComponentRegistry.register("channel_strategy", "vo_split")
class VOSplitStrategy(ChannelStrategy):
    """Velocity and onset split across two channels; third channel is zeros.

    Channel mapping:
        R = velocity (normalized to [-1, 1])
        G = onset_mask (binary, mapped to {-1, 1})
        B = zeros (constant -1 when normalize_low=-1)

    This strategy separates the "what" (velocity) from the "when" (onset),
    leaving the blue channel empty for potential future use.
    """

    def render(self, bar: BarData) -> torch.Tensor:
        """Render bar using velocity+onset split strategy.

        Args:
            bar: BarData with piano_roll and onset_mask arrays of shape (128, T).

        Returns:
            torch.Tensor of shape (3, H, W) with R=velocity, G=onset,
            B=zeros (normalize_low value).
        """
        vel = _unit_to_range(
            _velocity_to_unit(bar.piano_roll),
            low=self.normalize_low,
            high=self.normalize_high,
        )
        onset = _mask_to_range(
            bar.onset_mask,
            low=self.normalize_low,
            high=self.normalize_high,
        )
        zeros = np.full(bar.piano_roll.shape, self.normalize_low, dtype=np.float32)
        return self._finalize([vel, onset, zeros])


@ComponentRegistry.register("channel_strategy", "vos")
class VOSStrategy(ChannelStrategy):
    """Velocity, onset, and sustain across all three channels.

    Channel mapping:
        R = velocity (normalized to [-1, 1])
        G = onset_mask (binary, mapped to {-1, 1})
        B = sustain_mask (binary, mapped to {-1, 1})

    This strategy fully encodes the musical content: attack (onset),
    continuation (sustain), and intensity (velocity) in separate channels.
    """

    def render(self, bar: BarData) -> torch.Tensor:
        """Render bar using velocity+onset+sustain strategy.

        Args:
            bar: BarData with piano_roll, onset_mask, and sustain_mask
                 arrays of shape (128, T).

        Returns:
            torch.Tensor of shape (3, H, W) with R=velocity, G=onset,
            B=sustain.
        """
        vel = _unit_to_range(
            _velocity_to_unit(bar.piano_roll),
            low=self.normalize_low,
            high=self.normalize_high,
        )
        onset = _mask_to_range(
            bar.onset_mask,
            low=self.normalize_low,
            high=self.normalize_high,
        )
        sustain = _mask_to_range(
            bar.sustain_mask,
            low=self.normalize_low,
            high=self.normalize_high,
        )
        return self._finalize([vel, onset, sustain])


def build_strategy(
    name: str,
    pitch_axis: str = "height",
    normalize_low: float = -1.0,
    normalize_high: float = 1.0,
) -> ChannelStrategy:
    """Instantiate a registered channel strategy by name.

    Args:
        name: Registered strategy name ('velocity_only', 'vo_split', 'vos').
        pitch_axis: "height" or "width" for pitch axis orientation.
        normalize_low: Lower bound of the output normalization range.
        normalize_high: Upper bound of the output normalization range.

    Returns:
        An initialized ChannelStrategy instance.

    Raises:
        KeyError: If name is not a registered channel strategy.
    """
    cls = ComponentRegistry.get("channel_strategy", name)
    return cls(
        pitch_axis=pitch_axis,
        normalize_low=normalize_low,
        normalize_high=normalize_high,
    )
