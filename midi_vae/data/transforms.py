"""Image transform pipeline for piano-roll tensors.

Transforms operate on (3, H, W) float tensors and can be composed
sequentially. Each transform is a callable class with a documented
configuration interface.

Available transforms:
    ResizeTransform   — bilinear/nearest resize to a target resolution.
    NormalizeTransform — linearly rescale values to a target range.
    PadTransform      — zero-pad (or constant-pad) to a target square size.
    PitchAxisFlip     — flip the pitch axis (vertical flip) for display.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ResizeTransform:
    """Resize a (3, H, W) tensor to a fixed target resolution.

    Uses torch.nn.functional.interpolate under the hood, so the same
    method (bilinear, nearest, bicubic) is applied regardless of
    whether we are upsampling or downsampling.
    """

    _SUPPORTED_MODES = {"bilinear", "nearest", "bicubic", "area"}

    def __init__(
        self,
        target_resolution: tuple[int, int],
        method: str = "bilinear",
    ) -> None:
        """Initialize the resize transform.

        Args:
            target_resolution: (height, width) in pixels of the output tensor.
            method: Interpolation mode.  One of 'bilinear', 'nearest',
                    'bicubic', or 'area'.

        Raises:
            ValueError: If method is not supported or target_resolution
                        contains non-positive values.
        """
        if method not in self._SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported resize method '{method}'. "
                f"Choose from {sorted(self._SUPPORTED_MODES)}"
            )
        if target_resolution[0] <= 0 or target_resolution[1] <= 0:
            raise ValueError(
                f"target_resolution must be positive, got {target_resolution}"
            )
        self.target_resolution = target_resolution
        self.method = method

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the resize transform.

        Args:
            tensor: Input tensor of shape (3, H, W).

        Returns:
            Resized tensor of shape (3, target_H, target_W).
        """
        if tensor.ndim != 3:
            raise ValueError(
                f"Expected a (3, H, W) tensor, got shape {tuple(tensor.shape)}"
            )

        # interpolate requires a batch dimension: (1, 3, H, W)
        x = tensor.unsqueeze(0).float()

        align_corners: bool | None = None
        if self.method in {"bilinear", "bicubic"}:
            align_corners = False

        x = F.interpolate(
            x,
            size=self.target_resolution,
            mode=self.method,
            align_corners=align_corners,
        )
        return x.squeeze(0)

    def __repr__(self) -> str:
        return (
            f"ResizeTransform(target_resolution={self.target_resolution}, "
            f"method='{self.method}')"
        )


class NormalizeTransform:
    """Linearly rescale tensor values from one range to another.

    Given a tensor assumed to lie in [src_low, src_high], maps values to
    [dst_low, dst_high] using an affine transformation.  The default
    maps any input linearly to [-1, 1].
    """

    def __init__(
        self,
        dst_low: float = -1.0,
        dst_high: float = 1.0,
        src_low: float = 0.0,
        src_high: float = 1.0,
    ) -> None:
        """Initialize the normalization transform.

        Args:
            dst_low: Lower bound of the output range.
            dst_high: Upper bound of the output range.
            src_low: Lower bound of the assumed input range.
            src_high: Upper bound of the assumed input range.

        Raises:
            ValueError: If dst_low >= dst_high or src_low >= src_high.
        """
        if dst_low >= dst_high:
            raise ValueError(
                f"dst_low ({dst_low}) must be < dst_high ({dst_high})"
            )
        if src_low >= src_high:
            raise ValueError(
                f"src_low ({src_low}) must be < src_high ({src_high})"
            )
        self.dst_low = dst_low
        self.dst_high = dst_high
        self.src_low = src_low
        self.src_high = src_high

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the normalization transform.

        Args:
            tensor: Input tensor of arbitrary shape.

        Returns:
            Tensor rescaled to [dst_low, dst_high].
        """
        # Normalize to [0, 1] from src range, then to dst range.
        src_span = self.src_high - self.src_low
        dst_span = self.dst_high - self.dst_low

        tensor = tensor.float()
        normalized = (tensor - self.src_low) / src_span  # [0, 1]
        return normalized * dst_span + self.dst_low

    def __repr__(self) -> str:
        return (
            f"NormalizeTransform(dst=[{self.dst_low}, {self.dst_high}], "
            f"src=[{self.src_low}, {self.src_high}])"
        )


class PadTransform:
    """Pad a (3, H, W) tensor to a target square size.

    Padding is added symmetrically on each axis; if the difference is odd,
    the extra pixel goes on the right/bottom.  The pad value defaults to
    the minimum value of the target normalization range (-1 for [-1, 1]).
    """

    def __init__(
        self,
        target_size: int,
        pad_value: float = -1.0,
    ) -> None:
        """Initialize the pad transform.

        Args:
            target_size: Output size on both height and width dimensions.
            pad_value: Constant fill value for padded pixels.  Should match
                       the background level of the normalization range.

        Raises:
            ValueError: If target_size is not positive.
        """
        if target_size <= 0:
            raise ValueError(f"target_size must be positive, got {target_size}")
        self.target_size = target_size
        self.pad_value = pad_value

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the padding transform.

        Args:
            tensor: Input tensor of shape (3, H, W).

        Returns:
            Padded tensor of shape (3, target_size, target_size).

        Raises:
            ValueError: If the input tensor already exceeds target_size on
                        either spatial dimension.
        """
        if tensor.ndim != 3:
            raise ValueError(
                f"Expected a (3, H, W) tensor, got shape {tuple(tensor.shape)}"
            )
        _, h, w = tensor.shape

        if h > self.target_size or w > self.target_size:
            raise ValueError(
                f"Input ({h}x{w}) exceeds target_size={self.target_size}. "
                f"Use ResizeTransform first."
            )

        pad_h = self.target_size - h
        pad_w = self.target_size - w

        # F.pad uses (left, right, top, bottom) order for 2-D spatial padding
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return F.pad(
            tensor,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=self.pad_value,
        )

    def __repr__(self) -> str:
        return (
            f"PadTransform(target_size={self.target_size}, "
            f"pad_value={self.pad_value})"
        )


class PitchAxisFlip:
    """Flip the pitch axis of a piano-roll image tensor.

    In a standard piano-roll the lowest pitches sit at the bottom.
    After rendering with pitch_axis='height', low pitches are in the
    last rows (high row index).  Flipping produces an image where low
    pitches are at the top, matching a top-down sheet-music convention.
    """

    def __init__(self, pitch_axis: str = "height") -> None:
        """Initialize the pitch-axis flip transform.

        Args:
            pitch_axis: The axis along which pitch is encoded.
                        "height" flips vertically (dim 1 of a 3-D tensor),
                        "width" flips horizontally (dim 2).
        """
        if pitch_axis not in {"height", "width"}:
            raise ValueError(
                f"pitch_axis must be 'height' or 'width', got '{pitch_axis}'"
            )
        self.pitch_axis = pitch_axis

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the flip.

        Args:
            tensor: Input tensor of shape (3, H, W).

        Returns:
            Flipped tensor of the same shape.
        """
        if tensor.ndim != 3:
            raise ValueError(
                f"Expected a (3, H, W) tensor, got shape {tuple(tensor.shape)}"
            )
        flip_dim = 1 if self.pitch_axis == "height" else 2
        return tensor.flip(dims=[flip_dim])

    def __repr__(self) -> str:
        return f"PitchAxisFlip(pitch_axis='{self.pitch_axis}')"


class Compose:
    """Sequentially apply a list of transforms.

    Each transform must be callable and accept / return a torch.Tensor.
    """

    def __init__(self, transforms: Sequence) -> None:
        """Initialize the composition.

        Args:
            transforms: Ordered list of transform callables.
        """
        self.transforms = list(transforms)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply all transforms in sequence.

        Args:
            tensor: Input tensor.

        Returns:
            Transformed tensor.
        """
        for t in self.transforms:
            tensor = t(tensor)
        return tensor

    def __repr__(self) -> str:
        transform_reprs = "\n  ".join(repr(t) for t in self.transforms)
        return f"Compose([\n  {transform_reprs}\n])"
