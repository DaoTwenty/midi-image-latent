"""Tests for piano-roll rendering strategies in midi_vae/data/rendering.py.

Tests verify that channel strategy implementations produce correctly shaped
and normalized tensors from BarData inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from midi_vae.data.types import BarData

# ---------------------------------------------------------------------------
# Skip guard: skip entire module if rendering.py not yet implemented
# ---------------------------------------------------------------------------

rendering_available = False
try:
    from midi_vae.data import rendering as _rendering_module  # noqa: F401
    rendering_available = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not rendering_available,
    reason="midi_vae.data.rendering not yet implemented (BRAVO Sprint pending)",
)


# ---------------------------------------------------------------------------
# Lazy imports after skip guard
# ---------------------------------------------------------------------------


def _get_strategy(name: str):
    """Retrieve a channel strategy class from the registry."""
    from midi_vae.registry import ComponentRegistry
    return ComponentRegistry.get("channel_strategy", name)


def _make_strategy(name: str):
    """Instantiate a channel strategy with default settings."""
    cls = _get_strategy(name)
    return cls()


# ---------------------------------------------------------------------------
# VelocityOnlyStrategy tests
# ---------------------------------------------------------------------------


class TestVelocityOnlyStrategy:
    """Tests for the velocity_only channel strategy."""

    def test_output_is_tensor(self, synthetic_bar: BarData) -> None:
        """render() returns a torch.Tensor."""
        strategy = _make_strategy("velocity_only")
        result = strategy.render(synthetic_bar)
        assert isinstance(result, torch.Tensor)

    def test_output_shape(self, synthetic_bar: BarData) -> None:
        """Output tensor has shape (3, H, W)."""
        strategy = _make_strategy("velocity_only")
        result = strategy.render(synthetic_bar)
        assert result.ndim == 3
        assert result.shape[0] == 3

    def test_output_normalized_range(self, synthetic_bar: BarData) -> None:
        """Output tensor values are in [-1, 1]."""
        strategy = _make_strategy("velocity_only")
        result = strategy.render(synthetic_bar)
        assert result.min() >= -1.0 - 1e-5
        assert result.max() <= 1.0 + 1e-5

    def test_all_channels_equal(self, synthetic_bar: BarData) -> None:
        """For velocity_only, all three channels should be identical."""
        strategy = _make_strategy("velocity_only")
        result = strategy.render(synthetic_bar)
        assert torch.allclose(result[0], result[1])
        assert torch.allclose(result[0], result[2])

    def test_silent_bar_is_minimum(self) -> None:
        """A bar with no notes renders at minimum value."""
        silent_bar = BarData(
            bar_id="silent_0",
            song_id="silent",
            instrument="piano",
            program_number=0,
            piano_roll=np.zeros((128, 96), dtype=np.float32),
            onset_mask=np.zeros((128, 96), dtype=np.float32),
            sustain_mask=np.zeros((128, 96), dtype=np.float32),
            tempo=120.0,
            time_signature=(4, 4),
            metadata={},
        )
        strategy = _make_strategy("velocity_only")
        result = strategy.render(silent_bar)
        # Silent bar should have no positive activations
        assert result.max() <= 0.0 + 1e-5


# ---------------------------------------------------------------------------
# VOSplitStrategy tests
# ---------------------------------------------------------------------------


class TestVOSplitStrategy:
    """Tests for the vo_split channel strategy."""

    def test_output_shape(self, synthetic_bar: BarData) -> None:
        """Output tensor has shape (3, H, W)."""
        strategy = _make_strategy("vo_split")
        result = strategy.render(synthetic_bar)
        assert result.ndim == 3
        assert result.shape[0] == 3

    def test_output_normalized_range(self, synthetic_bar: BarData) -> None:
        """Output tensor is normalized to [-1, 1]."""
        strategy = _make_strategy("vo_split")
        result = strategy.render(synthetic_bar)
        assert result.min() >= -1.0 - 1e-5
        assert result.max() <= 1.0 + 1e-5

    def test_green_channel_encodes_onset(self, synthetic_bar: BarData) -> None:
        """G channel (index 1) contains onset information."""
        strategy = _make_strategy("vo_split")
        result = strategy.render(synthetic_bar)
        g_channel = result[1]
        # There are onsets in synthetic_bar, so G channel should have some signal
        assert g_channel.max() > g_channel.min()

    def test_blue_channel_is_zero(self, synthetic_bar: BarData) -> None:
        """B channel (index 2) should be at minimum value (zeros mapped to -1)."""
        strategy = _make_strategy("vo_split")
        result = strategy.render(synthetic_bar)
        b_channel = result[2]
        # B channel should be constant (all zeros mapped to normalize_low)
        assert b_channel.max() == pytest.approx(b_channel.min(), abs=1e-5)


# ---------------------------------------------------------------------------
# VOSStrategy tests
# ---------------------------------------------------------------------------


class TestVOSStrategy:
    """Tests for the vos (velocity/onset/sustain) channel strategy."""

    def test_output_shape(self, synthetic_bar: BarData) -> None:
        """Output tensor has shape (3, H, W)."""
        strategy = _make_strategy("vos")
        result = strategy.render(synthetic_bar)
        assert result.ndim == 3
        assert result.shape[0] == 3

    def test_output_normalized_range(self, synthetic_bar: BarData) -> None:
        """Output tensor is normalized to [-1, 1]."""
        strategy = _make_strategy("vos")
        result = strategy.render(synthetic_bar)
        assert result.min() >= -1.0 - 1e-5
        assert result.max() <= 1.0 + 1e-5

    def test_green_channel_encodes_onset(self, synthetic_bar: BarData) -> None:
        """G channel (index 1) uses onset_mask information."""
        strategy = _make_strategy("vos")
        result = strategy.render(synthetic_bar)
        g_channel = result[1]
        assert g_channel.max() > g_channel.min()

    def test_blue_channel_encodes_sustain(self, synthetic_bar: BarData) -> None:
        """B channel (index 2) uses sustain_mask information."""
        strategy = _make_strategy("vos")
        result = strategy.render(synthetic_bar)
        b_channel = result[2]
        assert b_channel.max() > b_channel.min()

    def test_channels_differ(self, synthetic_bar: BarData) -> None:
        """All three channels should have different content for VOS."""
        strategy = _make_strategy("vos")
        result = strategy.render(synthetic_bar)
        # At least G and B should differ since onset != sustain
        assert not torch.allclose(result[1], result[2])
