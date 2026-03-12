"""Threshold-based note detection methods.

Provides GlobalThresholdDetector and related threshold variants for
converting continuous-valued reconstructed piano-roll images back to
discrete MIDI note events.
"""

from __future__ import annotations

import torch
import numpy as np
from scipy import ndimage

from midi_vae.data.types import MidiNote, BarData
from midi_vae.note_detection.base import NoteDetector
from midi_vae.registry import ComponentRegistry


def _extract_velocity_channel(
    recon_image: torch.Tensor,
    channel_strategy: str,
) -> torch.Tensor:
    """Extract the velocity channel from a reconstructed image.

    For all channel strategies, the R (index 0) channel encodes velocity.
    The image is assumed to be in [-1, 1]; we remap to [0, 1].

    Args:
        recon_image: Continuous-valued image tensor of shape (3, H, W).
        channel_strategy: One of 'velocity_only', 'vo_split', 'vos'.

    Returns:
        2-D velocity map of shape (H, W) with values in [0, 1].
    """
    # Channel 0 is always velocity across all strategies
    vel = recon_image[0]  # (H, W)
    # Remap from [-1, 1] to [0, 1]
    vel = (vel + 1.0) / 2.0
    return vel.clamp(0.0, 1.0)


def _connected_components_1d(binary_row: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous runs of True values in a 1-D boolean array.

    Args:
        binary_row: 1-D boolean array of length T.

    Returns:
        List of (start_inclusive, end_exclusive) tuples for each run.
    """
    labeled, num_features = ndimage.label(binary_row)
    segments = []
    for label_id in range(1, num_features + 1):
        positions = np.where(labeled == label_id)[0]
        segments.append((int(positions[0]), int(positions[-1]) + 1))
    return segments


@ComponentRegistry.register('note_detector', 'global_threshold')
class GlobalThresholdDetector(NoteDetector):
    """Binary threshold note detector applied uniformly across all pitches.

    Applies a single threshold tau to the velocity channel of the
    reconstructed image. Connected components along the time axis
    within each pitch row are treated as individual note events.

    Parameters (from config['params']):
        threshold (float): Binary activation threshold in [0, 1]. Default 0.5.
        min_duration_steps (int): Minimum run length to be counted as a
            note. Shorter runs are discarded. Default 1.
        pitch_axis (str): 'height' (pitch = image row index) or 'width'
            (pitch = image column index). Default 'height'.
        num_pitches (int): Number of MIDI pitches represented in the image.
            Rows/columns are mapped linearly to this range. Default 128.
        pitch_offset (int): Lowest MIDI pitch number in the image. Default 0.
    """

    def __init__(self, params: dict | None = None) -> None:
        """Initialize the GlobalThresholdDetector.

        Args:
            params: Optional dict with keys 'threshold', 'min_duration_steps',
                'pitch_axis', 'num_pitches', 'pitch_offset'.
        """
        super().__init__(params)
        self._threshold: float = float(self.params.get('threshold', 0.5))
        self._min_dur: int = int(self.params.get('min_duration_steps', 1))
        self._pitch_axis: str = str(self.params.get('pitch_axis', 'height'))
        self._num_pitches: int = int(self.params.get('num_pitches', 128))
        self._pitch_offset: int = int(self.params.get('pitch_offset', 0))

    @property
    def needs_fitting(self) -> bool:
        """GlobalThreshold requires no fitting — returns False."""
        return False

    def detect(
        self,
        recon_image: torch.Tensor,
        channel_strategy: str,
    ) -> list[MidiNote]:
        """Detect MIDI notes from a reconstructed piano-roll image.

        Steps:
          1. Extract the velocity channel (R channel) and remap to [0, 1].
          2. Apply the binary threshold tau.
          3. For each pitch row, find contiguous True-regions along time.
          4. Each region becomes one MidiNote with mean activation scaled
             to a MIDI velocity (0–127).

        Args:
            recon_image: Continuous-valued image tensor of shape (3, H, W).
                Values are expected in [-1, 1].
            channel_strategy: The channel strategy used for rendering
                ('velocity_only', 'vo_split', 'vos').

        Returns:
            List of detected MidiNote objects sorted by onset_step.
        """
        vel_map = _extract_velocity_channel(recon_image, channel_strategy)
        # Shape: (H, W); convert to numpy for connected-component logic
        vel_np = vel_map.cpu().float().numpy()

        binary = (vel_np >= self._threshold)

        H, W = binary.shape

        if self._pitch_axis == 'height':
            # Each row corresponds to a pitch; time runs along columns
            num_rows = H
            num_time_steps = W
            def get_row(row_idx: int) -> np.ndarray:
                return binary[row_idx, :]

            def get_vel_row(row_idx: int) -> np.ndarray:
                return vel_np[row_idx, :]
        else:
            # pitch_axis == 'width': each column is a pitch; time along rows
            num_rows = W
            num_time_steps = H
            def get_row(row_idx: int) -> np.ndarray:
                return binary[:, row_idx]

            def get_vel_row(row_idx: int) -> np.ndarray:
                return vel_np[:, row_idx]

        notes: list[MidiNote] = []

        for row_idx in range(num_rows):
            # Map image row to MIDI pitch
            pitch = self._row_to_pitch(row_idx, num_rows)
            if pitch < 0 or pitch > 127:
                continue

            row = get_row(row_idx)
            vel_row = get_vel_row(row_idx)
            segments = _connected_components_1d(row)

            for onset, offset in segments:
                duration = offset - onset
                if duration < self._min_dur:
                    continue
                # Mean activation within the note region → MIDI velocity
                mean_activation = float(vel_row[onset:offset].mean())
                midi_velocity = max(1, min(127, round(mean_activation * 127)))
                try:
                    note = MidiNote(
                        pitch=pitch,
                        onset_step=onset,
                        offset_step=offset,
                        velocity=midi_velocity,
                    )
                    notes.append(note)
                except ValueError:
                    # Malformed note (e.g., zero-duration after rounding) — skip
                    continue

        notes.sort(key=lambda n: (n.onset_step, n.pitch))
        return notes

    def _row_to_pitch(self, row_idx: int, num_rows: int) -> int:
        """Map an image row index to a MIDI pitch number.

        Rows are mapped linearly across the self._num_pitches range
        starting at self._pitch_offset.

        Args:
            row_idx: Zero-based index into the pitch axis of the image.
            num_rows: Total number of rows along the pitch axis.

        Returns:
            Integer MIDI pitch in [0, 127].
        """
        # Linear interpolation: row 0 → pitch_offset, row num_rows-1 → pitch_offset + num_pitches - 1
        if num_rows == 1:
            return self._pitch_offset
        pitch = self._pitch_offset + round(
            row_idx * (self._num_pitches - 1) / (num_rows - 1)
        )
        return int(pitch)
