"""Morphological post-processing note detector.

Applies binary erosion and dilation (opening/closing) to a thresholded
piano-roll activation map before extracting note events. This removes
isolated noise pixels (erosion) and fills small gaps in sustained notes
(dilation).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

import torch

from midi_vae.data.types import MidiNote, BarData
from midi_vae.note_detection.base import NoteDetector
from midi_vae.note_detection.threshold import (
    _extract_velocity_channel,
    _connected_components_1d,
    _row_to_pitch_generic,
)
from midi_vae.registry import ComponentRegistry


@ComponentRegistry.register('note_detector', 'morphological')
class MorphologicalDetector(NoteDetector):
    """Note detector using morphological opening/closing as post-processing.

    Pipeline:
      1. Extract velocity channel and threshold at ``threshold``.
      2. Apply binary erosion along the time axis with ``erosion_size`` to
         remove isolated noise pixels (opening removes specks).
      3. Apply binary dilation along the time axis with ``dilation_size`` to
         fill small gaps in sustained notes (closing fills holes).
      4. Extract connected components per pitch row and emit MidiNote events.

    Parameters (from config['params']):
        threshold (float): Binary activation threshold in [0, 1]. Default 0.5.
        erosion_size (int): Structuring element size for erosion along time.
            Default 2.
        dilation_size (int): Structuring element size for dilation along time.
            Default 2.
        min_duration_steps (int): Minimum run length after morphological ops.
            Default 1.
        pitch_axis (str): 'height' or 'width'. Default 'height'.
        num_pitches (int): MIDI pitch range size. Default 128.
        pitch_offset (int): Lowest MIDI pitch number. Default 0.
    """

    def __init__(self, params: dict | None = None) -> None:
        """Initialise MorphologicalDetector.

        Args:
            params: Optional dict with keys 'threshold', 'erosion_size',
                'dilation_size', 'min_duration_steps', 'pitch_axis',
                'num_pitches', 'pitch_offset'.
        """
        super().__init__(params)
        self._threshold: float = float(self.params.get('threshold', 0.5))
        self._erosion_size: int = int(self.params.get('erosion_size', 2))
        self._dilation_size: int = int(self.params.get('dilation_size', 2))
        self._min_dur: int = int(self.params.get('min_duration_steps', 1))
        self._pitch_axis: str = str(self.params.get('pitch_axis', 'height'))
        self._num_pitches: int = int(self.params.get('num_pitches', 128))
        self._pitch_offset: int = int(self.params.get('pitch_offset', 0))

    @property
    def needs_fitting(self) -> bool:
        """Morphological detector requires no fitting — returns False."""
        return False

    def detect(
        self,
        recon_image: torch.Tensor,
        channel_strategy: str,
    ) -> list[MidiNote]:
        """Detect notes from a reconstructed piano-roll image.

        Applies morphological opening (erosion then dilation) along the time
        axis to clean up the binary activation map before note extraction.

        Args:
            recon_image: Continuous-valued image tensor of shape (3, H, W)
                with values in [-1, 1].
            channel_strategy: The channel strategy used for rendering.

        Returns:
            Sorted list of detected MidiNote objects.
        """
        vel_map = _extract_velocity_channel(recon_image, channel_strategy)
        vel_np = vel_map.cpu().float().numpy()

        # Threshold
        binary = (vel_np >= self._threshold)  # (H, W)

        H, W = binary.shape

        if self._pitch_axis == 'height':
            num_rows, num_time_steps = H, W
        else:
            num_rows, num_time_steps = W, H

        # Build 1-D structuring elements along the time axis
        if self._erosion_size > 1:
            if self._pitch_axis == 'height':
                erode_struct = np.ones((1, self._erosion_size), dtype=bool)
            else:
                erode_struct = np.ones((self._erosion_size, 1), dtype=bool)
            binary = ndimage.binary_erosion(binary, structure=erode_struct)

        if self._dilation_size > 1:
            if self._pitch_axis == 'height':
                dilate_struct = np.ones((1, self._dilation_size), dtype=bool)
            else:
                dilate_struct = np.ones((self._dilation_size, 1), dtype=bool)
            binary = ndimage.binary_dilation(binary, structure=dilate_struct)

        notes: list[MidiNote] = []

        for row_idx in range(num_rows):
            pitch = _row_to_pitch_generic(
                row_idx, num_rows, self._num_pitches, self._pitch_offset
            )
            if pitch < 0 or pitch > 127:
                continue

            if self._pitch_axis == 'height':
                row = binary[row_idx, :]
                vel_row = vel_np[row_idx, :]
            else:
                row = binary[:, row_idx]
                vel_row = vel_np[:, row_idx]

            segments = _connected_components_1d(row)

            for onset, offset in segments:
                duration = offset - onset
                if duration < self._min_dur:
                    continue
                mean_activation = float(vel_row[onset:offset].mean())
                midi_velocity = max(1, min(127, round(mean_activation * 127)))
                try:
                    notes.append(MidiNote(
                        pitch=pitch,
                        onset_step=onset,
                        offset_step=offset,
                        velocity=midi_velocity,
                    ))
                except ValueError:
                    continue

        notes.sort(key=lambda n: (n.onset_step, n.pitch))
        return notes
