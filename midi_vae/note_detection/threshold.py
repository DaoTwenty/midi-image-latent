"""Threshold-based note detection methods.

Provides GlobalThresholdDetector and related threshold variants for
converting continuous-valued reconstructed piano-roll images back to
discrete MIDI note events.
"""

from __future__ import annotations

import torch
import numpy as np

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


def _connected_components_1d_fast(binary_row: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous True runs using numpy diff — no scipy needed.

    Replaces the scipy.ndimage.label-based implementation with a pure numpy
    approach that avoids scipy overhead for 1-D data.

    Args:
        binary_row: 1-D boolean array of length T.  May be non-contiguous
            (e.g., a column-slice of a 2-D C-contiguous array).

    Returns:
        List of (start_inclusive, end_exclusive) tuples for each run.
    """
    if not binary_row.any():
        return []
    # Ensure a contiguous uint8 copy so that diff arithmetic is well-defined
    # regardless of the input array's memory layout.
    uint8_row = np.ascontiguousarray(binary_row, dtype=np.uint8)
    padded = np.empty(len(uint8_row) + 2, dtype=np.int8)
    padded[0] = 0
    padded[1:-1] = uint8_row
    padded[-1] = 0
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]  # already exclusive
    return list(zip(starts.tolist(), ends.tolist()))


def _connected_components_1d(binary_row: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous runs of True values in a 1-D boolean array.

    Thin wrapper kept for backwards compatibility; delegates to the fast
    numpy-only implementation.

    Args:
        binary_row: 1-D boolean array of length T.

    Returns:
        List of (start_inclusive, end_exclusive) tuples for each run.
    """
    return _connected_components_1d_fast(binary_row)


def _detect_notes_from_binary_2d(
    vel_np: np.ndarray,
    binary: np.ndarray,
    pitch_axis: str,
    min_dur: int,
    num_pitches: int,
    pitch_offset: int,
) -> list[MidiNote]:
    """Extract MidiNote objects from a 2-D binary activation map.

    Vectorized note detection that skips entirely-silent rows before
    entering the per-row loop and uses the fast numpy-based 1-D connected
    components finder.

    Args:
        vel_np: Float velocity map of shape (H, W) in [0, 1].
        binary: Boolean activation map of the same shape.
        pitch_axis: 'height' (pitch = row) or 'width' (pitch = column).
        min_dur: Minimum run length in time steps.
        num_pitches: Number of MIDI pitches covered.
        pitch_offset: Lowest MIDI pitch represented.

    Returns:
        List of MidiNote objects sorted by (onset_step, pitch).
    """
    H, W = binary.shape

    if pitch_axis == 'height':
        num_rows = H
    else:
        num_rows = W

    # Pre-compute the row-to-pitch mapping as an array for efficiency
    if num_rows == 1:
        pitch_map = np.array([pitch_offset], dtype=np.int32)
    else:
        indices = np.arange(num_rows, dtype=np.float64)
        pitch_map = (pitch_offset + np.round(
            indices * (num_pitches - 1) / (num_rows - 1)
        )).astype(np.int32)

    # Identify active rows up-front to skip silent ones
    if pitch_axis == 'height':
        active_rows = np.where(binary.any(axis=1))[0]
    else:
        active_rows = np.where(binary.any(axis=0))[0]

    notes: list[MidiNote] = []

    for row_idx in active_rows:
        pitch = int(pitch_map[row_idx])
        if pitch < 0 or pitch > 127:
            continue

        if pitch_axis == 'height':
            row = binary[row_idx, :]
            vel_row = vel_np[row_idx, :]
        else:
            row = binary[:, row_idx]
            vel_row = vel_np[:, row_idx]

        segments = _connected_components_1d_fast(row)

        for onset, offset in segments:
            duration = offset - onset
            if duration < min_dur:
                continue
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
                continue

    notes.sort(key=lambda n: (n.onset_step, n.pitch))
    return notes


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
          2. Apply the binary threshold tau to the full image at once.
          3. Skip entirely-silent rows, then for each active pitch row find
             contiguous True-regions using fast numpy diff.
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

        return _detect_notes_from_binary_2d(
            vel_np=vel_np,
            binary=binary,
            pitch_axis=self._pitch_axis,
            min_dur=self._min_dur,
            num_pitches=self._num_pitches,
            pitch_offset=self._pitch_offset,
        )

    def detect_batch(
        self,
        recon_images: list[torch.Tensor],
        channel_strategy: str,
    ) -> list[list[MidiNote]]:
        """Detect notes from multiple images in sequence.

        Amortizes the Python-level setup cost (numpy conversion, threshold
        application) across a batch of images.  Each image is processed
        independently so results are identical to repeated calls to
        :meth:`detect`.

        Args:
            recon_images: List of image tensors, each of shape (3, H, W)
                with values in [-1, 1].
            channel_strategy: Channel strategy used for all images in the
                batch ('velocity_only', 'vo_split', 'vos').

        Returns:
            List of note lists, one per input image, in the same order.
        """
        return [self.detect(img, channel_strategy) for img in recon_images]

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


def _row_to_pitch_generic(
    row_idx: int,
    num_rows: int,
    num_pitches: int,
    pitch_offset: int,
) -> int:
    """Map a row index to MIDI pitch.

    Args:
        row_idx: Zero-based index into the pitch axis.
        num_rows: Total rows in the pitch axis.
        num_pitches: Number of MIDI pitches represented.
        pitch_offset: Lowest MIDI pitch number.

    Returns:
        Integer MIDI pitch in [0, 127].
    """
    if num_rows == 1:
        return pitch_offset
    return int(pitch_offset + round(row_idx * (num_pitches - 1) / (num_rows - 1)))


@ComponentRegistry.register('note_detector', 'per_pitch_adaptive')
class PerPitchAdaptiveDetector(NoteDetector):
    """Per-pitch adaptive threshold note detector.

    Fits a separate activation threshold for each pitch row based on
    statistics (mean + k * std) accumulated from validation data.
    Requires a call to :meth:`fit` before :meth:`detect`.

    Parameters (from config['params']):
        k_sigma (float): Number of standard deviations above the mean to
            set the per-pitch threshold. Default 1.0.
        min_duration_steps (int): Minimum run length to be counted as a
            note. Default 1.
        pitch_axis (str): 'height' or 'width'. Default 'height'.
        num_pitches (int): MIDI pitch range size. Default 128.
        pitch_offset (int): Lowest MIDI pitch. Default 0.
    """

    def __init__(self, params: dict | None = None) -> None:
        """Initialise PerPitchAdaptiveDetector.

        Args:
            params: Optional dict with keys 'k_sigma', 'min_duration_steps',
                'pitch_axis', 'num_pitches', 'pitch_offset'.
        """
        super().__init__(params)
        self._k_sigma: float = float(self.params.get('k_sigma', 1.0))
        self._min_dur: int = int(self.params.get('min_duration_steps', 1))
        self._pitch_axis: str = str(self.params.get('pitch_axis', 'height'))
        self._num_pitches: int = int(self.params.get('num_pitches', 128))
        self._pitch_offset: int = int(self.params.get('pitch_offset', 0))
        # Per-row thresholds populated by fit(); default to 0.5
        self._thresholds: np.ndarray | None = None

    @property
    def needs_fitting(self) -> bool:
        """PerPitchAdaptive requires fitting — returns True."""
        return True

    def fit(self, validation_bars: list[tuple[BarData, torch.Tensor]]) -> None:
        """Fit per-pitch thresholds from validation reconstruction images.

        For each pitch row the threshold is: mean_activation + k_sigma * std_activation.

        Args:
            validation_bars: List of (ground_truth_bar, reconstructed_image) pairs.
                The images should have shape (3, H, W) with values in [-1, 1].
        """
        if not validation_bars:
            return

        # Accumulate per-row activations across all validation images
        all_vel: list[list[float]] = []
        num_rows: int | None = None

        for _, recon_img in validation_bars:
            vel_map = _extract_velocity_channel(recon_img, 'velocity_only')
            vel_np = vel_map.cpu().float().numpy()

            if self._pitch_axis == 'height':
                n_rows = vel_np.shape[0]
                rows = [vel_np[r, :] for r in range(n_rows)]
            else:
                n_rows = vel_np.shape[1]
                rows = [vel_np[:, r] for r in range(n_rows)]

            if num_rows is None:
                num_rows = n_rows
                all_vel = [[] for _ in range(n_rows)]

            for r_idx in range(min(n_rows, len(all_vel))):
                all_vel[r_idx].extend(rows[r_idx].tolist())

        if num_rows is None or num_rows == 0:
            return

        thresholds = np.zeros(num_rows, dtype=np.float32)
        for r_idx in range(num_rows):
            vals = np.array(all_vel[r_idx], dtype=np.float32)
            if len(vals) == 0:
                thresholds[r_idx] = 0.5
            else:
                thresholds[r_idx] = float(vals.mean() + self._k_sigma * vals.std())

        # Clip to valid range
        self._thresholds = np.clip(thresholds, 0.0, 1.0)

    def detect(
        self,
        recon_image: torch.Tensor,
        channel_strategy: str,
    ) -> list[MidiNote]:
        """Detect notes using per-pitch adaptive thresholds.

        Falls back to a global 0.5 threshold if :meth:`fit` has not been called.

        Args:
            recon_image: Continuous-valued image tensor of shape (3, H, W).
            channel_strategy: Channel strategy used for rendering.

        Returns:
            Sorted list of detected MidiNote objects.
        """
        vel_map = _extract_velocity_channel(recon_image, channel_strategy)
        vel_np = vel_map.cpu().float().numpy()
        H, W = vel_np.shape

        if self._pitch_axis == 'height':
            num_rows, num_time_steps = H, W
        else:
            num_rows, num_time_steps = W, H

        # Build threshold array for this image
        if self._thresholds is not None and len(self._thresholds) == num_rows:
            thresholds = self._thresholds
        else:
            # Fallback: uniform 0.5
            thresholds = np.full(num_rows, 0.5, dtype=np.float32)

        # Build full binary image by broadcasting per-row thresholds.
        # thresholds has shape (num_rows,); expand it for broadcasting.
        if self._pitch_axis == 'height':
            # thresholds[r] applies to row r, broadcast across time axis (W)
            tau_2d = thresholds[:, np.newaxis]   # (H, 1)
            binary = vel_np >= tau_2d            # (H, W)
        else:
            # thresholds[c] applies to column c, broadcast across time axis (H)
            tau_2d = thresholds[np.newaxis, :]   # (1, W)
            binary = vel_np >= tau_2d            # (H, W)

        return _detect_notes_from_binary_2d(
            vel_np=vel_np,
            binary=binary,
            pitch_axis=self._pitch_axis,
            min_dur=self._min_dur,
            num_pitches=self._num_pitches,
            pitch_offset=self._pitch_offset,
        )


@ComponentRegistry.register('note_detector', 'hysteresis')
class HysteresisDetector(NoteDetector):
    """Hysteresis threshold note detector.

    Uses a high threshold (tau_on) to detect note onsets and a lower
    threshold (tau_off) to allow notes to sustain once active. This
    avoids premature note-off events from small dips in the velocity
    channel.

    Parameters (from config['params']):
        tau_on (float): High onset threshold in [0, 1]. Default 0.6.
        tau_off (float): Low sustain threshold in [0, 1]. Default 0.3.
        min_duration_steps (int): Minimum run length. Default 1.
        pitch_axis (str): 'height' or 'width'. Default 'height'.
        num_pitches (int): MIDI pitch range size. Default 128.
        pitch_offset (int): Lowest MIDI pitch. Default 0.
    """

    def __init__(self, params: dict | None = None) -> None:
        """Initialise HysteresisDetector.

        Args:
            params: Optional dict with keys 'tau_on', 'tau_off',
                'min_duration_steps', 'pitch_axis', 'num_pitches', 'pitch_offset'.
        """
        super().__init__(params)
        self._tau_on: float = float(self.params.get('tau_on', 0.6))
        self._tau_off: float = float(self.params.get('tau_off', 0.3))
        self._min_dur: int = int(self.params.get('min_duration_steps', 1))
        self._pitch_axis: str = str(self.params.get('pitch_axis', 'height'))
        self._num_pitches: int = int(self.params.get('num_pitches', 128))
        self._pitch_offset: int = int(self.params.get('pitch_offset', 0))

    @property
    def needs_fitting(self) -> bool:
        """Hysteresis requires no fitting — returns False."""
        return False

    @staticmethod
    def _hysteresis_1d(row: np.ndarray, tau_on: float, tau_off: float) -> np.ndarray:
        """Apply hysteresis thresholding to a 1-D activation vector.

        A run starts when activation >= tau_on and ends when it drops below
        tau_off.

        Args:
            row: 1-D float array of activations in [0, 1].
            tau_on: High onset threshold.
            tau_off: Low sustain threshold.

        Returns:
            Boolean array of same length where active regions are True.
        """
        active = False
        result = np.zeros(len(row), dtype=bool)
        for t, val in enumerate(row):
            if not active and val >= tau_on:
                active = True
            elif active and val < tau_off:
                active = False
            result[t] = active
        return result

    def detect(
        self,
        recon_image: torch.Tensor,
        channel_strategy: str,
    ) -> list[MidiNote]:
        """Detect notes using hysteresis thresholding.

        Active rows (those with at least one value >= tau_on) are identified
        up-front to skip entirely-silent rows.  The hysteresis state machine
        and fast numpy-based connected-component finder are then applied only
        to active rows.

        Args:
            recon_image: Continuous-valued image tensor of shape (3, H, W).
            channel_strategy: Channel strategy used for rendering.

        Returns:
            Sorted list of detected MidiNote objects.
        """
        vel_map = _extract_velocity_channel(recon_image, channel_strategy)
        vel_np = vel_map.cpu().float().numpy()
        H, W = vel_np.shape

        if self._pitch_axis == 'height':
            num_rows = H
        else:
            num_rows = W

        # Pre-compute pitch map
        if num_rows == 1:
            pitch_map = np.array([self._pitch_offset], dtype=np.int32)
        else:
            indices = np.arange(num_rows, dtype=np.float64)
            pitch_map = (self._pitch_offset + np.round(
                indices * (self._num_pitches - 1) / (num_rows - 1)
            )).astype(np.int32)

        # Only visit rows that might produce a note (at least one value >= tau_on)
        if self._pitch_axis == 'height':
            candidate_rows = np.where((vel_np >= self._tau_on).any(axis=1))[0]
        else:
            candidate_rows = np.where((vel_np >= self._tau_on).any(axis=0))[0]

        notes: list[MidiNote] = []

        for row_idx in candidate_rows:
            pitch = int(pitch_map[row_idx])
            if pitch < 0 or pitch > 127:
                continue

            if self._pitch_axis == 'height':
                row = vel_np[row_idx, :]
            else:
                row = vel_np[:, row_idx]

            binary = self._hysteresis_1d(row, self._tau_on, self._tau_off)
            segments = _connected_components_1d_fast(binary)

            for onset, offset in segments:
                duration = offset - onset
                if duration < self._min_dur:
                    continue
                mean_activation = float(row[onset:offset].mean())
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


@ComponentRegistry.register('note_detector', 'velocity_aware')
class VelocityAwareDetector(NoteDetector):
    """Velocity-aware onset detector.

    Uses the velocity channel (R) to detect note regions and optionally
    weights onset detection by the magnitude of the velocity channel,
    preferring high-velocity activations for onset placement.

    When the channel strategy is 'vo_split' or 'vos', the onset channel
    (G, index 1) is used to refine onset boundaries.

    Parameters (from config['params']):
        threshold (float): Binary activation threshold. Default 0.5.
        onset_boost (float): Multiplier for the onset channel when present.
            Higher values make onset detection stricter. Default 2.0.
        min_duration_steps (int): Minimum run length. Default 1.
        pitch_axis (str): 'height' or 'width'. Default 'height'.
        num_pitches (int): MIDI pitch range size. Default 128.
        pitch_offset (int): Lowest MIDI pitch. Default 0.
    """

    def __init__(self, params: dict | None = None) -> None:
        """Initialise VelocityAwareDetector.

        Args:
            params: Optional dict with keys 'threshold', 'onset_boost',
                'min_duration_steps', 'pitch_axis', 'num_pitches', 'pitch_offset'.
        """
        super().__init__(params)
        self._threshold: float = float(self.params.get('threshold', 0.5))
        self._onset_boost: float = float(self.params.get('onset_boost', 2.0))
        self._min_dur: int = int(self.params.get('min_duration_steps', 1))
        self._pitch_axis: str = str(self.params.get('pitch_axis', 'height'))
        self._num_pitches: int = int(self.params.get('num_pitches', 128))
        self._pitch_offset: int = int(self.params.get('pitch_offset', 0))

    @property
    def needs_fitting(self) -> bool:
        """VelocityAware requires no fitting — returns False."""
        return False

    def detect(
        self,
        recon_image: torch.Tensor,
        channel_strategy: str,
    ) -> list[MidiNote]:
        """Detect notes using velocity magnitude-weighted thresholding.

        When channel_strategy provides a separate onset channel (index 1),
        the combined signal (velocity + onset_boost * onset_channel) is used
        to determine active regions. Otherwise the velocity channel alone is used.

        Args:
            recon_image: Continuous-valued image tensor of shape (3, H, W).
            channel_strategy: Channel strategy ('velocity_only', 'vo_split', 'vos').

        Returns:
            Sorted list of detected MidiNote objects.
        """
        # Extract velocity channel — always R (index 0)
        vel_map = _extract_velocity_channel(recon_image, channel_strategy)
        vel_np = vel_map.cpu().float().numpy()

        # Extract onset channel if available (G = index 1 for vo_split / vos)
        if channel_strategy in ('vo_split', 'vos'):
            onset_ch = recon_image[1].cpu().float()
            onset_np = ((onset_ch + 1.0) / 2.0).clamp(0.0, 1.0).numpy()
            combined = vel_np + self._onset_boost * onset_np
            # Threshold the combined signal; normalise by (1 + onset_boost)
            # so that a fully active velocity+onset still hits ~1.0
            combined = combined / (1.0 + self._onset_boost)
        else:
            combined = vel_np

        # Threshold the combined signal to produce a binary activation map for
        # region discovery.  Velocity is read from the raw vel_np channel.
        binary = (combined >= self._threshold)

        # Use _detect_notes_from_binary_2d but with vel_np supplying velocities.
        # The helper uses vel_np for velocity extraction and binary for region
        # discovery, which is exactly the semantics of VelocityAwareDetector.
        return _detect_notes_from_binary_2d(
            vel_np=vel_np,
            binary=binary,
            pitch_axis=self._pitch_axis,
            min_dur=self._min_dur,
            num_pitches=self._num_pitches,
            pitch_offset=self._pitch_offset,
        )
