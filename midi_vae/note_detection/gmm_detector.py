"""Gaussian Mixture Model onset detector.

Fits a 2-component GMM per pitch to learn data-driven thresholds that
distinguish note-on activations from silence. Requires a ``fit()`` call
on validation data before ``detect()`` can be used.
"""

from __future__ import annotations

import numpy as np
import torch

from midi_vae.data.types import BarData, MidiNote
from midi_vae.note_detection.base import NoteDetector
from midi_vae.note_detection.threshold import (
    _extract_velocity_channel,
    _connected_components_1d,
    _row_to_pitch_generic,
)
from midi_vae.registry import ComponentRegistry

try:
    from sklearn.mixture import GaussianMixture
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


@ComponentRegistry.register('note_detector', 'gmm')
class GMMDetector(NoteDetector):
    """Gaussian Mixture Model note detector.

    Fits a 2-component GMM per pitch row to model the bimodal distribution
    of activations (silence vs note-on). After fitting, the component with
    the higher mean is treated as the "note-on" component. The classification
    boundary (decision point between components) is used as the detection
    threshold.

    Requires ``fit()`` before ``detect()``. Falls back to a global threshold
    of 0.5 if fit has not been called or sklearn is unavailable.

    Parameters (from config['params']):
        n_components (int): Number of GMM components per pitch. Default 2.
        min_duration_steps (int): Minimum run length to count as a note. Default 1.
        pitch_axis (str): 'height' or 'width'. Default 'height'.
        num_pitches (int): MIDI pitch range size. Default 128.
        pitch_offset (int): Lowest MIDI pitch. Default 0.
        fallback_threshold (float): Global threshold to use if GMM fitting
            fails or sklearn is not available. Default 0.5.
        max_iter (int): Maximum EM iterations for sklearn GaussianMixture. Default 100.
        random_state (int): Random seed for GMM fitting reproducibility. Default 42.
    """

    def __init__(self, params: dict | None = None) -> None:
        """Initialise GMMDetector.

        Args:
            params: Optional dict with keys 'n_components', 'min_duration_steps',
                'pitch_axis', 'num_pitches', 'pitch_offset', 'fallback_threshold',
                'max_iter', 'random_state'.
        """
        super().__init__(params)
        self._n_components: int = int(self.params.get('n_components', 2))
        self._min_dur: int = int(self.params.get('min_duration_steps', 1))
        self._pitch_axis: str = str(self.params.get('pitch_axis', 'height'))
        self._num_pitches: int = int(self.params.get('num_pitches', 128))
        self._pitch_offset: int = int(self.params.get('pitch_offset', 0))
        self._fallback_threshold: float = float(self.params.get('fallback_threshold', 0.5))
        self._max_iter: int = int(self.params.get('max_iter', 100))
        self._random_state: int = int(self.params.get('random_state', 42))

        # Per-row thresholds, populated by fit()
        self._thresholds: np.ndarray | None = None

    @property
    def needs_fitting(self) -> bool:
        """GMM detector requires fitting — returns True."""
        return True

    def fit(self, validation_bars: list[tuple[BarData, torch.Tensor]]) -> None:
        """Fit a 2-component GMM per pitch row from validation reconstructions.

        For each pitch row, all activation values across all validation images
        are collected and a GMM is fitted. The decision threshold between the
        two components is estimated as the midpoint between the two component
        means (weighted by priors).

        If sklearn is unavailable or fitting fails for a row, the fallback
        threshold is used.

        Args:
            validation_bars: List of (ground_truth_bar, reconstructed_image) pairs.
                Images should be shape (3, H, W) with values in [-1, 1].
        """
        if not validation_bars:
            return

        if not _SKLEARN_AVAILABLE:
            return

        # Accumulate per-row activations across all validation images
        all_activations: list[list[float]] = []
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
                all_activations = [[] for _ in range(n_rows)]

            for r_idx in range(min(n_rows, len(all_activations))):
                all_activations[r_idx].extend(rows[r_idx].tolist())

        if num_rows is None or num_rows == 0:
            return

        thresholds = np.full(num_rows, self._fallback_threshold, dtype=np.float32)

        for r_idx in range(num_rows):
            vals = np.array(all_activations[r_idx], dtype=np.float64)
            if len(vals) < self._n_components * 2:
                # Too few samples to fit GMM — keep fallback
                continue

            try:
                gmm = GaussianMixture(
                    n_components=self._n_components,
                    max_iter=self._max_iter,
                    random_state=self._random_state,
                )
                gmm.fit(vals.reshape(-1, 1))

                # Identify the "note-on" component as the one with higher mean
                means = gmm.means_.flatten()
                weights = gmm.weights_.flatten()
                covars = gmm.covariances_.flatten()
                stds = np.sqrt(np.clip(covars, 1e-12, None))

                if self._n_components == 2:
                    # Binary case: threshold at the crossing point between
                    # the two Gaussian components (weighted midpoint by std)
                    low_idx = int(means.argmin())
                    high_idx = 1 - low_idx

                    mean_low = means[low_idx]
                    mean_high = means[high_idx]
                    std_low = stds[low_idx]
                    std_high = stds[high_idx]

                    # Weighted midpoint: closer to the tighter component
                    if std_low + std_high > 1e-10:
                        threshold = (mean_low * std_high + mean_high * std_low) / (
                            std_low + std_high
                        )
                    else:
                        threshold = (mean_low + mean_high) / 2.0
                else:
                    # For >2 components, use mean of the highest-mean component
                    # minus 2 std as threshold
                    high_idx = int(means.argmax())
                    threshold = float(means[high_idx] - 2.0 * stds[high_idx])

                thresholds[r_idx] = float(np.clip(threshold, 0.0, 1.0))

            except Exception:
                # On any fitting failure, keep fallback threshold
                pass

        self._thresholds = thresholds

    def detect(
        self,
        recon_image: torch.Tensor,
        channel_strategy: str,
    ) -> list[MidiNote]:
        """Detect notes using per-pitch GMM-derived thresholds.

        Falls back to the global fallback_threshold if ``fit()`` has not been
        called or if sklearn was unavailable.

        Args:
            recon_image: Continuous-valued image tensor of shape (3, H, W)
                with values in [-1, 1].
            channel_strategy: Channel strategy used for rendering
                ('velocity_only', 'vo_split', 'vos').

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

        # Select thresholds: fitted or fallback
        if self._thresholds is not None and len(self._thresholds) == num_rows:
            thresholds = self._thresholds
        else:
            thresholds = np.full(num_rows, self._fallback_threshold, dtype=np.float32)

        notes: list[MidiNote] = []

        for row_idx in range(num_rows):
            pitch = _row_to_pitch_generic(
                row_idx, num_rows, self._num_pitches, self._pitch_offset
            )
            if pitch < 0 or pitch > 127:
                continue

            if self._pitch_axis == 'height':
                row = vel_np[row_idx, :]
            else:
                row = vel_np[:, row_idx]

            tau = float(thresholds[row_idx])
            binary = row >= tau
            segments = _connected_components_1d(binary)

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
