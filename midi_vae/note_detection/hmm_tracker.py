"""HMM-based note tracker using 2-state Viterbi decoding per pitch row."""

from __future__ import annotations

import numpy as np
import torch

from midi_vae.data.types import MidiNote, BarData
from midi_vae.note_detection.base import NoteDetector
from midi_vae.note_detection.threshold import (
    _extract_velocity_channel,
    _connected_components_1d,
    _row_to_pitch_generic,
)
from midi_vae.registry import ComponentRegistry


def _viterbi_2state(
    obs: np.ndarray,
    log_trans: np.ndarray,
    log_emit_on: np.ndarray,
    log_emit_off: np.ndarray,
    log_init: np.ndarray,
) -> np.ndarray:
    """Viterbi decoding for a 2-state HMM (OFF=0, ON=1).

    Args:
        obs: Observation sequence of shape (T,), values in [0, 1].
        log_trans: Log transition matrix (2, 2). log_trans[i,j] = log P(j|i).
        log_emit_on: Log emission log-probs for ON state, shape (T,).
        log_emit_off: Log emission log-probs for OFF state, shape (T,).
        log_init: Log initial state probabilities, shape (2,).

    Returns:
        Integer state sequence of shape (T,) with values 0 or 1.
    """
    T = len(obs)
    viterbi = np.full((T, 2), -np.inf, dtype=np.float64)
    backptr = np.zeros((T, 2), dtype=np.int32)
    log_emit = np.stack([log_emit_off, log_emit_on], axis=1)  # (T, 2)

    viterbi[0, 0] = log_init[0] + log_emit[0, 0]
    viterbi[0, 1] = log_init[1] + log_emit[0, 1]

    for t in range(1, T):
        for j in range(2):
            candidates = viterbi[t - 1, :] + log_trans[:, j]
            best = int(np.argmax(candidates))
            viterbi[t, j] = candidates[best] + log_emit[t, j]
            backptr[t, j] = best

    states = np.zeros(T, dtype=np.int32)
    states[T - 1] = int(np.argmax(viterbi[T - 1, :]))
    for t in range(T - 2, -1, -1):
        states[t] = backptr[t + 1, states[t + 1]]
    return states


@ComponentRegistry.register('note_detector', 'hmm_tracker')
class HMMNoteTracker(NoteDetector):
    """2-state HMM note tracker with Viterbi decoding.

    Models each pitch row as an independent OFF/ON HMM. Emission
    probabilities are Gaussian. Parameters can be fitted from validation
    data or set via constructor params.

    Parameters (from config['params']):
        p_onset (float): P(OFF->ON) transition. Default 0.1.
        p_offset (float): P(ON->OFF) transition. Default 0.2.
        mu_on (float): Emission mean for ON state. Default 0.75.
        sigma_on (float): Emission std for ON state. Default 0.1.
        mu_off (float): Emission mean for OFF state. Default 0.15.
        sigma_off (float): Emission std for OFF state. Default 0.1.
        prior_on (float): Initial probability of ON state. Default 0.3.
        min_duration_steps (int): Minimum note length. Default 1.
        pitch_axis (str): 'height' or 'width'. Default 'height'.
        num_pitches (int): MIDI pitch range. Default 128.
        pitch_offset (int): Lowest MIDI pitch. Default 0.
    """

    def __init__(self, params: dict | None = None) -> None:
        """Initialise HMMNoteTracker.

        Args:
            params: Optional config dict with HMM parameter overrides.
        """
        super().__init__(params)
        self._p_onset: float = float(self.params.get('p_onset', 0.1))
        self._p_offset: float = float(self.params.get('p_offset', 0.2))
        self._mu_on: float = float(self.params.get('mu_on', 0.75))
        self._sigma_on: float = float(self.params.get('sigma_on', 0.1))
        self._mu_off: float = float(self.params.get('mu_off', 0.15))
        self._sigma_off: float = float(self.params.get('sigma_off', 0.1))
        self._prior_on: float = float(self.params.get('prior_on', 0.3))
        self._min_dur: int = int(self.params.get('min_duration_steps', 1))
        self._pitch_axis: str = str(self.params.get('pitch_axis', 'height'))
        self._num_pitches: int = int(self.params.get('num_pitches', 128))
        self._pitch_offset: int = int(self.params.get('pitch_offset', 0))

    @property
    def needs_fitting(self) -> bool:
        """HMMNoteTracker supports fitting but works without it — returns True."""
        return True

    def fit(self, validation_bars: list[tuple[BarData, torch.Tensor]]) -> None:
        """Estimate emission and transition parameters from validation data.

        Uses ground-truth piano_roll to label each (pitch, time) cell as
        ON or OFF, then computes mean/std of velocity-channel activations
        conditioned on the label and counts transitions.

        Args:
            validation_bars: List of (ground_truth_bar, reconstructed_image) pairs.
        """
        if not validation_bars:
            return

        on_vals: list[float] = []
        off_vals: list[float] = []
        on_to_off = off_to_on = on_stays = off_stays = 0

        for bar_data, recon_img in validation_bars:
            vel_np = _extract_velocity_channel(recon_img, 'velocity_only').cpu().float().numpy()
            gt_roll = bar_data.piano_roll  # (128, T)
            H, W = vel_np.shape
            T = gt_roll.shape[1]
            num_t = min(W, T)

            if self._pitch_axis == 'height':
                n_rows = H
                def get_row(r: int) -> np.ndarray:
                    return vel_np[r, :num_t]
            else:
                n_rows = W
                def get_row(r: int) -> np.ndarray:
                    return vel_np[:num_t, r]

            for row_idx in range(n_rows):
                pitch = _row_to_pitch_generic(
                    row_idx, n_rows, self._num_pitches, self._pitch_offset
                )
                if not 0 <= pitch <= 127:
                    continue
                obs_row = get_row(row_idx)
                gt_row = (gt_roll[pitch, :num_t] > 0).astype(int)
                on_vals.extend(obs_row[gt_row == 1].tolist())
                off_vals.extend(obs_row[gt_row == 0].tolist())
                for t in range(len(gt_row) - 1):
                    s, ns = gt_row[t], gt_row[t + 1]
                    if s == 1 and ns == 1:
                        on_stays += 1
                    elif s == 1 and ns == 0:
                        on_to_off += 1
                    elif s == 0 and ns == 1:
                        off_to_on += 1
                    else:
                        off_stays += 1

        eps = 1e-6
        if on_vals:
            a = np.array(on_vals, dtype=np.float32)
            self._mu_on = float(a.mean())
            self._sigma_on = float(max(a.std(), 0.05))
        if off_vals:
            a = np.array(off_vals, dtype=np.float32)
            self._mu_off = float(a.mean())
            self._sigma_off = float(max(a.std(), 0.05))
        if (on_stays + on_to_off) > 0:
            self._p_offset = float((on_to_off + eps) / (on_stays + on_to_off + 2 * eps))
        if (off_stays + off_to_on) > 0:
            self._p_onset = float((off_to_on + eps) / (off_stays + off_to_on + 2 * eps))

    def _log_gauss(self, obs: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """Compute log Gaussian density element-wise.

        Args:
            obs: Observation array of shape (T,).
            mu: Gaussian mean.
            sigma: Gaussian standard deviation (clamped >= 1e-6).

        Returns:
            Log-probability array of shape (T,).
        """
        sigma = max(sigma, 1e-6)
        return -0.5 * ((obs - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))

    def detect(
        self,
        recon_image: torch.Tensor,
        channel_strategy: str,
    ) -> list[MidiNote]:
        """Detect notes via Viterbi decoding on the velocity channel.

        Args:
            recon_image: Continuous-valued image tensor of shape (3, H, W).
            channel_strategy: Channel strategy used for rendering.

        Returns:
            Sorted list of detected MidiNote objects.
        """
        vel_np = _extract_velocity_channel(recon_image, channel_strategy).cpu().float().numpy()
        H, W = vel_np.shape

        if self._pitch_axis == 'height':
            num_rows = H
        else:
            num_rows = W

        p_onset = float(np.clip(self._p_onset, 1e-6, 1 - 1e-6))
        p_offset = float(np.clip(self._p_offset, 1e-6, 1 - 1e-6))
        log_trans = np.log([
            [1 - p_onset, p_onset],
            [p_offset, 1 - p_offset],
        ])
        prior_on = float(np.clip(self._prior_on, 1e-6, 1 - 1e-6))
        log_init = np.log([1.0 - prior_on, prior_on])

        notes: list[MidiNote] = []

        for row_idx in range(num_rows):
            pitch = _row_to_pitch_generic(
                row_idx, num_rows, self._num_pitches, self._pitch_offset
            )
            if not 0 <= pitch <= 127:
                continue

            obs = (vel_np[row_idx, :] if self._pitch_axis == 'height'
                   else vel_np[:, row_idx]).astype(np.float64)
            if len(obs) == 0:
                continue

            log_e_on = self._log_gauss(obs, self._mu_on, self._sigma_on)
            log_e_off = self._log_gauss(obs, self._mu_off, self._sigma_off)
            states = _viterbi_2state(obs, log_trans, log_e_on, log_e_off, log_init)

            segments = _connected_components_1d(states == 1)
            for onset, offset in segments:
                if (offset - onset) < self._min_dur:
                    continue
                mean_vel = float(obs[onset:offset].mean())
                midi_velocity = max(1, min(127, round(mean_vel * 127)))
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
