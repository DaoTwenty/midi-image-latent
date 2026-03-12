"""Reconstruction quality metrics: pixel-level image fidelity measures.

Implements PixelMSE, SSIM, and PSNR comparing the original rendered
piano-roll image against its VAE reconstruction.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from midi_vae.data.types import BarData, ReconstructedBar, PianoRollImage
from midi_vae.metrics.base import Metric
from midi_vae.registry import ComponentRegistry


def _to_float_tensor(t: torch.Tensor) -> torch.Tensor:
    """Return tensor as float32 on CPU."""
    return t.cpu().float()


@ComponentRegistry.register('metric', 'pixel_mse')
class PixelMSE(Metric):
    """Mean squared error between the original and reconstructed images.

    Both images are expected in [-1, 1] and are compared pixel-wise.
    The result is the average over all spatial positions and channels.

    Returns:
        {'pixel_mse': float}
    """

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'pixel_mse'

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
        gt_image: PianoRollImage | None = None,
    ) -> dict[str, float]:
        """Compute mean squared error between original and reconstructed images.

        Args:
            gt: Ground-truth bar data (not directly used; image comes from gt_image).
            recon: Reconstructed bar containing recon_image (3, H, W).
            gt_image: Optional rendered ground-truth image. If None, the metric
                returns NaN (caller is expected to supply the image).

        Returns:
            Dict with key 'pixel_mse' mapping to the MSE value.
        """
        if gt_image is None:
            return {'pixel_mse': float('nan')}

        orig = _to_float_tensor(gt_image.image)
        pred = _to_float_tensor(recon.recon_image)

        if orig.shape != pred.shape:
            pred = F.interpolate(
                pred.unsqueeze(0), size=orig.shape[-2:], mode='bilinear', align_corners=False
            ).squeeze(0)

        mse = float(((orig - pred) ** 2).mean().item())
        return {'pixel_mse': mse}


def _gaussian_kernel_2d(kernel_size: int, sigma: float) -> torch.Tensor:
    """Create a 2-D Gaussian kernel.

    Args:
        kernel_size: Side length of the square kernel (should be odd).
        sigma: Standard deviation of the Gaussian.

    Returns:
        2-D float tensor of shape (kernel_size, kernel_size) normalised to sum to 1.
    """
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    g_2d = g_1d[:, None] * g_1d[None, :]
    return g_2d / g_2d.sum()


def _ssim_single_channel(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    C1: float,
    C2: float,
) -> float:
    """Compute SSIM for a single-channel pair of images.

    Args:
        x: Ground-truth channel of shape (1, 1, H, W).
        y: Reconstructed channel of shape (1, 1, H, W).
        kernel: Gaussian kernel of shape (1, 1, K, K).
        C1: Stability constant for luminance.
        C2: Stability constant for contrast.

    Returns:
        Mean SSIM value as a Python float.
    """
    padding = kernel.shape[-1] // 2

    mu_x = F.conv2d(x, kernel, padding=padding)
    mu_y = F.conv2d(y, kernel, padding=padding)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x ** 2, kernel, padding=padding) - mu_x_sq
    sigma_y_sq = F.conv2d(y ** 2, kernel, padding=padding) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=padding) - mu_xy

    # Clamp negative variances caused by numerical noise
    sigma_x_sq = sigma_x_sq.clamp(min=0.0)
    sigma_y_sq = sigma_y_sq.clamp(min=0.0)

    numerator = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    ssim_map = numerator / (denominator + 1e-10)
    return float(ssim_map.mean().item())


@ComponentRegistry.register('metric', 'ssim')
class SSIM(Metric):
    """Structural Similarity Index (SSIM) between original and reconstructed images.

    Implemented with a Gaussian-weighted sliding window. Computed per-channel
    and averaged. Images should be in [-1, 1]; they are remapped to [0, 1]
    internally for the SSIM formula.

    Parameters (passed as constructor kwargs or via subclass override):
        kernel_size (int): Gaussian window size. Default 11.
        sigma (float): Gaussian standard deviation. Default 1.5.

    Returns:
        {'ssim': float}  — value in [-1, 1], higher is better.
    """

    def __init__(
        self,
        kernel_size: int = 11,
        sigma: float = 1.5,
    ) -> None:
        """Initialise SSIM metric.

        Args:
            kernel_size: Size of the Gaussian window (odd integer).
            sigma: Standard deviation of the Gaussian kernel.
        """
        self._kernel_size = kernel_size
        self._sigma = sigma
        # Stability constants — standard SSIM values for data_range=1
        self._C1 = (0.01) ** 2
        self._C2 = (0.03) ** 2

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'ssim'

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
        gt_image: PianoRollImage | None = None,
    ) -> dict[str, float]:
        """Compute SSIM between original and reconstructed images.

        Args:
            gt: Ground-truth bar data.
            recon: Reconstructed bar containing recon_image.
            gt_image: Rendered ground-truth image. Required; returns NaN if None.

        Returns:
            Dict with key 'ssim'.
        """
        if gt_image is None:
            return {'ssim': float('nan')}

        orig = _to_float_tensor(gt_image.image)
        pred = _to_float_tensor(recon.recon_image)

        if orig.shape != pred.shape:
            pred = F.interpolate(
                pred.unsqueeze(0), size=orig.shape[-2:], mode='bilinear', align_corners=False
            ).squeeze(0)

        # Remap [-1, 1] -> [0, 1]
        orig = (orig + 1.0) / 2.0
        pred = (pred + 1.0) / 2.0

        kernel_2d = _gaussian_kernel_2d(self._kernel_size, self._sigma)
        # Shape: (1, 1, K, K) for conv2d
        kernel = kernel_2d.unsqueeze(0).unsqueeze(0)

        C = orig.shape[0]
        ssim_sum = 0.0
        for c in range(C):
            x = orig[c].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            y = pred[c].unsqueeze(0).unsqueeze(0)
            ssim_sum += _ssim_single_channel(x, y, kernel, self._C1, self._C2)

        return {'ssim': ssim_sum / C}


@ComponentRegistry.register('metric', 'psnr')
class PSNR(Metric):
    """Peak Signal-to-Noise Ratio between original and reconstructed images.

    Computed as: PSNR = 10 * log10(MAX^2 / MSE)
    where MAX = 2.0 (data range for images in [-1, 1]).

    Returns:
        {'psnr': float}  — value in dB, higher is better. Returns inf if MSE == 0.
    """

    @property
    def name(self) -> str:
        """Metric identifier."""
        return 'psnr'

    def compute(
        self,
        gt: BarData,
        recon: ReconstructedBar,
        gt_image: PianoRollImage | None = None,
    ) -> dict[str, float]:
        """Compute PSNR between original and reconstructed images.

        Args:
            gt: Ground-truth bar data.
            recon: Reconstructed bar.
            gt_image: Rendered ground-truth image. Required; returns NaN if None.

        Returns:
            Dict with key 'psnr' (in dB).
        """
        if gt_image is None:
            return {'psnr': float('nan')}

        orig = _to_float_tensor(gt_image.image)
        pred = _to_float_tensor(recon.recon_image)

        if orig.shape != pred.shape:
            pred = F.interpolate(
                pred.unsqueeze(0), size=orig.shape[-2:], mode='bilinear', align_corners=False
            ).squeeze(0)

        mse = float(((orig - pred) ** 2).mean().item())
        if mse == 0.0:
            return {'psnr': float('inf')}

        # Data range is 2.0 (from -1 to 1)
        max_val = 2.0
        psnr = 10.0 * math.log10(max_val ** 2 / mse)
        return {'psnr': psnr}
