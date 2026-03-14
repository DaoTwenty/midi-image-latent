"""Device selection and management utilities."""

from __future__ import annotations

import torch


def get_device(requested: str = "cuda") -> torch.device:
    """Get the best available device matching the request.

    Falls back to CPU if CUDA is requested but unavailable.

    Args:
        requested: Device string ('cuda', 'cpu', 'cuda:0', etc.).

    Returns:
        torch.device for the selected device.
    """
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def get_device_info(device: torch.device) -> dict[str, str]:
    """Get device information for experiment tracking.

    Args:
        device: The torch device.

    Returns:
        Dict with device name, type, and GPU details if applicable.
    """
    info = {
        "device_type": device.type,
        "device_str": str(device),
    }
    if device.type == "cuda":
        idx = device.index or 0
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        info["gpu_memory_gb"] = f"{torch.cuda.get_device_properties(idx).total_memory / 1e9:.1f}"
        info["cuda_version"] = torch.version.cuda or "unknown"
    return info
