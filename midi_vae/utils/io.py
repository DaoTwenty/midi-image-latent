"""File I/O utilities for saving and loading artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist.

    Args:
        path: Directory path to create.

    Returns:
        The Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a dict as JSON.

    Args:
        data: Dict to serialize.
        path: Output file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file.

    Args:
        path: Input file path.

    Returns:
        Parsed dict.
    """
    with open(path) as f:
        return json.load(f)


def save_tensor(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a PyTorch tensor to disk.

    Args:
        tensor: Tensor to save.
        path: Output file path (.pt).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, p)


def load_tensor(path: str | Path) -> torch.Tensor:
    """Load a PyTorch tensor from disk.

    Args:
        path: Input file path (.pt).

    Returns:
        Loaded tensor.
    """
    return torch.load(path, weights_only=True)
