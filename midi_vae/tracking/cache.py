"""Content-addressed artifact caching for pipeline stages."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class ArtifactCache:
    """Content-addressed cache for pipeline artifacts.

    Uses SHA-256 hashes of inputs to create cache keys. Supports
    tensor and JSON data.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        """Initialize the cache.

        Args:
            cache_dir: Directory for cached artifacts.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def compute_key(self, *args: Any) -> str:
        """Compute a content hash key from arguments.

        Args:
            *args: Values to hash.

        Returns:
            16-char hex digest.
        """
        hasher = hashlib.sha256()
        for arg in args:
            hasher.update(str(arg).encode())
        return hasher.hexdigest()[:16]

    def has(self, key: str, name: str) -> bool:
        """Check if a cached artifact exists.

        Args:
            key: Cache key.
            name: Artifact name.

        Returns:
            True if the artifact is cached.
        """
        path = self.cache_dir / key / name
        return path.exists()

    def get_json(self, key: str, name: str) -> dict[str, Any] | None:
        """Load a cached JSON artifact.

        Args:
            key: Cache key.
            name: Artifact name (should end in .json).

        Returns:
            Parsed dict, or None if not cached.
        """
        path = self.cache_dir / key / name
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning(f"Corrupted cache entry: {path}")
            return None

    def put_json(self, key: str, name: str, data: dict[str, Any]) -> Path:
        """Save a JSON artifact to cache.

        Args:
            key: Cache key.
            name: Artifact name.
            data: Dict to cache.

        Returns:
            Path to cached file.
        """
        dir_path = self.cache_dir / key
        dir_path.mkdir(parents=True, exist_ok=True)
        path = dir_path / name
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def get_tensor(self, key: str, name: str) -> torch.Tensor | None:
        """Load a cached tensor.

        Args:
            key: Cache key.
            name: Artifact name (should end in .pt).

        Returns:
            Tensor, or None if not cached.
        """
        path = self.cache_dir / key / name
        if not path.exists():
            return None
        try:
            return torch.load(path, weights_only=True)
        except (RuntimeError, OSError):
            logger.warning(f"Corrupted tensor cache: {path}")
            return None

    def put_tensor(self, key: str, name: str, tensor: torch.Tensor) -> Path:
        """Save a tensor to cache.

        Args:
            key: Cache key.
            name: Artifact name.
            tensor: Tensor to cache.

        Returns:
            Path to cached file.
        """
        dir_path = self.cache_dir / key
        dir_path.mkdir(parents=True, exist_ok=True)
        path = dir_path / name
        torch.save(tensor, path)
        return path

    def clear(self) -> None:
        """Clear all cached artifacts."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
