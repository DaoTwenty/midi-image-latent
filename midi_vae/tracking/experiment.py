"""Experiment tracking with unique IDs, config snapshots, and metric logging."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from midi_vae.config import ExperimentConfig
from midi_vae.utils.io import ensure_dir, save_json


class ExperimentTracker:
    """Tracks experiment runs with unique IDs, saves configs, metrics, and artifacts.

    ID format: {experiment_name}_{YYYYMMDD_HHMMSS}_{4char_hash}

    Directory structure:
        {output_root}/{experiment_id}/
            config.yaml
            environment.json
            metrics/
            artifacts/
            figures/
            logs/
            jobs/
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize the tracker and create the experiment directory.

        Args:
            config: Full experiment configuration.
        """
        self.config = config
        self.experiment_id = self._generate_id(config.tracking.experiment_name)
        self.experiment_dir = Path(config.paths.output_root) / self.experiment_id

        # Create subdirectories
        self.metrics_dir = ensure_dir(self.experiment_dir / "metrics")
        self.artifacts_dir = ensure_dir(self.experiment_dir / "artifacts")
        self.figures_dir = ensure_dir(self.experiment_dir / "figures")
        self.logs_dir = ensure_dir(self.experiment_dir / "logs")
        self.jobs_dir = ensure_dir(self.experiment_dir / "jobs")

        # Save config and environment
        self._save_config()
        self._save_environment()

    def _generate_id(self, name: str) -> str:
        """Generate a unique experiment ID.

        Args:
            name: Experiment name from config.

        Returns:
            ID string: {name}_{YYYYMMDD_HHMMSS}_{4char_hash}
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{name}_{timestamp}_{id(self)}"
        short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:4]
        return f"{name}_{timestamp}_{short_hash}"

    def _save_config(self) -> None:
        """Save experiment config as JSON."""
        config_dict = self.config.model_dump()
        save_json(config_dict, self.experiment_dir / "config.json")

    def _save_environment(self) -> None:
        """Save environment info for reproducibility."""
        env_info: dict[str, Any] = {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "hostname": platform.node(),
            "platform": platform.platform(),
        }

        # Git hash
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            env_info["git_hash"] = result.stdout.strip() if result.returncode == 0 else "unknown"
        except (subprocess.SubprocessError, FileNotFoundError):
            env_info["git_hash"] = "unknown"

        # CUDA info
        if torch.cuda.is_available():
            env_info["cuda_version"] = torch.version.cuda or "unknown"
            env_info["gpu_name"] = torch.cuda.get_device_name(0)
            env_info["gpu_count"] = torch.cuda.device_count()

        save_json(env_info, self.experiment_dir / "environment.json")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None, tag: str = "eval") -> None:
        """Log metrics to disk.

        Args:
            metrics: Dict of metric name -> value.
            step: Optional step/iteration number.
            tag: Tag for grouping metrics (e.g., 'eval', 'train').
        """
        filename = f"{tag}_step{step}.json" if step is not None else f"{tag}.json"
        save_json(metrics, self.metrics_dir / filename)

    def save_artifact(self, data: Any, name: str) -> Path:
        """Save an artifact to the artifacts directory.

        Args:
            data: Data to save (if dict/list, saved as JSON; if Tensor, saved as .pt).
            name: Artifact filename.

        Returns:
            Path to the saved artifact.
        """
        path = self.artifacts_dir / name
        if isinstance(data, (dict, list)):
            save_json(data, path)
        elif isinstance(data, torch.Tensor):
            torch.save(data, path)
        else:
            with open(path, "w") as f:
                f.write(str(data))
        return path

    def log_figure(self, fig: Any, name: str) -> Path:
        """Save a matplotlib figure.

        Args:
            fig: Matplotlib figure object.
            name: Filename (should end in .png or .pdf).

        Returns:
            Path to the saved figure.
        """
        path = self.figures_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        return path
