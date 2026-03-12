"""Tests for Sprint 0 utilities: seed, device, io, tracking, and cache.

All tests target code from midi_vae/utils/ and midi_vae/tracking/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from midi_vae.utils.seed import set_global_seed
from midi_vae.utils.device import get_device, get_device_info
from midi_vae.utils.io import (
    ensure_dir,
    save_json,
    load_json,
    save_tensor,
    load_tensor,
)
from midi_vae.tracking.cache import ArtifactCache
from midi_vae.tracking.experiment import ExperimentTracker
from midi_vae.config import ExperimentConfig, PathsConfig


# ---------------------------------------------------------------------------
# set_global_seed tests
# ---------------------------------------------------------------------------


class TestSetGlobalSeed:
    """Tests for set_global_seed in midi_vae/utils/seed.py."""

    def test_seed_makes_torch_reproducible(self) -> None:
        """Two calls with the same seed produce identical torch tensors."""
        set_global_seed(42)
        t1 = torch.randn(10)

        set_global_seed(42)
        t2 = torch.randn(10)

        assert torch.allclose(t1, t2)

    def test_seed_makes_numpy_reproducible(self) -> None:
        """Two calls with the same seed produce identical numpy arrays."""
        set_global_seed(42)
        a1 = np.random.rand(10)

        set_global_seed(42)
        a2 = np.random.rand(10)

        np.testing.assert_array_equal(a1, a2)

    def test_different_seeds_produce_different_results(self) -> None:
        """Different seeds produce different random values."""
        set_global_seed(1)
        t1 = torch.randn(10)

        set_global_seed(2)
        t2 = torch.randn(10)

        assert not torch.allclose(t1, t2)

    def test_accepts_any_int_seed(self) -> None:
        """set_global_seed accepts any integer without raising."""
        set_global_seed(0)
        set_global_seed(12345)
        set_global_seed(2**31 - 1)


# ---------------------------------------------------------------------------
# get_device tests
# ---------------------------------------------------------------------------


class TestGetDevice:
    """Tests for get_device in midi_vae/utils/device.py."""

    def test_cpu_always_available(self) -> None:
        """Requesting 'cpu' always returns a CPU device."""
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_cuda_falls_back_to_cpu(self) -> None:
        """Requesting 'cuda' when CUDA is unavailable falls back to CPU."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available — fallback not tested")
        device = get_device("cuda")
        assert device.type == "cpu"

    def test_returns_torch_device(self) -> None:
        """get_device returns a torch.device instance."""
        device = get_device("cpu")
        assert isinstance(device, torch.device)

    def test_cuda_returned_when_available(self) -> None:
        """When CUDA is available, requesting 'cuda' returns a CUDA device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = get_device("cuda")
        assert device.type == "cuda"


class TestGetDeviceInfo:
    """Tests for get_device_info in midi_vae/utils/device.py."""

    def test_cpu_info_keys(self) -> None:
        """CPU device info includes required keys."""
        device = torch.device("cpu")
        info = get_device_info(device)
        assert "device_type" in info
        assert "device_str" in info
        assert info["device_type"] == "cpu"

    def test_cpu_no_gpu_keys(self) -> None:
        """CPU device info does not include GPU-specific keys."""
        device = torch.device("cpu")
        info = get_device_info(device)
        assert "gpu_name" not in info

    def test_cuda_info_keys(self) -> None:
        """CUDA device info includes GPU-specific keys."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device("cuda")
        info = get_device_info(device)
        assert "gpu_name" in info
        assert "gpu_memory_gb" in info
        assert "cuda_version" in info


# ---------------------------------------------------------------------------
# IO utilities tests
# ---------------------------------------------------------------------------


class TestEnsureDir:
    """Tests for ensure_dir in midi_vae/utils/io.py."""

    def test_creates_directory(self, tmp_path: Path) -> None:
        """ensure_dir creates a new directory."""
        new_dir = tmp_path / "a" / "b" / "c"
        result = ensure_dir(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_no_error_if_exists(self, tmp_path: Path) -> None:
        """ensure_dir does not raise if directory already exists."""
        existing = tmp_path / "existing"
        existing.mkdir()
        result = ensure_dir(existing)
        assert result == existing

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """ensure_dir accepts a string path."""
        path_str = str(tmp_path / "string_dir")
        result = ensure_dir(path_str)
        assert Path(path_str).exists()


class TestSaveLoadJson:
    """Tests for save_json / load_json in midi_vae/utils/io.py."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """Saving and loading a JSON dict returns the original data."""
        data = {"key": "value", "num": 42, "list": [1, 2, 3]}
        path = tmp_path / "data.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_json creates parent directories if missing."""
        path = tmp_path / "nested" / "dir" / "data.json"
        save_json({"x": 1}, path)
        assert path.exists()

    def test_nested_dict(self, tmp_path: Path) -> None:
        """Nested dicts are serialized and deserialized correctly."""
        data = {"outer": {"inner": {"deep": 3.14}}}
        path = tmp_path / "nested.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded["outer"]["inner"]["deep"] == pytest.approx(3.14)

    def test_file_is_valid_json(self, tmp_path: Path) -> None:
        """The saved file is valid JSON."""
        path = tmp_path / "valid.json"
        save_json({"a": 1}, path)
        with open(path) as f:
            parsed = json.load(f)
        assert parsed["a"] == 1


class TestSaveLoadTensor:
    """Tests for save_tensor / load_tensor in midi_vae/utils/io.py."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """Saving and loading a tensor returns the original values."""
        tensor = torch.randn(4, 16, 16)
        path = tmp_path / "tensor.pt"
        save_tensor(tensor, path)
        loaded = load_tensor(path)
        assert torch.allclose(tensor, loaded)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_tensor creates parent directories if missing."""
        path = tmp_path / "nested" / "tensor.pt"
        save_tensor(torch.zeros(3), path)
        assert path.exists()

    def test_preserves_dtype(self, tmp_path: Path) -> None:
        """Loaded tensor has the same dtype as the saved tensor."""
        tensor = torch.randn(4, 4).to(torch.float32)
        path = tmp_path / "float32.pt"
        save_tensor(tensor, path)
        loaded = load_tensor(path)
        assert loaded.dtype == torch.float32

    def test_preserves_shape(self, tmp_path: Path) -> None:
        """Loaded tensor has the same shape as the saved tensor."""
        shape = (2, 3, 64, 64)
        tensor = torch.zeros(*shape)
        path = tmp_path / "shaped.pt"
        save_tensor(tensor, path)
        loaded = load_tensor(path)
        assert loaded.shape == torch.Size(shape)


# ---------------------------------------------------------------------------
# ArtifactCache tests
# ---------------------------------------------------------------------------


class TestArtifactCache:
    """Tests for ArtifactCache in midi_vae/tracking/cache.py."""

    def test_creation(self, tmp_path: Path) -> None:
        """ArtifactCache creates the cache directory on init."""
        cache_dir = tmp_path / "cache"
        cache = ArtifactCache(cache_dir)
        assert cache_dir.exists()

    def test_compute_key_deterministic(self, tmp_path: Path) -> None:
        """compute_key returns the same hash for the same inputs."""
        cache = ArtifactCache(tmp_path)
        key1 = cache.compute_key("arg1", 42, "arg3")
        key2 = cache.compute_key("arg1", 42, "arg3")
        assert key1 == key2

    def test_compute_key_different_inputs(self, tmp_path: Path) -> None:
        """compute_key returns different hashes for different inputs."""
        cache = ArtifactCache(tmp_path)
        key1 = cache.compute_key("input_a")
        key2 = cache.compute_key("input_b")
        assert key1 != key2

    def test_compute_key_length(self, tmp_path: Path) -> None:
        """compute_key returns a 16-character hex string."""
        cache = ArtifactCache(tmp_path)
        key = cache.compute_key("test")
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)

    def test_has_returns_false_before_put(self, tmp_path: Path) -> None:
        """has() returns False before an artifact is cached."""
        cache = ArtifactCache(tmp_path)
        assert cache.has("nonexistent_key", "data.json") is False

    def test_json_round_trip(self, tmp_path: Path) -> None:
        """put_json + get_json round-trips data correctly."""
        cache = ArtifactCache(tmp_path)
        key = cache.compute_key("test_json")
        data = {"metric": 0.95, "epoch": 10}
        cache.put_json(key, "results.json", data)
        loaded = cache.get_json(key, "results.json")
        assert loaded == data

    def test_has_returns_true_after_put_json(self, tmp_path: Path) -> None:
        """has() returns True after put_json."""
        cache = ArtifactCache(tmp_path)
        key = cache.compute_key("json_exists")
        cache.put_json(key, "data.json", {"x": 1})
        assert cache.has(key, "data.json") is True

    def test_get_json_missing_returns_none(self, tmp_path: Path) -> None:
        """get_json returns None for a missing artifact."""
        cache = ArtifactCache(tmp_path)
        result = cache.get_json("bad_key", "missing.json")
        assert result is None

    def test_tensor_round_trip(self, tmp_path: Path) -> None:
        """put_tensor + get_tensor round-trips data correctly."""
        cache = ArtifactCache(tmp_path)
        key = cache.compute_key("test_tensor")
        tensor = torch.randn(4, 16, 16)
        cache.put_tensor(key, "latent.pt", tensor)
        loaded = cache.get_tensor(key, "latent.pt")
        assert loaded is not None
        assert torch.allclose(tensor, loaded)

    def test_has_returns_true_after_put_tensor(self, tmp_path: Path) -> None:
        """has() returns True after put_tensor."""
        cache = ArtifactCache(tmp_path)
        key = cache.compute_key("tensor_exists")
        cache.put_tensor(key, "z.pt", torch.zeros(4))
        assert cache.has(key, "z.pt") is True

    def test_get_tensor_missing_returns_none(self, tmp_path: Path) -> None:
        """get_tensor returns None for a missing artifact."""
        cache = ArtifactCache(tmp_path)
        result = cache.get_tensor("bad_key", "missing.pt")
        assert result is None

    def test_clear_removes_all_artifacts(self, tmp_path: Path) -> None:
        """clear() removes all cached artifacts."""
        cache = ArtifactCache(tmp_path / "cache")
        key = cache.compute_key("to_clear")
        cache.put_json(key, "data.json", {"x": 1})
        cache.clear()
        assert cache.get_json(key, "data.json") is None


# ---------------------------------------------------------------------------
# ExperimentTracker tests
# ---------------------------------------------------------------------------


class TestExperimentTracker:
    """Tests for ExperimentTracker in midi_vae/tracking/experiment.py."""

    @pytest.fixture
    def tracker(self, tmp_path: Path) -> ExperimentTracker:
        """Create an ExperimentTracker with tmp directories."""
        config = ExperimentConfig(
            paths=PathsConfig(
                data_root=str(tmp_path / "data"),
                output_root=str(tmp_path / "outputs"),
                cache_dir=str(tmp_path / "cache"),
            ),
        )
        return ExperimentTracker(config)

    def test_creates_experiment_directory(self, tracker: ExperimentTracker) -> None:
        """ExperimentTracker creates the experiment directory on init."""
        assert tracker.experiment_dir.exists()

    def test_experiment_id_contains_name(self, tracker: ExperimentTracker) -> None:
        """experiment_id includes the experiment name."""
        # Default name is "default"
        assert "default" in tracker.experiment_id

    def test_subdirectories_created(self, tracker: ExperimentTracker) -> None:
        """Subdirectories (metrics, artifacts, figures, logs, jobs) are created."""
        assert tracker.metrics_dir.exists()
        assert tracker.artifacts_dir.exists()
        assert tracker.figures_dir.exists()
        assert tracker.logs_dir.exists()
        assert tracker.jobs_dir.exists()

    def test_config_saved_on_init(self, tracker: ExperimentTracker) -> None:
        """config.json is created in the experiment directory."""
        config_file = tracker.experiment_dir / "config.json"
        assert config_file.exists()

    def test_environment_saved_on_init(self, tracker: ExperimentTracker) -> None:
        """environment.json is created in the experiment directory."""
        env_file = tracker.experiment_dir / "environment.json"
        assert env_file.exists()

    def test_environment_has_required_keys(self, tracker: ExperimentTracker) -> None:
        """environment.json contains required keys."""
        env_file = tracker.experiment_dir / "environment.json"
        env = json.loads(env_file.read_text())
        assert "python_version" in env
        assert "torch_version" in env
        assert "cuda_available" in env

    def test_log_metrics_saves_file(self, tracker: ExperimentTracker) -> None:
        """log_metrics creates a JSON file in the metrics directory."""
        tracker.log_metrics({"loss": 0.42, "f1": 0.85}, step=10, tag="eval")
        metric_file = tracker.metrics_dir / "eval_step10.json"
        assert metric_file.exists()
        data = json.loads(metric_file.read_text())
        assert data["loss"] == pytest.approx(0.42)

    def test_log_metrics_no_step(self, tracker: ExperimentTracker) -> None:
        """log_metrics without step uses tag-only filename."""
        tracker.log_metrics({"accuracy": 0.9}, tag="final")
        metric_file = tracker.metrics_dir / "final.json"
        assert metric_file.exists()

    def test_save_artifact_dict(self, tracker: ExperimentTracker) -> None:
        """save_artifact saves a dict as JSON."""
        path = tracker.save_artifact({"result": 42}, "results.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["result"] == 42

    def test_save_artifact_tensor(self, tracker: ExperimentTracker) -> None:
        """save_artifact saves a tensor as .pt file."""
        tensor = torch.randn(4, 16)
        path = tracker.save_artifact(tensor, "latent.pt")
        assert path.exists()
        loaded = torch.load(path, weights_only=True)
        assert torch.allclose(tensor, loaded)

    def test_unique_ids_per_instance(self, tmp_path: Path) -> None:
        """Two ExperimentTracker instances get different experiment IDs."""
        config = ExperimentConfig(
            paths=PathsConfig(
                data_root=str(tmp_path / "data"),
                output_root=str(tmp_path / "outputs"),
                cache_dir=str(tmp_path / "cache"),
            ),
        )
        tracker1 = ExperimentTracker(config)
        tracker2 = ExperimentTracker(config)
        # IDs should differ (hash includes id(self))
        assert tracker1.experiment_id != tracker2.experiment_id
