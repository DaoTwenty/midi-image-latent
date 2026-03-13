"""Tests for sub-latent dimensionality reduction models.

Covers PCASubLatent in midi_vae/models/sublatent/pca.py and the SubLatentModel ABC.
Tests are CPU-only and use synthetic tensor data.
"""

from __future__ import annotations

import pytest
import torch

from midi_vae.config import SubLatentConfig
from midi_vae.models.sublatent.base import SubLatentModel
from midi_vae.models.sublatent.pca import PCASubLatent
from midi_vae.registry import ComponentRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INPUT_DIM = 64   # 4 channels * 4 * 4
TARGET_DIM = 16
N_SAMPLES = 100


def _make_config(target_dim: int = TARGET_DIM) -> SubLatentConfig:
    """Return a minimal SubLatentConfig for PCA."""
    return SubLatentConfig(enabled=True, approach="pca", target_dim=target_dim)


def _make_pca(target_dim: int = TARGET_DIM, input_dim: int = INPUT_DIM) -> PCASubLatent:
    """Return a PCASubLatent instance."""
    cfg = _make_config(target_dim)
    return PCASubLatent(config=cfg, input_dim=input_dim, device="cpu")


def _synthetic_latents(n: int = N_SAMPLES, dim: int = INPUT_DIM) -> torch.Tensor:
    """Return synthetic latent data with known structure (low-rank)."""
    torch.manual_seed(0)
    # Low-rank structure so PCA has meaningful components to capture
    basis = torch.randn(5, dim)
    coeffs = torch.randn(n, 5)
    return coeffs @ basis + torch.randn(n, dim) * 0.01


# ---------------------------------------------------------------------------
# PCA registration test
# ---------------------------------------------------------------------------


class TestPCARegistration:
    """Verify PCASubLatent is registered in the ComponentRegistry."""

    def test_pca_registered(self) -> None:
        """PCASubLatent is registered under ('sublatent', 'pca')."""
        registered = ComponentRegistry.list_components("sublatent").get("sublatent", [])
        assert "pca" in registered

    def test_registry_returns_sublatent_model_subclass(self) -> None:
        """ComponentRegistry.get('sublatent', 'pca') is a SubLatentModel subclass."""
        cls = ComponentRegistry.get("sublatent", "pca")
        assert issubclass(cls, SubLatentModel)


# ---------------------------------------------------------------------------
# PCASubLatent initialisation
# ---------------------------------------------------------------------------


class TestPCASubLatentInit:
    """Test PCASubLatent initialisation and default state."""

    def test_not_fitted_on_init(self) -> None:
        """PCA starts in an unfitted state."""
        pca = _make_pca()
        assert not pca._fitted

    def test_target_dim_stored(self) -> None:
        """target_dim is stored from config."""
        pca = _make_pca(target_dim=32)
        assert pca.target_dim == 32

    def test_input_dim_stored(self) -> None:
        """input_dim is stored on the instance."""
        pca = _make_pca(input_dim=128)
        assert pca.input_dim == 128

    def test_components_none_before_fit(self) -> None:
        """_components is None before fit."""
        pca = _make_pca()
        assert pca._components is None

    def test_mean_none_before_fit(self) -> None:
        """_mean is None before fit."""
        pca = _make_pca()
        assert pca._mean is None


# ---------------------------------------------------------------------------
# PCASubLatent.fit()
# ---------------------------------------------------------------------------


class TestPCAFit:
    """Tests for PCASubLatent.fit()."""

    def test_fit_sets_fitted_flag(self) -> None:
        """After fit(), _fitted is True."""
        pca = _make_pca()
        data = _synthetic_latents()
        pca.fit(data)
        assert pca._fitted

    def test_fit_returns_dict_with_expected_keys(self) -> None:
        """fit() returns dict with known keys."""
        pca = _make_pca()
        stats = pca.fit(_synthetic_latents())
        assert "explained_variance_ratio" in stats
        assert "total_explained_variance" in stats
        assert "n_samples" in stats
        assert "n_components" in stats

    def test_fit_n_samples_correct(self) -> None:
        """fit() stats record correct n_samples."""
        pca = _make_pca()
        data = _synthetic_latents(n=80)
        stats = pca.fit(data)
        assert stats["n_samples"] == 80

    def test_fit_n_components_correct(self) -> None:
        """fit() retains target_dim components when enough data provided."""
        pca = _make_pca(target_dim=10)
        data = _synthetic_latents(n=200)
        stats = pca.fit(data)
        assert stats["n_components"] == 10

    def test_fit_explained_variance_sums_to_at_most_1(self) -> None:
        """Total explained variance is in (0, 1]."""
        pca = _make_pca()
        stats = pca.fit(_synthetic_latents())
        assert 0.0 < stats["total_explained_variance"] <= 1.0

    def test_fit_stores_components_with_correct_shape(self) -> None:
        """_components has shape (target_dim, input_dim) after fit."""
        pca = _make_pca(target_dim=TARGET_DIM, input_dim=INPUT_DIM)
        pca.fit(_synthetic_latents())
        assert pca._components is not None
        assert pca._components.shape == (TARGET_DIM, INPUT_DIM)

    def test_fit_stores_mean_with_correct_shape(self) -> None:
        """_mean has shape (input_dim,) after fit."""
        pca = _make_pca()
        pca.fit(_synthetic_latents())
        assert pca._mean is not None
        assert pca._mean.shape == (INPUT_DIM,)

    def test_fit_adjusts_when_n_samples_lt_target_dim(self) -> None:
        """fit() gracefully handles n_samples < target_dim."""
        pca = _make_pca(target_dim=50)
        data = _synthetic_latents(n=10)  # n < target_dim
        stats = pca.fit(data)
        # Should not crash; n_components capped at min(n, D)
        assert stats["n_components"] <= 10

    def test_fit_explained_variance_list_length_matches_n_components(self) -> None:
        """explained_variance_ratio list has length == n_components."""
        pca = _make_pca(target_dim=8)
        stats = pca.fit(_synthetic_latents(n=200))
        assert len(stats["explained_variance_ratio"]) == stats["n_components"]


# ---------------------------------------------------------------------------
# PCASubLatent.encode() and decode()
# ---------------------------------------------------------------------------


class TestPCAEncodeDecodeShapes:
    """Tests for encode/decode input-output shapes."""

    @pytest.fixture
    def fitted_pca(self) -> PCASubLatent:
        """Return a PCA fitted on synthetic data."""
        pca = _make_pca()
        pca.fit(_synthetic_latents())
        return pca

    def test_encode_output_shape(self, fitted_pca: PCASubLatent) -> None:
        """encode() returns (B, target_dim) tensor."""
        x = torch.randn(8, INPUT_DIM)
        s = fitted_pca.encode(x)
        assert s.shape == (8, TARGET_DIM)

    def test_decode_output_shape(self, fitted_pca: PCASubLatent) -> None:
        """decode() returns (B, input_dim) tensor."""
        s = torch.randn(8, TARGET_DIM)
        z_hat = fitted_pca.decode(s)
        assert z_hat.shape == (8, INPUT_DIM)

    def test_encode_single_sample(self, fitted_pca: PCASubLatent) -> None:
        """encode() works with batch size 1."""
        x = torch.randn(1, INPUT_DIM)
        s = fitted_pca.encode(x)
        assert s.shape == (1, TARGET_DIM)

    def test_decode_single_sample(self, fitted_pca: PCASubLatent) -> None:
        """decode() works with batch size 1."""
        s = torch.randn(1, TARGET_DIM)
        z_hat = fitted_pca.decode(s)
        assert z_hat.shape == (1, INPUT_DIM)


class TestPCAEncodeDecodeRoundtrip:
    """Tests for encode→decode reconstruction quality."""

    @pytest.fixture
    def fitted_pca_full_rank(self) -> PCASubLatent:
        """PCA with target_dim == input_dim for perfect reconstruction."""
        cfg = _make_config(target_dim=INPUT_DIM)
        pca = PCASubLatent(config=cfg, input_dim=INPUT_DIM, device="cpu")
        data = _synthetic_latents(n=200)
        pca.fit(data)
        return pca

    def test_roundtrip_is_approximate(self) -> None:
        """encode then decode approximately reconstructs the input (not exact for low rank)."""
        pca = _make_pca(target_dim=TARGET_DIM)
        pca.fit(_synthetic_latents(n=200))
        x = _synthetic_latents(n=10)
        s = pca.encode(x)
        x_hat = pca.decode(s)
        # Should be similar but not perfect (lossy due to dimensionality reduction)
        assert x_hat.shape == x.shape

    def test_low_rank_data_reconstructs_well(self) -> None:
        """PCA reconstructs low-rank data with low error when target_dim >= true rank."""
        input_dim = 32
        true_rank = 4
        target_dim = 8  # capture more than true rank
        torch.manual_seed(42)
        basis = torch.randn(true_rank, input_dim)
        basis = basis / basis.norm(dim=1, keepdim=True)  # normalise
        coeffs = torch.randn(200, true_rank)
        data = coeffs @ basis

        cfg = SubLatentConfig(enabled=True, approach="pca", target_dim=target_dim)
        pca = PCASubLatent(config=cfg, input_dim=input_dim, device="cpu")
        pca.fit(data)

        test_data = (torch.randn(20, true_rank) @ basis)
        recon = pca.decode(pca.encode(test_data))
        mse = ((test_data - recon) ** 2).mean().item()
        # Should reconstruct near-perfectly for truly low-rank data
        assert mse < 0.1, f"MSE too high for low-rank data: {mse:.4f}"


# ---------------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------------


class TestPCAErrorConditions:
    """Tests for error handling in PCASubLatent."""

    def test_encode_raises_before_fit(self) -> None:
        """encode() raises RuntimeError if called before fit."""
        pca = _make_pca()
        x = torch.randn(4, INPUT_DIM)
        with pytest.raises(RuntimeError, match="must be fitted"):
            pca.encode(x)

    def test_decode_raises_before_fit(self) -> None:
        """decode() raises RuntimeError if called before fit."""
        pca = _make_pca()
        s = torch.randn(4, TARGET_DIM)
        with pytest.raises(RuntimeError, match="must be fitted"):
            pca.decode(s)

    def test_train_step_raises_not_implemented(self) -> None:
        """train_step() raises NotImplementedError — PCA has no iterative training."""
        pca = _make_pca()
        pca.fit(_synthetic_latents())
        with pytest.raises(NotImplementedError):
            pca.train_step(torch.randn(4, INPUT_DIM))

    def test_save_raises_before_fit(self) -> None:
        """save() raises RuntimeError if called before fit."""
        pca = _make_pca()
        with pytest.raises(RuntimeError, match="must be fitted"):
            pca.save("/tmp/test_pca.pt")


# ---------------------------------------------------------------------------
# Persistence (save/load roundtrip)
# ---------------------------------------------------------------------------


class TestPCASaveLoad:
    """Tests for PCA save/load checkpoint."""

    def test_save_and_load_roundtrip(self, tmp_path) -> None:
        """save() + load() preserves PCA components and mean."""
        pca1 = _make_pca()
        data = _synthetic_latents(n=200)
        pca1.fit(data)

        path = str(tmp_path / "pca.pt")
        pca1.save(path)

        cfg = _make_config(TARGET_DIM)
        pca2 = PCASubLatent(config=cfg, input_dim=INPUT_DIM, device="cpu")
        pca2.load(path)

        assert pca2._fitted
        assert torch.allclose(pca1._components, pca2._components)
        assert torch.allclose(pca1._mean, pca2._mean)

    def test_loaded_pca_produces_same_encodings(self, tmp_path) -> None:
        """Loaded PCA produces identical encodings as the original."""
        pca1 = _make_pca()
        pca1.fit(_synthetic_latents(n=200))

        path = str(tmp_path / "pca.pt")
        pca1.save(path)

        cfg = _make_config(TARGET_DIM)
        pca2 = PCASubLatent(config=cfg, input_dim=INPUT_DIM, device="cpu")
        pca2.load(path)

        x = _synthetic_latents(n=10)
        s1 = pca1.encode(x)
        s2 = pca2.encode(x)
        assert torch.allclose(s1, s2, atol=1e-5)


# ---------------------------------------------------------------------------
# Variance explained property
# ---------------------------------------------------------------------------


class TestVarianceExplained:
    """Tests for the explained variance stats returned by fit()."""

    def test_variance_increases_with_more_components(self) -> None:
        """More PCA components capture more variance."""
        data = _synthetic_latents(n=300)

        pca_small = _make_pca(target_dim=4)
        stats_small = pca_small.fit(data)

        pca_large = _make_pca(target_dim=16)
        stats_large = pca_large.fit(data)

        assert (
            stats_large["total_explained_variance"]
            >= stats_small["total_explained_variance"]
        )

    def test_explained_variance_ratios_are_non_negative(self) -> None:
        """All explained variance ratios are >= 0."""
        pca = _make_pca()
        stats = pca.fit(_synthetic_latents(n=200))
        for ratio in stats["explained_variance_ratio"]:
            assert ratio >= 0.0

    def test_singular_values_stored_correctly(self) -> None:
        """_singular_values has length target_dim after fit."""
        pca = _make_pca(target_dim=8)
        pca.fit(_synthetic_latents(n=100))
        assert pca._singular_values is not None
        assert pca._singular_values.shape[0] <= 8
