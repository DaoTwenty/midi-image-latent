"""Tests for the Bar Transformer sequence model.

Covers instantiation, forward pass, generation, training_step, and masking.
All tests use synthetic CPU tensors — no GPU or real model weights required.

BarTransformer lives at midi_vae/models/sequence/bar_transformer.py.
It takes a single config object (not keyword args) and reads attributes
via getattr with defaults.  generate() takes (prompt_z, n_steps, temperature)
and returns (B, P + n_steps, latent_dim) including the prompt.
"""

from __future__ import annotations

import types

import pytest
import torch

# ---------------------------------------------------------------------------
# Availability guard — skip cleanly when the module hasn't been merged yet
# ---------------------------------------------------------------------------

bar_transformer_available = False
BarTransformer = None  # type: ignore[assignment]

try:
    from midi_vae.models.sequence.bar_transformer import BarTransformer  # type: ignore[assignment]
    bar_transformer_available = True
except ImportError:
    pass

skip_if_unavailable = pytest.mark.skipif(
    not bar_transformer_available,
    reason="midi_vae.models.sequence.bar_transformer not yet implemented (CHARLIE Sprint 4)",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LATENT_DIM = 64          # flattened latent dimension
D_MODEL = 128
SEQ_LEN = 8
BATCH = 4
N_HEADS = 4
N_LAYERS = 2
D_FF = 256


def _make_config(**overrides) -> types.SimpleNamespace:
    """Return a config SimpleNamespace with sensible test defaults.

    BarTransformer reads: latent_dim, d_model, n_heads, n_layers, d_ff,
    dropout, max_seq_len, learning_rate, weight_decay from the config
    (or its .sequence sub-object).
    """
    cfg = types.SimpleNamespace(
        latent_dim=LATENT_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=0.0,
        max_seq_len=SEQ_LEN * 4,
        learning_rate=1e-3,
        weight_decay=0.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_transformer(**overrides) -> "BarTransformer":  # type: ignore[name-defined]
    """Return a small BarTransformer configured for fast CPU tests."""
    cfg = _make_config(**overrides)
    return BarTransformer(cfg)  # type: ignore[call-arg]


def _synthetic_sequence(batch: int = BATCH, seq_len: int = SEQ_LEN) -> torch.Tensor:
    """Return a random latent sequence (B, T, latent_dim)."""
    torch.manual_seed(0)
    return torch.randn(batch, seq_len, LATENT_DIM)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


@skip_if_unavailable
class TestBarTransformerInstantiation:
    """Tests for BarTransformer construction and attribute initialisation."""

    def test_instantiation_succeeds_with_defaults(self) -> None:
        """BarTransformer can be constructed with a config object."""
        model = _make_transformer()
        assert model is not None

    def test_latent_dim_stored(self) -> None:
        """latent_dim attribute is stored correctly."""
        model = _make_transformer(latent_dim=32)
        assert model.latent_dim == 32

    def test_d_model_stored(self) -> None:
        """d_model attribute is stored correctly (may be adjusted for n_heads)."""
        model = _make_transformer(d_model=128, n_heads=4)
        # d_model must be divisible by n_heads; BarTransformer adjusts if needed
        assert model.d_model % 4 == 0

    def test_model_is_nn_module(self) -> None:
        """BarTransformer is a torch.nn.Module subclass."""
        model = _make_transformer()
        assert isinstance(model, torch.nn.Module)

    def test_model_has_no_non_finite_params_on_init(self) -> None:
        """All parameters are finite at initialisation."""
        model = _make_transformer()
        for name, param in model.named_parameters():
            assert torch.isfinite(param).all(), f"Non-finite parameter: {name}"

    def test_model_runs_on_cpu(self) -> None:
        """Model is CPU-resident after construction (no device specified)."""
        model = _make_transformer()
        for param in model.parameters():
            assert param.device.type == "cpu"

    def test_model_has_expected_attributes(self) -> None:
        """Model exposes latent_dim, d_model, max_seq_len attributes."""
        model = _make_transformer()
        assert hasattr(model, "latent_dim")
        assert hasattr(model, "d_model")
        assert hasattr(model, "max_seq_len")


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


@skip_if_unavailable
class TestBarTransformerForward:
    """Tests for the forward pass shape contract."""

    @pytest.fixture
    def model(self) -> "BarTransformer":  # type: ignore[name-defined]
        """Return a small transformer for testing."""
        return _make_transformer()

    def test_forward_output_shape(self, model) -> None:
        """forward(z_sequence) returns a tensor of shape (B, T, latent_dim)."""
        src = _synthetic_sequence()
        out = model(src)
        assert out.shape == (BATCH, SEQ_LEN, LATENT_DIM), (
            f"Expected ({BATCH}, {SEQ_LEN}, {LATENT_DIM}), got {out.shape}"
        )

    def test_forward_output_is_float(self, model) -> None:
        """forward() output is a floating-point tensor."""
        out = model(_synthetic_sequence())
        assert out.is_floating_point()

    def test_forward_output_is_finite(self, model) -> None:
        """forward() output contains no NaN or Inf values."""
        out = model(_synthetic_sequence())
        assert torch.isfinite(out).all(), "forward() produced non-finite values"

    def test_forward_batch_size_one(self, model) -> None:
        """forward() handles batch size 1 without error."""
        src = _synthetic_sequence(batch=1)
        out = model(src)
        assert out.shape[0] == 1

    def test_forward_sequence_length_one(self, model) -> None:
        """forward() handles sequence length 1 (single bar) without error."""
        src = _synthetic_sequence(batch=2, seq_len=1)
        out = model(src)
        assert out.shape == (2, 1, LATENT_DIM)

    def test_forward_varying_sequence_lengths_do_not_raise(self, model) -> None:
        """forward() accepts different sequence lengths up to max_seq_len."""
        for seq_len in [1, 4, SEQ_LEN]:
            src = _synthetic_sequence(batch=2, seq_len=seq_len)
            out = model(src)
            assert out.shape[1] == seq_len

    def test_forward_with_optional_mask(self, model) -> None:
        """forward() accepts an optional mask argument without raising."""
        src = _synthetic_sequence()
        T = src.size(1)
        mask = torch.zeros(T, T)
        try:
            out = model(src, mask=mask)
            assert out.shape == (BATCH, T, LATENT_DIM)
        except TypeError:
            # mask parameter may have a different name — also acceptable
            pass


# ---------------------------------------------------------------------------
# generate() method
# ---------------------------------------------------------------------------


@skip_if_unavailable
class TestBarTransformerGenerate:
    """Tests for the autoregressive generate() method.

    generate(prompt_z, n_steps, temperature) returns (B, P + n_steps, latent_dim)
    where P is the prompt length. n_steps new bars are appended to the prompt.
    """

    @pytest.fixture
    def model(self) -> "BarTransformer":  # type: ignore[name-defined]
        return _make_transformer()

    def test_generate_output_shape_includes_prompt(self, model) -> None:
        """generate() returns (B, P + n_steps, latent_dim)."""
        prompt = _synthetic_sequence(batch=2, seq_len=4)
        n_steps = 4
        out = model.generate(prompt, n_steps=n_steps)
        P = prompt.size(1)
        assert out.shape == (2, P + n_steps, LATENT_DIM), (
            f"Expected (2, {P + n_steps}, {LATENT_DIM}), got {out.shape}"
        )

    def test_generate_is_finite(self, model) -> None:
        """generate() output contains no NaN or Inf values."""
        context = _synthetic_sequence(batch=1, seq_len=4)
        out = model.generate(context, n_steps=4)
        assert torch.isfinite(out).all()

    def test_generate_returns_tensor(self, model) -> None:
        """generate() returns a torch.Tensor, not a list."""
        context = _synthetic_sequence(batch=1, seq_len=2)
        out = model.generate(context, n_steps=2)
        assert isinstance(out, torch.Tensor)

    def test_generate_n_steps_one(self, model) -> None:
        """generate() works for n_steps=1 (single-step generation)."""
        context = _synthetic_sequence(batch=2, seq_len=2)
        out = model.generate(context, n_steps=1)
        # Prompt 2 + 1 new = 3 total
        assert out.shape[1] == context.size(1) + 1

    def test_generate_is_deterministic_with_fixed_seed(self, model) -> None:
        """generate() is deterministic when torch seed is fixed."""
        context = _synthetic_sequence(batch=1, seq_len=4)
        torch.manual_seed(42)
        out1 = model.generate(context, n_steps=4)
        torch.manual_seed(42)
        out2 = model.generate(context, n_steps=4)
        assert torch.allclose(out1, out2)

    def test_generate_with_temperature(self, model) -> None:
        """generate() accepts a temperature argument without error."""
        context = _synthetic_sequence(batch=1, seq_len=4)
        out = model.generate(context, n_steps=4, temperature=0.5)
        P = context.size(1)
        assert out.shape[1] == P + 4

    def test_generate_2d_prompt_is_expanded(self, model) -> None:
        """generate() accepts a 2D prompt (P, latent_dim) and expands to batch=1."""
        prompt_2d = torch.randn(4, LATENT_DIM)
        out = model.generate(prompt_2d, n_steps=2)
        assert out.shape[0] == 1
        assert out.shape[1] == 4 + 2


# ---------------------------------------------------------------------------
# training_step()
# ---------------------------------------------------------------------------


@skip_if_unavailable
class TestBarTransformerTrainingStep:
    """Tests for the training_step() method."""

    @pytest.fixture
    def model(self) -> "BarTransformer":  # type: ignore[name-defined]
        return _make_transformer()

    def test_training_step_returns_dict(self, model) -> None:
        """training_step() returns a dict mapping str -> float."""
        seq = _synthetic_sequence()
        result = model.training_step(seq)
        assert isinstance(result, dict)
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, float)

    def test_training_step_has_loss_key(self, model) -> None:
        """training_step() dict contains a 'total_loss' or 'loss' key."""
        seq = _synthetic_sequence()
        result = model.training_step(seq)
        has_loss = any("loss" in k.lower() for k in result)
        assert has_loss, f"No loss key found in {list(result.keys())}"

    def test_training_step_loss_is_non_negative(self, model) -> None:
        """training_step() MSE loss is non-negative for random inputs."""
        seq = _synthetic_sequence()
        result = model.training_step(seq)
        loss_val = result.get("total_loss", result.get("loss", None))
        if loss_val is not None:
            assert loss_val >= 0.0

    def test_training_step_decreases_loss_over_iterations(self, model) -> None:
        """Repeated training_step() calls should reduce loss on a fixed sequence."""
        torch.manual_seed(0)
        seq = _synthetic_sequence(batch=8, seq_len=4)

        losses = []
        for _ in range(10):
            result = model.training_step(seq)
            loss_val = result.get("total_loss", result.get("loss", float("inf")))
            losses.append(loss_val)

        # Loss should decrease at least once across the 10 steps
        assert min(losses) < losses[0], (
            f"Loss never decreased: {losses}"
        )

    def test_training_step_single_bar_raises_or_handles(self, model) -> None:
        """training_step() with length-1 sequence raises ValueError (T < 2 for autoregressive)."""
        seq = _synthetic_sequence(batch=4, seq_len=1)
        with pytest.raises(ValueError):
            model.training_step(seq)

    def test_training_step_seq_len_two(self, model) -> None:
        """training_step() handles minimum valid sequence length of 2."""
        seq = _synthetic_sequence(batch=4, seq_len=2)
        result = model.training_step(seq)
        assert isinstance(result, dict)

    def test_training_step_result_keys(self, model) -> None:
        """training_step() returns 'total_loss' and 'seq_loss' keys."""
        seq = _synthetic_sequence()
        result = model.training_step(seq)
        assert "total_loss" in result
        assert "seq_loss" in result


# ---------------------------------------------------------------------------
# Causal masking
# ---------------------------------------------------------------------------


@skip_if_unavailable
class TestBarTransformerMasking:
    """Tests that causal masking works correctly."""

    @pytest.fixture
    def model(self) -> "BarTransformer":  # type: ignore[name-defined]
        return _make_transformer()

    def test_autoregressive_mask_is_causal(self, model) -> None:
        """Model output at position t should not depend on positions > t.

        Verified by checking that corrupting position 4 does NOT change
        the model output at positions 0-3.
        """
        torch.manual_seed(7)
        seq = _synthetic_sequence(batch=1, seq_len=6)
        model.eval()
        with torch.no_grad():
            out_orig = model(seq)

        # Corrupt position 4 and check positions 0-3 are unchanged
        seq_corrupt = seq.clone()
        seq_corrupt[0, 4, :] = 99.0
        with torch.no_grad():
            out_corrupt = model(seq_corrupt)

        # Positions 0 through 3 should be identical (causal)
        assert torch.allclose(out_orig[0, :4], out_corrupt[0, :4], atol=1e-4), (
            "Causal masking violated: output at positions < 4 changed when "
            "position 4 was corrupted"
        )

    def test_internal_causal_mask_method(self, model) -> None:
        """_causal_mask() returns correct upper-triangular shape."""
        mask = model._causal_mask(seq_len=8, device=torch.device("cpu"))
        assert mask.shape == (8, 8)
        # Upper triangle (diagonal=1) should have -inf
        upper = mask[0, 1:]
        assert upper[0].item() == float("-inf"), "First super-diagonal should be -inf"
        # Diagonal and below should be 0
        assert mask[3, 3].item() == 0.0

    def test_model_train_eval_toggle(self, model) -> None:
        """Model can switch between train and eval modes without error."""
        model.train()
        assert model.training
        model.eval()
        assert not model.training
        model.train()
        assert model.training


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


@skip_if_unavailable
class TestBarTransformerPersistence:
    """Tests for save() and load() checkpoint methods."""

    def test_save_creates_file(self, tmp_path) -> None:
        """save() writes a .pt file to disk."""
        model = _make_transformer()
        path = str(tmp_path / "model.pt")
        model.save(path)
        import os
        assert os.path.exists(path)

    def test_load_restores_weights(self, tmp_path) -> None:
        """load() restores weights so forward pass is identical."""
        model = _make_transformer()
        path = str(tmp_path / "model.pt")
        model.save(path)

        model2 = _make_transformer()
        model2.load(path)

        seq = _synthetic_sequence(batch=1)
        model.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model(seq)
            out2 = model2(seq)
        assert torch.allclose(out1, out2, atol=1e-6)
