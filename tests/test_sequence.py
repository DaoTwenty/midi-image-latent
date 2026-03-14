"""Tests for the Bar Transformer sequence model.

Covers instantiation, forward pass, generation, training_step, and masking.
All tests use synthetic CPU tensors — no GPU or real model weights required.

The bar_transformer module is expected at midi_vae/models/sequence/bar_transformer.py
and will be built by CHARLIE in Sprint 4.  Tests are decorated with
``@pytest.mark.skipif`` so the suite stays green until the module lands.
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LATENT_DIM = 64          # flattened latent dimension (e.g. from sub-latent 4C)
D_MODEL = 128
SEQ_LEN = 8
BATCH = 4
NHEAD = 4
N_LAYERS = 2
DIM_FF = 256


def _make_transformer(**kwargs) -> "BarTransformer":  # type: ignore[name-defined]
    """Return a small BarTransformer configured for fast CPU tests."""
    defaults = dict(
        latent_dim=LATENT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=N_LAYERS,
        num_decoder_layers=N_LAYERS,
        dim_feedforward=DIM_FF,
        dropout=0.0,
        max_seq_len=SEQ_LEN * 2,
    )
    defaults.update(kwargs)
    return BarTransformer(**defaults)  # type: ignore[call-arg]


def _synthetic_sequence(batch: int = BATCH, seq_len: int = SEQ_LEN) -> torch.Tensor:
    """Return a random latent sequence (B, T, latent_dim)."""
    torch.manual_seed(0)
    return torch.randn(batch, seq_len, LATENT_DIM)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not bar_transformer_available,
    reason="midi_vae.models.sequence.bar_transformer not yet implemented (CHARLIE Sprint 4)",
)
class TestBarTransformerInstantiation:
    """Tests for BarTransformer construction and attribute initialisation."""

    def test_instantiation_succeeds_with_defaults(self) -> None:
        """BarTransformer can be constructed with minimal config."""
        model = _make_transformer()
        assert model is not None

    def test_latent_dim_stored(self) -> None:
        """latent_dim attribute is stored correctly."""
        model = _make_transformer(latent_dim=32)
        assert model.latent_dim == 32

    def test_d_model_stored(self) -> None:
        """d_model attribute is stored correctly."""
        model = _make_transformer(d_model=256)
        assert model.d_model == 256

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


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not bar_transformer_available,
    reason="midi_vae.models.sequence.bar_transformer not yet implemented (CHARLIE Sprint 4)",
)
class TestBarTransformerForward:
    """Tests for the forward pass shape contract."""

    @pytest.fixture
    def model(self) -> "BarTransformer":  # type: ignore[name-defined]
        """Return a small transformer for testing."""
        return _make_transformer()

    def test_forward_output_shape(self, model) -> None:
        """forward() returns a tensor of shape (B, T, latent_dim)."""
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

    def test_forward_with_tgt_shifted_input(self, model) -> None:
        """forward() accepts an explicit tgt argument (teacher-forced decode)."""
        src = _synthetic_sequence()
        tgt = _synthetic_sequence()  # same shape as teacher-forcing target
        try:
            out = model(src, tgt=tgt)
            assert out.shape == (BATCH, SEQ_LEN, LATENT_DIM)
        except TypeError:
            # Model may not accept tgt — that's also valid if it's encoder-only
            pass


# ---------------------------------------------------------------------------
# generate() method
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not bar_transformer_available,
    reason="midi_vae.models.sequence.bar_transformer not yet implemented (CHARLIE Sprint 4)",
)
class TestBarTransformerGenerate:
    """Tests for the autoregressive generate() method."""

    @pytest.fixture
    def model(self) -> "BarTransformer":  # type: ignore[name-defined]
        return _make_transformer()

    def test_generate_output_shape(self, model) -> None:
        """generate() returns (B, n_bars, latent_dim)."""
        context = _synthetic_sequence(batch=2, seq_len=4)
        n_bars = 8
        out = model.generate(context, n_bars=n_bars)
        assert out.shape == (2, n_bars, LATENT_DIM), (
            f"Expected (2, {n_bars}, {LATENT_DIM}), got {out.shape}"
        )

    def test_generate_is_finite(self, model) -> None:
        """generate() output contains no NaN or Inf values."""
        context = _synthetic_sequence(batch=1, seq_len=4)
        out = model.generate(context, n_bars=4)
        assert torch.isfinite(out).all()

    def test_generate_returns_tensor(self, model) -> None:
        """generate() returns a torch.Tensor, not a list."""
        context = _synthetic_sequence(batch=1, seq_len=2)
        out = model.generate(context, n_bars=2)
        assert isinstance(out, torch.Tensor)

    def test_generate_n_bars_one(self, model) -> None:
        """generate() works for n_bars=1 (single-step generation)."""
        context = _synthetic_sequence(batch=2, seq_len=2)
        out = model.generate(context, n_bars=1)
        assert out.shape[1] == 1

    def test_generate_is_deterministic_with_fixed_seed(self, model) -> None:
        """generate() is deterministic when torch seed is fixed."""
        context = _synthetic_sequence(batch=1, seq_len=4)
        torch.manual_seed(42)
        out1 = model.generate(context, n_bars=4)
        torch.manual_seed(42)
        out2 = model.generate(context, n_bars=4)
        assert torch.allclose(out1, out2)

    def test_generate_with_temperature(self, model) -> None:
        """generate() accepts a temperature argument without error."""
        context = _synthetic_sequence(batch=1, seq_len=4)
        try:
            out = model.generate(context, n_bars=4, temperature=0.5)
            assert out.shape[1] == 4
        except TypeError:
            pass  # temperature may not be supported yet


# ---------------------------------------------------------------------------
# training_step()
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not bar_transformer_available,
    reason="midi_vae.models.sequence.bar_transformer not yet implemented (CHARLIE Sprint 4)",
)
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
        """training_step() dict contains a 'loss' or 'total_loss' key."""
        seq = _synthetic_sequence()
        result = model.training_step(seq)
        has_loss = any("loss" in k.lower() for k in result)
        assert has_loss, f"No loss key found in {list(result.keys())}"

    def test_training_step_loss_is_positive(self, model) -> None:
        """training_step() loss is non-negative for random inputs."""
        seq = _synthetic_sequence()
        result = model.training_step(seq)
        loss_val = result.get("loss", result.get("total_loss", None))
        if loss_val is not None:
            assert loss_val >= 0.0

    def test_training_step_decreases_loss_over_iterations(self, model) -> None:
        """Repeated training_step() calls should reduce loss on a fixed sequence."""
        torch.manual_seed(0)
        seq = _synthetic_sequence(batch=8, seq_len=4)

        losses = []
        for _ in range(10):
            result = model.training_step(seq)
            loss_val = result.get("loss", result.get("total_loss", float("inf")))
            losses.append(loss_val)

        # Loss should decrease at least once across the 10 steps
        assert min(losses) < losses[0], (
            f"Loss never decreased: {losses}"
        )

    def test_training_step_single_bar_sequence(self, model) -> None:
        """training_step() handles length-1 sequences (edge case)."""
        seq = _synthetic_sequence(batch=4, seq_len=1)
        result = model.training_step(seq)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not bar_transformer_available,
    reason="midi_vae.models.sequence.bar_transformer not yet implemented (CHARLIE Sprint 4)",
)
class TestBarTransformerMasking:
    """Tests that causal / padding masking works correctly."""

    @pytest.fixture
    def model(self) -> "BarTransformer":  # type: ignore[name-defined]
        return _make_transformer()

    def test_forward_with_padding_mask(self, model) -> None:
        """forward() accepts a key_padding_mask without error."""
        seq = _synthetic_sequence(batch=2, seq_len=6)
        # Mask last 2 positions in second batch element
        padding_mask = torch.zeros(2, 6, dtype=torch.bool)
        padding_mask[1, 4:] = True  # pad last 2 positions
        try:
            out = model(seq, src_key_padding_mask=padding_mask)
            assert out.shape == (2, 6, LATENT_DIM)
        except TypeError:
            # Parameter name may differ — acceptable
            pass

    def test_autoregressive_mask_is_causal(self, model) -> None:
        """Model output at position t should not depend on positions > t.

        This is checked by verifying that modifying token at position t+1
        does NOT change the model output at position t.
        """
        torch.manual_seed(7)
        seq = _synthetic_sequence(batch=1, seq_len=6)
        out_orig = model(seq)

        # Corrupt position 4 and check positions 0-3 are unchanged
        seq_corrupt = seq.clone()
        seq_corrupt[0, 4, :] = 99.0
        out_corrupt = model(seq_corrupt)

        # Positions 0 through 3 should be identical
        assert torch.allclose(out_orig[0, :4], out_corrupt[0, :4], atol=1e-4), (
            "Causal masking violated: output at positions < 4 changed when "
            "position 4 was corrupted"
        )

    def test_causal_mask_helper_shape(self, model) -> None:
        """generate_causal_mask() (if present) returns correct upper-triangular shape."""
        if not hasattr(model, "generate_causal_mask"):
            pytest.skip("generate_causal_mask not present on this model")
        mask = model.generate_causal_mask(seq_len=8)
        assert mask.shape == (8, 8)
        # Upper triangle (excluding diagonal) should be True (masked)
        upper = mask.triu(diagonal=1)
        assert upper.any(), "Causal mask should mask future positions"
