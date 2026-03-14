#!/usr/bin/env python
"""GPU smoke test: verify CUDA, test VAE loading, run GPU integration tests."""

import os
import sys
import gc
import torch

# Load .env if present (for HF_TOKEN)
_env_path = os.path.join(os.path.dirname(__file__), os.pardir, ".env")
if os.path.isfile(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())


def check_cuda():
    """Verify CUDA is available and print GPU info."""
    print("=" * 60)
    print("CUDA CHECK")
    print("=" * 60)
    assert torch.cuda.is_available(), "CUDA not available!"
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  VRAM: {props.total_memory / 1e9:.1f} GB")
    print()


def test_single_vae():
    """Smoke test: load the smallest VAE, encode/decode a synthetic image."""
    print("=" * 60)
    print("PHASE 1: Single VAE Smoke Test (sd_vae_ft_mse)")
    print("=" * 60)

    from midi_vae.registry import ComponentRegistry
    from midi_vae.config import VAEConfig

    # Trigger registration
    import midi_vae.models.vae_registry  # noqa: F401

    cfg = VAEConfig(
        model_id="stabilityai/sd-vae-ft-mse",
        name="sd_vae_ft_mse",
        latent_type="mean",
        dtype="float32",
        batch_size=4,
    )

    vae = ComponentRegistry.get("vae", "sd_vae_ft_mse")(config=cfg, device="cuda")
    vae.load_model()

    x = torch.randn(2, 3, 128, 128).cuda()
    z_mu, z_sigma = vae.encode(x)
    print(f"  Latent shape: {z_mu.shape}")  # Expect (2, 4, 16, 16)
    assert z_mu.shape == (2, 4, 16, 16), f"Unexpected latent shape: {z_mu.shape}"

    recon = vae.decode(z_mu)
    print(f"  Recon shape: {recon.shape}")  # Expect (2, 3, 128, 128)
    assert recon.shape == (2, 3, 128, 128), f"Unexpected recon shape: {recon.shape}"

    assert not torch.isnan(z_mu).any(), "NaN in latents!"
    assert not torch.isnan(recon).any(), "NaN in reconstruction!"
    print(f"  Recon range: [{recon.min():.2f}, {recon.max():.2f}]")
    print("  PASSED!")
    print()

    del vae, x, z_mu, z_sigma, recon
    gc.collect()
    torch.cuda.empty_cache()


def test_all_vaes():
    """Load each VAE, verify shapes, free memory between models."""
    print("=" * 60)
    print("PHASE 2: All 12 VAEs")
    print("=" * 60)

    from midi_vae.registry import ComponentRegistry
    from midi_vae.config import VAEConfig
    import midi_vae.models.vae_registry  # noqa: F401

    VAES_TO_TEST = [
        ("sd_vae_ft_mse", "stabilityai/sd-vae-ft-mse", "float32", 4),
        ("sdxl_vae", "stabilityai/sdxl-vae", "float32", 4),
        ("eq_vae_ema", "zelaki/eq-vae-ema", "float32", 4),
        ("eq_sdxl_vae", "KBlueLeaf/EQ-SDXL-VAE", "float32", 4),
        ("sd_v1_4", "CompVis/stable-diffusion-v1-4", "float32", 4),
        ("playground_v25", "playgroundai/playground-v2.5-1024px-aesthetic", "float32", 4),
        ("sd3_medium", "stabilityai/stable-diffusion-3-medium-diffusers", "float32", 16),
        ("flux1_dev", "black-forest-labs/FLUX.1-dev", "bfloat16", 16),
        ("flux1_kontext", "black-forest-labs/FLUX.1-Kontext-dev", "bfloat16", 16),
        ("flux2_dev", "black-forest-labs/FLUX.2-dev", "bfloat16", 16),
        ("flux2_tiny", "fal/FLUX.2-Tiny-AutoEncoder", "bfloat16", 128),
        ("cogview4", "THUDM/CogView4-6B", "bfloat16", 16),
    ]

    passed = []
    failed = []

    for name, model_id, dtype, expected_ch in VAES_TO_TEST:
        print(f"\n  --- Testing {name} ---")
        try:
            cfg = VAEConfig(
                model_id=model_id,
                name=name,
                dtype=dtype,
                batch_size=2,
                latent_type="mean",
            )
            vae = ComponentRegistry.get("vae", name)(config=cfg, device="cuda")
            vae.load_model()
            x = torch.randn(1, 3, 128, 128).cuda()
            z_mu, z_sigma = vae.encode(x)
            recon = vae.decode(z_mu)
            assert z_mu.shape[1] == expected_ch, f"Expected {expected_ch} channels, got {z_mu.shape[1]}"
            assert not torch.isnan(z_mu).any(), "NaN in latents"
            print(f"    OK: latent={z_mu.shape}, recon={recon.shape}")
            passed.append(name)
        except Exception as e:
            print(f"    FAILED: {e}")
            failed.append((name, str(e)))
        finally:
            try:
                del vae
            except NameError:
                pass
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n  Results: {len(passed)} passed, {len(failed)} failed")
    if failed:
        for name, err in failed:
            print(f"    FAIL: {name} — {err}")
    print()
    return len(failed) == 0


def run_gpu_tests():
    """Run pytest GPU-marked tests."""
    print("=" * 60)
    print("PHASE 3: GPU Integration Tests")
    print("=" * 60)
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-m", "gpu", "-v", "--tb=long"],
        cwd="/scratch/triana24/midi-image-latent",
    )
    return result.returncode == 0


if __name__ == "__main__":
    check_cuda()

    phase = sys.argv[1] if len(sys.argv) > 1 else "all"

    if phase in ("1", "smoke", "all"):
        test_single_vae()

    if phase in ("2", "vaes", "all"):
        test_all_vaes()

    if phase in ("3", "tests", "all"):
        run_gpu_tests()
