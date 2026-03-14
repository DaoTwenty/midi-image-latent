# GPU-Deferred Tasks

Tasks that require GPU hardware. All codebase implementation is **complete** (526 tests passing, 12 GPU tests passing).

> **See also**: `docs/GPU_SESSION_GUIDE.md` for detailed execution instructions, VRAM budgets, and troubleshooting tips.

---

## Phase 1: Validation — COMPLETE (12/12 VAEs)

### 1. VAE Model Weight Downloads & Real Inference
- **Status**: 12/12 COMPLETE
- **Verified on GPU** (H100 MIG 1g.10gb, 10.5 GB VRAM):
  - sd_vae_ft_mse (4ch, float32) — latent (B,4,16,16)
  - sdxl_vae (4ch, float32) — latent (B,4,16,16)
  - eq_vae_ema (4ch, float32) — latent (B,4,16,16)
  - eq_sdxl_vae (4ch, float32) — latent (B,4,16,16)
  - sd_v1_4 (4ch, float32) — latent (B,4,16,16)
  - playground_v25 (4ch, float32) — latent (B,4,16,16)
  - sd3_medium (16ch, float32) — latent (B,16,16,16)
  - flux1_dev (16ch, bfloat16) — latent (B,16,16,16)
  - flux1_kontext (16ch, bfloat16) — latent (B,16,16,16)
  - cogview4 (16ch, bfloat16) — latent (B,16,16,16)
  - flux2_dev (32ch, bfloat16) — latent (B,32,16,16) (expanded from FLUX.1's 16ch)
  - flux2_tiny (128ch, bfloat16) — latent (B,128,8,8), custom loader, scale_factor=16
- **Fixes applied**:
  - `device.py`: `total_mem` -> `total_memory` (PyTorch 2.10 API)
  - `flux2_tiny`: Complete rewrite — uses custom `Flux2TinyAutoEncoder` from HF repo (not `transformers.AutoModel`), 128 latent channels (not 16), scale_factor=16 (not 8), encode returns `.latent` (deterministic, no distribution)
  - `flux2_dev`: Corrected from 16 to 32 latent channels (FLUX.2 expanded latent space)

### 2. Full Encode-Decode Pipeline Integration Test
- **Status**: COMPLETE
- 12 GPU tests pass for all VAEs (`pytest tests/test_integration_encode_decode.py -m gpu --run-gated`)
- Auto-skip infrastructure: `@pytest.mark.gpu` auto-skips on CPU, `@pytest.mark.gated` requires `--run-gated` flag

---

## Phase 2: Data Pipeline — COMPLETE

### 3. Real Dataset Download & Pipeline Validation
- **Status**: COMPLETE (using MAESTRO v3 — LPD5 cleansed URL is 404/down)
- **Dataset**: MAESTRO v3.0.0 (1276 classical piano MIDI files, 58 MB) downloaded to `data/maestro/maestro-v3.0.0/`
- **Validated**: Full end-to-end pipeline with real data:
  - Ingest: 20 files -> 1000 bars (piano, 50 bars/instrument, min 2 notes)
  - Render: All 3 channel strategies (velocity_only, vo_split, vos) produce (3, 128, 128) tensors
  - Encode: sd_vae_ft_mse latent shape (B,4,16,16), range [-20.6, 14.2]
  - Decode: Reconstruction shape (B,3,128,128), pixel MSE = 0.0113
  - Detect: GlobalThresholdDetector recovers 15-35 notes per bar from reconstructions
  - MaestroDataset class works correctly with max_files limiter
- **Note**: LPD5 cleansed URL (hog.ee.columbia.edu) returns 404. The full LMD (.mid files, 1.7 GB) is available but not needed — MAESTRO validates the full pipeline. LPD5 .npz ingestion path is tested with synthetic data in unit tests.

---

## Phase 3: Core Experiments

### 4. Experiment 1A — VAE Comparison (the flagship experiment)
- **Owner**: ALPHA (sweep executor)
- **Current state**: Config at `configs/experiments/exp_1a_vae_comparison.yaml`. SweepExecutor, all pipeline stages, metrics, and detection are implemented.
- **Action**: Run the full sweep: 12 VAEs x 5 instruments x 3 channel strategies = 180 conditions. This produces the core research results.
- **Estimated time**: 45-90 min on A100/4090.

### 5. Experiment 1B — Detection Methods
- **Owner**: ALPHA
- **Depends on**: Exp 1A results (best VAE).
- **Action**: Update config with best VAE from 1A, run 8 detection methods comparison.

### 6. Experiments 2 & 3 — Resolution & Channel Strategy Studies
- **Owner**: ALPHA
- **Action**: Run configs `exp_2_resolution_study.yaml` and `exp_3_channel_strategy.yaml`.

---

## Phase 4: Advanced Experiments

### 7. Sub-Latent Model Training (Exp 4C)
- **Owner**: CHARLIE
- **Current state**: MLP, PCA, sub-VAE training pipelines implemented, tested with tiny synthetic data on CPU.
- **Action**: Train 4 approaches x 5 target dims = 20 conditions using real VAE latents. Config: `exp_4c_sublatent.yaml`.
- **Estimated time**: 2-3 hours.

### 8. Conditioning Experiments (Exp 4D)
- **Owner**: CHARLIE
- **Depends on**: Best sub-latent from Exp 4C.
- **Action**: Run conditioned sub-latent training with 3 conditioning families. Config: `exp_4d_conditioning.yaml`.

### 9. CNN Note Detector Training
- **Owner**: DELTA
- **Current state**: Architecture defined, forward pass tested on CPU.
- **Action**: Train CNN segmenter on rendered piano-roll / ground-truth note pairs from real data.

### 10. Sequence Transformer Training (Exp 5)
- **Owner**: CHARLIE
- **Current state**: BarTransformer and TrainSequenceStage fully implemented (Sprint 4), tested with synthetic data.
- **Action**: Train on real bar latent sequences, generate bars, evaluate with generative metrics. Config: `exp_5_sequence_generation.yaml`.
- **Estimated time**: 1-2 hours.

### 11. Latent Space Probes & Silhouette Metrics (Exp 4A/4B)
- **Owner**: DELTA
- **Current state**: Metric classes implemented, tested with synthetic latent vectors.
- **Action**: Run on real encoded latents across all 12 VAEs. Configs: `exp_4_latent_analysis.yaml`, `exp_4b_latent_structure.yaml`.

---

## Phase 5: Sweep Executor & Memory Management

### 12. Verify SweepExecutor Memory Cleanup
- **Owner**: ALPHA
- **Action**: Ensure `midi_vae/pipelines/sweep.py` calls `del vae; gc.collect(); torch.cuda.empty_cache()` between VAE conditions. If not, add it. Running 12 VAEs sequentially without cleanup will OOM.

### 13. CLI Entry Point
- **Owner**: ECHO
- **Action**: Create `scripts/run_experiment.py` (or `midi_vae/__main__.py`) so experiments can be launched via `python -m midi_vae configs/experiments/exp_1a_vae_comparison.yaml`.

---

## Completion Checklist

- [x] PyTorch with CUDA installed in `.venv`
- [x] HuggingFace auth configured (all licenses accepted)
- [x] All 12 VAEs load and produce correct latent shapes
- [x] All 12 GPU integration tests pass
- [x] Real dataset (MAESTRO v3) downloaded and full pipeline validated
- [ ] Exp 1A complete (core research results)
- [ ] Remaining experiments (1B, 2, 3, 4A-D, 5) run
- [ ] All metrics computed on real data
- [ ] Results saved via ExperimentTracker
