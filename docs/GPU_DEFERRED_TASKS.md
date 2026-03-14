# GPU-Deferred Tasks

Tasks that require GPU hardware. All codebase implementation is **complete** (517 tests passing). These tasks are about running real models and experiments.

> **See also**: `docs/GPU_SESSION_GUIDE.md` for detailed execution instructions, VRAM budgets, and troubleshooting tips.

---

## Phase 1: Validation (Do First)

### 1. VAE Model Weight Downloads & Real Inference
- **Owner**: CHARLIE
- **Current state**: All 12 VAE wrappers implemented with correct API shapes. Tests use synthetic tensors and mock the diffusers autoencoder.
- **Action**: Download weights for all 12 models, run real encode/decode on synthetic 128x128 images, verify latent shapes match spec (4-ch models -> (B,4,16,16), 16-ch models -> (B,16,16,16)).
- **Watch for**: HuggingFace gated model auth (FLUX, SD3 need license acceptance). `flux2_tiny` uses `transformers.AutoModel` not `diffusers.AutoencoderKL` — may need API adjustments.

### 2. Full Encode-Decode Pipeline Integration Test
- **Owner**: ECHO
- **Current state**: Integration test at `tests/test_integration_encode_decode.py:409` is marked `@pytest.mark.gpu` and skipped.
- **Action**: Run `pytest tests/test_integration_encode_decode.py -m gpu -v --tb=long`. Fix any issues.

---

## Phase 2: Data Pipeline

### 3. LPD5 Dataset Download & Ingest Test
- **Owner**: BRAVO
- **Current state**: `LPD5Dataset` and `MidiIngestor` implemented and tested with synthetic data. The real LPD5 dataset has not been downloaded.
- **Action**: Download LPD5 cleansed (~3.3 GB). Extract to `data/lpd5/`. Run IngestStage + RenderStage on a small subset (20 songs) to validate the full MIDI -> BarData -> PianoRollImage pipeline with real data.

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

- [ ] PyTorch with CUDA installed in `.venv`
- [ ] HuggingFace auth configured (gated models)
- [ ] All 12 VAEs load and produce correct latent shapes
- [ ] GPU integration test passes
- [ ] LPD5 dataset downloaded and ingest verified
- [ ] Exp 1A complete (core research results)
- [ ] Remaining experiments (1B, 2, 3, 4A-D, 5) run
- [ ] All metrics computed on real data
- [ ] Results saved via ExperimentTracker
