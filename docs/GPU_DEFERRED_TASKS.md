# GPU-Deferred Tasks

Tasks that require GPU hardware. Currently **CPU-only** — these are parked until GPU access is available.

## How to Use This File
- Agents: check this file before implementing GPU-dependent features. If a task is listed here, implement a **CPU stub/mock** and note the limitation.
- Orchestrator: when GPU becomes available, work through this list and spawn the appropriate agent to complete each item.

---

## Deferred Tasks

### 1. VAE Model Weight Downloads & Real Inference
- **Owner**: CHARLIE
- **Why GPU**: All 12 VAE wrappers (SD 1.5/2.1/XL, FLUX, CogView4, SD3 etc.) require GPU VRAM for `encode()`/`decode()`. HuggingFace diffusers will OOM or refuse on CPU for large models.
- **Current state**: Wrappers are implemented with correct API shapes; tests use synthetic tensors and mock the diffusers autoencoder.
- **When GPU available**: Run real encode/decode on sample images, verify latent shapes match spec, benchmark throughput.

### 2. Full Encode-Decode Pipeline Integration Test
- **Owner**: ECHO
- **Why GPU**: Needs real VAE forward passes to validate the full ingest -> render -> encode -> decode -> detect -> evaluate pipeline.
- **Current state**: Integration test stubs exist using mocked VAE outputs.
- **When GPU available**: Run `tests/test_integration_encode_decode.py` with `@pytest.mark.gpu` enabled.

### 3. Sub-Latent Model Training (MLP, Sub-VAE)
- **Owner**: CHARLIE
- **Why GPU**: Training loops for MLP projection and sub-VAE require GPU for reasonable speed on real latent data.
- **Current state**: Training pipeline code is written, tested with tiny synthetic data on CPU.
- **When GPU available**: Train on real VAE latents from LPD5 subset, validate reconstruction quality.

### 4. CNN Note Detector Training & Inference
- **Owner**: DELTA
- **Why GPU**: CNN segmenter requires GPU for training and fast inference.
- **Current state**: Architecture defined, forward pass tested on CPU with small inputs.
- **When GPU available**: Train on rendered piano-roll / ground-truth note pairs.

### 5. Sequence Transformer (Bar-Level)
- **Owner**: CHARLIE
- **Why GPU**: Transformer training on latent bar sequences needs GPU.
- **Current state**: Not yet implemented (Sprint 3).
- **When GPU available**: Train on sequences of bar latents, evaluate generative metrics.

### 6. Latent Space Probes & Silhouette Metrics
- **Owner**: DELTA
- **Why GPU**: Computing latent embeddings for probe classifiers and silhouette scores over full dataset.
- **Current state**: Metric classes implemented, tested with synthetic latent vectors.
- **When GPU available**: Run on real encoded latents across all 12 VAEs.

### 7. Full Experiment Sweeps
- **Owner**: ALPHA (sweep executor)
- **Why GPU**: Running all 5 experiments across 12 VAEs with full datasets.
- **Current state**: Sweep executor and pipeline stages being built (Sprint 3).
- **When GPU available**: Execute experiment YAML configs end-to-end.

---

## Adding New Deferred Tasks
Append to this list with:
```
### N. Task Name
- **Owner**: AGENT_NAME
- **Why GPU**: reason
- **Current state**: what exists now
- **When GPU available**: what to do
```
