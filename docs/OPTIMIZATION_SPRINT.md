# Optimization Sprint — CPU Pipeline Performance

**Date**: 2026-03-18
**Goal**: Maximize CPU throughput for all non-GPU pipeline stages
**Environment**: 8 CPUs, 16 GB RAM, no GPU (interactive node)

## Pipeline Stages Optimized

| Stage | What it does | Original Bottleneck |
|-------|-------------|---------------------|
| **Ingest** | Parse MIDI files → BarData objects | `pretty_midi` Python parsing, sequential file loading |
| **Render** | BarData → (3,H,W) image tensors | Per-bar numpy→torch, per-image F.interpolate |
| **Detect** | Reconstructed images → MidiNote lists | Per-row `scipy.ndimage.label` (128 calls/bar) |
| **Evaluate** | (gt, recon) pairs → metric dicts | Sequential per-pair metric computation |

## Optimizations Applied

### 1. symusic Backend for MIDI Parsing
- **Before**: `pretty_midi.PrettyMIDI()` — pure Python MIDI parser
- **After**: `symusic.Score()` — Rust-based parser + custom numpy pianoroll builder
- **Measured**: 45x faster file parsing (0.36ms vs 16ms), 12x faster end-to-end per file
- **Implementation**: `_ingest_midi_symusic()` with automatic fallback to `pretty_midi`

### 2. Parallel MIDI Ingestion
- **Before**: Sequential `for f in files: ingest_file(f)`
- **After**: `concurrent.futures.ProcessPoolExecutor` with configurable workers
- **Implementation**: `MidiIngestor.ingest_files_parallel()`, activated when `config.num_workers > 1`
- **Key detail**: Deterministic ordering by sorted file path

### 3. Batched Rendering
- **Before**: Per-bar `strategy.render(bar)` + per-image `resize(tensor)`
- **After**: `strategy.render_batch(bars)` → single `np.stack` + vectorized normalize + one `torch.from_numpy`, then `resize.batch_call(batch)` → single `F.interpolate`
- **Implementation**: `ChannelStrategy.render_batch()` on all 3 strategies, `ResizeTransform.batch_call()`

### 4. Vectorized Note Detection
- **Before**: `scipy.ndimage.label` called per pitch row (128 times per image)
- **After**: Pure numpy `np.diff`-based connected components, skip silent rows via `np.any(binary, axis=1)`, shared `_detect_notes_from_binary_2d` helper
- **Implementation**: `_connected_components_1d_fast()` replaces scipy, `detect_batch()` for multiple images

### 5. Parallel Metrics Computation
- **Before**: Sequential `engine.evaluate_batch(pairs)`
- **After**: `engine.evaluate_batch_parallel(pairs, max_workers=N)` using `ThreadPoolExecutor`
- **Implementation**: Per-pair evaluation is independent → trivially parallelizable

## Benchmark Results

**Config**: 20 MIDI files from Lakh dataset, 300 bars, 80 metric pairs, 3 repeats, 4 workers

| Stage | Sequential | Optimized | Speedup |
|-------|-----------|-----------|---------|
| Ingestion (pretty_midi) | 1.43s (14 files/s) | 0.70s (29 files/s) | **2.05x** (parallel only) |
| Rendering | 0.033s (9.2K bars/s) | 0.012s (25.5K bars/s) | **2.75x** |
| Detection | 0.018s (16.3K imgs/s) | 0.018s (16.6K imgs/s) | 1.02x |
| Metrics | 1.29s (62 pairs/s) | 0.24s (334 pairs/s) | **5.38x** |

### symusic vs pretty_midi (micro-benchmark, 100 iterations)

| Operation | pretty_midi | symusic | Speedup |
|-----------|------------|---------|---------|
| File parse only | 16.06ms/file | 0.36ms/file | **45x** |
| Parse + pianoroll | 21.23ms/file | 1.77ms/file | **12x** |

## How to Run Benchmarks

```bash
# Full benchmark with before/after comparison
.venv/bin/python benchmarks/bench_pipeline.py --files 20 --repeats 3 --workers 4

# Quick smoke test
.venv/bin/python benchmarks/bench_pipeline.py --files 5 --repeats 1

# Results saved to benchmarks/results/<tag>.json
.venv/bin/python benchmarks/bench_pipeline.py --tag my_test
```

## Files Changed

| File | Change |
|------|--------|
| `midi_vae/data/preprocessing.py` | `ingest_files_parallel()`, `_ingest_midi_symusic()` |
| `midi_vae/data/rendering.py` | `render_batch()` on all 3 strategies |
| `midi_vae/data/transforms.py` | `ResizeTransform.batch_call()` |
| `midi_vae/note_detection/threshold.py` | `_connected_components_1d_fast()`, `_detect_notes_from_binary_2d()`, `detect_batch()` |
| `midi_vae/metrics/base.py` | `MetricsEngine.evaluate_batch_parallel()` |
| `midi_vae/pipelines/ingest.py` | Uses parallel path when `num_workers > 1` |
| `midi_vae/pipelines/render.py` | Uses batched render + resize |
| `midi_vae/pipelines/detect.py` | Uses `detect_batch` when available |
| `midi_vae/pipelines/evaluate.py` | Uses parallel evaluation when `num_workers > 1` |
| `benchmarks/bench_pipeline.py` | Comprehensive benchmark script |
| `docs/OPTIMIZATION_SPRINT.md` | This document |
