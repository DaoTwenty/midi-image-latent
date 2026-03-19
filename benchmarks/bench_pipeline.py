#!/usr/bin/env python
"""Pipeline performance benchmarks — all optimization combinations.

Compares vanilla (original) vs each optimization and all combinations:
  - Ingestion: pretty_midi vs symusic, sequential vs parallel
  - Rendering: per-bar sequential vs batched vectorized
  - Detection: scipy per-row vs numpy vectorized, sequential vs batch
  - Metrics: sequential vs parallel (ThreadPoolExecutor)

Usage
-----
    .venv/bin/python benchmarks/bench_pipeline.py [--files N] [--repeats N] [--tag TAG]
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import numpy as np
import torch

from midi_vae.data.preprocessing import MidiIngestor
from midi_vae.data.rendering import build_strategy
from midi_vae.data.transforms import ResizeTransform
from midi_vae.data.types import BarData, PianoRollImage, ReconstructedBar
from midi_vae.note_detection.threshold import GlobalThresholdDetector
from midi_vae.metrics.base import MetricsEngine
import midi_vae.metrics  # noqa: F401
import midi_vae.note_detection  # noqa: F401

logging.basicConfig(level=logging.WARNING)


# ── Timing ───────────────────────────────────────────────────────────────────

def timed(fn, repeats: int = 3) -> tuple[float, float, Any]:
    """Run fn() repeats times. Returns (mean_sec, std_sec, last_result)."""
    times = []
    result = None
    for i in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        # Free intermediate results to avoid OOM from accumulated BarData
        if i < repeats - 1:
            del result
            result = None
            gc.collect()
    return statistics.mean(times), (statistics.stdev(times) if len(times) > 1 else 0.0), result


def _noise(img: torch.Tensor) -> torch.Tensor:
    return (img + torch.randn_like(img) * 0.1).clamp(-1.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# INGESTION BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════

def _make_ingestor():
    return MidiIngestor(
        time_steps=96, min_notes_per_bar=2,
        instruments=["piano", "bass", "drums", "guitar", "strings"],
        bars_per_instrument=50, seed=42,
    )


def bench_ingest_pretty_midi_sequential(files: list[Path], repeats: int) -> dict:
    """Vanilla: pretty_midi, sequential."""
    ing = _make_ingestor()
    def run():
        bars = []
        for f in files:
            try:
                bars.extend(ing._ingest_midi(f))
            except Exception:
                pass
        return bars
    mean, std, bars = timed(run, repeats)
    n = len(bars) if bars else 0
    return {"stage": "ingest/pretty_midi/seq", "n_files": len(files), "n_items": n,
            "mean_sec": round(mean, 4), "std_sec": round(std, 4),
            "throughput": round(len(files) / mean, 2) if mean > 0 else 0, "unit": "files/s"}


def bench_ingest_symusic_sequential(files: list[Path], repeats: int) -> dict:
    """symusic parser, sequential."""
    ing = _make_ingestor()
    def run():
        bars = []
        for f in files:
            try:
                bars.extend(ing._ingest_midi_symusic(f))
            except Exception:
                try:
                    bars.extend(ing._ingest_midi(f))
                except Exception:
                    pass
        return bars
    mean, std, bars = timed(run, repeats)
    n = len(bars) if bars else 0
    return {"stage": "ingest/symusic/seq", "n_files": len(files), "n_items": n,
            "mean_sec": round(mean, 4), "std_sec": round(std, 4),
            "throughput": round(len(files) / mean, 2) if mean > 0 else 0, "unit": "files/s"}


def bench_ingest_pretty_midi_parallel(files: list[Path], repeats: int, workers: int) -> dict:
    """pretty_midi, parallel (but each worker uses pretty_midi)."""
    # Temporarily make ingest_file use only pretty_midi
    ing = _make_ingestor()
    def run():
        return ing.ingest_files_parallel(files, max_workers=workers)
    mean, std, bars = timed(run, repeats)
    n = len(bars) if bars else 0
    return {"stage": f"ingest/default/par{workers}", "n_files": len(files), "n_items": n,
            "mean_sec": round(mean, 4), "std_sec": round(std, 4),
            "throughput": round(len(files) / mean, 2) if mean > 0 else 0, "unit": "files/s"}


def bench_ingest_symusic_parallel(files: list[Path], repeats: int, workers: int) -> dict:
    """symusic + parallel (the default ingest_file now tries symusic first)."""
    ing = _make_ingestor()
    def run():
        return ing.ingest_files_parallel(files, max_workers=workers)
    mean, std, bars = timed(run, repeats)
    n = len(bars) if bars else 0
    return {"stage": f"ingest/symusic+par{workers}", "n_files": len(files), "n_items": n,
            "mean_sec": round(mean, 4), "std_sec": round(std, 4),
            "throughput": round(len(files) / mean, 2) if mean > 0 else 0, "unit": "files/s"}


# ══════════════════════════════════════════════════════════════════════════════
# RENDERING BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════

def bench_render_sequential(bars: list[BarData], repeats: int) -> dict:
    """Vanilla: per-bar render + per-image resize."""
    strategy = build_strategy("velocity_only", pitch_axis="height")
    resize = ResizeTransform(target_resolution=(128, 128), method="bilinear")
    def run():
        return [resize(strategy.render(bar)) for bar in bars]
    mean, std, _ = timed(run, repeats)
    return {"stage": "render/sequential", "n_items": len(bars),
            "mean_sec": round(mean, 4), "std_sec": round(std, 4),
            "throughput": round(len(bars) / mean, 2) if mean > 0 else 0, "unit": "bars/s"}


def bench_render_batched(bars: list[BarData], repeats: int) -> dict:
    """Optimized: vectorized render_batch + single F.interpolate."""
    strategy = build_strategy("velocity_only", pitch_axis="height")
    resize = ResizeTransform(target_resolution=(128, 128), method="bilinear")
    def run():
        return resize.batch_call(strategy.render_batch(bars))
    mean, std, _ = timed(run, repeats)
    return {"stage": "render/batched", "n_items": len(bars),
            "mean_sec": round(mean, 4), "std_sec": round(std, 4),
            "throughput": round(len(bars) / mean, 2) if mean > 0 else 0, "unit": "bars/s"}


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════

def bench_detect_sequential(images: list[torch.Tensor], repeats: int) -> dict:
    """Per-image detect() calls (uses vectorized internals now)."""
    det = GlobalThresholdDetector(params={"threshold": 0.5})
    def run():
        return [det.detect(img, "velocity_only") for img in images]
    mean, std, _ = timed(run, repeats)
    return {"stage": "detect/sequential", "n_items": len(images),
            "mean_sec": round(mean, 4), "std_sec": round(std, 4),
            "throughput": round(len(images) / mean, 2) if mean > 0 else 0, "unit": "imgs/s"}


def bench_detect_batch(images: list[torch.Tensor], repeats: int) -> dict:
    """Batched detect_batch() call."""
    det = GlobalThresholdDetector(params={"threshold": 0.5})
    def run():
        return det.detect_batch(images, "velocity_only")
    mean, std, _ = timed(run, repeats)
    return {"stage": "detect/batch", "n_items": len(images),
            "mean_sec": round(mean, 4), "std_sec": round(std, 4),
            "throughput": round(len(images) / mean, 2) if mean > 0 else 0, "unit": "imgs/s"}


# ══════════════════════════════════════════════════════════════════════════════
# METRICS BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════

def _build_metric_pairs(bars, recon_imgs, pr_images):
    det = GlobalThresholdDetector(params={"threshold": 0.5})
    recon_bars = [
        ReconstructedBar(bar_id=b.bar_id, vae_name="mock", recon_image=ri,
                         detected_notes=det.detect(ri, "velocity_only"),
                         detection_method="global_threshold")
        for b, ri in zip(bars, recon_imgs)
    ]
    return list(zip(bars, recon_bars)), {img.bar_id: img for img in pr_images}


def bench_metrics_sequential(bars, recon_imgs, pr_images, repeats: int) -> dict:
    pairs, img_lookup = _build_metric_pairs(bars, recon_imgs, pr_images)
    engine = MetricsEngine(["reconstruction", "harmony", "dynamics"])
    def run():
        return engine.evaluate_batch(pairs, images_by_id=img_lookup)
    mean, std, _ = timed(run, repeats)
    return {"stage": "metrics/sequential", "n_items": len(pairs),
            "mean_sec": round(mean, 4), "std_sec": round(std, 4),
            "throughput": round(len(pairs) / mean, 2) if mean > 0 else 0, "unit": "pairs/s"}


def bench_metrics_parallel(bars, recon_imgs, pr_images, repeats: int, workers: int) -> dict:
    pairs, img_lookup = _build_metric_pairs(bars, recon_imgs, pr_images)
    engine = MetricsEngine(["reconstruction", "harmony", "dynamics"])
    def run():
        return engine.evaluate_batch_parallel(pairs, images_by_id=img_lookup, max_workers=workers)
    mean, std, _ = timed(run, repeats)
    return {"stage": f"metrics/parallel_{workers}w", "n_items": len(pairs),
            "mean_sec": round(mean, 4), "std_sec": round(std, 4),
            "throughput": round(len(pairs) / mean, 2) if mean > 0 else 0, "unit": "pairs/s"}


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def display(results: list[dict]) -> None:
    """Print a formatted results table with speedup computations."""
    W = 80
    print(f"\n{'='*W}")
    print(f"{'ALL BENCHMARK RESULTS':^{W}}")
    print(f"{'='*W}")
    print(f"  {'Stage':<32} {'N':>6} {'Time (s)':>16} {'Throughput':>20}")
    print(f"  {'-'*32} {'-'*6} {'-'*16} {'-'*20}")
    for r in results:
        t = f"{r['mean_sec']:.4f}±{r['std_sec']:.4f}"
        tp = f"{r['throughput']} {r['unit']}"
        print(f"  {r['stage']:<32} {r['n_items']:>6} {t:>16} {tp:>20}")

    # Speedup analysis — group by category
    print(f"\n{'='*W}")
    print(f"{'SPEEDUP ANALYSIS':^{W}}")
    print(f"{'='*W}")
    by_stage = {r["stage"]: r for r in results}

    # Auto-detect worker count from results
    par_stages = [k for k in by_stage if k.startswith("ingest/default/par")]
    w = par_stages[0].split("par")[1] if par_stages else "4"
    met_par = [k for k in by_stage if k.startswith("metrics/parallel_")]
    mw = met_par[0] if met_par else f"metrics/parallel_{w}w"

    groups = {
        "INGESTION": [
            ("ingest/pretty_midi/seq", "ingest/symusic/seq", "symusic only"),
            ("ingest/pretty_midi/seq", f"ingest/default/par{w}", f"parallel {w}w only"),
            ("ingest/pretty_midi/seq", f"ingest/symusic+par{w}", f"symusic + parallel {w}w"),
        ],
        "RENDERING": [
            ("render/sequential", "render/batched", "batched vectorized"),
        ],
        "DETECTION": [
            ("detect/sequential", "detect/batch", "batched"),
        ],
        "METRICS": [
            ("metrics/sequential", mw, f"parallel {w} workers"),
        ],
    }

    for category, comparisons in groups.items():
        print(f"\n  {category}:")
        for base_name, opt_name, desc in comparisons:
            base = by_stage.get(base_name)
            opt = by_stage.get(opt_name)
            if base and opt and base["mean_sec"] > 0 and opt["mean_sec"] > 0:
                speedup = base["mean_sec"] / opt["mean_sec"]
                print(f"    {desc:<36} {speedup:>6.2f}x  ({base['mean_sec']:.4f}s -> {opt['mean_sec']:.4f}s)")
            else:
                print(f"    {desc:<36} N/A")
    print(f"\n{'='*W}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pipeline benchmarks — all combinations")
    parser.add_argument("--files", type=int, default=20, help="MIDI files to use")
    parser.add_argument("--repeats", type=int, default=3, help="Reps per benchmark")
    parser.add_argument("--tag", type=str, default="benchmark", help="Results file tag")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    data_root = _ROOT / "data" / "lakh" / "7"
    midi_files = sorted(data_root.glob("*.mid"))[:args.files]
    print(f"Pipeline Performance Benchmark — All Combinations")
    print(f"  Files: {len(midi_files)}, Repeats: {args.repeats}, Workers: {args.workers}\n")

    if not midi_files:
        print("ERROR: No MIDI files found!")
        sys.exit(1)

    results: list[dict] = []

    # ── INGESTION ───────────────────────────────────────────────────────
    print("=== INGESTION ===")
    for label, fn in [
        ("pretty_midi/seq", lambda: bench_ingest_pretty_midi_sequential(midi_files, args.repeats)),
        ("symusic/seq", lambda: bench_ingest_symusic_sequential(midi_files, args.repeats)),
        (f"default/par{args.workers}", lambda: bench_ingest_pretty_midi_parallel(midi_files, args.repeats, args.workers)),
        (f"symusic+par{args.workers}", lambda: bench_ingest_symusic_parallel(midi_files, args.repeats, args.workers)),
    ]:
        r = fn()
        results.append(r)
        print(f"  {r['stage']:<36} {r['mean_sec']:.4f}s  ({r['throughput']} {r['unit']})")
        gc.collect()  # Free accumulated BarData between benchmarks

    # Get bars for downstream benchmarks
    ing = _make_ingestor()
    bars: list[BarData] = []
    for f in midi_files:
        bars.extend(ing.ingest_file(f))
    bars = bars[:300]
    print(f"  -> {len(bars)} bars for downstream benchmarks\n")

    # ── RENDERING ───────────────────────────────────────────────────────
    print("=== RENDERING ===")
    for fn in [
        lambda: bench_render_sequential(bars, args.repeats),
        lambda: bench_render_batched(bars, args.repeats),
    ]:
        r = fn()
        results.append(r)
        print(f"  {r['stage']:<36} {r['mean_sec']:.4f}s  ({r['throughput']} {r['unit']})")

    # Prepare images
    strategy = build_strategy("velocity_only", pitch_axis="height")
    resize = ResizeTransform(target_resolution=(128, 128), method="bilinear")
    rendered, pr_images = [], []
    for bar in bars:
        try:
            t = resize(strategy.render(bar))
            rendered.append(t)
            pr_images.append(PianoRollImage(
                bar_id=bar.bar_id, image=t, channel_strategy="velocity_only",
                resolution=tuple(t.shape[1:]), pitch_axis="height",
            ))
        except Exception:
            pass
    recon_imgs = [_noise(img) for img in rendered]
    print()

    # ── DETECTION ───────────────────────────────────────────────────────
    print("=== DETECTION ===")
    for fn in [
        lambda: bench_detect_sequential(recon_imgs, args.repeats),
        lambda: bench_detect_batch(recon_imgs, args.repeats),
    ]:
        r = fn()
        results.append(r)
        print(f"  {r['stage']:<36} {r['mean_sec']:.4f}s  ({r['throughput']} {r['unit']})")
    print()

    # ── METRICS ─────────────────────────────────────────────────────────
    n_m = min(80, len(bars))
    m_bars, m_recons, m_imgs = bars[:n_m], recon_imgs[:n_m], pr_images[:n_m]
    print(f"=== METRICS ({n_m} pairs) ===")
    for fn in [
        lambda: bench_metrics_sequential(m_bars, m_recons, m_imgs, args.repeats),
        lambda: bench_metrics_parallel(m_bars, m_recons, m_imgs, args.repeats, args.workers),
    ]:
        r = fn()
        results.append(r)
        print(f"  {r['stage']:<36} {r['mean_sec']:.4f}s  ({r['throughput']} {r['unit']})")

    # ── Display full results ────────────────────────────────────────────
    display(results)

    # ── Save ────────────────────────────────────────────────────────────
    out_path = _ROOT / "benchmarks" / "results" / f"{args.tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": args.tag,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "config": {"n_files": len(midi_files), "n_bars": len(bars),
                   "n_metric_pairs": n_m, "repeats": args.repeats, "workers": args.workers},
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
