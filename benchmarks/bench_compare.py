"""Comparison script for before/after benchmark results.

Loads baseline.json and optimized.json, then prints a side-by-side
comparison table with speedup ratios highlighting the most improved stages.

Usage:
    .venv/bin/python benchmarks/bench_compare.py
    .venv/bin/python benchmarks/bench_compare.py --baseline path/to/baseline.json \\
                                                    --optimized path/to/optimized.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _PROJECT_ROOT / "benchmarks" / "results"

DEFAULT_BASELINE = _RESULTS_DIR / "baseline.json"
DEFAULT_OPTIMIZED = _RESULTS_DIR / "optimized.json"

# ANSI colour codes (skipped when stdout is not a tty)
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_RESET = "\033[0m"
_BOLD = "\033[1m"

_USE_COLOR = sys.stdout.isatty()


def _color(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}" if _USE_COLOR else text


def _load(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as fh:
        return json.load(fh)


def _build_index(data: dict) -> dict[str, dict]:
    """Return stage_name -> result_dict mapping from a results JSON."""
    return {r["stage"]: r for r in data.get("results", [])}


def _throughput(result: dict) -> tuple[float, str]:
    """Extract the primary throughput value and its unit label."""
    stage = result.get("stage", "")
    if stage == "ingest":
        return result.get("files_per_sec", float("nan")), "files/s"
    if stage == "evaluate":
        return result.get("pairs_per_sec", float("nan")), "pairs/s"
    return result.get("bars_per_sec", float("nan")), "bars/s"


def _speedup_label(ratio: float) -> str:
    """Return a coloured speedup label."""
    label = f"{ratio:.2f}x"
    if ratio >= 1.5:
        return _color(label, _GREEN)
    if ratio >= 1.1:
        return _color(label, _YELLOW)
    return _color(label, _RED)


def compare(baseline_path: Path, optimized_path: Path) -> None:
    baseline = _load(baseline_path)
    optimized = _load(optimized_path)

    b_index = _build_index(baseline)
    o_index = _build_index(optimized)

    all_stages = list(b_index.keys())
    # Include any stages that exist only in optimized (e.g., new stages).
    for s in o_index:
        if s not in all_stages:
            all_stages.append(s)

    print(_color("\nPIPELINE BENCHMARK COMPARISON", _BOLD))
    print(f"  Baseline:  {baseline_path}")
    print(f"  Optimized: {optimized_path}")
    print(f"  Baseline  timestamp: {baseline.get('timestamp', 'n/a')}")
    print(f"  Optimized timestamp: {optimized.get('timestamp', 'n/a')}")

    col_w = [22, 12, 12, 12, 12, 9]
    header = (
        f"{'Stage':<{col_w[0]}}"
        f"{'Base(s)':>{col_w[1]}}"
        f"{'Opt(s)':>{col_w[2]}}"
        f"{'Base tput':>{col_w[3]}}"
        f"{'Opt tput':>{col_w[4]}}"
        f"{'Speedup':>{col_w[5]}}"
    )
    sep = "-" * sum(col_w)
    print("\n" + sep)
    print(header)
    print(sep)

    speedups: list[tuple[str, float]] = []

    for stage in all_stages:
        br = b_index.get(stage)
        or_ = o_index.get(stage)

        b_mean = br["mean_s"] if br else float("nan")
        o_mean = or_["mean_s"] if or_ else float("nan")

        b_tput, unit = _throughput(br) if br else (float("nan"), "?/s")
        o_tput, _ = _throughput(or_) if or_ else (float("nan"), unit)

        if b_mean > 0 and o_mean > 0:
            ratio = b_mean / o_mean
        else:
            ratio = float("nan")

        speedups.append((stage, ratio))

        b_mean_str = f"{b_mean:.3f}" if br else "n/a"
        o_mean_str = f"{o_mean:.3f}" if or_ else "n/a"
        b_tput_str = f"{b_tput:.1f} {unit}" if br else "n/a"
        o_tput_str = f"{o_tput:.1f} {unit}" if or_ else "n/a"
        ratio_str = _speedup_label(ratio) if ratio == ratio else "n/a"  # nan check

        print(
            f"{stage:<{col_w[0]}}"
            f"{b_mean_str:>{col_w[1]}}"
            f"{o_mean_str:>{col_w[2]}}"
            f"{b_tput_str:>{col_w[3]}}"
            f"{o_tput_str:>{col_w[4]}}"
            f"{ratio_str:>{col_w[5]}}"
        )

    print(sep)

    # Most improved stage
    valid = [(s, r) for s, r in speedups if r == r and r > 0]
    if valid:
        best_stage, best_ratio = max(valid, key=lambda x: x[1])
        print(
            f"\nMost improved: {_color(best_stage, _BOLD)}  "
            f"({_speedup_label(best_ratio)} faster)"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two pipeline benchmark JSON result files."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Path to baseline results JSON (default: benchmarks/results/baseline.json)",
    )
    parser.add_argument(
        "--optimized",
        type=Path,
        default=DEFAULT_OPTIMIZED,
        help="Path to optimized results JSON (default: benchmarks/results/optimized.json)",
    )
    args = parser.parse_args()
    compare(args.baseline, args.optimized)


if __name__ == "__main__":
    main()
