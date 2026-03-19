#!/usr/bin/env python
"""Merge per-rank sweep_summary_rank*.json files into a single sweep_summary.json.

This utility is called automatically by ``run_experiment_distributed.py`` after
all single-node subprocesses complete.  For multi-node SLURM runs it must be
invoked manually after all ``srun`` ranks have written their rank files.

Usage
-----
Merge all rank files found under an output directory:

    python scripts/merge_sweep_results.py /scratch/.../outputs/exp_1a/

Explicit output path:

    python scripts/merge_sweep_results.py /scratch/.../outputs/exp_1a/ \\
        --output /scratch/.../outputs/exp_1a/sweep_summary.json

Scan rank subdirectories (created by ``run_experiment_distributed.py``):

    python scripts/merge_sweep_results.py /scratch/.../outputs/exp_1a/ \\
        --scan-rank-dirs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge per-rank sweep summary JSON files into one.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "output_dir",
        help=(
            "Directory to scan for sweep_summary_rank*.json files.  "
            "When --scan-rank-dirs is given, subdirectories named rank_XXXX/ "
            "are also searched."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Destination path for the merged JSON file.  "
            "Defaults to <output_dir>/sweep_summary.json."
        ),
    )
    parser.add_argument(
        "--scan-rank-dirs",
        action="store_true",
        default=False,
        help=(
            "Also search rank_XXXX/ subdirectories under output_dir for "
            "sweep_summary_rank*.json files.  This matches the directory layout "
            "produced by run_experiment_distributed.py."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-file details while merging.",
    )
    return parser.parse_args()


def _find_rank_files(output_dir: Path, scan_rank_dirs: bool) -> list[Path]:
    """Collect all sweep_summary_rank*.json files.

    Args:
        output_dir: Base directory to scan.
        scan_rank_dirs: If True, also look inside rank_XXXX/ subdirectories.

    Returns:
        Sorted list of Path objects pointing to rank summary files.
    """
    found: list[Path] = []

    # Direct files in output_dir
    found.extend(sorted(output_dir.glob("sweep_summary_rank*.json")))

    if scan_rank_dirs:
        # rank_XXXX/ subdirectories created by the distributed launcher
        for rank_dir in sorted(output_dir.glob("rank_*")):
            if rank_dir.is_dir():
                found.extend(sorted(rank_dir.glob("sweep_summary_rank*.json")))

    return found


def merge_rank_files(rank_files: list[Path], verbose: bool = False) -> dict:
    """Read and merge a list of rank summary JSON files.

    The merge is additive:
    * ``conditions`` lists are concatenated in the order files are provided.
    * ``metrics_summary`` dicts are merged (later files overwrite duplicate keys).
    * ``elapsed_seconds`` values are summed to give total wall time.

    Args:
        rank_files: Ordered list of rank summary file paths.
        verbose: Print per-file info to stdout.

    Returns:
        Merged summary dict ready to be written as JSON.
    """
    all_conditions: list[str] = []
    all_metrics: dict[str, dict] = {}
    all_condition_indices: list[int] = []
    total_elapsed = 0.0
    num_ranks_seen = 0

    for path in rank_files:
        try:
            with path.open() as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
            continue

        rank = data.get("rank", "?")
        conditions = data.get("conditions", [])
        metrics = data.get("metrics_summary", {})
        elapsed = data.get("elapsed_seconds", 0.0)
        indices = data.get("condition_indices", [])

        if verbose:
            print(f"  Rank {rank}: {len(conditions)} condition(s), {elapsed:.1f}s")

        all_conditions.extend(conditions)
        all_metrics.update(metrics)
        all_condition_indices.extend(indices)
        total_elapsed += elapsed
        num_ranks_seen += 1

    return {
        "num_ranks": num_ranks_seen,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "num_conditions": len(all_conditions),
        "condition_indices": sorted(all_condition_indices),
        "conditions": all_conditions,
        "metrics_summary": all_metrics,
    }


def main() -> int:
    """Entry point.  Returns exit code."""
    args = parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"ERROR: output_dir does not exist: {output_dir}", file=sys.stderr)
        return 1

    rank_files = _find_rank_files(output_dir, args.scan_rank_dirs)

    if not rank_files:
        print(
            f"ERROR: No sweep_summary_rank*.json files found under {output_dir}. "
            "Did all rank processes complete?",
            file=sys.stderr,
        )
        return 1

    print(f"Found {len(rank_files)} rank file(s):")
    for f in rank_files:
        print(f"  {f}")
    print()

    merged = merge_rank_files(rank_files, verbose=args.verbose)

    output_path = Path(args.output) if args.output else output_dir / "sweep_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as fh:
        json.dump(merged, fh, indent=2, default=str)

    print(f"\nMerged summary ({merged['num_conditions']} conditions, "
          f"{merged['num_ranks']} rank(s)) written to:")
    print(f"  {output_path}")

    if merged["metrics_summary"]:
        print("\nMetrics summary (per-condition):")
        for label, metrics in merged["metrics_summary"].items():
            print(f"  {label}")
            for k, v in metrics.items():
                fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
                print(f"    {k}: {fmt}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
