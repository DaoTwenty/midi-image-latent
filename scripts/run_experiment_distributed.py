#!/usr/bin/env python
"""Distributed multi-GPU experiment runner.

Splits sweep conditions across multiple GPUs (one subprocess per GPU, each
subprocess runs an independent subset of conditions).  Works both for same-node
multi-GPU and for multi-node setups via SLURM environment variables.

Single-node example (auto-detect GPUs):
    python scripts/run_experiment_distributed.py configs/experiments/exp_1a_vae_comparison.yaml

Single-node example (explicit GPU count):
    python scripts/run_experiment_distributed.py configs/experiments/exp_1a_vae_comparison.yaml \\
        --num-gpus 4 --sweep-strategies

Multi-node (called via srun — each rank handles its own slice):
    srun python scripts/run_experiment_distributed.py \\
        configs/experiments/exp_1a_vae_comparison.yaml \\
        --sweep-strategies --multi-node

Merge results after a multi-node run:
    python scripts/merge_sweep_results.py outputs/exp_1a/ --output outputs/exp_1a/sweep_summary.json

Design notes
------------
* subprocess-based: each GPU process is completely independent — a crash in one
  process does not kill others.
* CUDA_VISIBLE_DEVICES is set per subprocess so the VAE loads on the right GPU.
* Conditions are assigned round-robin across ranks for balanced workload.
* Each rank writes its results to ``<output_root>/rank_<rank>/sweep_summary_rank<rank>.json``.
* When all ranks finish the launcher merges those files into a single
  ``<output_root>/sweep_summary.json``.
* For multi-node SLURM runs each ``srun`` invocation is its own rank; the
  launcher exits immediately after spawning (no waiting) and the merge step
  must be run separately via ``scripts/merge_sweep_results.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Repo root and .env loading (same pattern as run_experiment.py)
# ---------------------------------------------------------------------------

def _load_dotenv(env_path: Path) -> None:
    """Parse key=value pairs from a .env file and set missing env vars."""
    if not env_path.exists():
        return
    with env_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_REPO_ROOT = Path(__file__).resolve().parent.parent
_load_dotenv(_REPO_ROOT / ".env")

# ---------------------------------------------------------------------------
# Imports (safe after .env is loaded)
# ---------------------------------------------------------------------------

from omegaconf import OmegaConf, DictConfig, open_dict  # noqa: E402

from midi_vae.config import ExperimentConfig  # noqa: E402
from midi_vae.pipelines.sweep import SweepExecutor  # noqa: E402
from midi_vae.utils.logging import get_logger, setup_logging  # noqa: E402


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the distributed runner."""
    parser = argparse.ArgumentParser(
        description="Distributed multi-GPU MIDI Image VAE experiment runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("config", help="Path to experiment YAML config file.")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of GPUs to use.  Auto-detects if not specified.  "
            "Set to 1 to run all conditions on a single GPU sequentially."
        ),
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        default=False,
        help="Mini mode: 20 bars, 5 files, first 2 VAEs only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print conditions and GPU assignments without running.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        metavar="PATH",
        help="Override paths.data_root in the config.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        metavar="PATH",
        help="Override paths.output_root in the config.",
    )
    parser.add_argument(
        "--override-config",
        default=None,
        metavar="PATH",
        help="Path to a secondary YAML whose values override the primary config.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        metavar="STAGE",
        help="Pipeline stage name to resume from (skips earlier stages).",
    )
    parser.add_argument(
        "--sweep-detectors",
        action="store_true",
        default=False,
        help="Sweep over all detection_methods listed in the config.",
    )
    parser.add_argument(
        "--sweep-strategies",
        action="store_true",
        default=False,
        help="Sweep over all channel_strategies listed in the config.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        metavar="N",
        help="Limit number of files loaded per instrument.",
    )
    parser.add_argument(
        "--subset-dir",
        default=None,
        metavar="SUBDIR",
        help="Only load files from this subdirectory under data_root.",
    )
    parser.add_argument(
        "--subset-fraction",
        type=float,
        default=None,
        metavar="FRAC",
        help="Random fraction of files to use (e.g. 0.1 for 10%%).",
    )
    parser.add_argument(
        "--subset-file-list",
        default=None,
        metavar="PATH",
        help="Path to a text file listing specific files to use.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level (default: INFO).",
    )
    parser.add_argument(
        "--multi-node",
        action="store_true",
        default=False,
        help=(
            "Multi-node mode: read SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID "
            "to determine this rank's GPU and condition subset.  The script runs "
            "its own slice and exits — no subprocess spawning."
        ),
    )
    parser.add_argument(
        "--assign-strategy",
        choices=["round-robin", "contiguous", "cost-balanced"],
        default="cost-balanced",
        help=(
            "How to assign conditions to ranks. "
            "'round-robin': cycle through ranks in index order. "
            "'contiguous': give each rank a contiguous block of conditions. "
            "'cost-balanced': greedy LPT algorithm using estimated VAE compute "
            "costs for even wall-clock load distribution (default)."
        ),
    )
    parser.add_argument(
        "--no-cpu-pinning",
        action="store_true",
        default=False,
        help=(
            "Disable taskset CPU core pinning when launching GPU subprocesses. "
            "Use this on systems where taskset is unavailable or restricted."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config helpers (mirrors run_experiment.py)
# ---------------------------------------------------------------------------

def _apply_mini_overrides(raw: DictConfig, max_files: int | None) -> DictConfig:
    OmegaConf.update(raw, "data.bars_per_instrument", 20, merge=True)
    if OmegaConf.select(raw, "vaes") and len(raw.vaes) > 2:
        with open_dict(raw):
            raw.vaes = list(raw.vaes)[:2]
    effective_max = max_files if max_files is not None else 5
    OmegaConf.update(raw, "data.max_files", effective_max, merge=True)
    return raw


def _extract_sweep_detectors(raw: DictConfig) -> list[str] | None:
    methods_cfg = OmegaConf.select(raw, "detection_methods")
    if not methods_cfg:
        return None
    return [entry.method for entry in methods_cfg]


def _extract_sweep_strategies(raw: DictConfig) -> list[str] | None:
    strategies_cfg = OmegaConf.select(raw, "channel_strategies")
    if not strategies_cfg:
        return None
    return [entry.channel_strategy for entry in strategies_cfg]


_EXTRA_KEYS = {
    "detection_methods", "render_variants", "channel_strategies",
    "encoding_variants", "latent_analysis", "sublatent_variants",
    "sublatent_base", "conditioning_variants", "sequence_variants",
    "transformer", "generation", "sequence_training",
}


def _build_pydantic_config(raw: DictConfig) -> ExperimentConfig:
    container = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(container, dict)
    for key in _EXTRA_KEYS:
        container.pop(key, None)
    return ExperimentConfig(**container)


def _load_raw_config(args: argparse.Namespace) -> DictConfig:
    """Load, merge, and apply all CLI overrides to the raw OmegaConf config."""
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    raw: DictConfig = OmegaConf.load(str(config_path))

    if args.override_config:
        override_path = Path(args.override_config)
        if not override_path.exists():
            print(f"ERROR: Override config not found: {override_path}", file=sys.stderr)
            sys.exit(1)
        raw = OmegaConf.merge(raw, OmegaConf.load(str(override_path)))

    if args.data_root:
        OmegaConf.update(raw, "paths.data_root", args.data_root, merge=True)
    if args.output_root:
        OmegaConf.update(raw, "paths.output_root", args.output_root, merge=True)
    if args.subset_dir:
        OmegaConf.update(raw, "data.subset.subdirectory", args.subset_dir, merge=True)
    if args.subset_fraction is not None:
        OmegaConf.update(raw, "data.subset.fraction", args.subset_fraction, merge=True)
    if args.subset_file_list:
        OmegaConf.update(raw, "data.subset.file_list", args.subset_file_list, merge=True)

    if args.mini:
        raw = _apply_mini_overrides(raw, args.max_files)
    elif args.max_files is not None:
        OmegaConf.update(raw, "data.max_files", args.max_files, merge=True)

    return raw


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _detect_gpu_count() -> int:
    """Return the number of CUDA GPUs visible in this environment.

    Falls back to 1 if torch is unavailable or no CUDA devices are found.
    """
    try:
        import torch
        n = torch.cuda.device_count()
        return n if n > 0 else 1
    except ImportError:
        return 1


# ---------------------------------------------------------------------------
# Condition assignment
# ---------------------------------------------------------------------------

def _assign_conditions(
    num_conditions: int,
    num_ranks: int,
    strategy: str = "cost-balanced",
    cost_weights: list[float] | None = None,
) -> list[list[int]]:
    """Partition condition indices across ranks.

    Args:
        num_conditions: Total number of sweep conditions.
        num_ranks: Number of GPU ranks (workers).
        strategy: One of "round-robin", "contiguous", or "cost-balanced".
        cost_weights: Per-condition estimated cost weights, required (and only
            used) when strategy is "cost-balanced".  If None and strategy is
            "cost-balanced", falls back to round-robin behaviour.

    Returns:
        List of length ``num_ranks``.  Each element is the list of
        zero-based condition indices assigned to that rank.  Some
        ranks may receive an empty list if ``num_conditions < num_ranks``.
    """
    assignments: list[list[int]] = [[] for _ in range(num_ranks)]

    if strategy == "round-robin":
        for idx in range(num_conditions):
            assignments[idx % num_ranks].append(idx)

    elif strategy == "contiguous":
        chunk, remainder = divmod(num_conditions, num_ranks)
        start = 0
        for rank in range(num_ranks):
            size = chunk + (1 if rank < remainder else 0)
            assignments[rank] = list(range(start, start + size))
            start += size

    else:  # cost-balanced (LPT — Longest Processing Time first)
        if cost_weights is None or len(cost_weights) != num_conditions:
            # Graceful fallback: treat all conditions as equal cost (round-robin)
            for idx in range(num_conditions):
                assignments[idx % num_ranks].append(idx)
        else:
            # Sort condition indices by descending cost
            sorted_indices = sorted(
                range(num_conditions),
                key=lambda i: cost_weights[i],
                reverse=True,
            )
            # Track total cost accumulated per rank for the greedy assignment
            rank_loads: list[float] = [0.0] * num_ranks
            for idx in sorted_indices:
                # Assign to the rank with the lowest current load
                cheapest_rank = min(range(num_ranks), key=lambda r: rank_loads[r])
                assignments[cheapest_rank].append(idx)
                rank_loads[cheapest_rank] += cost_weights[idx]

    return assignments


# ---------------------------------------------------------------------------
# Rank-local output directory
# ---------------------------------------------------------------------------

def _rank_output_root(base_output_root: str, rank: int) -> str:
    """Return the output root for a specific rank's subprocess."""
    return str(Path(base_output_root) / f"rank_{rank:04d}")


# ---------------------------------------------------------------------------
# Subprocess-based single-node launcher
# ---------------------------------------------------------------------------

def _build_worker_command(
    args: argparse.Namespace,
    rank: int,
    condition_indices: list[int],
    output_root: str,
) -> list[str]:
    """Build the command list for a single worker subprocess.

    The worker re-invokes THIS script in ``--multi-node`` mode but with the
    SLURM variables pre-set so it picks up the right rank and conditions.
    We pass the conditions explicitly via ``--condition-indices`` to avoid
    re-enumerating inside the worker (the worker may have different visible GPUs).

    We instead call ``run_experiment.py`` directly with ``--condition-indices``
    injected via environment variable, because ``run_experiment.py`` already
    handles all config loading.  To keep things simple and avoid a circular
    call, the worker re-invokes ``run_experiment_distributed.py`` with
    ``--multi-node`` and appropriate SLURM env vars.
    """
    python = str(_REPO_ROOT / ".venv" / "bin" / "python")
    script = str(Path(__file__).resolve())

    cmd = [python, script, args.config]

    # Pass all config-affecting flags through unchanged
    if args.mini:
        cmd.append("--mini")
    if args.data_root:
        cmd += ["--data-root", args.data_root]
    if args.output_root:
        cmd += ["--output-root", output_root]
    else:
        cmd += ["--output-root", output_root]
    if args.override_config:
        cmd += ["--override-config", args.override_config]
    if args.resume_from:
        cmd += ["--resume-from", args.resume_from]
    if args.sweep_detectors:
        cmd.append("--sweep-detectors")
    if args.sweep_strategies:
        cmd.append("--sweep-strategies")
    if args.max_files:
        cmd += ["--max-files", str(args.max_files)]
    if args.subset_dir:
        cmd += ["--subset-dir", args.subset_dir]
    if args.subset_fraction is not None:
        cmd += ["--subset-fraction", str(args.subset_fraction)]
    if args.subset_file_list:
        cmd += ["--subset-file-list", args.subset_file_list]
    cmd += ["--log-level", args.log_level]
    cmd += ["--assign-strategy", args.assign_strategy]

    # Signal to the child that it should run only specific conditions.
    # We pass this via env var to avoid changing the CLI interface of the
    # script (and to keep arg parsing clean in the worker path).
    # The worker reads _CONDITION_INDICES_ENV to get its subset.
    cmd.append("--multi-node")

    return cmd


_CONDITION_INDICES_ENV = "MIDI_VAE_CONDITION_INDICES"
_RANK_ENV = "MIDI_VAE_RANK"
_WORLD_SIZE_ENV = "MIDI_VAE_WORLD_SIZE"


# ---------------------------------------------------------------------------
# Worker (multi-node or subprocess worker) entry point
# ---------------------------------------------------------------------------

def _run_worker(
    args: argparse.Namespace,
    raw: DictConfig,
    config: ExperimentConfig,
    rank: int,
    local_gpu_id: int,
    condition_indices: list[int],
) -> int:
    """Run the assigned subset of conditions on the local GPU.

    This function is called in the worker process (either a subprocess started
    by the launcher, or the main process in true multi-node mode where each
    ``srun`` rank IS the worker).

    Args:
        args: Parsed CLI arguments.
        raw: Raw OmegaConf config (already merged and overridden).
        config: Validated Pydantic ExperimentConfig.
        rank: Global rank of this worker.
        local_gpu_id: GPU index within CUDA_VISIBLE_DEVICES for this worker.
        condition_indices: Condition indices this worker must run.

    Returns:
        Exit code (0 = success).
    """
    from midi_vae.utils.seed import set_global_seed

    logger = get_logger(__name__)

    if not condition_indices:
        logger.info("Rank %d: no conditions assigned — exiting early.", rank)
        return 0

    logger.info(
        "Rank %d: assigned %d conditions: %s",
        rank,
        len(condition_indices),
        condition_indices,
    )

    # Extract sweep axes before building the executor
    sweep_detectors: list[str] | None = None
    sweep_strategies: list[str] | None = None
    if args.sweep_detectors:
        sweep_detectors = _extract_sweep_detectors(raw)
    if args.sweep_strategies:
        sweep_strategies = _extract_sweep_strategies(raw)

    # Build executor
    executor = SweepExecutor(
        config=config,
        sweep_strategies=sweep_strategies,
        sweep_detectors=sweep_detectors,
    )

    set_global_seed(config.seed)

    # Run only this rank's conditions
    t_start = time.monotonic()
    try:
        results = executor.run_subset(
            condition_indices=condition_indices,
            resume_from=args.resume_from,
        )
    except KeyboardInterrupt:
        logger.warning("Rank %d: interrupted by user.", rank)
        return 130
    except Exception as exc:
        logger.error("Rank %d: sweep failed: %s", rank, exc, exc_info=True)
        return 1

    elapsed = time.monotonic() - t_start
    logger.info("Rank %d: finished in %.1fs", rank, elapsed)

    # Write per-rank summary JSON
    output_root = Path(config.paths.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_path = output_root / f"sweep_summary_rank{rank:04d}.json"
    try:
        all_conditions = executor.conditions()
        assigned_conditions = [all_conditions[i] for i in condition_indices]
        estimated_total_cost = sum(c.estimated_cost() for c in assigned_conditions)
        summary_data = {
            "rank": rank,
            "local_gpu_id": local_gpu_id,
            "elapsed_seconds": round(elapsed, 2),
            "condition_indices": condition_indices,
            "conditions": [all_conditions[i].label for i in condition_indices],
            "estimated_total_cost": round(estimated_total_cost, 4),
            "metrics_summary": results.get("__summary__", {}),
        }
        with summary_path.open("w") as fh:
            json.dump(summary_data, fh, indent=2, default=str)
        logger.info("Rank %d: summary written to %s", rank, summary_path)
    except Exception as exc:
        logger.warning("Rank %d: could not write summary JSON: %s", rank, exc)

    return 0


# ---------------------------------------------------------------------------
# Single-node launcher: spawn one subprocess per GPU
# ---------------------------------------------------------------------------

def _compute_cpu_affinity(num_gpus: int) -> tuple[int, list[tuple[int, int]]]:
    """Compute CPU core ranges to pin to each GPU process.

    Divides all available CPU cores evenly across GPU processes.

    Args:
        num_gpus: Number of GPU subprocess workers.

    Returns:
        Tuple of (cores_per_gpu, [(start_core, end_core), ...]) where each
        element of the list corresponds to a GPU rank (0-based).  The
        start/end values are inclusive core indices for taskset.
    """
    total_cores = os.cpu_count() or 1
    cores_per_gpu = max(1, total_cores // num_gpus)
    ranges: list[tuple[int, int]] = []
    for rank in range(num_gpus):
        start_core = rank * cores_per_gpu
        # Last rank gets any leftover cores
        if rank == num_gpus - 1:
            end_core = total_cores - 1
        else:
            end_core = start_core + cores_per_gpu - 1
        ranges.append((start_core, end_core))
    return cores_per_gpu, ranges


_PROGRESS_POLL_INTERVAL = 30  # seconds between progress reports


def _launch_single_node(
    args: argparse.Namespace,
    num_gpus: int,
    assignments: list[list[int]],
) -> int:
    """Spawn one subprocess per GPU on this node and wait for all to finish.

    Each subprocess is given:
    * Its own CUDA_VISIBLE_DEVICES (one GPU each).
    * Thread-count env vars (OMP_NUM_THREADS, MKL_NUM_THREADS, etc.) set to
      ``total_cores // num_gpus`` to prevent thread over-subscription.
    * taskset CPU pinning (Linux only) unless ``--no-cpu-pinning`` is set.

    After launching, the function polls subprocess status every
    ``_PROGRESS_POLL_INTERVAL`` seconds and prints a running summary.
    A final per-rank summary table is printed after all workers finish.

    Args:
        args: Parsed CLI arguments.
        num_gpus: Number of GPUs (and subprocesses) to spawn.
        assignments: Per-rank condition index lists.

    Returns:
        Exit code: 0 if all subprocesses succeeded, 1 otherwise.
    """
    logger = get_logger(__name__)

    base_output_root = args.output_root  # may be None; workers write to rank subdirs

    # Compute CPU affinity parameters once for all ranks
    cores_per_gpu, core_ranges = _compute_cpu_affinity(num_gpus)
    is_linux = platform.system() == "Linux"
    use_taskset = is_linux and not getattr(args, "no_cpu_pinning", False)

    # Check taskset availability on Linux
    if use_taskset:
        try:
            subprocess.run(
                ["taskset", "--version"],
                capture_output=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.warning("taskset not available — disabling CPU pinning.")
            use_taskset = False

    processes: list[tuple[int, subprocess.Popen, object, float]] = []  # (rank, proc, log_fh, start_time)

    for rank, condition_indices in enumerate(assignments):
        if not condition_indices:
            logger.info("GPU %d: no conditions assigned — skipping.", rank)
            continue

        # Each subprocess sees only one GPU
        gpu_env = os.environ.copy()
        gpu_env["CUDA_VISIBLE_DEVICES"] = str(rank)

        # Tell the worker its rank and condition slice via env vars
        gpu_env[_RANK_ENV] = str(rank)
        gpu_env[_WORLD_SIZE_ENV] = str(num_gpus)
        gpu_env[_CONDITION_INDICES_ENV] = ",".join(str(i) for i in condition_indices)

        # Set thread-count environment variables to prevent over-subscription
        threads_str = str(cores_per_gpu)
        gpu_env["OMP_NUM_THREADS"] = threads_str
        gpu_env["MKL_NUM_THREADS"] = threads_str
        gpu_env["OPENBLAS_NUM_THREADS"] = threads_str
        gpu_env["NUMEXPR_NUM_THREADS"] = threads_str

        # Rank-specific output directory (avoids file conflicts)
        rank_output = _rank_output_root(base_output_root or "outputs", rank)

        cmd = _build_worker_command(args, rank, condition_indices, rank_output)

        # Prepend taskset for CPU core pinning on Linux
        if use_taskset:
            start_core, end_core = core_ranges[rank]
            cmd = ["taskset", "-c", f"{start_core}-{end_core}"] + cmd

        log_path = Path(rank_output) / f"worker_rank{rank:04d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Launching GPU %d (CUDA_VISIBLE_DEVICES=%d): %d conditions -> %s",
            rank,
            rank,
            len(condition_indices),
            condition_indices,
        )

        log_fh = log_path.open("w")
        proc = subprocess.Popen(
            cmd,
            env=gpu_env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        processes.append((rank, proc, log_fh, time.monotonic()))  # type: ignore[arg-type]

    if not processes:
        logger.warning("No subprocesses launched — all ranks have empty condition lists.")
        return 0

    # Print launch summary table
    print(f"\n{'='*70}")
    print(f"  Distributed sweep: {len(processes)} GPU worker(s) launched")
    print(f"  CPU cores available: {os.cpu_count() or 1}  |  cores/GPU: {cores_per_gpu}")
    print(f"  CPU pinning (taskset): {'enabled' if use_taskset else 'disabled'}")
    for rank, proc, _, _ in processes:
        start_core, end_core = core_ranges[rank]
        core_info = f"  cores {start_core}-{end_core}" if use_taskset else ""
        print(
            f"    GPU {rank} (PID {proc.pid}){core_info}: "
            f"{len(assignments[rank])} conditions {assignments[rank]}"
        )
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Poll subprocess status every _PROGRESS_POLL_INTERVAL seconds
    # ------------------------------------------------------------------
    exit_codes: dict[int, int | None] = {rank: None for rank, _, _, _ in processes}
    active = {rank: (proc, log_fh, t0) for rank, proc, log_fh, t0 in processes}

    while active:
        time.sleep(_PROGRESS_POLL_INTERVAL)

        still_running: list[int] = []
        newly_done: list[int] = []

        for rank, (proc, log_fh, _t0) in list(active.items()):
            rc = proc.poll()
            if rc is not None:
                exit_codes[rank] = rc
                log_fh.close()
                del active[rank]
                newly_done.append(rank)
            else:
                still_running.append(rank)

        t0_by_rank = {rank: t0 for rank, _p, _fh, t0 in processes}

        if newly_done:
            for rank in newly_done:
                rc = exit_codes[rank]
                elapsed = round(time.monotonic() - t0_by_rank[rank], 1)
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                print(f"  [progress] GPU {rank} finished after {elapsed}s — {status}")

        if still_running:
            running_info = ", ".join(
                f"GPU {r} ({round(time.monotonic() - t0_by_rank[r], 0):.0f}s)"
                for r in sorted(still_running)
            )
            print(f"  [progress] Still running: {running_info}")

    # ------------------------------------------------------------------
    # Final summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  Per-rank final summary:")
    print(f"  {'GPU':>4}  {'#Conds':>7}  {'Elapsed':>9}  {'Exit':>6}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*9}  {'-'*6}")

    t0_by_rank = {rank: t0 for rank, _p, _fh, t0 in processes}
    elapsed_by_rank = {
        rank: round(time.monotonic() - t0_by_rank[rank], 1)
        for rank in exit_codes
    }

    failed_ranks: list[int] = []
    for rank in sorted(exit_codes):
        rc = exit_codes[rank]
        n_conds = len(assignments[rank])
        elapsed = elapsed_by_rank[rank]
        rc_str = "0 (OK)" if rc == 0 else str(rc)
        print(f"  {rank:>4}  {n_conds:>7}  {elapsed:>8.1f}s  {rc_str:>6}")
        if rc != 0:
            failed_ranks.append(rank)

    print(f"{'='*70}\n")

    if failed_ranks:
        logger.error("Failed ranks: %s", failed_ranks)
        return 1

    logger.info("All GPU workers finished successfully.")
    return 0


# ---------------------------------------------------------------------------
# Merge rank summaries
# ---------------------------------------------------------------------------

def _merge_rank_summaries(output_root: str, num_ranks: int, assignments: list[list[int]]) -> None:
    """Merge per-rank sweep_summary_rank*.json files into a single summary.

    Args:
        output_root: Base output root (parent of rank_XXXX/ dirs).
        num_ranks: Total number of ranks (to know which files to look for).
        assignments: Per-rank condition index lists (for ordering).
    """
    logger = get_logger(__name__)
    base = Path(output_root)

    all_metrics: dict[str, dict] = {}
    all_conditions_ordered: list[str] = []
    total_elapsed = 0.0

    # Collect rank summaries in rank order so the final summary is deterministic
    for rank in range(num_ranks):
        if not assignments[rank]:
            continue
        rank_dir = base / f"rank_{rank:04d}"
        summary_path = rank_dir / f"sweep_summary_rank{rank:04d}.json"
        if not summary_path.exists():
            logger.warning("Rank %d summary not found at %s", rank, summary_path)
            continue
        with summary_path.open() as fh:
            data = json.load(fh)
        total_elapsed += data.get("elapsed_seconds", 0.0)
        metrics = data.get("metrics_summary", {})
        all_metrics.update(metrics)
        all_conditions_ordered.extend(data.get("conditions", []))

    merged = {
        "num_ranks": num_ranks,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "num_conditions": len(all_conditions_ordered),
        "conditions": all_conditions_ordered,
        "metrics_summary": all_metrics,
    }

    merged_path = base / "sweep_summary.json"
    with merged_path.open("w") as fh:
        json.dump(merged, fh, indent=2, default=str)
    logger.info("Merged summary written to %s", merged_path)
    print(f"\nMerged sweep summary -> {merged_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Entry point.  Returns exit code."""
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Check if this process is a worker (spawned by the launcher)
    # ------------------------------------------------------------------
    condition_indices_env = os.environ.get(_CONDITION_INDICES_ENV)
    rank_env = os.environ.get(_RANK_ENV)
    if condition_indices_env is not None and rank_env is not None and args.multi_node:
        # Worker path: run only the conditions specified in the env var
        rank = int(rank_env)
        condition_indices = [int(x) for x in condition_indices_env.split(",") if x]
        local_gpu_id = 0  # CUDA_VISIBLE_DEVICES already set to a single GPU

        raw = _load_raw_config(args)

        # Collect sweep axes
        sweep_detectors: list[str] | None = None
        sweep_strategies: list[str] | None = None
        if args.sweep_detectors:
            sweep_detectors = _extract_sweep_detectors(raw)
        if args.sweep_strategies:
            sweep_strategies = _extract_sweep_strategies(raw)

        try:
            config = _build_pydantic_config(raw)
        except Exception as exc:
            print(f"ERROR: Config validation failed: {exc}", file=sys.stderr)
            return 1

        # Ensure rank-specific output dir is used
        # (already set by the launcher via --output-root)

        return _run_worker(
            args=args,
            raw=raw,
            config=config,
            rank=rank,
            local_gpu_id=local_gpu_id,
            condition_indices=condition_indices,
        )

    # ------------------------------------------------------------------
    # Check if this is a true SLURM multi-node srun rank (--multi-node
    # without our env vars, but with SLURM_ vars)
    # ------------------------------------------------------------------
    if args.multi_node and condition_indices_env is None:
        slurm_procid = os.environ.get("SLURM_PROCID")
        slurm_ntasks = os.environ.get("SLURM_NTASKS")
        slurm_localid = os.environ.get("SLURM_LOCALID")

        if slurm_procid is None or slurm_ntasks is None:
            print(
                "ERROR: --multi-node requires either SLURM_PROCID/SLURM_NTASKS env vars "
                "or MIDI_VAE_CONDITION_INDICES/MIDI_VAE_RANK env vars.",
                file=sys.stderr,
            )
            return 1

        global_rank = int(slurm_procid)
        world_size = int(slurm_ntasks)
        local_gpu_id = int(slurm_localid) if slurm_localid is not None else global_rank

        # Set CUDA_VISIBLE_DEVICES to the local GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_gpu_id)

        # In multi-node SLURM mode, trust SLURM's CPU binding but ensure
        # thread-count env vars are set so libraries don't over-subscribe.
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus:
            for thread_var in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                if thread_var not in os.environ:
                    os.environ[thread_var] = slurm_cpus

        raw = _load_raw_config(args)

        # Collect sweep axes
        sweep_detectors = None
        sweep_strategies = None
        if args.sweep_detectors:
            sweep_detectors = _extract_sweep_detectors(raw)
        if args.sweep_strategies:
            sweep_strategies = _extract_sweep_strategies(raw)

        try:
            config = _build_pydantic_config(raw)
        except Exception as exc:
            print(f"ERROR: Config validation failed: {exc}", file=sys.stderr)
            return 1

        # Build a temp executor just to count conditions
        temp_executor = SweepExecutor(
            config=config,
            sweep_strategies=sweep_strategies,
            sweep_detectors=sweep_detectors,
        )
        all_conditions = temp_executor.conditions()
        num_conditions = len(all_conditions)

        slurm_cost_weights: list[float] | None = None
        if args.assign_strategy == "cost-balanced":
            slurm_cost_weights = [c.estimated_cost() for c in all_conditions]

        assignments = _assign_conditions(
            num_conditions, world_size, args.assign_strategy, slurm_cost_weights
        )
        condition_indices = assignments[global_rank]

        logger.info(
            "SLURM multi-node rank %d/%d, local GPU %d: %d conditions",
            global_rank,
            world_size,
            local_gpu_id,
            len(condition_indices),
        )

        # Override output root to be rank-specific
        base_output_root = config.paths.output_root
        rank_output_root = _rank_output_root(base_output_root, global_rank)
        OmegaConf.update(raw, "paths.output_root", rank_output_root, merge=True)
        try:
            config = _build_pydantic_config(raw)
        except Exception as exc:
            print(f"ERROR: Config re-validation failed: {exc}", file=sys.stderr)
            return 1

        return _run_worker(
            args=args,
            raw=raw,
            config=config,
            rank=global_rank,
            local_gpu_id=local_gpu_id,
            condition_indices=condition_indices,
        )

    # ------------------------------------------------------------------
    # Single-node launcher path: spawn subprocesses
    # ------------------------------------------------------------------
    raw = _load_raw_config(args)

    # Collect sweep axes
    sweep_detectors = None
    sweep_strategies = None
    if args.sweep_detectors:
        sweep_detectors = _extract_sweep_detectors(raw)
    if args.sweep_strategies:
        sweep_strategies = _extract_sweep_strategies(raw)

    try:
        config = _build_pydantic_config(raw)
    except Exception as exc:
        print(f"ERROR: Config validation failed: {exc}", file=sys.stderr)
        return 1

    if not config.vaes:
        print("ERROR: No VAEs defined in config.vaes — nothing to run.", file=sys.stderr)
        return 1

    # Enumerate all conditions to compute assignments
    executor = SweepExecutor(
        config=config,
        sweep_strategies=sweep_strategies,
        sweep_detectors=sweep_detectors,
    )
    all_conditions = executor.conditions()
    num_conditions = len(all_conditions)

    # Determine number of GPUs
    if args.num_gpus is not None:
        num_gpus = args.num_gpus
    else:
        num_gpus = _detect_gpu_count()
        logger.info("Auto-detected %d GPU(s).", num_gpus)

    # Cap num_gpus to num_conditions (no point having idle ranks)
    effective_ranks = min(num_gpus, num_conditions) if num_conditions > 0 else 1
    if effective_ranks < num_gpus:
        logger.info(
            "Capping ranks from %d to %d (fewer conditions than GPUs).",
            num_gpus,
            effective_ranks,
        )

    # Compute per-condition cost weights for cost-balanced strategy
    cost_weights: list[float] | None = None
    if args.assign_strategy == "cost-balanced":
        cost_weights = [c.estimated_cost() for c in all_conditions]

    assignments = _assign_conditions(
        num_conditions, effective_ranks, args.assign_strategy, cost_weights
    )

    # Print assignment table
    print(f"\n{'='*70}")
    print(f"  Distributed sweep: {num_conditions} conditions -> {effective_ranks} GPU(s)")
    print(f"  Assignment strategy: {args.assign_strategy}")
    print(f"{'='*70}")
    for rank_idx, indices in enumerate(assignments):
        labels = [all_conditions[i].label for i in indices]
        rank_cost = (
            sum(cost_weights[i] for i in indices) if cost_weights is not None else None
        )
        cost_str = f"  (est. cost: {rank_cost:.2f})" if rank_cost is not None else ""
        print(f"  GPU {rank_idx}: {len(indices)} conditions{cost_str}")
        for lbl in labels:
            print(f"        {lbl}")
    print(f"{'='*70}\n")

    if args.dry_run:
        logger.info("Dry-run: no subprocesses will be launched.")
        return 0

    base_output_root = args.output_root or config.paths.output_root

    rc = _launch_single_node(args, effective_ranks, assignments)

    # Merge results even if some ranks failed (partial merge)
    if num_conditions > 0:
        try:
            _merge_rank_summaries(base_output_root, effective_ranks, assignments)
        except Exception as exc:
            logger.warning("Could not merge rank summaries: %s", exc)

    return rc


if __name__ == "__main__":
    sys.exit(main())
