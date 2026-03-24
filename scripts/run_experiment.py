#!/usr/bin/env python
"""Run a MIDI Image VAE experiment from a YAML config.

Usage:
    python scripts/run_experiment.py configs/experiments/exp_1a_vae_comparison.yaml
    python scripts/run_experiment.py configs/experiments/exp_1a_vae_comparison.yaml --mini
    python scripts/run_experiment.py configs/experiments/exp_1a_vae_comparison.yaml --dry-run
    python scripts/run_experiment.py configs/experiments/exp_1a_vae_comparison.yaml --mini --data-root data/maestro/maestro-v3.0.0
    python scripts/run_experiment.py configs/experiments/exp_1b_detection_methods.yaml --sweep-detectors
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env before importing anything that might need HF_TOKEN
# ---------------------------------------------------------------------------

def _load_dotenv(env_path: Path) -> None:
    """Parse key=value pairs from a .env file and export them to os.environ.

    Only sets variables that are not already present in the environment,
    preserving any values that were set by the scheduler or the user before
    launching this script.

    Args:
        env_path: Path to the .env file.
    """
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


# Resolve the repo root relative to this script so it works regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_load_dotenv(_REPO_ROOT / ".env")

# ---------------------------------------------------------------------------
# Now safe to import project modules
# ---------------------------------------------------------------------------

from omegaconf import OmegaConf, DictConfig, open_dict  # noqa: E402

from midi_vae.config import ExperimentConfig  # noqa: E402
from midi_vae.pipelines.sweep import SweepExecutor  # noqa: E402
from midi_vae.utils.device import get_device, get_device_info  # noqa: E402
from midi_vae.utils.logging import get_logger, setup_logging  # noqa: E402
from midi_vae.utils.seed import set_global_seed  # noqa: E402


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a MIDI Image VAE experiment from a YAML config file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "config",
        help="Path to experiment YAML config file.",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        default=False,
        help=(
            "Run a small version of the experiment for quick testing: "
            "bars_per_instrument=20, max_files=5, first 2 VAEs only."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Enumerate all sweep conditions and print them without running.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        metavar="PATH",
        help=(
            "Override paths.data_root in the config. "
            "Use this to point at the actual dataset directory "
            "(e.g., data/maestro/maestro-v3.0.0)."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        metavar="N",
        help="Limit number of files loaded per instrument (overrides --mini default of 5).",
    )
    parser.add_argument(
        "--override-config",
        default=None,
        metavar="PATH",
        help=(
            "Path to a second YAML file whose values override the primary config. "
            "Useful for per-variant overrides in Exp 2 / custom one-off runs."
        ),
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
        help=(
            "Sweep over all detection_methods listed in the config "
            "(used for exp_1b_detection_methods.yaml)."
        ),
    )
    parser.add_argument(
        "--sweep-strategies",
        action="store_true",
        default=False,
        help=(
            "Sweep over all channel_strategies listed in the config "
            "(used for exp_3_channel_strategy.yaml)."
        ),
    )
    parser.add_argument(
        "--sweep-render-variants",
        action="store_true",
        default=False,
        help=(
            "Sweep over all render_variants listed in the config "
            "(used for exp_2_resolution_orientation.yaml). "
            "Takes priority over --sweep-strategies when both are given."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level (default: INFO).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        metavar="PATH",
        help="Override paths.output_root in the config.",
    )
    parser.add_argument(
        "--subset-dir",
        default=None,
        metavar="SUBDIR",
        help=(
            "Only load files from this subdirectory under data_root "
            "(e.g. 'f/' for the Lakh 'f' partition)."
        ),
    )
    parser.add_argument(
        "--subset-fraction",
        type=float,
        default=None,
        metavar="FRAC",
        help="Random fraction of files to use, e.g. 0.1 for 10%% (range: 0.0 < FRAC <= 1.0).",
    )
    parser.add_argument(
        "--subset-file-list",
        default=None,
        metavar="PATH",
        help="Path to a text file listing specific files to use (one filepath per line).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config manipulation helpers
# ---------------------------------------------------------------------------


def _apply_mini_overrides(raw: DictConfig, max_files: int | None) -> DictConfig:
    """Return a new DictConfig with mini-mode overrides applied.

    Mini mode uses 20 bars per instrument, 5 files maximum, and the
    first 2 VAEs only.  Pass max_files to override the default of 5.

    Args:
        raw: The merged OmegaConf DictConfig loaded from YAML.
        max_files: File cap (None -> use mini default of 5).

    Returns:
        Updated DictConfig suitable for Pydantic validation.
    """
    OmegaConf.update(raw, "data.bars_per_instrument", 20, merge=True)
    # Truncate VAE list to the first 2 entries
    if OmegaConf.select(raw, "vaes") and len(raw.vaes) > 2:
        with open_dict(raw):
            raw.vaes = list(raw.vaes)[:2]
    # Honour explicit --max-files
    effective_max = max_files if max_files is not None else 5
    OmegaConf.update(raw, "data.max_files", effective_max, merge=True)
    return raw


def _apply_data_root(raw: DictConfig, data_root: str) -> DictConfig:
    """Override paths.data_root in the raw DictConfig.

    Args:
        raw: OmegaConf DictConfig.
        data_root: New data root path string.

    Returns:
        Updated DictConfig.
    """
    OmegaConf.update(raw, "paths.data_root", data_root, merge=True)
    return raw


def _apply_output_root(raw: DictConfig, output_root: str) -> DictConfig:
    """Override paths.output_root in the raw DictConfig.

    Args:
        raw: OmegaConf DictConfig.
        output_root: New output root path string.

    Returns:
        Updated DictConfig.
    """
    OmegaConf.update(raw, "paths.output_root", output_root, merge=True)
    return raw


def _extract_sweep_detectors(raw: DictConfig) -> list[str] | None:
    """Extract detection method names from the detection_methods sweep list.

    Experiments like exp_1b use a top-level ``detection_methods`` list in the
    YAML.  This helper extracts the method names for passing to SweepExecutor.

    Args:
        raw: The raw DictConfig from the YAML file.

    Returns:
        List of method name strings, or None if no detection_methods key found.
    """
    methods_cfg = OmegaConf.select(raw, "detection_methods")
    if not methods_cfg:
        return None
    return [entry.method for entry in methods_cfg]


def _extract_sweep_strategies(raw: DictConfig) -> list[str] | None:
    """Extract channel strategy names from the channel_strategies sweep list.

    Experiments like exp_3 use a top-level ``channel_strategies`` list.

    Args:
        raw: The raw DictConfig from the YAML file.

    Returns:
        List of strategy name strings, or None if no channel_strategies key found.
    """
    strategies_cfg = OmegaConf.select(raw, "channel_strategies")
    if not strategies_cfg:
        return None
    return [entry.channel_strategy for entry in strategies_cfg]


def _extract_sweep_render_variants(raw: DictConfig) -> list[dict] | None:
    """Extract render variant dicts from the render_variants sweep list.

    Experiments like exp_2 use a top-level ``render_variants`` list in the
    YAML.  Each entry is a dict with a ``name`` key plus optional overrides for
    ``channel_strategy``, ``pitch_axis``, ``target_resolution``,
    ``normalize_range``, and ``resize_method``.

    Args:
        raw: The raw DictConfig from the YAML file.

    Returns:
        List of render variant dicts, or None if no render_variants key found.
    """
    variants_cfg = OmegaConf.select(raw, "render_variants")
    if not variants_cfg:
        return None
    # Convert each entry to a plain dict for SweepExecutor
    container = OmegaConf.to_container(variants_cfg, resolve=True)
    if not isinstance(container, list):
        return None
    return [dict(entry) for entry in container]  # type: ignore[arg-type]


def _build_pydantic_config(raw: DictConfig) -> ExperimentConfig:
    """Convert an OmegaConf DictConfig to a validated ExperimentConfig.

    Strips keys that are not part of the Pydantic schema (e.g. the sweep
    variant lists that live in the YAML for documentation purposes).

    Args:
        raw: The fully merged and overridden OmegaConf DictConfig.

    Returns:
        A validated, frozen ExperimentConfig.

    Raises:
        pydantic.ValidationError: If the config fails validation.
    """
    # Keys that are experiment-level documentation / sweep axes — not part of
    # the Pydantic schema and must be stripped before validation.
    _EXTRA_KEYS = {
        "detection_methods",
        "render_variants",
        "channel_strategies",
        "encoding_variants",
        "latent_analysis",
        "sublatent_variants",
        "sublatent_base",
        "conditioning_variants",
        "sequence_variants",
        "transformer",
        "generation",
        "sequence_training",
    }
    container = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(container, dict)
    for key in _EXTRA_KEYS:
        container.pop(key, None)
    # max_files is now part of the DataConfig Pydantic schema
    return ExperimentConfig(**container)


# ---------------------------------------------------------------------------
# GPU memory cleanup
# ---------------------------------------------------------------------------


def _free_gpu_memory() -> None:
    """Release cached GPU memory between conditions.

    Calls gc.collect() and torch.cuda.empty_cache() when CUDA is available.
    Safe to call on CPU-only machines.
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def _print_conditions(conditions: list) -> None:
    """Print a formatted table of sweep conditions.

    Args:
        conditions: List of SweepCondition objects from SweepExecutor.conditions().
    """
    print(f"\n{'='*70}")
    print(f"  Sweep conditions ({len(conditions)} total)")
    print(f"{'='*70}")
    for i, cond in enumerate(conditions, 1):
        print(f"  [{i:3d}] {cond.label}")
    print(f"{'='*70}\n")


def _print_summary(results: dict, elapsed: float) -> None:
    """Print a human-readable summary of the sweep results.

    Args:
        results: Dict returned by SweepExecutor.run().
        elapsed: Wall-clock seconds the sweep took.
    """
    print(f"\n{'='*70}")
    print("  Sweep complete")
    print(f"{'='*70}")
    print(f"  Wall time : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    summary = results.get("__summary__", {})
    if summary:
        print(f"  Conditions: {len(summary)}")
        print()
        for label, metrics in summary.items():
            print(f"  {label}")
            for k, v in metrics.items():
                print(f"      {k}: {v:.4f}" if isinstance(v, float) else f"      {k}: {v}")
    else:
        n = sum(1 for k in results if not k.startswith("__"))
        print(f"  Conditions ran: {n}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Piano roll image logging helper
# ---------------------------------------------------------------------------


def _log_piano_roll_examples(
    wandb_logger: "WandbLogger",
    all_results: dict,
    max_examples: int = 4,
) -> None:
    """Log GT vs reconstructed piano roll comparison figures to wandb.

    For each condition in *all_results*, picks up to *max_examples* bars per
    VAE and uploads side-by-side comparison figures as ``wandb.Image`` objects.
    Figures are closed immediately after logging to avoid memory leaks.

    This function is silently skipped when:
      * wandb logging is not active.
      * The context dict for a condition has no ``images`` or
        ``reconstructed_bars`` (this is the case for chunked pipeline runs
        where heavy tensors are freed after each chunk).
      * Any unexpected error occurs (non-critical — logged as a warning).

    Args:
        wandb_logger: An active :class:`WandbLogger` instance.
        all_results: The dict returned by :meth:`SweepExecutor.run`.
            Keys are condition labels; the ``"__summary__"`` key is skipped.
        max_examples: Maximum number of bar comparisons to log per
            (condition, VAE) pair.  Defaults to 4.
    """
    if not wandb_logger.enabled:
        return

    try:
        import matplotlib.pyplot as plt

        from midi_vae.visualization.piano_roll import plot_gt_vs_recon
    except ImportError as exc:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "Skipping piano roll image logging — import failed: %s", exc
        )
        return

    import logging as _logging
    _log = _logging.getLogger(__name__)

    for label, context in all_results.items():
        if label.startswith("__"):
            continue

        if not isinstance(context, dict):
            continue

        images = context.get("images", [])
        reconstructed_bars = context.get("reconstructed_bars", {})
        visual_examples = context.get("_visual_examples", {})

        if not images or not reconstructed_bars:
            if not visual_examples:
                # No pre-collected examples either — nothing to log.
                _log.debug(
                    "Skipping piano roll logging for condition '%s': "
                    "images or reconstructed_bars not in context (chunked run?)",
                    label,
                )
                continue

            # Chunked pipeline: use pre-collected (GT, recon) tensor pairs.
            safe_label = label.replace("/", "_")
            total_logged = 0
            for vae_name, examples in visual_examples.items():
                logged = 0
                for bar_id, gt_tensor, recon_tensor, channel_strategy in examples:
                    if logged >= max_examples:
                        break
                    try:
                        fig = plot_gt_vs_recon(
                            gt_image=gt_tensor,
                            recon_image=recon_tensor,
                            bar_id=bar_id,
                            vae_name=vae_name,
                            channel_strategy=channel_strategy,
                        )
                        key = f"examples/{safe_label}/{vae_name}/bar_{logged}"
                        wandb_logger.log_image(key, fig)
                        plt.close(fig)
                        logged += 1
                        total_logged += 1
                    except Exception as exc:
                        _log.warning(
                            "Failed to log piano roll for bar '%s' (VAE '%s'): %s",
                            bar_id,
                            vae_name,
                            exc,
                        )
            if total_logged:
                _log.info(
                    "Logged %d piano roll examples for condition '%s' (from pre-collected chunks)",
                    total_logged,
                    label,
                )
            continue

        # Build bar_id -> PianoRollImage lookup
        gt_lookup = {img.bar_id: img for img in images}

        for vae_name, recon_list in reconstructed_bars.items():
            logged = 0
            for recon in recon_list:
                if logged >= max_examples:
                    break

                gt_img = gt_lookup.get(recon.bar_id)
                if gt_img is None:
                    continue

                try:
                    fig = plot_gt_vs_recon(
                        gt_image=gt_img.image,
                        recon_image=recon.recon_image,
                        bar_id=recon.bar_id,
                        vae_name=vae_name,
                        channel_strategy=gt_img.channel_strategy,
                    )

                    safe_label = label.replace("/", "_")
                    key = f"examples/{safe_label}/{vae_name}/bar_{logged}"
                    wandb_logger.log_image(key, fig)
                    plt.close(fig)
                    logged += 1
                except Exception as exc:
                    _log.warning(
                        "Failed to log piano roll for bar '%s' (VAE '%s'): %s",
                        recon.bar_id,
                        vae_name,
                        exc,
                    )

        if logged_for_condition := sum(
            min(max_examples, len(v)) for v in reconstructed_bars.values()
        ):
            _log.info(
                "Logged up to %d piano roll examples for condition '%s'",
                logged_for_condition,
                label,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point.  Returns exit code."""
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = get_logger(__name__)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Load raw YAML so we can apply overrides before Pydantic validation
    # ------------------------------------------------------------------
    logger.info("Loading config", path=str(config_path))
    raw: DictConfig = OmegaConf.load(str(config_path))

    # Merge secondary override config if provided (e.g. per-variant configs for Exp 2)
    if args.override_config:
        override_path = Path(args.override_config)
        if not override_path.exists():
            print(f"ERROR: Override config not found: {override_path}", file=sys.stderr)
            return 1
        logger.info("Merging override config", path=str(override_path))
        override_raw: DictConfig = OmegaConf.load(str(override_path))
        raw = OmegaConf.merge(raw, override_raw)

    # Apply data-root override first (before mini truncation so paths are right)
    if args.data_root:
        raw = _apply_data_root(raw, args.data_root)
        logger.info("data_root overridden", data_root=args.data_root)

    if args.output_root:
        raw = _apply_output_root(raw, args.output_root)
        logger.info("output_root overridden", output_root=args.output_root)

    # Apply subset CLI flags — update the raw OmegaConf before Pydantic validation.
    if args.subset_dir:
        OmegaConf.update(raw, "data.subset.subdirectory", args.subset_dir, merge=True)
        logger.info("subset.subdirectory overridden", subdirectory=args.subset_dir)
    if args.subset_fraction is not None:
        OmegaConf.update(raw, "data.subset.fraction", args.subset_fraction, merge=True)
        logger.info("subset.fraction overridden", fraction=args.subset_fraction)
    if args.subset_file_list:
        OmegaConf.update(raw, "data.subset.file_list", args.subset_file_list, merge=True)
        logger.info("subset.file_list overridden", file_list=args.subset_file_list)

    # Collect sweep axes before mini truncates the VAE list
    sweep_detectors: list[str] | None = None
    sweep_strategies: list[str] | None = None
    sweep_render_variants: list[dict] | None = None

    if args.sweep_detectors:
        sweep_detectors = _extract_sweep_detectors(raw)
        if sweep_detectors:
            logger.info("Sweeping detectors", methods=sweep_detectors)

    if args.sweep_render_variants:
        sweep_render_variants = _extract_sweep_render_variants(raw)
        if sweep_render_variants:
            logger.info(
                "Sweeping render variants",
                variants=[v.get("name") for v in sweep_render_variants],
            )
    elif args.sweep_strategies:
        sweep_strategies = _extract_sweep_strategies(raw)
        if sweep_strategies:
            logger.info("Sweeping channel strategies", strategies=sweep_strategies)

    if args.mini:
        raw = _apply_mini_overrides(raw, args.max_files)
        logger.info(
            "Mini mode active",
            bars_per_instrument=20,
            max_files=args.max_files or 5,
            num_vaes=len(raw.vaes),
        )
    elif args.max_files is not None:
        OmegaConf.update(raw, "data.max_files", args.max_files, merge=True)
        logger.info("max_files overridden", max_files=args.max_files)

    # ------------------------------------------------------------------
    # Build validated Pydantic config
    # ------------------------------------------------------------------
    try:
        config = _build_pydantic_config(raw)
    except Exception as exc:
        print(f"ERROR: Config validation failed: {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Setup reproducibility & device
    # ------------------------------------------------------------------
    set_global_seed(config.seed)
    device = get_device(config.device)
    device_info = get_device_info(device)
    logger.info("Experiment setup", seed=config.seed, **device_info)

    if not config.vaes:
        print("ERROR: No VAEs defined in config.vaes — nothing to run.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Ensure output directories exist
    # ------------------------------------------------------------------
    Path(config.paths.output_root).mkdir(parents=True, exist_ok=True)
    Path(config.paths.cache_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build SweepExecutor
    # ------------------------------------------------------------------
    executor = SweepExecutor(
        config=config,
        sweep_strategies=sweep_strategies,
        sweep_detectors=sweep_detectors,
        sweep_render_variants=sweep_render_variants,
    )

    all_conditions = executor.conditions()
    _print_conditions(all_conditions)

    if args.dry_run:
        logger.info("Dry-run requested — no pipelines will be executed.")
        print("Dry-run complete. Conditions listed above.")
        return 0

    # ------------------------------------------------------------------
    # Initialize wandb (if enabled)
    # ------------------------------------------------------------------
    wandb_logger = None
    if config.tracking.wandb_enabled:
        from midi_vae.tracking.wandb_logger import WandbLogger
        wandb_logger = WandbLogger(
            config=config,
            experiment_id=config.tracking.experiment_name,
        )

    # ------------------------------------------------------------------
    # Run sweep
    # ------------------------------------------------------------------
    logger.info(
        "Starting sweep",
        experiment=config.tracking.experiment_name,
        num_conditions=len(all_conditions),
        data_root=config.paths.data_root,
        output_root=config.paths.output_root,
    )

    t_start = time.monotonic()
    try:
        results = executor.run(
            resume_from=args.resume_from,
            dry_run=False,
            wandb_logger=wandb_logger,  # images are logged per-condition inside the sweep
        )
        # Clean GPU memory after the sweep completes
        _free_gpu_memory()
    except KeyboardInterrupt:
        logger.warning("Sweep interrupted by user.")
        _free_gpu_memory()
        if wandb_logger:
            wandb_logger.finish()
        return 130
    except Exception as exc:
        logger.error("Sweep failed", error=str(exc), exc_info=True)
        _free_gpu_memory()
        if wandb_logger:
            wandb_logger.finish()
        return 1

    elapsed = time.monotonic() - t_start
    _print_summary(results, elapsed)

    # ------------------------------------------------------------------
    # Write a lightweight JSON summary next to the outputs
    # ------------------------------------------------------------------
    summary_path = Path(config.paths.output_root) / "sweep_summary.json"
    try:
        summary_data = {
            "experiment": config.tracking.experiment_name,
            "elapsed_seconds": round(elapsed, 2),
            "num_conditions": len(all_conditions),
            "conditions": [c.label for c in all_conditions],
            "metrics_summary": results.get("__summary__", {}),
        }
        with summary_path.open("w") as fh:
            json.dump(summary_data, fh, indent=2, default=str)
        logger.info("Summary written", path=str(summary_path))
    except Exception as exc:
        logger.warning("Could not write summary JSON", error=str(exc))

    # ------------------------------------------------------------------
    # Log to wandb and finalize
    # ------------------------------------------------------------------
    if wandb_logger and wandb_logger.enabled:
        # Log per-condition metrics
        sweep_summary = results.get("__summary__", {})
        for label, metrics in sweep_summary.items():
            wandb_logger.log_metrics(
                {f"condition/{label}/{k}": v for k, v in metrics.items()},
            )

        # Log summary table
        if sweep_summary:
            # Collect all metric keys
            all_keys = sorted(
                set(k for m in sweep_summary.values() for k in m.keys())
            )
            columns = ["condition"] + all_keys
            rows = [
                [label] + [metrics.get(k, float("nan")) for k in all_keys]
                for label, metrics in sweep_summary.items()
            ]
            wandb_logger.log_table("sweep_results", columns=columns, data=rows)

        # Note: piano roll image logging is now handled per-condition inside
        # SweepExecutor.run() via _log_visual_examples(), so images appear in
        # wandb as the sweep progresses rather than only at the very end.

        # Log final summary
        wandb_logger.log_summary({
            "elapsed_seconds": round(elapsed, 2),
            "num_conditions": len(all_conditions),
        })

        # Log the JSON summary as artifact
        if summary_path.exists():
            wandb_logger.log_artifact(summary_path, name="sweep_summary", artifact_type="result")

        wandb_logger.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
