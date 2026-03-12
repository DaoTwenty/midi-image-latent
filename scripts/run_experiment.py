#!/usr/bin/env python
"""CLI entry point for running MIDI Image VAE experiments.

Usage:
    python scripts/run_experiment.py configs/base.yaml configs/experiments/exp1a.yaml
    python scripts/run_experiment.py configs/base.yaml --override data.dataset=pop909 seed=123
"""

from __future__ import annotations

import argparse
import sys

from midi_vae.config import load_config
from midi_vae.utils.logging import setup_logging, get_logger
from midi_vae.utils.seed import set_global_seed
from midi_vae.utils.device import get_device


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a MIDI Image VAE experiment",
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="YAML config file paths (later files override earlier)",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Dotlist overrides (e.g., data.dataset=pop909 seed=123)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Pipeline stage name to resume from",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(level=args.log_level)
    logger = get_logger(__name__)

    # Load config
    logger.info("Loading configuration", configs=args.configs, overrides=args.override)
    config = load_config(paths=args.configs, overrides=args.override)

    # Setup reproducibility and device
    set_global_seed(config.seed)
    device = get_device(config.device)
    logger.info("Experiment setup", seed=config.seed, device=str(device))

    # TODO: Build pipeline stages based on config
    # TODO: Create ExperimentTracker
    # TODO: Run pipeline
    logger.info("Pipeline execution not yet implemented — foundation only")


if __name__ == "__main__":
    main()
