"""CLI script for preprocessing a MIDI dataset into BarData and optionally HDF5.

Usage::

    python scripts/preprocess_dataset.py \\
        --config configs/base.yaml configs/data/lakh.yaml \\
        --data-root data/lakh \\
        --output-dir outputs/bars \\
        [--max-files N] \\
        [--save-hdf5] \\
        [--overrides key=value ...]

The script runs IngestStage followed by RenderStage and reports statistics.
If --save-hdf5 is passed, rendered tensors are saved to an HDF5 file in
output-dir grouped by instrument.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("preprocess_dataset")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess a MIDI dataset into BarData objects.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        nargs="+",
        default=["configs/base.yaml"],
        metavar="YAML",
        help="One or more YAML config files to merge (later files override earlier).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override paths.data_root from config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/preprocessed",
        help="Directory to write statistics and optional HDF5 output.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit the number of source files processed (for debugging).",
    )
    parser.add_argument(
        "--save-hdf5",
        action="store_true",
        default=False,
        help="Save rendered tensors to an HDF5 file.",
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
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="OmegaConf dotlist overrides (e.g. data.min_notes_per_bar=4).",
    )
    return parser.parse_args(argv)


def save_hdf5(images: list, output_dir: Path) -> None:
    """Save rendered PianoRollImage tensors to an HDF5 file.

    The file is organised as::

        /instrument_name/bar_id  -> tensor (3, H, W)

    Args:
        images: List of PianoRollImage objects.
        output_dir: Directory where the HDF5 file will be written.
    """
    try:
        import h5py
        import numpy as np
    except ImportError:
        logger.error("h5py is required for HDF5 output. Install with: pip install h5py")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = output_dir / "bars.h5"

    logger.info("Saving %d images to %s", len(images), hdf5_path)

    with h5py.File(str(hdf5_path), "w") as f:
        for img in images:
            # Group by instrument extracted from bar_id (format: song_instrument_NNNN)
            parts = img.bar_id.split("_")
            instrument = parts[1] if len(parts) >= 3 else "unknown"

            grp = f.require_group(instrument)
            arr = img.image.numpy()
            grp.create_dataset(img.bar_id, data=arr, compression="gzip", compression_opts=4)

    logger.info("HDF5 file written: %s", hdf5_path)


def main(argv: list[str] | None = None) -> int:
    """Run the preprocessing pipeline.

    Args:
        argv: CLI arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 = success).
    """
    args = parse_args(argv)

    # Build overrides list
    overrides = list(args.overrides)
    if args.data_root:
        overrides.append(f"paths.data_root={args.data_root}")
    if args.subset_dir:
        overrides.append(f"data.subset.subdirectory={args.subset_dir}")
    if args.subset_fraction is not None:
        overrides.append(f"data.subset.fraction={args.subset_fraction}")
    if args.subset_file_list:
        overrides.append(f"data.subset.file_list={args.subset_file_list}")

    # Load config
    try:
        from midi_vae.config import load_config
        config = load_config(paths=args.config, overrides=overrides)
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import pipeline stages (triggers registry population)
    from midi_vae.pipelines.ingest import IngestStage
    from midi_vae.pipelines.render import RenderStage

    ingest = IngestStage(config, max_files=args.max_files)
    render = RenderStage(config)

    # Run ingest
    logger.info("=== IngestStage ===")
    ingest_outputs = ingest.run({})
    bars = ingest_outputs.get("bars", [])
    logger.info("Extracted %d bars", len(bars))

    if not bars:
        logger.warning("No bars extracted. Check data_root and dataset config.")
        return 0

    # Run render
    logger.info("=== RenderStage ===")
    render_outputs = render.run({"bars": bars})
    images = render_outputs.get("images", [])
    logger.info("Rendered %d images", len(images))

    # Statistics
    instruments: dict[str, int] = {}
    for bar in bars:
        instruments[bar.instrument] = instruments.get(bar.instrument, 0) + 1

    logger.info("Bar counts by instrument:")
    for inst, count in sorted(instruments.items()):
        logger.info("  %-12s %d bars", inst, count)

    # Optional HDF5 save
    if args.save_hdf5 and images:
        save_hdf5(images, output_dir)

    logger.info("Preprocessing complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
