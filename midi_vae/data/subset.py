"""File subset filtering for dataset ingestion.

Provides a single public function, :func:`apply_subset`, that takes a list of
discovered MIDI file paths and applies the filtering rules encoded in a
:class:`~midi_vae.config.SubsetConfig` instance.  The function is a pure
transformation — it has no side effects and can be tested independently.
"""

from __future__ import annotations

import random
from pathlib import Path

from midi_vae.config import SubsetConfig


def apply_subset(
    files: list[Path],
    subset: SubsetConfig,
    data_root: Path,
    seed: int,
) -> list[Path]:
    """Filter a list of file paths according to a SubsetConfig.

    Filters are applied in this order:

    1. **subdirectory** — keep only files whose path relative to *data_root*
       starts with ``subset.subdirectory``.
    2. **glob_pattern** — discover files using ``data_root.glob(subset.glob_pattern)``
       and intersect with *files*.
    3. **file_list** — read the text file, resolve each line relative to
       *data_root*, and intersect with *files*.
    4. **fraction** — randomly sample ``int(len(files) * subset.fraction)``
       files using ``random.Random(subset.fraction_seed or seed)``.

    The returned list is always sorted for determinism.

    Args:
        files: Initial list of candidate file paths (typically from a glob).
        subset: Subset configuration describing which filters to apply.
        data_root: Root directory of the dataset.  Used to resolve relative
            paths in ``subdirectory`` and ``file_list``.
        seed: Fallback RNG seed used when ``subset.fraction_seed`` is None.

    Returns:
        Sorted list of file paths after all filters have been applied.
    """
    result: list[Path] = list(files)

    # 1. Subdirectory filter: keep only files inside data_root/subdirectory.
    if subset.subdirectory is not None:
        prefix = data_root / subset.subdirectory
        result = [f for f in result if _is_relative_to(f, prefix)]

    # 2. Glob-pattern filter: intersect with files discovered by a custom glob.
    if subset.glob_pattern is not None:
        glob_set: set[Path] = set(data_root.glob(subset.glob_pattern))
        result = [f for f in result if f in glob_set]

    # 3. File-list filter: intersect with an explicit allowlist.
    # Both sides are resolved so symlinks and relative '..' components do not
    # produce false mismatches.
    if subset.file_list is not None:
        allowlist = _read_file_list(Path(subset.file_list), data_root)
        result = [f for f in result if f.resolve() in allowlist]

    # 4. Fraction filter: deterministic random sampling.
    if subset.fraction is not None:
        rng = random.Random(subset.fraction_seed if subset.fraction_seed is not None else seed)
        k = max(1, int(len(result) * subset.fraction))
        result = rng.sample(result, min(k, len(result)))

    return sorted(result)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_relative_to(path: Path, prefix: Path) -> bool:
    """Return True if *path* is inside *prefix* (inclusive).

    Supports Python 3.8 where ``Path.is_relative_to`` does not exist.

    Args:
        path: The path to test.
        prefix: The directory that must be an ancestor of *path*.

    Returns:
        True if *path* starts with *prefix*.
    """
    try:
        path.relative_to(prefix)
        return True
    except ValueError:
        return False


def _read_file_list(file_list_path: Path, data_root: Path) -> set[Path]:
    """Read an allowlist file and resolve each entry relative to *data_root*.

    Lines starting with ``#`` and blank lines are ignored.  Absolute paths are
    used as-is; relative paths are resolved against *data_root*.

    Args:
        file_list_path: Path to the text file (one filepath per line).
        data_root: Dataset root used for resolving relative paths.

    Returns:
        Set of resolved absolute ``Path`` objects for O(1) membership checks.

    Raises:
        FileNotFoundError: If *file_list_path* does not exist.
    """
    if not file_list_path.exists():
        raise FileNotFoundError(f"subset.file_list not found: {file_list_path}")

    resolved: set[Path] = set()
    with file_list_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            if not p.is_absolute():
                p = data_root / p
            resolved.add(p.resolve())
    return resolved
