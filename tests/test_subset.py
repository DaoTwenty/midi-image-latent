"""Tests for SubsetConfig validation and apply_subset filtering logic.

Covers midi_vae/config.py::SubsetConfig and midi_vae/data/subset.py::apply_subset,
plus the IngestStage cache key invalidation when subset parameters change.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from midi_vae.config import (
    DataConfig,
    ExperimentConfig,
    PathsConfig,
    SubsetConfig,
)
from midi_vae.data.subset import apply_subset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_files(tmp_path: Path, paths: list[str]) -> list[Path]:
    """Create empty files at the given relative paths under tmp_path.

    Returns the list of created absolute Paths.
    """
    result: list[Path] = []
    for rel in paths:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        result.append(p)
    return sorted(result)


# ---------------------------------------------------------------------------
# apply_subset — subdirectory filter
# ---------------------------------------------------------------------------


class TestApplySubsetSubdirectory:
    """apply_subset with subdirectory filtering."""

    def test_returns_only_files_in_subdir(self, tmp_path: Path) -> None:
        """Only files whose path relative to data_root starts with subdirectory are kept."""
        files = _make_files(
            tmp_path,
            [
                "f/song1.mid",
                "f/song2.mid",
                "g/song3.mid",
                "song4.mid",
            ],
        )
        subset = SubsetConfig(subdirectory="f/")
        result = apply_subset(files, subset, tmp_path, seed=42)
        result_names = {p.name for p in result}
        assert result_names == {"song1.mid", "song2.mid"}

    def test_subdirectory_without_trailing_slash(self, tmp_path: Path) -> None:
        """subdirectory filter works regardless of trailing slash convention."""
        files = _make_files(tmp_path, ["alpha/a.mid", "beta/b.mid"])
        subset = SubsetConfig(subdirectory="alpha")
        result = apply_subset(files, subset, tmp_path, seed=42)
        assert len(result) == 1
        assert result[0].name == "a.mid"

    def test_subdirectory_nested(self, tmp_path: Path) -> None:
        """Files in nested subdirectories under the target dir are included."""
        files = _make_files(
            tmp_path,
            [
                "a/b/deep.mid",
                "a/shallow.mid",
                "c/other.mid",
            ],
        )
        subset = SubsetConfig(subdirectory="a/")
        result = apply_subset(files, subset, tmp_path, seed=42)
        result_names = {p.name for p in result}
        assert result_names == {"deep.mid", "shallow.mid"}

    def test_subdirectory_no_match_returns_empty(self, tmp_path: Path) -> None:
        """Empty list returned when no files match the subdirectory."""
        files = _make_files(tmp_path, ["x/song.mid"])
        subset = SubsetConfig(subdirectory="y/")
        result = apply_subset(files, subset, tmp_path, seed=42)
        assert result == []

    def test_result_is_sorted(self, tmp_path: Path) -> None:
        """Result is sorted for determinism."""
        files = _make_files(tmp_path, ["f/z.mid", "f/a.mid", "f/m.mid"])
        subset = SubsetConfig(subdirectory="f/")
        result = apply_subset(files, subset, tmp_path, seed=42)
        assert result == sorted(result)


# ---------------------------------------------------------------------------
# apply_subset — glob pattern filter
# ---------------------------------------------------------------------------


class TestApplySubsetGlobPattern:
    """apply_subset with custom glob_pattern filtering."""

    def test_glob_filters_by_extension(self, tmp_path: Path) -> None:
        """glob_pattern can filter to a specific file extension."""
        files = _make_files(tmp_path, ["a/song.mid", "a/song.midi", "a/notes.txt"])
        subset = SubsetConfig(glob_pattern="a/**/*.mid")
        result = apply_subset(files, subset, tmp_path, seed=42)
        result_names = {p.name for p in result}
        assert result_names == {"song.mid"}

    def test_glob_intersects_with_input_files(self, tmp_path: Path) -> None:
        """apply_subset intersects glob results with the provided files list."""
        # Create extra files on disk not in the files list
        _make_files(tmp_path, ["a/extra.mid", "a/included.mid"])
        included_only = [tmp_path / "a" / "included.mid"]
        subset = SubsetConfig(glob_pattern="a/**/*.mid")
        result = apply_subset(included_only, subset, tmp_path, seed=42)
        assert len(result) == 1
        assert result[0].name == "included.mid"

    def test_glob_across_subdirs(self, tmp_path: Path) -> None:
        """glob_pattern can select files from multiple subdirectories."""
        files = _make_files(tmp_path, ["x/a.mid", "y/b.mid", "z/c.txt"])
        subset = SubsetConfig(glob_pattern="**/*.mid")
        result = apply_subset(files, subset, tmp_path, seed=42)
        result_names = {p.name for p in result}
        assert result_names == {"a.mid", "b.mid"}

    def test_glob_no_match_returns_empty(self, tmp_path: Path) -> None:
        """Empty list returned when glob matches no files in the input list."""
        files = _make_files(tmp_path, ["a/song.mid"])
        subset = SubsetConfig(glob_pattern="**/*.flac")
        result = apply_subset(files, subset, tmp_path, seed=42)
        assert result == []


# ---------------------------------------------------------------------------
# apply_subset — file list filter
# ---------------------------------------------------------------------------


class TestApplySubsetFileList:
    """apply_subset with explicit file_list filtering."""

    def test_file_list_relative_paths(self, tmp_path: Path) -> None:
        """File list with relative paths is resolved relative to data_root."""
        files = _make_files(tmp_path, ["a/song1.mid", "a/song2.mid", "b/song3.mid"])
        file_list_path = tmp_path / "my_list.txt"
        # Write relative paths (relative to data_root = tmp_path)
        file_list_path.write_text("a/song1.mid\nb/song3.mid\n")

        subset = SubsetConfig(file_list=str(file_list_path))
        result = apply_subset(files, subset, tmp_path, seed=42)
        result_names = {p.name for p in result}
        assert result_names == {"song1.mid", "song3.mid"}

    def test_file_list_absolute_paths(self, tmp_path: Path) -> None:
        """File list with absolute paths is handled correctly."""
        files = _make_files(tmp_path, ["a/song1.mid", "a/song2.mid"])
        file_list_path = tmp_path / "absolute_list.txt"
        # Write absolute paths
        abs_paths = "\n".join([str(tmp_path / "a" / "song1.mid")])
        file_list_path.write_text(abs_paths + "\n")

        subset = SubsetConfig(file_list=str(file_list_path))
        result = apply_subset(files, subset, tmp_path, seed=42)
        assert len(result) == 1
        assert result[0].name == "song1.mid"

    def test_file_list_intersection_only(self, tmp_path: Path) -> None:
        """Files in the list but not in the input files are ignored."""
        files = _make_files(tmp_path, ["a/present.mid"])
        file_list_path = tmp_path / "list.txt"
        # List includes present.mid plus a ghost entry that was never in files
        file_list_path.write_text("a/present.mid\na/ghost.mid\n")

        subset = SubsetConfig(file_list=str(file_list_path))
        result = apply_subset(files, subset, tmp_path, seed=42)
        assert len(result) == 1
        assert result[0].name == "present.mid"

    def test_file_list_no_match_returns_empty(self, tmp_path: Path) -> None:
        """Empty result when the file list has no overlap with input files."""
        files = _make_files(tmp_path, ["a/song.mid"])
        file_list_path = tmp_path / "empty_overlap.txt"
        file_list_path.write_text("b/other.mid\n")

        subset = SubsetConfig(file_list=str(file_list_path))
        result = apply_subset(files, subset, tmp_path, seed=42)
        assert result == []

    def test_file_list_blank_lines_ignored(self, tmp_path: Path) -> None:
        """Blank lines in the file list are ignored without error."""
        files = _make_files(tmp_path, ["a/song.mid"])
        file_list_path = tmp_path / "blanks.txt"
        file_list_path.write_text("\na/song.mid\n\n")

        subset = SubsetConfig(file_list=str(file_list_path))
        result = apply_subset(files, subset, tmp_path, seed=42)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# apply_subset — fraction filter
# ---------------------------------------------------------------------------


class TestApplySubsetFraction:
    """apply_subset with random fraction sampling."""

    def _make_n_files(self, tmp_path: Path, n: int) -> list[Path]:
        """Create n dummy files and return their sorted Paths."""
        return _make_files(tmp_path, [f"song_{i:04d}.mid" for i in range(n)])

    def test_fraction_50_percent_of_100(self, tmp_path: Path) -> None:
        """Sampling 50% of 100 files returns approximately 50 files."""
        files = self._make_n_files(tmp_path, 100)
        subset = SubsetConfig(fraction=0.5)
        result = apply_subset(files, subset, tmp_path, seed=42)
        assert len(result) == 50

    def test_fraction_deterministic_with_same_seed(self, tmp_path: Path) -> None:
        """Same seed produces the same sampled files across two calls."""
        files = self._make_n_files(tmp_path, 100)
        subset = SubsetConfig(fraction=0.3)
        result1 = apply_subset(files, subset, tmp_path, seed=7)
        result2 = apply_subset(files, subset, tmp_path, seed=7)
        assert result1 == result2

    def test_fraction_result_is_subset_of_input(self, tmp_path: Path) -> None:
        """Every file in the fraction result is from the original input."""
        files = self._make_n_files(tmp_path, 50)
        subset = SubsetConfig(fraction=0.4)
        result = apply_subset(files, subset, tmp_path, seed=99)
        assert all(p in files for p in result)

    def test_fraction_1_returns_all(self, tmp_path: Path) -> None:
        """fraction=1.0 returns all files."""
        files = self._make_n_files(tmp_path, 20)
        subset = SubsetConfig(fraction=1.0)
        result = apply_subset(files, subset, tmp_path, seed=42)
        assert len(result) == len(files)

    def test_fraction_uses_fraction_seed_when_provided(self, tmp_path: Path) -> None:
        """fraction_seed takes precedence over the top-level seed parameter."""
        files = self._make_n_files(tmp_path, 100)
        subset_with_fseed = SubsetConfig(fraction=0.5, fraction_seed=123)
        # Same fraction_seed + different top-level seed should produce same result
        result_seed42 = apply_subset(files, subset_with_fseed, tmp_path, seed=42)
        result_seed99 = apply_subset(files, subset_with_fseed, tmp_path, seed=99)
        assert result_seed42 == result_seed99


# ---------------------------------------------------------------------------
# apply_subset — fraction_seed produces different sample
# ---------------------------------------------------------------------------


class TestApplySubsetFractionSeed:
    """Different fraction_seed values produce different samples."""

    def test_different_fraction_seeds_produce_different_samples(
        self, tmp_path: Path
    ) -> None:
        """Two different fraction_seed values give different selections from the same files."""
        files = _make_files(tmp_path, [f"song_{i:04d}.mid" for i in range(100)])
        subset_a = SubsetConfig(fraction=0.5, fraction_seed=1)
        subset_b = SubsetConfig(fraction=0.5, fraction_seed=2)
        result_a = apply_subset(files, subset_a, tmp_path, seed=42)
        result_b = apply_subset(files, subset_b, tmp_path, seed=42)
        # With 100 files and 50 sampled, the probability of identical results
        # is astronomically small — treat any difference as a pass
        assert result_a != result_b

    def test_no_fraction_seed_respects_top_level_seed(self, tmp_path: Path) -> None:
        """When fraction_seed is None, the top-level seed drives sampling."""
        files = _make_files(tmp_path, [f"s_{i}.mid" for i in range(80)])
        subset = SubsetConfig(fraction=0.5)
        result_seed1 = apply_subset(files, subset, tmp_path, seed=1)
        result_seed2 = apply_subset(files, subset, tmp_path, seed=2)
        # Different top-level seeds must yield different samples
        assert result_seed1 != result_seed2


# ---------------------------------------------------------------------------
# apply_subset — composed (subdirectory + fraction)
# ---------------------------------------------------------------------------


class TestApplySubsetComposed:
    """apply_subset with multiple filters active simultaneously."""

    def test_subdirectory_then_fraction(self, tmp_path: Path) -> None:
        """subdirectory filter runs before fraction, fraction samples from the narrowed list."""
        # 10 files in 'a/', 10 files in 'b/'
        files = _make_files(
            tmp_path,
            [f"a/song_{i}.mid" for i in range(10)]
            + [f"b/song_{i}.mid" for i in range(10)],
        )
        # Keep only 'a/' subdir, then take 50%
        subset = SubsetConfig(subdirectory="a/", fraction=0.5)
        result = apply_subset(files, subset, tmp_path, seed=42)
        # Must be from 'a/' only
        assert all("a" in str(p) for p in result)
        # Must be ~50% of the 10 files in 'a/'
        assert len(result) == 5

    def test_file_list_then_fraction(self, tmp_path: Path) -> None:
        """file_list filter intersects first, then fraction samples from the intersection."""
        files = _make_files(tmp_path, [f"song_{i:03d}.mid" for i in range(20)])
        # Allowlist: first 10 songs
        file_list_path = tmp_path / "allowlist.txt"
        file_list_path.write_text(
            "\n".join(f"song_{i:03d}.mid" for i in range(10)) + "\n"
        )
        subset = SubsetConfig(file_list=str(file_list_path), fraction=0.5)
        result = apply_subset(files, subset, tmp_path, seed=42)
        # Fraction of 10 = 5
        assert len(result) == 5
        # All from allowlisted files
        allowlisted = {tmp_path / f"song_{i:03d}.mid" for i in range(10)}
        assert all(p in allowlisted for p in result)


# ---------------------------------------------------------------------------
# apply_subset — None config returns original list
# ---------------------------------------------------------------------------


class TestApplySubsetNoneReturnsAll:
    """apply_subset with an empty SubsetConfig returns the original file list."""

    def test_empty_subset_returns_all_files(self, tmp_path: Path) -> None:
        """A SubsetConfig with all-None fields returns the full input list unchanged."""
        files = _make_files(tmp_path, ["a.mid", "b.mid", "c.mid"])
        subset = SubsetConfig()
        result = apply_subset(files, subset, tmp_path, seed=42)
        assert sorted(result) == sorted(files)

    def test_empty_subset_preserves_count(self, tmp_path: Path) -> None:
        """The count of returned files equals the count of input files."""
        files = _make_files(tmp_path, [f"song_{i}.mid" for i in range(15)])
        subset = SubsetConfig()
        result = apply_subset(files, subset, tmp_path, seed=42)
        assert len(result) == 15


# ---------------------------------------------------------------------------
# SubsetConfig — validation: mutual exclusivity
# ---------------------------------------------------------------------------


class TestSubsetConfigValidationMutualExclusion:
    """SubsetConfig rejects simultaneous subdirectory + glob_pattern."""

    def test_both_subdirectory_and_glob_raises(self) -> None:
        """Setting both subdirectory and glob_pattern raises ValidationError."""
        with pytest.raises(ValidationError, match="mutually exclusive"):
            SubsetConfig(subdirectory="f/", glob_pattern="**/*.mid")

    def test_subdirectory_alone_valid(self) -> None:
        """subdirectory alone is valid."""
        cfg = SubsetConfig(subdirectory="f/")
        assert cfg.subdirectory == "f/"
        assert cfg.glob_pattern is None

    def test_glob_pattern_alone_valid(self) -> None:
        """glob_pattern alone is valid."""
        cfg = SubsetConfig(glob_pattern="a/**/*.mid")
        assert cfg.glob_pattern == "a/**/*.mid"
        assert cfg.subdirectory is None

    def test_neither_is_valid(self) -> None:
        """Neither subdirectory nor glob_pattern set is valid (all-None)."""
        cfg = SubsetConfig()
        assert cfg.subdirectory is None
        assert cfg.glob_pattern is None


# ---------------------------------------------------------------------------
# SubsetConfig — validation: fraction range
# ---------------------------------------------------------------------------


class TestSubsetConfigValidationFractionRange:
    """SubsetConfig enforces fraction in (0.0, 1.0]."""

    def test_fraction_zero_raises(self) -> None:
        """fraction=0.0 is rejected (exclusive lower bound)."""
        with pytest.raises(ValidationError):
            SubsetConfig(fraction=0.0)

    def test_fraction_above_one_raises(self) -> None:
        """fraction=1.5 is rejected (above 1.0)."""
        with pytest.raises(ValidationError):
            SubsetConfig(fraction=1.5)

    def test_fraction_negative_raises(self) -> None:
        """fraction=-0.1 is rejected (below 0.0)."""
        with pytest.raises(ValidationError):
            SubsetConfig(fraction=-0.1)

    def test_fraction_0_5_valid(self) -> None:
        """fraction=0.5 is accepted."""
        cfg = SubsetConfig(fraction=0.5)
        assert cfg.fraction == pytest.approx(0.5)

    def test_fraction_1_0_valid(self) -> None:
        """fraction=1.0 is accepted (upper bound inclusive)."""
        cfg = SubsetConfig(fraction=1.0)
        assert cfg.fraction == pytest.approx(1.0)

    def test_fraction_small_positive_valid(self) -> None:
        """A small positive fraction like 0.01 is accepted."""
        cfg = SubsetConfig(fraction=0.01)
        assert cfg.fraction == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# DataConfig — subset field
# ---------------------------------------------------------------------------


class TestDataConfigSubsetField:
    """DataConfig correctly accepts and stores SubsetConfig."""

    def test_default_subset_is_none(self) -> None:
        """DataConfig.subset defaults to None."""
        cfg = DataConfig()
        assert cfg.subset is None

    def test_subset_stored_correctly(self) -> None:
        """DataConfig.subset stores the provided SubsetConfig."""
        subset = SubsetConfig(subdirectory="f/")
        cfg = DataConfig(subset=subset)
        assert cfg.subset is not None
        assert cfg.subset.subdirectory == "f/"


# ---------------------------------------------------------------------------
# IngestStage — cache key changes with subset
# ---------------------------------------------------------------------------


class TestCacheKeyChangesWithSubset:
    """IngestStage.cache_key() differs when subset config changes."""

    def _make_stage(
        self, tmp_path: Path, subset: SubsetConfig | None = None
    ):  # type: ignore[return]
        """Build an IngestStage with optional subset config.

        Imports IngestStage inside the function so the test still works
        even if ingest.py changes signature; we only care about cache_key().
        """
        from midi_vae.pipelines.ingest import IngestStage

        data_root = tmp_path / "data"
        data_root.mkdir(parents=True, exist_ok=True)
        # Create at least one dummy file so data_root.exists() == True
        (data_root / "dummy.mid").touch()

        config = ExperimentConfig(
            paths=PathsConfig(
                data_root=str(data_root),
                output_root=str(tmp_path / "outputs"),
                cache_dir=str(tmp_path / "cache"),
            ),
            data=DataConfig(subset=subset),
        )
        return IngestStage(config)

    def test_no_subset_produces_cache_key(self, tmp_path: Path) -> None:
        """IngestStage without subset config returns a non-None cache key."""
        stage = self._make_stage(tmp_path, subset=None)
        key = stage.cache_key({})
        assert key is not None
        assert isinstance(key, str)
        assert len(key) > 0

    def test_different_subsets_produce_different_cache_keys(
        self, tmp_path: Path
    ) -> None:
        """Two IngestStages with different SubsetConfig produce different cache keys."""
        stage_no_subset = self._make_stage(tmp_path, subset=None)
        stage_with_subset = self._make_stage(
            tmp_path, subset=SubsetConfig(subdirectory="f/")
        )
        key_no_subset = stage_no_subset.cache_key({})
        key_with_subset = stage_with_subset.cache_key({})
        assert key_no_subset != key_with_subset

    def test_different_subdirectories_produce_different_keys(
        self, tmp_path: Path
    ) -> None:
        """Two IngestStages with different subdirectories have different cache keys."""
        stage_f = self._make_stage(tmp_path, subset=SubsetConfig(subdirectory="f/"))
        stage_g = self._make_stage(tmp_path, subset=SubsetConfig(subdirectory="g/"))
        assert stage_f.cache_key({}) != stage_g.cache_key({})

    def test_different_fractions_produce_different_keys(self, tmp_path: Path) -> None:
        """Two IngestStages with different fractions have different cache keys."""
        stage_10 = self._make_stage(tmp_path, subset=SubsetConfig(fraction=0.1))
        stage_50 = self._make_stage(tmp_path, subset=SubsetConfig(fraction=0.5))
        assert stage_10.cache_key({}) != stage_50.cache_key({})

    def test_same_subset_produces_same_cache_key(self, tmp_path: Path) -> None:
        """Two IngestStages with identical SubsetConfig produce the same cache key."""
        subset = SubsetConfig(subdirectory="f/", fraction=0.3, fraction_seed=7)
        stage_a = self._make_stage(tmp_path, subset=subset)
        stage_b = self._make_stage(tmp_path, subset=subset)
        assert stage_a.cache_key({}) == stage_b.cache_key({})

    def test_different_file_lists_produce_different_keys(
        self, tmp_path: Path
    ) -> None:
        """Two IngestStages with different file_list paths have different cache keys."""
        list_a = tmp_path / "list_a.txt"
        list_b = tmp_path / "list_b.txt"
        list_a.touch()
        list_b.touch()
        stage_a = self._make_stage(tmp_path, subset=SubsetConfig(file_list=str(list_a)))
        stage_b = self._make_stage(tmp_path, subset=SubsetConfig(file_list=str(list_b)))
        assert stage_a.cache_key({}) != stage_b.cache_key({})
