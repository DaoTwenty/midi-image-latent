# Plan: Data Subset Feature

**Branch**: `feature/data-subset`
**Goal**: Let users run experiments or training on a subset of a dataset (e.g., only files from `data/lakh/f/`, a random 10% sample, or an explicit file list).

---

## Overview

Add a `SubsetConfig` to `DataConfig` that controls which files are selected from a dataset before ingestion. Supports five modes: subdirectory filtering, first-N selection, random sampling by fraction, custom glob patterns, and an explicit file list. The filtering is applied in `IngestStage.run()` after initial file discovery but before loading, and the subset parameters are folded into the cache key to prevent stale cache hits.

---

## 1. Config Schema Changes (`midi_vae/config.py`)

Add a new Pydantic model `SubsetConfig` and a `subset` field on `DataConfig`:

```python
class SubsetConfig(BaseModel):
    """Controls which files from a dataset are selected for processing."""

    subdirectory: str | None = None       # e.g. "f/" — only files under data_root/subdirectory
    glob_pattern: str | None = None       # e.g. "a/**/*.mid" — custom glob relative to data_root
    file_list: str | None = None          # path to a text file with one filepath per line
    fraction: float | None = None         # e.g. 0.1 — random 10% sample (0.0-1.0 exclusive)
    fraction_seed: int | None = None      # separate seed for fraction sampling; defaults to top-level seed

    model_config = {"frozen": True}
```

On `DataConfig`, add:

```python
class DataConfig(BaseModel):
    # ... existing fields ...
    max_files: int | None = None
    subset: SubsetConfig | None = None    # NEW
```

Add a `model_validator` on `SubsetConfig`:

```python
@model_validator(mode="after")
def _validate_exclusivity(self) -> SubsetConfig:
    if self.subdirectory and self.glob_pattern:
        raise ValueError("subset.subdirectory and subset.glob_pattern are mutually exclusive")
    if self.fraction is not None and not (0.0 < self.fraction <= 1.0):
        raise ValueError("subset.fraction must be in (0.0, 1.0]")
    return self
```

---

## 2. File Filtering Logic — New File (`midi_vae/data/subset.py`)

Create a pure function that takes a list of file paths and a `SubsetConfig` plus a seed, and returns the filtered list. This keeps the logic testable and reusable across `IngestStage`, the three `Dataset` classes, and `preprocess_dataset.py`.

```python
def apply_subset(
    files: list[Path],
    subset: SubsetConfig,
    data_root: Path,
    seed: int,
) -> list[Path]:
```

**Filter application order:**

1. **subdirectory**: Keep only files whose path relative to `data_root` starts with `subset.subdirectory`
2. **glob_pattern**: Discover files using `data_root.glob(subset.glob_pattern)` and intersect with `files`
3. **file_list**: Read the text file, resolve paths relative to `data_root`, and intersect with `files`
4. **fraction**: Use `random.Random(subset.fraction_seed or seed)` to sample `int(len(files) * subset.fraction)` files deterministically

Returns a sorted list for determinism.

---

## 3. IngestStage Changes (`midi_vae/pipelines/ingest.py`)

Modify `IngestStage.run()` to call `apply_subset` after initial file discovery and before `max_files` truncation:

```python
from midi_vae.data.subset import apply_subset

all_files = sorted(data_root.glob(glob_pattern))
# MAESTRO fallback...

# NEW: Apply subset filtering
if data_cfg.subset is not None:
    all_files = apply_subset(all_files, data_cfg.subset, data_root, self.config.seed)

# EXISTING: max_files truncation (applied last)
if self._max_files is not None:
    all_files = all_files[: self._max_files]
```

When `subset.glob_pattern` is set, replace the default `_DATASET_GLOB` pattern:

```python
if data_cfg.subset and data_cfg.subset.glob_pattern:
    glob_pattern = data_cfg.subset.glob_pattern
else:
    glob_pattern = _DATASET_GLOB.get(dataset_name, "**/*.mid")
```

Update `cache_key()` to include subset parameters:

```python
subset_key = ""
if self.config.data.subset:
    s = self.config.data.subset
    subset_key = f"{s.subdirectory}|{s.glob_pattern}|{s.file_list}|{s.fraction}|{s.fraction_seed}"

return compute_hash(
    str(data_root),
    self.config.data.dataset,
    ...,
    subset_key,         # NEW
    self._max_files,    # BUG FIX: was missing
)
```

---

## 4. Dataset Class Changes (`midi_vae/data/datasets.py`)

Add a helper method to the `MidiDataset` base class (or each dataset class) that applies subset filtering, then call it after file discovery and before `max_files` slicing in `_load_bars()`.

---

## 5. CLI Changes (`scripts/run_experiment.py`)

Add three new CLI flags:

```
--subset-dir SUBDIR       Only load files from this subdirectory under data_root
--subset-fraction FRAC    Random fraction of files to use (e.g. 0.1 for 10%)
--subset-file-list PATH   Path to a text file listing specific files (one per line)
```

In `main()`, apply them to the raw OmegaConf config before Pydantic validation:

```python
if args.subset_dir:
    OmegaConf.update(raw, "data.subset.subdirectory", args.subset_dir, merge=True)
if args.subset_fraction is not None:
    OmegaConf.update(raw, "data.subset.fraction", args.subset_fraction, merge=True)
if args.subset_file_list:
    OmegaConf.update(raw, "data.subset.file_list", args.subset_file_list, merge=True)
```

Add the same flags to `scripts/preprocess_dataset.py`.

---

## 6. YAML Configuration Examples

**Subdirectory in experiment config:**
```yaml
data:
  dataset: lakh
  subset:
    subdirectory: "f/"
```

**Override file (`configs/overrides/lakh_subdir_f.yaml`):**
```yaml
data:
  subset:
    subdirectory: "f/"
```

**File list:**
```yaml
data:
  subset:
    file_list: "configs/file_lists/curated_100.txt"
```

**Fraction with cap:**
```yaml
data:
  subset:
    fraction: 0.1
  max_files: 200
```

**CLI examples:**
```bash
# Only files from the 'f' subdirectory of lakh
python scripts/run_experiment.py configs/experiments/exp_1a_vae_comparison.yaml \
    --data-root data/lakh --subset-dir f/

# Random 10% of files
python scripts/run_experiment.py configs/experiments/exp_1a_vae_comparison.yaml \
    --data-root data/lakh --subset-fraction 0.1

# Explicit file list
python scripts/run_experiment.py configs/experiments/exp_1a_vae_comparison.yaml \
    --data-root data/lakh --subset-file-list configs/file_lists/debug_50.txt
```

---

## 7. Composition with Existing Features

Filtering order ensures clean composition:

```
1. glob_pattern (or default dataset glob) — which files exist
2. subdirectory — narrow to a subtree
3. file_list — intersect with an explicit allowlist
4. fraction — randomly sample from what remains
5. max_files — hard cap on final count (existing)
6. bars_per_instrument — per-instrument sampling within each file (existing, in MidiIngestor)
```

`--mini` continues to set `max_files=5` + `bars_per_instrument=20`. Running `--mini --subset-fraction 0.5` gives 50% of files, capped at 5.

---

## 8. Testing Plan (`tests/test_subset.py`)

| Test | What it verifies |
|------|-----------------|
| `test_apply_subset_subdirectory` | Filters to files under a subdirectory |
| `test_apply_subset_glob_pattern` | Custom glob works |
| `test_apply_subset_file_list` | Reads file list and intersects |
| `test_apply_subset_fraction` | Deterministic sampling at given fraction |
| `test_apply_subset_fraction_seed` | Different seed produces different sample |
| `test_apply_subset_composed` | subdirectory + fraction together |
| `test_apply_subset_none` | No subset returns original list |
| `test_subset_config_validation` | Mutual exclusivity of subdirectory and glob_pattern |
| `test_cache_key_changes_with_subset` | IngestStage cache key differs with subset |

---

## 9. Files to Create and Modify

| File | Action | What changes |
|------|--------|-------------|
| `midi_vae/config.py` | Modify | Add `SubsetConfig`, add `subset` field to `DataConfig` |
| `midi_vae/data/subset.py` | Create | `apply_subset()` function |
| `midi_vae/pipelines/ingest.py` | Modify | Call `apply_subset`, update `cache_key`, handle glob override |
| `midi_vae/data/datasets.py` | Modify | Add subset support to all 3 dataset classes |
| `scripts/run_experiment.py` | Modify | Add 3 CLI flags, apply to raw config |
| `scripts/preprocess_dataset.py` | Modify | Add same 3 CLI flags |
| `tests/test_subset.py` | Create | Unit tests for `apply_subset` and config validation |
| `EXPERIMENTS.md` | Modify | Document subset usage |
