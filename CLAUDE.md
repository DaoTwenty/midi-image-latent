# MIDI Image VAE — Project Memory

## Project Summary
Research project investigating whether frozen pretrained image VAEs (Stable Diffusion, FLUX, CogView4 families) can encode MIDI piano-roll images into musically useful latent spaces. 12 VAEs, 5 experiments, modular Python/PyTorch codebase.

## Architecture Principles
- **Configuration-first**: All params in YAML. No magic constants. OmegaConf + Pydantic schemas.
- **Registry pattern**: All swappable components use `@ComponentRegistry.register(type, name)`. New implementations = new decorated class, zero pipeline changes.
- **Pipeline composition**: Experiments are DAGs of `PipelineStage` nodes. Stages declare typed inputs/outputs via frozen dataclasses. Runner resolves dependencies, caches results.
- **Typed contracts**: All inter-stage data uses frozen dataclasses (`BarData`, `PianoRollImage`, `LatentEncoding`, `ReconstructedBar`, `MidiNote`).

## Tech Stack
- Python 3.11+, PyTorch >= 2.2, HuggingFace diffusers >= 0.28
- pypianoroll for MIDI parsing, OmegaConf for config, Pydantic v2 for validation
- structlog for logging, h5py for storage, pandas/pyarrow for metrics tables
- Optional: wandb for remote tracking

## Directory Layout
```
midi_vae/              # Core library package
  config.py            # OmegaConf + Pydantic config loader
  registry.py          # ComponentRegistry with @register decorator
  data/
    types.py           # ALL dataclasses (BarData, PianoRollImage, etc.)
    preprocessing.py   # MIDI ingest, bar segmentation
    rendering.py       # ChannelStrategy implementations (velocity_only, vo_split, vos)
    transforms.py      # Resize, normalize, pad transforms
    datasets.py        # LPD5Dataset, Pop909Dataset, MaestroDataset
  models/
    vae_wrapper.py     # FrozenImageVAE ABC
    vae_registry.py    # 12 concrete VAE wrappers
    sublatent/
      base.py          # SubLatentModel ABC
      pca.py, mlp.py, sub_vae.py, conditioning.py
    sequence/
      bar_transformer.py
  note_detection/
    base.py            # NoteDetector ABC
    threshold.py       # 4 threshold methods
    hmm_tracker.py, cnn_segmenter.py, gmm_detector.py, morphological.py
  metrics/
    base.py            # Metric ABC + MetricsEngine
    reconstruction.py, harmony.py, rhythm.py, dynamics.py
    information.py, latent_space.py, conditioning.py, generative.py
  pipelines/
    base.py            # PipelineStage ABC + PipelineRunner
    ingest.py, render.py, encode.py, decode.py, detect.py, evaluate.py
    train_sublatent.py, train_sequence.py
  tracking/
    experiment.py, job.py, cache.py, wandb_logger.py
  visualization/
    piano_roll.py, latent_plots.py, metric_dashboards.py
  utils/
    seed.py, device.py, io.py, logging.py
configs/               # All YAML configs
tests/                 # All tests
scripts/               # CLI entry points
```

## Git Conventions
- Branches: `feature/<role>/<description>` (e.g., `feature/alpha/config-system`)
- Commits: `[ROLE] <type>: <description>` (e.g., `[ALPHA] feat: add config loader`)
- Types: feat, fix, refactor, test, docs
- Squash-merge to main. One logical change per PR.

## Code Conventions
- Every public class/method has a docstring
- Type annotations on all function signatures
- All params come from config — no hardcoded paths, thresholds, model IDs
- Components register via `@ComponentRegistry.register('type', 'name')`
- Import only ABCs and data types across module boundaries — never internal implementations

## Protected Interfaces (require team consensus to change)
- `midi_vae/data/types.py` — all dataclass contracts
- `midi_vae/registry.py` — ComponentRegistry
- `midi_vae/config.py` — config schemas
- `midi_vae/models/vae_wrapper.py` — FrozenImageVAE ABC
- `midi_vae/note_detection/base.py` — NoteDetector ABC
- `midi_vae/metrics/base.py` — Metric ABC + MetricsEngine
- `midi_vae/models/sublatent/base.py` — SubLatentModel ABC
- `midi_vae/pipelines/base.py` — PipelineStage ABC + PipelineRunner

## Module Ownership
| Role    | Writes to                                    | Imports from (read-only)         |
|---------|----------------------------------------------|----------------------------------|
| ALPHA   | config, registry, types, pipelines/base, tracking, utils | —                   |
| BRAVO   | data/ (except types.py), configs/data/       | types, config, registry          |
| CHARLIE | models/                                       | types, config, registry          |
| DELTA   | note_detection/, metrics/                     | types, config, registry          |
| ECHO    | tests/, visualization/, configs/experiments/ | everything (read for testing)    |

## Reference Documents
- Full implementation spec: `specs/implementation_spec.md`
- Team coordination rules: `specs/coordination.md`
- Role-specific briefs: `specs/task_*.md`

## Testing
- `pytest` from repo root. All tests in `tests/`.
- Every public method needs at least 1 test.
- Use fixtures from `tests/conftest.py` for synthetic data.
- Run: `make test` (quick) or `make test-cov` (with coverage).

## Agent Teams Orchestration

This project uses Claude Code Agent Teams (experimental). The orchestrator agent manages 5 specialist teammates.

### How It Works
- The **orchestrator** (Opus) is the interactive session you talk to
- It spawns **teammates** (Sonnet) that work in parallel with worktree isolation
- Teammates can message each other directly for coordination
- You only talk to the orchestrator — it relays instructions and synthesizes results

### Launch Command
```bash
bash scripts/launch.sh          # Fresh start
bash scripts/launch.sh resume   # Resume previous session
```

### Talking to the Orchestrator
- `start` — Begin Sprint 0
- `status` — Check all agent progress
- `pause` — Gracefully stop after current tasks finish
- `resume` — Pick up from last checkpoint
- `skip to sprint N` — Jump to a specific sprint
- Any natural language instruction — the orchestrator adapts

### State Persistence
The orchestrator tracks state through git. After each sprint:
- All work is committed on feature branches
- Merged work is on main
- `git log --oneline -20 main` shows what's done
- `python -m pytest tests/ -x` shows what works
To resume: the orchestrator reads git state and picks up where it left off.
