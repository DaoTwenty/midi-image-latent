# Team Coordination — Quick Reference

## Team Structure

| Role | Callsign | Owns | Key Deliverable |
|------|----------|------|-----------------|
| Core Infrastructure | ALPHA | config, registry, types, ABCs, pipeline, tracking | Foundation skeleton |
| Data Pipeline | BRAVO | preprocessing, rendering, transforms, datasets | MIDI → image tensors |
| Model Integration | CHARLIE | 12 VAE wrappers, sub-latent models, Transformer | Model layer |
| Algorithms | DELTA | 8 note detectors, 45+ metrics | Research instruments |
| QA & Integration | ECHO | tests, viz, experiment configs | Quality gate |

## Sprint Schedule

| Sprint | Days | Focus |
|--------|------|-------|
| S0 | 1–3 | ALPHA delivers skeleton. Others: env setup, read specs, write test stubs. |
| S1 | 4–10 | Parallel build. BRAVO: preprocessing + channels. CHARLIE: first 6 VAEs. DELTA: P0 metrics + threshold detector. ECHO: integration test stubs. |
| S2 | 11–17 | Continue parallel. CHARLIE: remaining VAEs + PCA. DELTA: advanced detectors + full metrics. ECHO: pipeline integration tests. |
| S3 | 18–24 | Integration. Wire everything. Exp 1A dry run. CHARLIE: sub-latent models. DELTA: information metrics. |
| S4 | 25+ | Run experiments. CHARLIE: conditioning + Transformer. DELTA: generative metrics. ECHO: all 9 experiment configs. |

## Git Rules

- Branch: `feature/<role>/<description>`
- Commit: `[ROLE] <type>: <description>`
- Squash-merge to main
- Rebase daily onto main before starting work
- No branch lives >3 days without merging

## Module Ownership (write access)

| Role | Directories |
|------|-------------|
| ALPHA | midi_vae/config.py, registry.py, data/types.py, pipelines/base.py, tracking/, utils/ |
| BRAVO | midi_vae/data/ (except types.py), configs/data/ |
| CHARLIE | midi_vae/models/ |
| DELTA | midi_vae/note_detection/, midi_vae/metrics/ |
| ECHO | tests/, midi_vae/visualization/, configs/experiments/, scripts/ |

## Protected Files (change requires consensus)
- midi_vae/data/types.py
- midi_vae/registry.py
- midi_vae/config.py
- All ABC base.py files (vae_wrapper, note_detection/base, metrics/base, sublatent/base, pipelines/base)

## PR Requirements
1. pytest passes
2. ECHO reviews
3. One other role reviews
4. No cross-boundary internal imports
5. Docstrings on public API
