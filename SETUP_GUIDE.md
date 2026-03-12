# Multi-Agent Claude Code Setup — Operational Guide

## MIDI Image VAE Project

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Architecture Overview](#2-architecture-overview)
3. [File Inventory](#3-file-inventory)
4. [Step-by-Step Setup](#4-step-by-step-setup)
5. [How to Launch Agents](#5-how-to-launch-agents)
6. [Orchestrator Workflow](#6-orchestrator-workflow)
7. [Agent Workflow (per role)](#7-agent-workflow)
8. [Sprint 0 Playbook](#8-sprint-0-playbook)
9. [Merge & Integration Workflow](#9-merge--integration-workflow)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| **git** | Version control + worktrees | Pre-installed on most systems |
| **Python 3.11+** | Runtime | `pyenv install 3.11` or system Python |
| **Node.js 18+** | Required by Claude Code | `nvm install 18` or system Node |
| **Claude Code CLI** | Agent runtime | `npm install -g @anthropic-ai/claude-code` |
| **tmux** (recommended) | Multi-terminal multiplexer | `brew install tmux` / `sudo apt install tmux` |

Verify installation:

```bash
git --version          # >= 2.20 (worktree support)
python3 --version      # >= 3.11
node --version         # >= 18
claude --version       # Any recent version
tmux -V                # Optional but recommended
```

---

## 2. Architecture Overview

### The Core Idea

Each agent role runs as a **separate Claude Code session** in its **own git worktree**. Worktrees are lightweight checkouts of the same repository — they share history and remote connections but have independent working directories and branches. This means five agents can write code simultaneously without any file conflicts.

```
┌─────────────────────────────────────────────────────────────┐
│  REPO: midi-image-vae (main branch)                         │
│  ┌──────────────┐                                           │
│  │ ORCHESTRATOR  │  ← You sit here. Coordinates all agents. │
│  │ (main branch) │    Merges PRs. Runs /project:status.     │
│  └──────┬───────┘                                           │
│         │ git worktree                                      │
│  ┌──────┴──────────────────────────────────────────────┐    │
│  │ .claude/worktrees/                                   │    │
│  │  ├── alpha/  ← feature/alpha/sprint0 branch          │    │
│  │  ├── bravo/  ← feature/bravo/sprint0 branch          │    │
│  │  ├── charlie/← feature/charlie/sprint0 branch         │    │
│  │  ├── delta/  ← feature/delta/sprint0 branch           │    │
│  │  └── echo/   ← feature/echo/sprint0 branch            │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Why This Works

1. **Git worktrees** give each agent isolated file access — no merge conflicts during development
2. **`.claude/agents/*.md`** files define each agent's persona, owned files, and constraints
3. **`claude --agent <role>`** launches Claude Code with that agent's system prompt as the primary identity
4. **CLAUDE.md** at repo root provides shared project context to every session
5. **`.claude/commands/`** provides consistent workflow operations across all agents
6. **Orchestrator on main** coordinates merges and dispatches sprint tasks

### What Claude Code Reads (in order)

```
~/.claude/CLAUDE.md            ← User-level (your personal preferences)
<repo-root>/CLAUDE.md          ← Project-level (shared team context) ✓ WE SET THIS
.claude/settings.json          ← Permissions and env vars           ✓ WE SET THIS
.claude/agents/<name>.md       ← Agent definitions (for delegation) ✓ WE SET THIS
.claude/commands/<name>.md     ← Slash commands (/project:sync etc) ✓ WE SET THIS
```

---

## 3. File Inventory

Every file you need is in the setup package. Here's what each does:

### Root Level

| File | Purpose |
|------|---------|
| `CLAUDE.md` | **Shared project memory.** Loaded by every Claude Code session. Contains architecture principles, directory layout, code conventions, module ownership table, and git rules. |
| `.gitignore` | Standard Python + Claude Code ignores (`.claude/worktrees/`, `outputs/`, etc.) |

### `.claude/` Directory

| File | Purpose |
|------|---------|
| `settings.json` | **Project permissions.** Allows common Bash commands (python, pytest, git, make), Read/Write/Edit. Denies destructive commands. Shared via git. |
| `agents/alpha.md` | ALPHA agent definition: system prompt, owned files, forbidden files, implementation references |
| `agents/bravo.md` | BRAVO agent: data pipeline focus |
| `agents/charlie.md` | CHARLIE agent: model integration focus |
| `agents/delta.md` | DELTA agent: algorithms (detection + metrics) focus |
| `agents/echo.md` | ECHO agent: QA & integration focus |
| `commands/sync.md` | `/project:sync` — Rebase current branch onto latest main |
| `commands/pr.md` | `/project:pr` — Validate and prepare a pull request |
| `commands/check.md` | `/project:check` — Run tests and verify imports |
| `commands/status.md` | `/project:status` — Show all worktrees, branches, recent activity |
| `commands/kickoff.md` | `/project:kickoff <role>` — Assign next sprint task to a role |

### `specs/` Directory

| File | Purpose |
|------|---------|
| `implementation_spec.md` | Condensed reference from the full implementation spec. Config schemas, data types, ABCs, VAE registry, note detection methods, pipeline mapping. Agents read this instead of the full 20-page doc. |
| `coordination.md` | Team structure, sprint schedule, git rules, module ownership, PR requirements. |

### `scripts/` Directory

| File | Purpose |
|------|---------|
| `setup_multiagent.sh` | **Master setup script.** Initializes repo, creates skeleton, copies configs, makes initial commit, creates all 5 worktrees. Run once. |
| `launch_all_agents.sh` | Creates a tmux session with 6 windows (orchestrator + 5 agents). Convenience script. |

---

## 4. Step-by-Step Setup

### Step 1: Clone or Create the Setup Package

```bash
# Option A: If you have the setup package as a directory
cp -r multiagent-setup/ ~/midi-vae-setup
cd ~/midi-vae-setup

# Option B: If starting fresh, create the files manually
# (copy each file from this guide into the correct path)
```

### Step 2: Run the Setup Script

```bash
chmod +x scripts/setup_multiagent.sh
./scripts/setup_multiagent.sh ~/midi-image-vae
```

This script will:
1. Create `~/midi-image-vae/` with a git repo
2. Build the full directory skeleton (`midi_vae/`, `configs/`, `tests/`, etc.)
3. Copy all Claude Code configs (CLAUDE.md, agents, commands, settings)
4. Copy spec reference files
5. Create `pyproject.toml` and `Makefile`
6. Make an initial commit on `main`
7. Create 5 git worktrees under `.claude/worktrees/`

### Step 3: Set Up Python Environment

```bash
cd ~/midi-image-vae
make setup
source .venv/bin/activate
```

### Step 4: Verify the Setup

```bash
# Check worktrees exist
git worktree list

# Should show:
# /home/you/midi-image-vae              <hash> [main]
# /home/you/midi-image-vae/.claude/worktrees/alpha   <hash> [feature/alpha/sprint0]
# /home/you/midi-image-vae/.claude/worktrees/bravo   <hash> [feature/bravo/sprint0]
# /home/you/midi-image-vae/.claude/worktrees/charlie  <hash> [feature/charlie/sprint0]
# /home/you/midi-image-vae/.claude/worktrees/delta   <hash> [feature/delta/sprint0]
# /home/you/midi-image-vae/.claude/worktrees/echo    <hash> [feature/echo/sprint0]

# Check agent definitions exist
ls .claude/agents/
# alpha.md  bravo.md  charlie.md  delta.md  echo.md

# Check commands exist
ls .claude/commands/
# check.md  kickoff.md  pr.md  status.md  sync.md
```

### Step 5: (Optional) Verify Claude Code Reads Configs

```bash
cd ~/midi-image-vae
claude
# In the Claude session, type:
#   "What agents are available? List them."
# Claude should list: alpha, bravo, charlie, delta, echo
#
# Type: /project:status
# Should show worktree list and branch info
```

---

## 5. How to Launch Agents

### Option A: Manual Terminals (Simple)

Open 6 terminal windows/tabs. In each:

```bash
# Terminal 1: ORCHESTRATOR
cd ~/midi-image-vae
claude

# Terminal 2: ALPHA
cd ~/midi-image-vae/.claude/worktrees/alpha
claude --agent alpha

# Terminal 3: BRAVO
cd ~/midi-image-vae/.claude/worktrees/bravo
claude --agent bravo

# Terminal 4: CHARLIE
cd ~/midi-image-vae/.claude/worktrees/charlie
claude --agent charlie

# Terminal 5: DELTA
cd ~/midi-image-vae/.claude/worktrees/delta
claude --agent delta

# Terminal 6: ECHO
cd ~/midi-image-vae/.claude/worktrees/echo
claude --agent echo
```

### Option B: Built-in Worktree Mode

Claude Code has native `--worktree` support that creates worktrees automatically:

```bash
# These create worktrees in .claude/worktrees/ automatically
claude --worktree alpha --agent alpha
claude --worktree bravo --agent bravo
# ... etc.
```

### Option C: tmux (Recommended for Monitoring)

```bash
chmod +x scripts/launch_all_agents.sh
./scripts/launch_all_agents.sh

# Attach to the session
tmux attach -t midi-vae-agents

# Navigate: Ctrl+B then n (next) or p (previous) or 0-5 (jump)
# Detach: Ctrl+B then d (agents keep running)
```

### Option D: Orchestrator-Driven Subagents

From the orchestrator session on main, you can delegate directly:

```
> Use the alpha subagent to implement the ComponentRegistry class in midi_vae/registry.py.
  It should have register(), get(), and list() class methods as described in specs/implementation_spec.md Section 4.1.
```

Claude will spawn ALPHA as a subagent with its own context window and worktree isolation.

---

## 6. Orchestrator Workflow

The orchestrator (you or a Claude session on `main`) coordinates all work.

### Daily Rhythm

```
Morning:
1. /project:status                    ← See what each agent did
2. For each role with pending work:
   - Review their branch: git log feature/<role>/sprint0 --oneline -5
   - If ready: merge to main (see Section 9)
3. /project:kickoff <role>            ← Assign next task

During the day:
4. Monitor agents: switch between tmux windows
5. Answer agent questions when they get stuck
6. Merge completed work to main as it comes in

End of day:
7. /project:status                    ← Final check
8. Ensure all agents committed their WIP
```

### Key Orchestrator Commands

| Command | When to Use |
|---------|-------------|
| `/project:status` | Check all agent progress |
| `/project:kickoff alpha` | Assign ALPHA their next task |
| `git merge feature/alpha/sprint0` | Merge completed ALPHA work |
| `git log --all --oneline --graph -20` | Visualize branch state |

### Sprint 0 Orchestrator Checklist

```
[ ] Run setup_multiagent.sh
[ ] Launch ALPHA agent
[ ] Give ALPHA the Sprint 0 prompt (see Section 8)
[ ] While ALPHA works: launch ECHO to set up conftest.py and test stubs
[ ] When ALPHA finishes: review and merge skeleton to main
[ ] Announce to all agents: "main has the skeleton, start building"
[ ] Launch BRAVO, CHARLIE, DELTA with Sprint 1 tasks
```

---

## 7. Agent Workflow

Every agent follows the same cycle, regardless of role:

### Per-Feature Cycle

```
1. Sync       → /project:sync         # Get latest main
2. Build      → Implement the feature  # Write code in your owned files
3. Test       → /project:check         # Verify tests pass
4. Commit     → git add -A && git commit -m "[ROLE] feat: description"
5. Prepare    → /project:pr            # Validate ownership + push
6. Wait       → Orchestrator merges    # Move to next feature
```

### What Each Agent Says on Launch

When you start an agent with `claude --agent <role>`, give it a task prompt:

```
ALPHA (Sprint 0):
"Read specs/implementation_spec.md. Implement Sprint 0 deliverables:
 1. midi_vae/data/types.py — all frozen dataclasses
 2. midi_vae/registry.py — ComponentRegistry
 3. midi_vae/config.py — load_config with OmegaConf + Pydantic
 4. All ABC files (vae_wrapper.py, note_detection/base.py, metrics/base.py, sublatent/base.py, pipelines/base.py)
 5. configs/base.yaml
 Commit each deliverable separately. Run tests after each."

BRAVO (Sprint 1):
"Read specs/implementation_spec.md Section 5. Implement:
 1. midi_vae/data/preprocessing.py — MidiPreprocessor
 2. midi_vae/data/rendering.py — 3 ChannelStrategy implementations
 3. midi_vae/data/transforms.py — 4 transform classes
 Start with rendering.py since it has no external data dependency."

CHARLIE (Sprint 1):
"Read specs/implementation_spec.md Section 6. Implement:
 1. DiffusersVAE shared base class in vae_registry.py
 2. First 6 VAE wrappers: sd_v1_4, sd_vae_ft_mse, sdxl_vae, eq_vae_ema, eq_sdxl_vae, playground_v25
 All direct AutoencoderKL loads. Register each with ComponentRegistry."

DELTA (Sprint 1):
"Read specs/implementation_spec.md Sections 7-8. Implement P0 priority:
 1. GlobalThreshold in threshold.py
 2. OnsetF1, OnsetPrecision, OnsetRecall in reconstruction.py
 3. PixelMSE, SSIM, PSNR in reconstruction.py
 4. NoteDensityPearson in rhythm.py
 Each metric needs compute(gt, recon) → dict. Write tests for each."

ECHO (Sprint 0-1):
"Read specs/coordination.md. Set up:
 1. tests/conftest.py with all shared fixtures
 2. tests/stubs/ with stub implementations of every ABC
 3. tests/test_config.py, test_registry.py — test ALPHA's Sprint 0 work
 4. First integration test: test_ingest_render (can use stubs initially)"
```

---

## 8. Sprint 0 Playbook

Sprint 0 is the critical foundation phase. Here's the exact sequence:

### Hour 0-1: Setup
```bash
./scripts/setup_multiagent.sh ~/midi-image-vae
cd ~/midi-image-vae
make setup && source .venv/bin/activate
```

### Hour 1-4: ALPHA Builds Skeleton

Launch ALPHA and give the Sprint 0 prompt. ALPHA should deliver in this order:

1. **`midi_vae/data/types.py`** — All 5 dataclasses (BarData, PianoRollImage, LatentEncoding, ReconstructedBar, MidiNote). This unblocks everyone.

2. **`midi_vae/registry.py`** — ComponentRegistry with register/get/list. This unblocks BRAVO, CHARLIE, DELTA.

3. **`midi_vae/config.py`** — Pydantic schemas + OmegaConf loader. This unblocks experiment configs.

4. **All ABCs** — FrozenImageVAE, ChannelStrategy, NoteDetector, Metric, SubLatentModel, PipelineStage.

5. **`configs/base.yaml`** + **`scripts/run_experiment.py`** — Prove the config chain works end-to-end.

### Hour 1-4: ECHO Writes Test Stubs (Parallel)

While ALPHA works, ECHO can start immediately by reading the spec and writing test fixtures and stubs. ECHO works against the spec's type definitions, not ALPHA's actual code.

### Hour 4: Merge & Fan Out

```bash
# Orchestrator: merge ALPHA's skeleton to main
cd ~/midi-image-vae
git merge feature/alpha/sprint0
git push origin main

# Each agent syncs:
# (in their terminal) /project:sync
```

### Hour 4+: Sprint 1 Begins

All 4 feature agents start their Sprint 1 tasks in parallel. ECHO continues writing tests.

---

## 9. Merge & Integration Workflow

### How an Agent's Work Gets to Main

```
Agent's worktree (feature/bravo/sprint0)
    │
    ├── Agent commits: [BRAVO] feat: add ChannelStrategy impls
    ├── Agent runs: /project:pr  (validates + pushes)
    │
    ▼
Orchestrator (on main)
    │
    ├── Reviews: git log feature/bravo/sprint0 --oneline
    ├── Checks:  git diff main..feature/bravo/sprint0 --stat
    ├── Tests:   git merge --no-commit feature/bravo/sprint0 && pytest
    ├── Merges:  git merge --squash feature/bravo/sprint0
    ├── Commits: git commit -m "[BRAVO] feat: channel strategies + transforms"
    │
    ▼
Other agents sync: /project:sync (rebase onto updated main)
```

### Handling Merge Conflicts

Conflicts should be rare because agents own different files. If they occur:

1. The orchestrator identifies conflicting files
2. Determine which agent's version is correct (usually the file owner's)
3. Resolve on main, commit with `[ORCHESTRATOR] fix: resolve merge conflict in <file>`

### When to Merge

- Merge early and often. Don't let branches diverge.
- Merge after each logical deliverable (e.g., "all 3 channel strategies"), not after the entire sprint.
- Rule of thumb: if the branch has >5 commits, it should be merged.

---

## 10. Troubleshooting

### Agent Can't See Updated Main

```bash
# In the agent's worktree:
git fetch origin
git rebase origin/main
# Or use: /project:sync
```

### Agent Modified a Protected File

```bash
# Check what the agent changed:
git diff main -- midi_vae/data/types.py

# If it's a legitimate need, discuss and update on main
# If it's a mistake, reset:
git checkout main -- midi_vae/data/types.py
```

### Worktree in Bad State

```bash
# List worktrees
git worktree list

# Remove and recreate a broken worktree
git worktree remove .claude/worktrees/alpha --force
git worktree add .claude/worktrees/alpha -b feature/alpha/sprint0
```

### Agent Context Window Full

If an agent's context gets too large:
1. Have the agent commit its current work
2. Exit the session
3. Restart: `claude --agent <role> --resume` or start fresh
4. Point the agent at its current task: "Continue from the last commit. Run `git log --oneline -3` to see where you left off."

### Tests Fail After Merge

```bash
# On main, identify the failing test:
pytest tests/ -x --tb=long

# Determine which module is responsible from the traceback
# Assign the fix to the owning agent
```

### Two Agents Need the Same File Changed

This should not happen if ownership is respected. If it does:
1. One agent implements their change and merges first
2. The other agent syncs (`/project:sync`) and works on top of the merged version
3. Never have two agents editing the same file simultaneously
