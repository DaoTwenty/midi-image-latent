#!/usr/bin/env bash
# ============================================================================
# MIDI Image VAE — Multi-Agent Claude Code Setup
# ============================================================================
# Run from inside the extracted midi-vae-multiagent-setup/ directory.
#
# Usage:
#   cd midi-vae-multiagent-setup
#   chmod +x scripts/setup_multiagent.sh
#   ./scripts/setup_multiagent.sh [project_dir]
#
# Default project_dir: ~/midi-image-vae
# ============================================================================

set -euo pipefail

PROJECT_DIR="${1:-$HOME/midi-image-vae}"
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROLES=(alpha bravo charlie delta echo)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'
ROLE_COLORS=("$PURPLE" "$GREEN" "$YELLOW" "$BLUE" "$RED")

log()  { echo -e "${GREEN}[SETUP]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── Preflight ──────────────────────────────────────────────────────
log "Preflight checks..."
command -v git >/dev/null 2>&1    || err "git is required."
command -v python3 >/dev/null 2>&1 || err "python3 is required."
command -v claude >/dev/null 2>&1  || warn "claude CLI not found. Install: npm install -g @anthropic-ai/claude-code"

# Verify we're running from the setup package
if [ ! -f "$SETUP_DIR/CLAUDE.md" ]; then
    err "Cannot find CLAUDE.md. Run this script from inside the extracted setup directory."
fi
if [ ! -d "$SETUP_DIR/claude-config/agents" ]; then
    err "Cannot find claude-config/agents/. The setup package is incomplete."
fi

AGENT_COUNT=$(ls "$SETUP_DIR/claude-config/agents/"*.md 2>/dev/null | wc -l)
CMD_COUNT=$(ls "$SETUP_DIR/claude-config/commands/"*.md 2>/dev/null | wc -l)
log "Found $AGENT_COUNT agent definitions, $CMD_COUNT commands."

if [ "$AGENT_COUNT" -lt 5 ]; then
    err "Expected 5 agent files, found $AGENT_COUNT. Package is corrupted."
fi

# ── Step 1: Create or reuse the project repo ──────────────────────
if [ -d "$PROJECT_DIR/.git" ]; then
    warn "Repository $PROJECT_DIR already exists. Updating configs."
    cd "$PROJECT_DIR"
else
    log "Creating project: $PROJECT_DIR"
    mkdir -p "$PROJECT_DIR"
    cd "$PROJECT_DIR"
    git init
    log "Git repository initialized."
fi

REPO_ROOT=$(pwd)
log "Working in: $REPO_ROOT"

# ── Step 2: Directory skeleton ─────────────────────────────────────
log "Creating directory skeleton..."

mkdir -p midi_vae/data
mkdir -p midi_vae/models/sublatent
mkdir -p midi_vae/models/sequence
mkdir -p midi_vae/note_detection
mkdir -p midi_vae/metrics
mkdir -p midi_vae/pipelines
mkdir -p midi_vae/tracking
mkdir -p midi_vae/visualization
mkdir -p midi_vae/utils
mkdir -p configs/data
mkdir -p configs/vae
mkdir -p configs/experiments
mkdir -p configs/note_detection
mkdir -p tests/stubs
mkdir -p scripts
mkdir -p specs
mkdir -p outputs/experiments
mkdir -p outputs/cache

for d in $(find midi_vae -type d); do
    touch "$d/__init__.py"
done

log "Skeleton created."

# ── Step 3: Install Claude Code config (dotfiles) ─────────────────
# This is the critical step: we read from claude-config/ (non-dot)
# and write to .claude/ (dotdir) in the project.
log "Installing Claude Code configuration into .claude/ ..."

mkdir -p .claude/agents
mkdir -p .claude/commands

for f in "$SETUP_DIR/claude-config/agents/"*.md; do
    cp "$f" ".claude/agents/$(basename "$f")"
done
log "  Agents:   $(ls .claude/agents/*.md | wc -l) files → .claude/agents/"

for f in "$SETUP_DIR/claude-config/commands/"*.md; do
    cp "$f" ".claude/commands/$(basename "$f")"
done
log "  Commands: $(ls .claude/commands/*.md | wc -l) files → .claude/commands/"

cp "$SETUP_DIR/claude-config/settings.json" ".claude/settings.json"
log "  Settings: .claude/settings.json"

cp "$SETUP_DIR/CLAUDE.md" "./CLAUDE.md"
log "  Memory:   CLAUDE.md"

cp "$SETUP_DIR/gitignore.txt" "./.gitignore"
log "  Ignore:   .gitignore"

cp "$SETUP_DIR/specs/"*.md ./specs/
log "  Specs:    $(ls specs/*.md | wc -l) files → specs/"

cp "$SETUP_DIR/SETUP_GUIDE.md" "./SETUP_GUIDE.md"

# ── Step 4: Verify ─────────────────────────────────────────────────
log "Verifying installation..."
ERRORS=0

check_exists() {
    if [ ! -e "$1" ]; then
        echo -e "  ${RED}MISSING${NC}: $1"
        ERRORS=$((ERRORS + 1))
    else
        echo -e "  ${GREEN}OK${NC}:      $1"
    fi
}

check_exists "CLAUDE.md"
check_exists ".claude/settings.json"
check_exists ".claude/agents/alpha.md"
check_exists ".claude/agents/bravo.md"
check_exists ".claude/agents/charlie.md"
check_exists ".claude/agents/delta.md"
check_exists ".claude/agents/echo.md"
check_exists ".claude/commands/sync.md"
check_exists ".claude/commands/pr.md"
check_exists ".claude/commands/check.md"
check_exists ".claude/commands/status.md"
check_exists ".claude/commands/kickoff.md"
check_exists "specs/implementation_spec.md"
check_exists "specs/coordination.md"
check_exists ".gitignore"

if [ "$ERRORS" -gt 0 ]; then
    err "$ERRORS files missing."
fi
log "All 15 critical files verified."

# ── Step 5: pyproject.toml ─────────────────────────────────────────
log "Creating pyproject.toml..."

cat > pyproject.toml << 'PYPROJECT'
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "midi-image-vae"
version = "0.1.0"
description = "Pretrained Image VAEs for MIDI Encoding via Piano-Roll Representations"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "diffusers>=0.28",
    "transformers>=4.40",
    "accelerate>=0.30",
    "safetensors",
    "pypianoroll>=1.0",
    "pretty_midi>=0.2.10",
    "mido",
    "numpy>=1.26",
    "scipy>=1.12",
    "pandas>=2.2",
    "scikit-learn>=1.4",
    "omegaconf>=2.3",
    "pydantic>=2.6",
    "structlog>=24.1",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "h5py>=3.10",
    "pyarrow>=15.0",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.4"]
extra = ["wandb>=0.16", "umap-learn>=0.5", "hmmlearn>=0.3"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-x --tb=short -q"

[tool.ruff]
line-length = 120
target-version = "py311"

[tool.setuptools.packages.find]
include = ["midi_vae*"]
PYPROJECT

# ── Step 6: Makefile ───────────────────────────────────────────────
log "Creating Makefile..."

cat > Makefile << 'MAKEFILE'
.PHONY: setup test test-cov lint preprocess run clean status

setup:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev,extra]"
	@echo "Activate: source .venv/bin/activate"

test:
	python -m pytest tests/ -x --tb=short -q

test-cov:
	python -m pytest tests/ --cov=midi_vae --cov-report=html --cov-report=term

lint:
	ruff check midi_vae/ tests/

status:
	@echo "=== Worktrees ===" && git worktree list
	@echo "" && echo "=== Branches ===" && git branch -a --sort=-committerdate | head -15
	@echo "" && echo "=== Main log ===" && git log --oneline -5 main 2>/dev/null || echo "(empty)"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage dist build *.egg-info
MAKEFILE

# ── Step 7: Copy launch script ─────────────────────────────────────
# Copy all scripts
for script_file in "$SETUP_DIR/scripts/"*.sh; do
    if [ -f "$script_file" ]; then
        cp "$script_file" "scripts/$(basename $script_file)"
        chmod +x "scripts/$(basename $script_file)"
    fi
done

if false; then
    cp "$SETUP_DIR/scripts/launch_all_agents.sh" scripts/launch_all_agents.sh
    chmod +x scripts/launch_all_agents.sh
fi

# ── Step 8: Initial commit ─────────────────────────────────────────
log "Creating initial commit..."

git add -A
git commit -m "[SETUP] Initialize multi-agent project

- .claude/agents/ (5 roles: alpha, bravo, charlie, delta, echo)
- .claude/commands/ (sync, pr, check, status, kickoff)
- .claude/settings.json
- CLAUDE.md, specs/, pyproject.toml, Makefile"

MAIN_BRANCH=$(git branch --show-current)

# ── Step 9: Create worktrees ──────────────────────────────────────
log "Creating agent worktrees..."
echo ""

for i in "${!ROLES[@]}"; do
    role="${ROLES[$i]}"
    color="${ROLE_COLORS[$i]}"
    branch="feature/${role}/sprint0"
    wt=".claude/worktrees/${role}"

    if git worktree list | grep -q "worktrees/${role}"; then
        warn "Worktree $role exists."
    else
        git worktree add "$wt" -b "$branch"
        echo -e "  ${color}✓ ${role^^}${NC}  →  $wt"
    fi
done

# ── Done ───────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo -e " ${GREEN}SETUP COMPLETE${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Project: $REPO_ROOT"
echo ""
echo "Verified contents:"
echo "  .claude/agents/:   $(ls .claude/agents/ | tr '\n' ' ')"
echo "  .claude/commands/: $(ls .claude/commands/ | tr '\n' ' ')"
echo ""
git worktree list
echo ""
echo "NEXT:"
echo "  1.  cd $REPO_ROOT && make setup && source .venv/bin/activate"
echo "  2.  claude                                    # orchestrator"
echo "  3.  cd .claude/worktrees/alpha && claude --agent alpha"
echo "  4.  cd .claude/worktrees/bravo && claude --agent bravo"
echo "  5.  (same for charlie, delta, echo)"
echo "════════════════════════════════════════════════════════════════"
