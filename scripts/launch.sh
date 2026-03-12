#!/usr/bin/env bash
# ============================================================================
# MIDI Image VAE — Launch Orchestrated Multi-Agent Session
# ============================================================================
#
# This script:
#   1. Opens a tmux session with split panes for monitoring
#   2. Starts the orchestrator agent (interactive — you talk to it)
#   3. The orchestrator spawns and manages teammates automatically
#   4. You watch teammate activity in monitoring panes
#   5. You can interject with the orchestrator at any time
#
# Usage:
#   cd ~/midi-image-vae
#   bash scripts/launch.sh [start|resume]
#
# Controls:
#   - Type in the orchestrator pane (left) to give instructions
#   - Watch agent logs in the right panes
#   - Ctrl+B, arrow keys to switch panes
#   - Ctrl+B, z to zoom into a pane (toggle)
#   - Ctrl+B, d to detach (everything keeps running)
#   - To reattach: tmux attach -t midi-vae
#
# To stop: type "pause" or "stop" to the orchestrator
# To resume later: bash scripts/launch.sh resume
# ============================================================================

set -euo pipefail

SESSION="midi-vae"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
MODE="${1:-start}"
LOG_DIR="$REPO_ROOT/outputs/agent-logs"
mkdir -p "$LOG_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[LAUNCH]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── Preflight ──────────────────────────────────────────────────────
command -v tmux >/dev/null 2>&1   || err "tmux required: brew install tmux / sudo apt install tmux"
command -v claude >/dev/null 2>&1 || err "claude required: npm install -g @anthropic-ai/claude-code"

if [ ! -f "$REPO_ROOT/CLAUDE.md" ]; then
    err "Not in the project root. cd to your midi-image-vae directory first."
fi

if [ ! -f "$REPO_ROOT/.claude/agents/orchestrator.md" ]; then
    err "Orchestrator agent not found. Run setup_multiagent.sh first."
fi

# ── Kill existing session if starting fresh ────────────────────────
if [ "$MODE" = "start" ]; then
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    log "Starting fresh session..."
elif [ "$MODE" = "resume" ]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        log "Reattaching to existing session..."
        exec tmux attach -t "$SESSION"
    else
        log "No existing session found. Starting fresh..."
    fi
fi

# ── Create tmux layout ────────────────────────────────────────────
# Layout:
# ┌─────────────────────┬──────────────────────┐
# │                     │    Agent Activity     │
# │   ORCHESTRATOR      │    (log tail)         │
# │   (interactive)     ├──────────────────────┤
# │                     │    Git Status         │
# │   You talk here.    │    (watch loop)       │
# │                     ├──────────────────────┤
# │                     │    Test Status        │
# │                     │    (on-demand)        │
# └─────────────────────┴──────────────────────┘

tmux new-session -d -s "$SESSION" -n "main" -c "$REPO_ROOT" -x 220 -y 55

# Split: left 60% = orchestrator, right 40% = monitoring
tmux split-window -h -t "$SESSION:main" -l "40%" -c "$REPO_ROOT"

# Split right pane into 3 vertical sections
tmux split-window -v -t "$SESSION:main.1" -l "66%" -c "$REPO_ROOT"
tmux split-window -v -t "$SESSION:main.2" -l "50%" -c "$REPO_ROOT"

# ── Pane 0 (left): Orchestrator agent ─────────────────────────────
# This is the interactive session the user controls
ORCHESTRATOR_CMD="claude --agent orchestrator"
if [ "$MODE" = "resume" ]; then
    ORCHESTRATOR_CMD="claude --agent orchestrator --continue"
fi

tmux send-keys -t "$SESSION:main.0" "clear" Enter
tmux send-keys -t "$SESSION:main.0" "echo ''" Enter
tmux send-keys -t "$SESSION:main.0" "echo '  ╔══════════════════════════════════════════════╗'" Enter
tmux send-keys -t "$SESSION:main.0" "echo '  ║  ORCHESTRATOR — Interactive Session          ║'" Enter
tmux send-keys -t "$SESSION:main.0" "echo '  ║                                              ║'" Enter
tmux send-keys -t "$SESSION:main.0" "echo '  ║  Commands:                                   ║'" Enter
tmux send-keys -t "$SESSION:main.0" "echo '  ║    \"start\"  — Begin Sprint 0                  ║'" Enter
tmux send-keys -t "$SESSION:main.0" "echo '  ║    \"status\" — Check all agent progress        ║'" Enter
tmux send-keys -t "$SESSION:main.0" "echo '  ║    \"pause\"  — Pause after current tasks       ║'" Enter
tmux send-keys -t "$SESSION:main.0" "echo '  ║    \"resume\" — Resume from where we left off   ║'" Enter
tmux send-keys -t "$SESSION:main.0" "echo '  ║                                              ║'" Enter
tmux send-keys -t "$SESSION:main.0" "echo '  ║  Ctrl+B, →  — Switch to monitoring panes     ║'" Enter
tmux send-keys -t "$SESSION:main.0" "echo '  ║  Ctrl+B, z  — Zoom/unzoom current pane       ║'" Enter
tmux send-keys -t "$SESSION:main.0" "echo '  ║  Ctrl+B, d  — Detach (keeps running)         ║'" Enter
tmux send-keys -t "$SESSION:main.0" "echo '  ╚══════════════════════════════════════════════╝'" Enter
tmux send-keys -t "$SESSION:main.0" "echo ''" Enter
tmux send-keys -t "$SESSION:main.0" "$ORCHESTRATOR_CMD" Enter

# ── Pane 1 (top-right): Agent activity log ────────────────────────
tmux send-keys -t "$SESSION:main.1" "echo '── Agent Activity (auto-refreshes) ──'" Enter
tmux send-keys -t "$SESSION:main.1" "watch -n 5 'echo \"=== Worktrees ===\"; git worktree list 2>/dev/null; echo \"\"; echo \"=== Recent commits (all branches) ===\"; git log --all --oneline --graph -15 2>/dev/null; echo \"\"; echo \"=== Active branches ===\"; git branch -a --sort=-committerdate 2>/dev/null | head -10'" Enter

# ── Pane 2 (mid-right): File change watcher ──────────────────────
tmux send-keys -t "$SESSION:main.2" "echo '── File Changes (auto-refreshes) ──'" Enter
tmux send-keys -t "$SESSION:main.2" "watch -n 10 'echo \"=== Files on main ===\"; git diff --stat HEAD~1 HEAD 2>/dev/null || echo \"(only initial commit)\"; echo \"\"; echo \"=== Python modules ===\"; find midi_vae -name \"*.py\" ! -name \"__init__.py\" -newer CLAUDE.md 2>/dev/null | head -20 || echo \"(none yet)\"; echo \"\"; echo \"=== Test count ===\"; find tests -name \"test_*.py\" 2>/dev/null | wc -l | xargs -I {} echo \"{} test files\"'" Enter

# ── Pane 3 (bottom-right): Test runner (on-demand) ───────────────
tmux send-keys -t "$SESSION:main.3" "echo '── Test Runner (manual) ──'" Enter
tmux send-keys -t "$SESSION:main.3" "echo 'Run tests:  python -m pytest tests/ -x --tb=short'" Enter
tmux send-keys -t "$SESSION:main.3" "echo 'Watch mode: watch -n 30 python -m pytest tests/ -x --tb=short -q 2>&1'" Enter
tmux send-keys -t "$SESSION:main.3" "echo ''" Enter

# ── Select orchestrator pane and attach ────────────────────────────
tmux select-pane -t "$SESSION:main.0"

echo ""
log "════════════════════════════════════════════════════════"
log " tmux session '$SESSION' ready"
log "════════════════════════════════════════════════════════"
log ""
log " Attaching now. The orchestrator is starting in the left pane."
log " Type 'start' to begin the project."
log ""
log " To detach: Ctrl+B, d"
log " To reattach: tmux attach -t $SESSION"
log " To resume later: bash scripts/launch.sh resume"
log ""

sleep 1
exec tmux attach -t "$SESSION"
