#!/usr/bin/env bash
# ============================================================================
# MIDI Image VAE — Launch Orchestrated Session
# ============================================================================
#
# Opens a tmux session: orchestrator on the left, monitor on the right.
# You talk to the orchestrator. It spawns and manages agent teammates.
#
# Usage:   cd ~/midi-image-vae && bash scripts/launch.sh
# Resume:  tmux attach -t midi-vae
# Stop:    tell the orchestrator "pause", then Ctrl+B, d to detach
#
# Inside tmux:
#   Ctrl+B, ←/→    Switch panes
#   Ctrl+B, z       Zoom/unzoom current pane
#   Ctrl+B, d       Detach (keeps running in background)
# ============================================================================

set -euo pipefail

SESSION="midi-vae"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# ── Preflight ──────────────────────────────────────────────────────
command -v tmux >/dev/null 2>&1 || { echo "ERROR: tmux required (brew install tmux)"; exit 1; }
command -v claude >/dev/null 2>&1 || { echo "ERROR: claude required (npm install -g @anthropic-ai/claude-code)"; exit 1; }
[ -f "$REPO_ROOT/CLAUDE.md" ] || { echo "ERROR: Run from the project root (where CLAUDE.md is)"; exit 1; }
[ -f "$REPO_ROOT/.claude/agents/orchestrator.md" ] || { echo "ERROR: orchestrator.md missing. Run setup_multiagent.sh first."; exit 1; }

# ── Kill old session if exists ─────────────────────────────────────
tmux kill-session -t "$SESSION" 2>/dev/null || true

# ── Write helper scripts to disk (avoids send-keys problems) ──────
mkdir -p "$REPO_ROOT/outputs"

# Monitor script (replaces `watch` which doesn't exist on macOS)
cat > "$REPO_ROOT/outputs/.monitor.sh" << 'MONITOR'
#!/usr/bin/env bash
cd "$(git rev-parse --show-toplevel)"
while true; do
    clear
    printf '\033[1;36m═══ AGENT MONITOR ═══\033[0m  %s\n\n' "$(date +%H:%M:%S)"

    printf '\033[1;33m── Worktrees ──\033[0m\n'
    git worktree list 2>/dev/null || echo "(none)"
    echo ""

    printf '\033[1;33m── Recent Commits ──\033[0m\n'
    git log --all --oneline --graph --decorate -15 2>/dev/null || echo "(no commits)"
    echo ""

    printf '\033[1;33m── Modules Created ──\033[0m\n'
    find midi_vae -name "*.py" ! -name "__init__.py" ! -empty 2>/dev/null | sort | head -15 || echo "(none)"
    echo ""

    printf '\033[1;33m── Tests ──\033[0m\n'
    count=$(find tests -name "test_*.py" 2>/dev/null | wc -l | tr -d ' ')
    echo "$count test files"
    echo ""

    printf '\033[0;90mRefreshes every 1s\033[0m\n'
    sleep 1
done
MONITOR
chmod +x "$REPO_ROOT/outputs/.monitor.sh"

# Orchestrator launcher (shows banner, then starts claude)
cat > "$REPO_ROOT/outputs/.start_orchestrator.sh" << 'ORCH'
#!/usr/bin/env bash
cd "$(git rev-parse --show-toplevel)"
clear
printf '\n'
printf '  \033[1;36m╔══════════════════════════════════════════════════╗\033[0m\n'
printf '  \033[1;36m║\033[0m  MIDI Image VAE \033[1m— Orchestrator\033[0m                   \033[1;36m║\033[0m\n'
printf '  \033[1;36m╠══════════════════════════════════════════════════╣\033[0m\n'
printf '  \033[1;36m║\033[0m                                                  \033[1;36m║\033[0m\n'
printf '  \033[1;36m║\033[0m  Say \033[1;32m"start"\033[0m to begin building the project.      \033[1;36m║\033[0m\n'
printf '  \033[1;36m║\033[0m  Say \033[1;33m"status"\033[0m to check progress.                 \033[1;36m║\033[0m\n'
printf '  \033[1;36m║\033[0m  Say \033[1;31m"pause"\033[0m to stop gracefully.                 \033[1;36m║\033[0m\n'
printf '  \033[1;36m║\033[0m                                                  \033[1;36m║\033[0m\n'
printf '  \033[1;36m║\033[0m  Ctrl+B, →  \033[0;90mswitch to monitor pane\033[0m               \033[1;36m║\033[0m\n'
printf '  \033[1;36m║\033[0m  Ctrl+B, z  \033[0;90mzoom/unzoom pane\033[0m                     \033[1;36m║\033[0m\n'
printf '  \033[1;36m║\033[0m  Ctrl+B, d  \033[0;90mdetach (keeps running)\033[0m               \033[1;36m║\033[0m\n'
printf '  \033[1;36m║\033[0m                                                  \033[1;36m║\033[0m\n'
printf '  \033[1;36m╚══════════════════════════════════════════════════╝\033[0m\n'
printf '\n'

# Start the orchestrator agent on the main branch
exec claude --agent orchestrator
ORCH
chmod +x "$REPO_ROOT/outputs/.start_orchestrator.sh"

# ── Create tmux session ────────────────────────────────────────────
# Left pane (70%): orchestrator   |   Right pane (30%): monitor

tmux new-session -d -s "$SESSION" -n "main" -c "$REPO_ROOT"
tmux split-window -h -t "$SESSION:main" -p 30 -c "$REPO_ROOT"

# Right pane: run monitor
tmux send-keys -t "$SESSION:main.1" "bash outputs/.monitor.sh" Enter

# Left pane: run orchestrator launcher
tmux send-keys -t "$SESSION:main.0" "bash outputs/.start_orchestrator.sh" Enter

# Focus orchestrator pane
tmux select-pane -t "$SESSION:main.0"

# ── Attach ─────────────────────────────────────────────────────────
echo ""
echo "  Launching..."
echo "  Left pane:  Orchestrator (you talk here)"
echo "  Right pane: Agent monitor (auto-refreshes)"
echo ""
echo "  Detach: Ctrl+B, d  |  Reattach: tmux attach -t $SESSION"
echo ""

exec tmux attach -t "$SESSION"