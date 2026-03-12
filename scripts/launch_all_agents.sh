#!/usr/bin/env bash
# ============================================================================
# Launch all 5 agents in separate tmux panes
# ============================================================================
# Usage: ./scripts/launch_all_agents.sh
# Requires: tmux, claude CLI
#
# Creates a tmux session "midi-vae-agents" with 6 panes:
#   - Pane 0: Orchestrator (main branch)
#   - Pane 1-5: ALPHA through ECHO (each in their worktree)
# ============================================================================

set -euo pipefail

SESSION="midi-vae-agents"
REPO_ROOT="$(git rev-parse --show-toplevel)"
ROLES=("alpha" "bravo" "charlie" "delta" "echo")

# Check prerequisites
command -v tmux >/dev/null 2>&1 || { echo "tmux required. Install: sudo apt install tmux / brew install tmux"; exit 1; }
command -v claude >/dev/null 2>&1 || { echo "claude CLI required. Install: npm install -g @anthropic-ai/claude-code"; exit 1; }

# Kill existing session if present
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create session with orchestrator pane
tmux new-session -d -s "$SESSION" -n "orchestrator" -c "$REPO_ROOT"
tmux send-keys -t "$SESSION:orchestrator" "echo '🎯 ORCHESTRATOR — main branch'" Enter
tmux send-keys -t "$SESSION:orchestrator" "echo 'Run: claude'" Enter

# Create a pane for each agent
for role in "${ROLES[@]}"; do
    WORKTREE="$REPO_ROOT/.claude/worktrees/${role}"
    if [ ! -d "$WORKTREE" ]; then
        echo "Warning: worktree for $role not found at $WORKTREE. Run setup_multiagent.sh first."
        continue
    fi

    tmux new-window -t "$SESSION" -n "${role}" -c "$WORKTREE"
    tmux send-keys -t "$SESSION:${role}" "echo '🤖 ${role^^} agent — branch: $(cd "$WORKTREE" && git branch --show-current)'" Enter
    tmux send-keys -t "$SESSION:${role}" "echo 'Run: claude --agent ${role}'" Enter
done

# Select the orchestrator pane
tmux select-window -t "$SESSION:orchestrator"

echo "════════════════════════════════════════════════════════════"
echo " tmux session '$SESSION' created with 6 windows"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Attach: tmux attach -t $SESSION"
echo ""
echo "In each window, start the agent with:"
echo "  Orchestrator: claude"
echo "  Agents:       claude --agent <role>"
echo ""
echo "tmux shortcuts:"
echo "  Ctrl+B, n     — Next window"
echo "  Ctrl+B, p     — Previous window"
echo "  Ctrl+B, 0-5   — Jump to window N"
echo "  Ctrl+B, d     — Detach (agents keep running)"
echo "════════════════════════════════════════════════════════════"
