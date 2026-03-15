#!/bin/bash
# Sync offline wandb runs to the cloud from a login node.
#
# Usage:
#   bash scripts/wandb_sync.sh                  # sync all offline runs
#   bash scripts/wandb_sync.sh <run_dir>        # sync a specific run
#   bash scripts/wandb_sync.sh --clean          # sync all, then remove synced dirs
#   bash scripts/wandb_sync.sh --list           # list offline runs without syncing
#
# This script is for when you run experiments with WANDB_MODE=offline
# (e.g., on compute nodes without internet). After the job finishes,
# run this from a login node to upload results to wandb cloud.
#
# Prerequisites:
#   - wandb installed: pip install wandb
#   - API key set in .env (WANDB_API_KEY=...) or via: wandb login
#
# Offline run data is stored in:
#   outputs/wandb/  (default WANDB_DIR from SLURM scripts)

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"

# Activate project venv
source "$REPO/.venv/bin/activate"

# Load .env for WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY
if [[ -f "$REPO/.env" ]]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"
        # Only set if not already in environment (env vars take precedence)
        if [[ -z "${!key:-}" ]]; then
            export "$key"="$value"
        fi
    done < <(grep -v '^#' "$REPO/.env" | grep '=')
fi

# Validate API key
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[wandb_sync] ERROR: WANDB_API_KEY not set."
    echo "  Set it in $REPO/.env or run: wandb login"
    exit 1
fi

# Ensure we're in online mode for syncing (override any leftover offline setting)
export WANDB_MODE=online

# Parse flags
CLEAN=false
LIST_ONLY=false
SPECIFIC_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=true
            shift
            ;;
        --list)
            LIST_ONLY=true
            shift
            ;;
        *)
            SPECIFIC_DIR="$1"
            shift
            ;;
    esac
done

# -----------------------------------------------------------------------
# Sync a specific run directory
# -----------------------------------------------------------------------
if [[ -n "$SPECIFIC_DIR" ]]; then
    if [[ ! -d "$SPECIFIC_DIR" ]]; then
        echo "[wandb_sync] ERROR: Directory not found: $SPECIFIC_DIR"
        exit 1
    fi
    echo "[wandb_sync] Syncing: $SPECIFIC_DIR"
    wandb sync "$SPECIFIC_DIR"
    if $CLEAN; then
        echo "[wandb_sync] Cleaning: $SPECIFIC_DIR"
        rm -rf "$SPECIFIC_DIR"
    fi
    exit 0
fi

# -----------------------------------------------------------------------
# Find all offline runs under outputs/ (and any subdirectories)
# -----------------------------------------------------------------------
SEARCH_DIRS=("$REPO/outputs")

echo "[wandb_sync] Scanning for offline runs..."
for dir in "${SEARCH_DIRS[@]}"; do
    echo "  -> $dir"
done

# Collect all offline run directories
OFFLINE_RUNS=()
for search_dir in "${SEARCH_DIRS[@]}"; do
    if [[ ! -d "$search_dir" ]]; then
        continue
    fi
    while IFS= read -r -d '' run_dir; do
        OFFLINE_RUNS+=("$run_dir")
    done < <(find "$search_dir" -path "*/wandb/offline-run-*" -type d -print0 2>/dev/null)
done

if [[ ${#OFFLINE_RUNS[@]} -eq 0 ]]; then
    echo "[wandb_sync] No offline runs found."
    echo "[wandb_sync] Tip: offline runs are created when WANDB_MODE=offline"
    exit 0
fi

# -----------------------------------------------------------------------
# List mode
# -----------------------------------------------------------------------
if $LIST_ONLY; then
    echo "[wandb_sync] Found ${#OFFLINE_RUNS[@]} offline run(s):"
    for run_dir in "${OFFLINE_RUNS[@]}"; do
        # Show size
        size=$(du -sh "$run_dir" 2>/dev/null | cut -f1)
        echo "  [$size] $run_dir"
    done
    exit 0
fi

# -----------------------------------------------------------------------
# Sync all
# -----------------------------------------------------------------------
echo "[wandb_sync] Found ${#OFFLINE_RUNS[@]} offline run(s). Syncing..."
echo ""

synced=0
failed=0
for run_dir in "${OFFLINE_RUNS[@]}"; do
    echo "[wandb_sync] Syncing: $run_dir"
    if wandb sync "$run_dir"; then
        synced=$((synced + 1))
        if $CLEAN; then
            echo "[wandb_sync] Cleaning: $run_dir"
            rm -rf "$run_dir"
        fi
    else
        echo "[wandb_sync] FAILED: $run_dir"
        failed=$((failed + 1))
    fi
    echo ""
done

echo "=========================================="
echo "[wandb_sync] Synced: $synced  Failed: $failed  Total: ${#OFFLINE_RUNS[@]}"
if $CLEAN && [[ $synced -gt 0 ]]; then
    echo "[wandb_sync] Cleaned $synced synced run directories."
fi
echo "=========================================="
