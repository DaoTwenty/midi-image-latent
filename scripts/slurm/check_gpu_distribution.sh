#!/bin/bash
# =============================================================================
# check_gpu_distribution.sh — Diagnostic: verify GPU/CPU binding before a run
# =============================================================================
#
# Run this script directly (NOT as a SLURM batch job) to inspect the GPU and
# CPU resources available and confirm that the per-process binding will work
# as expected before submitting a full experiment.
#
# USAGE
# -----
#   # On a login node (shows CPU info only — no GPU access):
#   bash scripts/slurm/check_gpu_distribution.sh
#
#   # Inside an interactive SLURM allocation with GPUs:
#   salloc --nodes=1 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=8 \
#          --account=def-pasquier --time=00:10:00
#   bash scripts/slurm/check_gpu_distribution.sh
#
#   # Or submit as a quick batch job for a non-interactive check:
#   sbatch --nodes=1 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=8 \
#          --account=def-pasquier --time=00:05:00 \
#          --output=/tmp/gpu_check_%j.log \
#          --wrap="bash /scratch/triana24/midi-image-latent/scripts/slurm/check_gpu_distribution.sh"
#
# WHAT IT CHECKS
# --------------
#   1. SLURM allocation info (if inside a SLURM job)
#   2. GPU count via nvidia-smi and PyTorch
#   3. CPU topology (nproc, lscpu)
#   4. Thread environment variables
#   5. Spawn N worker processes (one per GPU) — each prints GPU id + CPU affinity

set -euo pipefail

REPO=/scratch/triana24/midi-image-latent
PYTHON="$REPO/.venv/bin/python"

echo "============================================================"
echo "  GPU / CPU Distribution Diagnostic"
echo "  $(date)"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# 1. SLURM allocation info
# ---------------------------------------------------------------------------
echo "--- SLURM Allocation Info ---"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Running INSIDE a SLURM job"
    echo "  SLURM_JOB_ID:          ${SLURM_JOB_ID}"
    echo "  SLURM_JOB_NODELIST:    ${SLURM_JOB_NODELIST:-<unset>}"
    echo "  SLURM_NNODES:          ${SLURM_NNODES:-<unset>}"
    echo "  SLURM_NTASKS:          ${SLURM_NTASKS:-<unset>}"
    echo "  SLURM_NTASKS_PER_NODE: ${SLURM_NTASKS_PER_NODE:-<unset>}"
    echo "  SLURM_CPUS_PER_TASK:   ${SLURM_CPUS_PER_TASK:-<unset>}"
    echo "  SLURM_PROCID:          ${SLURM_PROCID:-<unset>}"
    echo "  SLURM_LOCALID:         ${SLURM_LOCALID:-<unset>}"
    echo "  CUDA_VISIBLE_DEVICES:  ${CUDA_VISIBLE_DEVICES:-<unset>}"
else
    echo "NOT inside a SLURM job (running on login or interactive node)"
fi
echo ""

# ---------------------------------------------------------------------------
# 2. GPU info
# ---------------------------------------------------------------------------
echo "--- GPU Info ---"
if command -v nvidia-smi &>/dev/null; then
    echo "nvidia-smi output:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free \
               --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' '{printf "  GPU %s: %s  total=%s MiB  free=%s MiB\n", $1, $2, $3, $4}' \
        || echo "  (nvidia-smi query failed — no GPUs visible)"
    echo ""
    echo "nvidia-smi summary:"
    nvidia-smi -L 2>/dev/null || echo "  (nvidia-smi -L failed)"
else
    echo "  nvidia-smi not found (no GPU driver / not on a GPU node)"
fi
echo ""

echo "PyTorch GPU count:"
if [[ -x "$PYTHON" ]]; then
    "$PYTHON" - <<'EOF'
import sys
try:
    import torch
    n = torch.cuda.device_count()
    print(f"  torch.cuda.device_count() = {n}")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1024**3
        print(f"  GPU {i}: {name}  ({mem_gb:.1f} GB)")
    if n == 0:
        print("  No CUDA GPUs visible (CPU-only mode or no GPUs allocated)")
except ImportError:
    print("  torch not importable — check venv")
    sys.exit(1)
EOF
else
    echo "  $PYTHON not found — cannot run PyTorch check"
fi
echo ""

# ---------------------------------------------------------------------------
# 3. CPU topology
# ---------------------------------------------------------------------------
echo "--- CPU Info ---"
echo "  nproc (logical CPUs visible to this process): $(nproc)"
echo ""

if command -v lscpu &>/dev/null; then
    echo "lscpu summary:"
    lscpu 2>/dev/null | grep -E "^(CPU\(s\)|Thread|Core|Socket|NUMA|Model name)" \
        | awk '{printf "  %s\n", $0}' || true
fi
echo ""

# ---------------------------------------------------------------------------
# 4. Thread environment variables
# ---------------------------------------------------------------------------
echo "--- Thread Environment Variables ---"
for var in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS; do
    val="${!var:-<unset>}"
    echo "  $var = $val"
done
echo ""

# ---------------------------------------------------------------------------
# 5. Spawn one process per GPU to verify binding
# ---------------------------------------------------------------------------
echo "--- Per-GPU Process Binding Test ---"

# Determine how many GPUs to test against
NUM_GPUS=0
if [[ -x "$PYTHON" ]]; then
    NUM_GPUS=$("$PYTHON" -c "
import sys
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
" 2>/dev/null) || NUM_GPUS=0
fi

if [[ "$NUM_GPUS" -eq 0 ]]; then
    echo "  No CUDA GPUs detected — skipping per-GPU binding test."
    echo "  (Run this script inside a SLURM GPU allocation to test binding.)"
else
    echo "  Spawning $NUM_GPUS worker processes (one per GPU)..."
    echo ""

    # Each worker is a Python one-liner that:
    #   - Prints its PID, GPU id, and CPU affinity
    #   - Does a small tensor operation on the assigned GPU to confirm access
    WORKER_SCRIPT=$(cat <<'PYEOF'
import os, sys
try:
    import torch
    gpu_id = int(os.environ.get("WORKER_GPU_ID", "0"))
    rank   = int(os.environ.get("WORKER_RANK", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    # CPU affinity
    try:
        import psutil
        aff = sorted(psutil.Process().cpu_affinity())
    except Exception:
        aff = "psutil not available"
    # Quick tensor smoke test
    if torch.cuda.is_available():
        x = torch.ones(64, 64, device="cuda:0")
        result = float(x.sum())
        smoke = f"tensor sum={result:.0f} OK"
    else:
        smoke = "CUDA not available"
    print(f"  rank={rank}  GPU={gpu_id}  name={gpu_name}  cpu_affinity={aff}  [{smoke}]")
except Exception as e:
    print(f"  rank={os.environ.get('WORKER_RANK','?')}  ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
)

    PIDS=()
    for GPU_ID in $(seq 0 $(( NUM_GPUS - 1 ))); do
        WORKER_GPU_ID=$GPU_ID WORKER_RANK=$GPU_ID \
            "$PYTHON" -c "$WORKER_SCRIPT" &
        PIDS+=($!)
    done

    # Wait for all workers and collect exit codes
    FAILURES=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            FAILURES=$(( FAILURES + 1 ))
        fi
    done

    echo ""
    if [[ "$FAILURES" -eq 0 ]]; then
        echo "  All $NUM_GPUS worker processes completed successfully."
    else
        echo "  WARNING: $FAILURES worker(s) failed — review output above."
    fi
fi

echo ""
echo "============================================================"
echo "  Diagnostic complete: $(date)"
echo "============================================================"
