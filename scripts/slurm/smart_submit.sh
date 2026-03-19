#!/bin/bash
# =============================================================================
# smart_submit.sh — Probe GPU availability and submit the optimal job
# =============================================================================
#
# Queries sinfo/scontrol to find the best available GPU configuration on Fir,
# then submits exp_1a with the right partition, GPU type, and node/task layout.
#
# USAGE
#   bash scripts/slurm/smart_submit.sh                # auto-detect best config
#   bash scripts/slurm/smart_submit.sh --dry-run      # show what would be submitted
#   bash scripts/slurm/smart_submit.sh --mini         # pass --mini to the experiment
#   bash scripts/slurm/smart_submit.sh --prefer mig   # prefer MIG slices
#   bash scripts/slurm/smart_submit.sh --prefer full  # prefer full H100s
#   bash scripts/slurm/smart_submit.sh --max-gpus 4   # cap GPU count
#
# STRATEGY
#   1. Check full H100 nodes (bynode partitions) for free GPUs
#   2. Check MIG 3g.40gb slices (bygpu partitions) for free slices
#   3. Pick the config that gives the most GPUs with shortest wait
#   4. Fall back to single-GPU if nothing multi-GPU is available
#   5. Submit with the right partition, time limit, and GRES string
#
# GPU TYPES ON FIR
#   gpu:h100                                  — Full H100 80GB (bynode partitions)
#   gpu:nvidia_h100_80gb_hbm3_3g.40gb        — MIG 3g.40GB slice (bygpu partitions)
#   gpu:nvidia_h100_80gb_hbm3_2g.20gb        — MIG 2g.20GB slice (bygpu partitions)
#   gpu:nvidia_h100_80gb_hbm3_1g.10gb        — MIG 1g.10GB slice (bygpu partitions)
#
# For VAE encode/decode, 3g.40GB MIG slices are sufficient (largest models need ~10GB).
# Full H100s are overkill but schedule faster when bynode partitions have capacity.

set -euo pipefail

REPO=/scratch/triana24/midi-image-latent
CONFIG="$REPO/configs/experiments/exp_1a_vae_comparison.yaml"
ACCOUNT="def-pasquier_gpu"
PYTHON="$REPO/.venv/bin/python"

# ── Defaults ──────────────────────────────────────────────────────────────────
DRY_RUN=false
PREFER=""          # "", "full", or "mig"
MAX_GPUS=4         # max GPUs to request (more = faster but harder to schedule)
EXTRA_ARGS=""      # forwarded to run_experiment_distributed.py
NUM_CONDITIONS=36  # 12 VAEs x 3 channel strategies

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --prefer)    PREFER="$2"; shift 2 ;;
        --max-gpus)  MAX_GPUS="$2"; shift 2 ;;
        --config)    CONFIG="$2"; shift 2 ;;
        *)           EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# ── Helper: probe free GPUs of a given type (single scan) ────────────────────
# Sets global variables: _PROBE_TOTAL_FREE, _PROBE_BEST_FREE, _PROBE_BEST_NODE,
# _PROBE_NODES_WITH_FREE
probe_free_gpus() {
    local partition="$1"
    local gres_type="$2"   # e.g. "h100" or "nvidia_h100_80gb_hbm3_3g.40gb"
    local gres_key="gres/gpu:${gres_type}"

    _PROBE_TOTAL_FREE=0
    _PROBE_BEST_FREE=0
    _PROBE_BEST_NODE=""
    _PROBE_NODES_WITH_FREE=""

    # Get all nodes in the partition that are not down/drained
    while read -r node; do
        [[ -z "$node" ]] && continue
        local info
        info=$(scontrol show node "$node" 2>/dev/null)

        local cfg_gpus alloc_gpus
        cfg_gpus=$(echo "$info" | grep -oP "CfgTRES=.*${gres_key}=\K[0-9]+" 2>/dev/null || echo "0")
        alloc_gpus=$(echo "$info" | grep -oP "AllocTRES=.*${gres_key}=\K[0-9]+" 2>/dev/null || echo "0")

        local free=$(( cfg_gpus - alloc_gpus ))
        if [[ $free -gt 0 ]]; then
            _PROBE_TOTAL_FREE=$(( _PROBE_TOTAL_FREE + free ))
            _PROBE_NODES_WITH_FREE="$_PROBE_NODES_WITH_FREE $node($free)"
            if [[ $free -gt $_PROBE_BEST_FREE ]]; then
                _PROBE_BEST_FREE=$free
                _PROBE_BEST_NODE=$node
            fi
        fi
    done < <(sinfo -p "$partition" -N --noheader -o "%n %t" 2>/dev/null \
             | grep -vE "drain|down|comp|inval" \
             | awk '{print $1}' | sort -u)
}

# ── Helper: choose time limit based on estimated runtime ─────────────────────
choose_time_limit() {
    local num_gpus="$1"
    # 36 conditions, ~5min each on H100, plus overhead
    local conditions_per_gpu=$(( (NUM_CONDITIONS + num_gpus - 1) / num_gpus ))
    local est_minutes=$(( conditions_per_gpu * 6 + 15 ))  # 6min/condition + 15min overhead
    local est_hours=$(( (est_minutes + 59) / 60 ))

    # Round up to nice boundaries and add safety margin
    if [[ $est_hours -le 3 ]]; then
        echo "03:00:00"
    elif [[ $est_hours -le 6 ]]; then
        echo "06:00:00"
    elif [[ $est_hours -le 12 ]]; then
        echo "12:00:00"
    else
        echo "1-00:00:00"
    fi
}

# NOTE: We omit --partition from sbatch and let the scheduler auto-select
# based on GRES type + time limit. This avoids account/partition mismatches.

# =============================================================================
# MAIN: Probe GPU availability and decide
# =============================================================================

echo "============================================================"
echo "  smart_submit.sh — Exp 1A GPU availability probe"
echo "  $(date)"
echo "============================================================"
echo ""

# ── Probe full H100 nodes (bynode) ──────────────────────────────────────────
echo "Probing full H100 availability (bynode partitions)..."
probe_free_gpus "gpubase_bynode_b1" "h100"
FULL_FREE=$_PROBE_TOTAL_FREE
FULL_BEST_FREE=$_PROBE_BEST_FREE
FULL_BEST_NODE=$_PROBE_BEST_NODE
FULL_NODES=$_PROBE_NODES_WITH_FREE

echo "  Total free full H100s: $FULL_FREE"
if [[ $FULL_FREE -gt 0 ]]; then
    echo "  Best single node: $FULL_BEST_NODE ($FULL_BEST_FREE free)"
    echo "  Nodes with free GPUs:$FULL_NODES"
fi
echo ""

# ── Probe MIG 3g.40gb slices (bygpu) ────────────────────────────────────────
echo "Probing MIG 3g.40gb availability (bygpu partitions)..."
probe_free_gpus "gpubase_bygpu_b1" "nvidia_h100_80gb_hbm3_3g.40gb"
MIG_FREE=$_PROBE_TOTAL_FREE
MIG_BEST_FREE=$_PROBE_BEST_FREE
MIG_BEST_NODE=$_PROBE_BEST_NODE
MIG_NODES=$_PROBE_NODES_WITH_FREE

echo "  Total free MIG 3g.40gb: $MIG_FREE"
if [[ $MIG_FREE -gt 0 ]]; then
    echo "  Best single node: $MIG_BEST_NODE ($MIG_BEST_FREE free)"
    echo "  Nodes with free slices:$MIG_NODES"
fi
echo ""

# ── Decision logic ──────────────────────────────────────────────────────────
# Priority:
#   1. If --prefer is set, honour it
#   2. Multi-GPU on single node (avoids NFS coordination overhead)
#   3. Full H100 > MIG (full VRAM, no MIG overhead)
#   4. More GPUs > fewer (faster sweep)
#   5. If no free GPUs, submit single-GPU to shortest queue

GPU_TYPE=""        # "full" or "mig"
GPU_GRES=""        # SLURM --gres string
NUM_GPUS=1
NUM_NODES=1
TASKS_PER_NODE=1
CPUS_PER_TASK=8
MEM_PER_CPU="4G"
MODE=""            # "single", "distributed"

decide() {
    local full_avail=$FULL_BEST_FREE
    local mig_avail=$MIG_BEST_FREE

    # Cap to MAX_GPUS
    [[ $full_avail -gt $MAX_GPUS ]] && full_avail=$MAX_GPUS
    [[ $mig_avail -gt $MAX_GPUS ]] && mig_avail=$MAX_GPUS

    # Apply preference
    if [[ "$PREFER" == "full" ]]; then
        mig_avail=0
    elif [[ "$PREFER" == "mig" ]]; then
        full_avail=0
    fi

    if [[ $full_avail -ge 2 ]]; then
        GPU_TYPE="full"
        NUM_GPUS=$full_avail
        NUM_NODES=1
        TASKS_PER_NODE=$NUM_GPUS
        GPU_GRES="gpu:h100:${NUM_GPUS}"
        MODE="distributed"
        echo "DECISION: ${NUM_GPUS}x full H100 on 1 node (best available)"

    elif [[ $mig_avail -ge 2 ]]; then
        GPU_TYPE="mig"
        NUM_GPUS=$mig_avail
        NUM_NODES=1
        TASKS_PER_NODE=$NUM_GPUS
        GPU_GRES="gpu:nvidia_h100_80gb_hbm3_3g.40gb:${NUM_GPUS}"
        MODE="distributed"
        echo "DECISION: ${NUM_GPUS}x MIG 3g.40gb on 1 node (best available)"

    elif [[ $full_avail -ge 1 ]]; then
        GPU_TYPE="full"
        NUM_GPUS=1
        NUM_NODES=1
        TASKS_PER_NODE=1
        GPU_GRES="gpu:h100:1"
        MODE="single"
        echo "DECISION: 1x full H100 (only 1 free, running sequentially)"

    elif [[ $mig_avail -ge 1 ]]; then
        GPU_TYPE="mig"
        NUM_GPUS=1
        NUM_NODES=1
        TASKS_PER_NODE=1
        GPU_GRES="gpu:nvidia_h100_80gb_hbm3_3g.40gb:1"
        MODE="single"
        echo "DECISION: 1x MIG 3g.40gb (only 1 free, running sequentially)"

    else
        # Nothing free right now — submit single GPU to b1 (shortest queue)
        GPU_TYPE="full"
        NUM_GPUS=1
        NUM_NODES=1
        TASKS_PER_NODE=1
        GPU_GRES="gpu:h100:1"
        MODE="single"
        echo "DECISION: No free GPUs detected — submitting 1x H100 to queue"
        echo "          (scheduler will start it when a GPU becomes available)"
    fi

    # Choose time limit
    local time_limit
    time_limit=$(choose_time_limit $NUM_GPUS)

    # Override: for single GPU with 36 conditions, we need more time
    if [[ "$MODE" == "single" ]]; then
        time_limit="12:00:00"
    fi

    TIME_LIMIT="$time_limit"
}

decide
echo ""

# ── Build sbatch command ─────────────────────────────────────────────────────
EXP_NAME=$(basename "$CONFIG" .yaml)
OUTPUT_ROOT="$SCRATCH/midi-image-latent/outputs/${EXP_NAME}_distributed"

echo "------------------------------------------------------------"
echo "  Configuration:"
echo "    GPU type:     $GPU_TYPE ($GPU_GRES)"
echo "    GPUs:         $NUM_GPUS"
echo "    Nodes:        $NUM_NODES"
echo "    Tasks/node:   $TASKS_PER_NODE"
echo "    Time limit:   $TIME_LIMIT"
echo "    CPUs/task:    $CPUS_PER_TASK"
echo "    Mem/CPU:      $MEM_PER_CPU"
echo "    Mode:         $MODE"
echo "    Config:       $CONFIG"
echo "    Output:       $OUTPUT_ROOT"
echo "    Extra args:   ${EXTRA_ARGS:-<none>}"
echo "------------------------------------------------------------"
echo ""

mkdir -p "$SCRATCH/midi-image-latent/outputs/logs"

if [[ "$MODE" == "distributed" ]]; then
    # Multi-GPU: use srun + run_experiment_distributed.py
    SBATCH_CMD=(
        sbatch
        --job-name="exp1a-smart"
        --account="$ACCOUNT"

        --nodes="$NUM_NODES"
        --ntasks-per-node="$TASKS_PER_NODE"
        --gpus-per-task=1
        --cpus-per-task="$CPUS_PER_TASK"
        --mem-per-cpu="$MEM_PER_CPU"
        --time="$TIME_LIMIT"
        --output="$SCRATCH/midi-image-latent/outputs/logs/exp1a_smart_%j.log"
        --error="$SCRATCH/midi-image-latent/outputs/logs/exp1a_smart_%j.log"
    )

    # Build the inline script
    JOB_SCRIPT=$(cat <<'JOBEOF'
#!/bin/bash
set -euo pipefail

REPO=/scratch/triana24/midi-image-latent
PYTHON="$REPO/.venv/bin/python"
CONFIG="__CONFIG__"
OUTPUT_ROOT="__OUTPUT_ROOT__"

echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Head node:    $SLURMD_NODENAME"
echo "Nodes:        $SLURM_NNODES"
echo "Tasks total:  $SLURM_NTASKS (${SLURM_NTASKS_PER_NODE:-?} per node)"
echo "CPUs/task:    ${SLURM_CPUS_PER_TASK:-8}"
echo "GPU GRES:     __GPU_GRES__"
echo "Start:        $(date)"
echo "=========================================="

module purge
module load python/3.11 gcc/12 cuda/12.6 arrow/23.0.1

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export HF_HOME="$SCRATCH/.cache/huggingface"
export WANDB_MODE=offline
export WANDB_DIR="$SCRATCH/midi-image-latent/outputs"

if [[ -f "$REPO/.env" ]]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"
        [[ -n "$key" ]] && export "$key"="$value"
    done < <(grep -v '^#' "$REPO/.env" | grep '=')
fi

source "$REPO/.venv/bin/activate"
cd "$REPO"

echo "Python:     $(which python)"
echo "HF_HOME:    $HF_HOME"
echo ""

mkdir -p "$OUTPUT_ROOT/logs"

T_START=$(date +%s)

# Prevent SLURM_MEM_PER_CPU/GPU/NODE mutual exclusion error in srun.
# sbatch sets SLURM_MEM_PER_CPU; srun with --gpus-per-task can conflict.
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

srun \
    --output="$OUTPUT_ROOT/logs/rank_%t.log" \
    --error="$OUTPUT_ROOT/logs/rank_%t.log" \
    "$PYTHON" "$REPO/scripts/run_experiment_distributed.py" \
        "$CONFIG" \
        --output-root "$OUTPUT_ROOT" \
        --sweep-strategies \
        --multi-node \
        --assign-strategy cost-balanced \
        --log-level INFO \
        __EXTRA_ARGS__

T_SRUN=$(date +%s)
echo ""
echo "All ranks complete in $(( T_SRUN - T_START ))s."

echo ""
echo "Merging per-rank results..."
"$PYTHON" "$REPO/scripts/merge_sweep_results.py" \
    "$OUTPUT_ROOT" \
    --scan-rank-dirs \
    --verbose \
    --output "$OUTPUT_ROOT/sweep_summary.json"

T_END=$(date +%s)
echo ""
echo "=========================================="
echo "Done:       $(date)"
echo "Elapsed:    $(( T_END - T_START ))s"
echo "Summary:    $OUTPUT_ROOT/sweep_summary.json"
echo "Rank logs:  $OUTPUT_ROOT/logs/"
echo "=========================================="
JOBEOF
)

    # Substitute placeholders
    JOB_SCRIPT="${JOB_SCRIPT//__CONFIG__/$CONFIG}"
    JOB_SCRIPT="${JOB_SCRIPT//__OUTPUT_ROOT__/$OUTPUT_ROOT}"
    JOB_SCRIPT="${JOB_SCRIPT//__GPU_GRES__/$GPU_GRES}"
    JOB_SCRIPT="${JOB_SCRIPT//__EXTRA_ARGS__/$EXTRA_ARGS}"

else
    # Single GPU: use run_experiment.py directly
    SBATCH_CMD=(
        sbatch
        --job-name="exp1a-smart"
        --account="$ACCOUNT"

        --nodes=1
        --gpus-per-node="$GPU_GRES"
        --cpus-per-task="$CPUS_PER_TASK"
        --mem="64G"
        --time="$TIME_LIMIT"
        --output="$SCRATCH/midi-image-latent/outputs/logs/exp1a_smart_%j.log"
        --error="$SCRATCH/midi-image-latent/outputs/logs/exp1a_smart_%j.log"
    )

    JOB_SCRIPT=$(cat <<'JOBEOF'
#!/bin/bash
set -euo pipefail

REPO=/scratch/triana24/midi-image-latent

echo "=========================================="
echo "Job:        $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPU GRES:   __GPU_GRES__"
echo "Start:      $(date)"
echo "=========================================="

module purge
module load python/3.11 gcc/12 cuda/12.6 arrow/23.0.1

export HF_HOME="$SCRATCH/.cache/huggingface"
export WANDB_MODE=offline
export WANDB_DIR="$REPO/outputs"

if [[ -f "$REPO/.env" ]]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"
        export "$key"="$value"
    done < <(grep -v '^#' "$REPO/.env" | grep '=')
fi

source "$REPO/.venv/bin/activate"
cd "$REPO"

echo "Python: $(which python)"
echo "CUDA:   $(python -c 'import torch; print(torch.cuda.device_count(), "GPU(s)")')"

python scripts/run_experiment.py \
    __CONFIG__ \
    --sweep-strategies \
    --log-level INFO \
    __EXTRA_ARGS__

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
JOBEOF
)

    JOB_SCRIPT="${JOB_SCRIPT//__CONFIG__/$CONFIG}"
    JOB_SCRIPT="${JOB_SCRIPT//__GPU_GRES__/$GPU_GRES}"
    JOB_SCRIPT="${JOB_SCRIPT//__EXTRA_ARGS__/$EXTRA_ARGS}"
fi

# ── Submit or dry-run ────────────────────────────────────────────────────────
if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN — would submit:"
    echo ""
    echo "${SBATCH_CMD[*]} <<'EOF'"
    echo "$JOB_SCRIPT"
    echo "EOF"
    echo ""
    echo "(use without --dry-run to actually submit)"
else
    echo "Submitting job..."
    JOB_ID=$(echo "$JOB_SCRIPT" | "${SBATCH_CMD[@]}" --parsable)
    echo ""
    echo "============================================================"
    echo "  Submitted job $JOB_ID"
    echo "  GPUs:      $NUM_GPUS x $GPU_TYPE"
    echo "  Mode:      $MODE"
    echo "  Log:       $SCRATCH/midi-image-latent/outputs/logs/exp1a_smart_${JOB_ID}.log"
    echo ""
    echo "  Monitor:   squeue -j $JOB_ID"
    echo "  Cancel:    scancel $JOB_ID"
    echo "  Log tail:  tail -f $SCRATCH/midi-image-latent/outputs/logs/exp1a_smart_${JOB_ID}.log"
    echo "============================================================"
fi
