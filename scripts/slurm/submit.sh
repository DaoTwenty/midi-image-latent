#!/bin/bash
# =============================================================================
# submit.sh -- Single entrypoint for all MIDI Image VAE experiments
# =============================================================================
#
# Submits any experiment config as a SLURM job. Handles single-GPU, multi-GPU,
# multi-node, and mini mode via command-line flags.
#
# USAGE
# -----
#   bash scripts/slurm/submit.sh <config.yaml> [options] [-- extra-python-args...]
#
# EXAMPLES
# --------
#   # Single GPU (default):
#   bash scripts/slurm/submit.sh configs/experiments/exp_1a_vae_comparison.yaml
#
#   # Mini smoke test:
#   bash scripts/slurm/submit.sh configs/experiments/exp_1a_vae_comparison.yaml --mini
#
#   # 4 GPUs on one node:
#   bash scripts/slurm/submit.sh configs/experiments/exp_1a_vae_comparison.yaml --gpus 4
#
#   # 2 nodes x 4 GPUs (8 total):
#   bash scripts/slurm/submit.sh configs/experiments/exp_1a_vae_comparison.yaml --gpus 4 --nodes 2
#
#   # Auto-detect best available GPU config:
#   bash scripts/slurm/submit.sh configs/experiments/exp_1a_vae_comparison.yaml --auto
#
#   # MIG slices instead of full H100s:
#   bash scripts/slurm/submit.sh configs/experiments/exp_1a_vae_comparison.yaml --gpu-type mig
#
#   # Custom time limit:
#   bash scripts/slurm/submit.sh configs/experiments/exp_1a_vae_comparison.yaml --time 12:00:00
#
#   # Dry run (show sbatch command without submitting):
#   bash scripts/slurm/submit.sh configs/experiments/exp_1a_vae_comparison.yaml --gpus 4 --dry-run
#
#   # Forward extra args to the Python runner (after --):
#   bash scripts/slurm/submit.sh configs/experiments/exp_1a_vae_comparison.yaml --gpus 4 \
#       -- --sweep-strategies --assign-strategy cost-balanced
#
#   # Run all experiments in mini mode (smoke test):
#   bash scripts/slurm/submit.sh --all-mini

set -euo pipefail

REPO=/scratch/triana24/midi-image-latent
ACCOUNT="def-pasquier"

# ── Defaults ─────────────────────────────────────────────────────────────────
CONFIG=""
NUM_GPUS=1
NUM_NODES=1
CPUS_PER_TASK=8
MEM_PER_CPU="4G"
TIME_LIMIT=""
GPU_TYPE="h100"          # "h100" or "mig"
DRY_RUN=false
AUTO_DETECT=false
MINI=false
ALL_MINI=false
EXTRA_ARGS=""

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)       NUM_GPUS="$2"; shift 2 ;;
        --nodes)      NUM_NODES="$2"; shift 2 ;;
        --cpus)       CPUS_PER_TASK="$2"; shift 2 ;;
        --mem)        MEM_PER_CPU="$2"; shift 2 ;;
        --time)       TIME_LIMIT="$2"; shift 2 ;;
        --gpu-type)   GPU_TYPE="$2"; shift 2 ;;
        --account)    ACCOUNT="$2"; shift 2 ;;
        --mini)       MINI=true; shift ;;
        --auto)       AUTO_DETECT=true; shift ;;
        --dry-run)    DRY_RUN=true; shift ;;
        --all-mini)   ALL_MINI=true; shift ;;
        --)           shift; EXTRA_ARGS="$*"; break ;;
        --*)          echo "Unknown option: $1"; exit 1 ;;
        *)
            if [[ -z "$CONFIG" ]]; then
                CONFIG="$1"; shift
            else
                echo "Unexpected argument: $1"; exit 1
            fi
            ;;
    esac
done

# ── Validate ─────────────────────────────────────────────────────────────────
if [[ "$ALL_MINI" == true ]]; then
    # --all-mini mode doesn't need a config
    :
elif [[ -z "$CONFIG" ]]; then
    echo "Usage: bash scripts/slurm/submit.sh <config.yaml> [options] [-- extra-args...]"
    echo "       bash scripts/slurm/submit.sh --all-mini"
    exit 1
elif [[ ! -f "$CONFIG" ]]; then
    # Try relative to repo
    if [[ -f "$REPO/$CONFIG" ]]; then
        CONFIG="$REPO/$CONFIG"
    else
        echo "Config file not found: $CONFIG"
        exit 1
    fi
fi

# ── GPU auto-detection ───────────────────────────────────────────────────────
probe_free_gpus() {
    local partition="$1"
    local gres_type="$2"
    local gres_key="gres/gpu:${gres_type}"

    _PROBE_BEST_FREE=0

    while read -r node; do
        [[ -z "$node" ]] && continue
        local info
        info=$(scontrol show node "$node" 2>/dev/null)

        local cfg_gpus alloc_gpus
        cfg_gpus=$(echo "$info" | grep -oP "CfgTRES=.*${gres_key}=\K[0-9]+" 2>/dev/null || echo "0")
        alloc_gpus=$(echo "$info" | grep -oP "AllocTRES=.*${gres_key}=\K[0-9]+" 2>/dev/null || echo "0")

        local free=$(( cfg_gpus - alloc_gpus ))
        if [[ $free -gt $_PROBE_BEST_FREE ]]; then
            _PROBE_BEST_FREE=$free
        fi
    done < <(sinfo -p "$partition" -N --noheader -o "%n %t" 2>/dev/null \
             | grep -vE "drain|down|comp|inval" \
             | awk '{print $1}' | sort -u)
}

if [[ "$AUTO_DETECT" == true && "$ALL_MINI" == false ]]; then
    echo "Probing GPU availability..."

    probe_free_gpus "gpubase_bynode_b1" "h100"
    FULL_FREE=$_PROBE_BEST_FREE

    probe_free_gpus "gpubase_bygpu_b1" "nvidia_h100_80gb_hbm3_1g.10gb"
    MIG_FREE=$_PROBE_BEST_FREE

    # Pick the best available count (full H100s preferred, then MIG)
    if [[ $FULL_FREE -ge 2 ]]; then
        NUM_GPUS=$FULL_FREE
        GPU_TYPE="h100"
        echo "  Auto-detected: ${NUM_GPUS}x full H100 available"
    elif [[ $MIG_FREE -ge 2 ]]; then
        NUM_GPUS=$MIG_FREE
        GPU_TYPE="mig"
        echo "  Auto-detected: ${NUM_GPUS}x MIG 1g.10gb available"
    elif [[ $FULL_FREE -ge 1 ]]; then
        NUM_GPUS=1
        GPU_TYPE="h100"
        echo "  Auto-detected: 1x full H100 available"
    elif [[ $MIG_FREE -ge 1 ]]; then
        NUM_GPUS=1
        GPU_TYPE="mig"
        echo "  Auto-detected: 1x MIG 1g.10gb available"
    else
        NUM_GPUS=1
        GPU_TYPE="h100"
        echo "  No free GPUs detected -- submitting 1x h100 to queue"
    fi

    # Cap at 4 GPUs (diminishing returns beyond that for our workload)
    [[ $NUM_GPUS -gt 4 ]] && NUM_GPUS=4
    echo ""
fi

# ── Resolve GPU GRES string ─────────────────────────────────────────────────
# Always specify GPU type (h100 or full MIG identifier).
if [[ "$GPU_TYPE" == "mig" ]]; then
    GPU_GRES="nvidia_h100_80gb_hbm3_1g.10gb:${NUM_GPUS}"
else
    GPU_GRES="h100:${NUM_GPUS}"
fi

# ── Compute time limit if not set ────────────────────────────────────────────
if [[ -z "$TIME_LIMIT" ]]; then
    if [[ "$MINI" == true || "$ALL_MINI" == true ]]; then
        TIME_LIMIT="01:00:00"
    else
        TOTAL_GPUS=$(( NUM_GPUS * NUM_NODES ))
        if [[ $TOTAL_GPUS -ge 4 ]]; then
            TIME_LIMIT="06:00:00"
        elif [[ $TOTAL_GPUS -ge 2 ]]; then
            TIME_LIMIT="12:00:00"
        else
            TIME_LIMIT="1-00:00:00"
        fi
    fi
fi

# ── Determine mode ───────────────────────────────────────────────────────────
TOTAL_GPUS=$(( NUM_GPUS * NUM_NODES ))
if [[ $TOTAL_GPUS -gt 1 ]]; then
    MODE="distributed"
else
    MODE="single"
fi

# ── Handle --all-mini ────────────────────────────────────────────────────────
if [[ "$ALL_MINI" == true ]]; then
    _submit_all_mini() {
        local configs=(
            "configs/experiments/exp_1a_vae_comparison.yaml --sweep-strategies"
            "configs/experiments/exp_1b_detection_methods.yaml --sweep-detectors"
            "configs/experiments/exp_2_resolution_study.yaml"
            "configs/experiments/exp_3_channel_strategy.yaml --sweep-strategies"
            "configs/experiments/exp_4_latent_analysis.yaml"
            "configs/experiments/exp_4b_latent_structure.yaml"
            "configs/experiments/exp_4c_sublatent.yaml"
            "configs/experiments/exp_4d_conditioning.yaml"
            "configs/experiments/exp_5_sequence_generation.yaml"
        )

        EXP_NAME="all_mini"
        LOG_FILE="$SCRATCH/midi-image-latent/outputs/logs/${EXP_NAME}_%j.log"

        # Build inline script that runs all experiments sequentially
        local run_cmds=""
        for entry in "${configs[@]}"; do
            read -r cfg extra <<< "$entry"
            run_cmds+="
echo ''
echo '=== $(basename "$cfg" .yaml) (mini) ==='
python scripts/run_experiment.py $cfg --mini --data-root data/lakh --log-level INFO ${extra:-}
"
        done

        JOB_SCRIPT=$(cat <<JOBEOF
#!/bin/bash
set -euo pipefail
REPO=/scratch/triana24/midi-image-latent

echo "=========================================="
echo "Job:        \$SLURM_JOB_ID"
echo "Node:       \$SLURMD_NODENAME"
echo "Start:      \$(date)"
echo "Mode:       ALL EXPERIMENTS (mini)"
echo "=========================================="

module purge
module load python/3.11 gcc/12 cuda/12.6 arrow/23.0.1

export HF_HOME="\$SCRATCH/.cache/huggingface"
export WANDB_MODE=offline
export WANDB_DIR="\$REPO/outputs"

if [[ -f "\$REPO/.env" ]]; then
    while IFS='=' read -r key value; do
        [[ "\$key" =~ ^#.*\$ || -z "\$key" ]] && continue
        value="\${value%\\\"}"
        value="\${value#\\\"}"
        value="\${value%\\'}"
        value="\${value#\\'}"
        export "\$key"="\$value"
    done < <(grep -v '^#' "\$REPO/.env" | grep '=')
fi

source "\$REPO/.venv/bin/activate"
cd "\$REPO"

echo "Python: \$(which python)"
${run_cmds}

echo ""
echo "=========================================="
echo "All mini experiments complete: \$(date)"
echo "=========================================="
JOBEOF
        )

        SBATCH_CMD=(
            sbatch
            --job-name="midi-vae-all-mini"
            --account="$ACCOUNT"
            --gpus-per-node="${GPU_GRES}"
            --cpus-per-task=4
            --mem-per-cpu="$MEM_PER_CPU"
            --time="02:00:00"
            --output="$LOG_FILE"
            --error="$LOG_FILE"
        )

        if [[ "$DRY_RUN" == true ]]; then
            echo "DRY RUN -- would submit all-mini job:"
            echo ""
            echo "${SBATCH_CMD[*]} <<'EOF'"
            echo "$JOB_SCRIPT"
            echo "EOF"
        else
            mkdir -p "$SCRATCH/midi-image-latent/outputs/logs"
            JOB_ID=$(echo "$JOB_SCRIPT" | "${SBATCH_CMD[@]}" --parsable)
            echo "Submitted all-mini job $JOB_ID"
            echo "  Log: $SCRATCH/midi-image-latent/outputs/logs/${EXP_NAME}_${JOB_ID}.log"
            echo "  Monitor: squeue -j $JOB_ID"
        fi
    }

    _submit_all_mini
    exit 0
fi

# ── Derive experiment name ───────────────────────────────────────────────────
EXP_NAME=$(basename "$CONFIG" .yaml)
if [[ "$MINI" == true ]]; then
    EXP_NAME="${EXP_NAME}_mini"
fi
OUTPUT_ROOT="$SCRATCH/midi-image-latent/outputs/${EXP_NAME}"
LOG_FILE="$SCRATCH/midi-image-latent/outputs/logs/${EXP_NAME}_%j.log"

# Add --mini to extra args if requested
if [[ "$MINI" == true ]]; then
    EXTRA_ARGS="--mini $EXTRA_ARGS"
fi

# ── Print plan ───────────────────────────────────────────────────────────────
echo "============================================================"
echo "  MIDI Image VAE -- Job Submission"
echo "  $(date)"
echo "------------------------------------------------------------"
echo "  Config:       $CONFIG"
echo "  Experiment:   $EXP_NAME"
echo "  Mode:         $MODE"
echo "  GRES:         $GPU_GRES"
echo "  GPUs/node:    $NUM_GPUS"
echo "  Nodes:        $NUM_NODES"
echo "  Total GPUs:   $TOTAL_GPUS"
echo "  CPUs/task:    $CPUS_PER_TASK"
echo "  Mem/CPU:      $MEM_PER_CPU"
echo "  Time limit:   $TIME_LIMIT"
echo "  Output:       $OUTPUT_ROOT"
echo "  Extra args:   ${EXTRA_ARGS:-<none>}"
echo "============================================================"
echo ""

# ── Build the job script ─────────────────────────────────────────────────────
# Common preamble shared by both single and distributed modes
read -r -d '' PREAMBLE <<'PREAMBLE_EOF' || true
set -euo pipefail
REPO=/scratch/triana24/midi-image-latent

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
PREAMBLE_EOF

if [[ "$MODE" == "distributed" ]]; then
    # ── Multi-GPU job ────────────────────────────────────────────────────────
    SBATCH_CMD=(
        sbatch
        --job-name="midi-vae-${EXP_NAME}"
        --account="$ACCOUNT"
        --nodes="$NUM_NODES"
        --gpus-per-node="${GPU_GRES}"
        --ntasks-per-node="$NUM_GPUS"
        --cpus-per-task="$CPUS_PER_TASK"
        --mem-per-cpu="$MEM_PER_CPU"
        --time="$TIME_LIMIT"
        --output="$LOG_FILE"
        --error="$LOG_FILE"
    )

    JOB_SCRIPT=$(cat <<'JOBEOF'
#!/bin/bash
__PREAMBLE__

echo "=========================================="
echo "Job:        $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Nodes:      $SLURM_NNODES"
echo "Tasks:      $SLURM_NTASKS"
echo "Start:      $(date)"
echo "Config:     __CONFIG__"
echo "=========================================="

mkdir -p "__OUTPUT_ROOT__/logs"

T_START=$(date +%s)

# Prevent SLURM_MEM_PER_CPU/GPU/NODE mutual exclusion error in srun
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

srun \
    --output="__OUTPUT_ROOT__/logs/rank_%t.log" \
    --error="__OUTPUT_ROOT__/logs/rank_%t.log" \
    "$REPO/.venv/bin/python" "$REPO/scripts/run_experiment_distributed.py" \
        "__CONFIG__" \
        --output-root "__OUTPUT_ROOT__" \
        --multi-node \
        --assign-strategy cost-balanced \
        --log-level INFO __EXTRA_ARGS__

T_SRUN=$(date +%s)
echo ""
echo "All ranks complete in $(( T_SRUN - T_START ))s."

echo ""
echo "Merging per-rank results..."
"$REPO/.venv/bin/python" "$REPO/scripts/merge_sweep_results.py" \
    "__OUTPUT_ROOT__" \
    --scan-rank-dirs \
    --verbose \
    --output "__OUTPUT_ROOT__/sweep_summary.json"

T_END=$(date +%s)
echo ""
echo "=========================================="
echo "Done:       $(date)"
echo "Elapsed:    $(( T_END - T_START ))s"
echo "Summary:    __OUTPUT_ROOT__/sweep_summary.json"
echo "Rank logs:  __OUTPUT_ROOT__/logs/"
echo "=========================================="
JOBEOF
    )

    JOB_SCRIPT="${JOB_SCRIPT//__PREAMBLE__/$PREAMBLE}"
    JOB_SCRIPT="${JOB_SCRIPT//__CONFIG__/$CONFIG}"
    JOB_SCRIPT="${JOB_SCRIPT//__OUTPUT_ROOT__/$OUTPUT_ROOT}"
    JOB_SCRIPT="${JOB_SCRIPT//__EXTRA_ARGS__/$EXTRA_ARGS}"

else
    # ── Single-GPU job ───────────────────────────────────────────────────────
    SBATCH_CMD=(
        sbatch
        --job-name="midi-vae-${EXP_NAME}"
        --account="$ACCOUNT"
        --gpus-per-node="${GPU_GRES}"
        --cpus-per-task="$CPUS_PER_TASK"
        --mem-per-cpu="$MEM_PER_CPU"
        --time="$TIME_LIMIT"
        --output="$LOG_FILE"
        --error="$LOG_FILE"
    )

    JOB_SCRIPT=$(cat <<'JOBEOF'
#!/bin/bash
__PREAMBLE__

echo "=========================================="
echo "Job:        $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start:      $(date)"
echo "Config:     __CONFIG__"
echo "=========================================="

echo "Python: $(which python)"
echo "CUDA:   $(python -c 'import torch; print(torch.cuda.device_count(), "GPU(s)")')"
echo ""

python scripts/run_experiment.py \
    "__CONFIG__" \
    --log-level INFO __EXTRA_ARGS__

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
JOBEOF
    )

    JOB_SCRIPT="${JOB_SCRIPT//__PREAMBLE__/$PREAMBLE}"
    JOB_SCRIPT="${JOB_SCRIPT//__CONFIG__/$CONFIG}"
    JOB_SCRIPT="${JOB_SCRIPT//__EXTRA_ARGS__/$EXTRA_ARGS}"
fi

# ── Submit or dry-run ────────────────────────────────────────────────────────
if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN -- would submit:"
    echo ""
    echo "${SBATCH_CMD[*]} <<'EOF'"
    echo "$JOB_SCRIPT"
    echo "EOF"
    echo ""
    echo "(remove --dry-run to actually submit)"
else
    mkdir -p "$SCRATCH/midi-image-latent/outputs/logs"
    JOB_ID=$(echo "$JOB_SCRIPT" | "${SBATCH_CMD[@]}" --parsable)
    echo "Submitted job $JOB_ID"
    echo ""
    echo "  Monitor:  squeue -j $JOB_ID"
    echo "  Cancel:   scancel $JOB_ID"
    echo "  Log:      tail -f $SCRATCH/midi-image-latent/outputs/logs/${EXP_NAME}_${JOB_ID}.log"
    if [[ "$MODE" == "distributed" ]]; then
        echo "  Ranks:    ls $OUTPUT_ROOT/logs/"
    fi
    echo ""
fi
