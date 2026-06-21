#!/bin/bash

# =============================================================================
# SC-ERL MuJoCo Experiment Matrix — SLURM Array Job (Single Env Mode)
# =============================================================================
# Matrix for ONE environment: 8 algorithms × 5 seeds = 40 runs
#
# Execution (default Ant-v5):
#   sbatch --array=0-39 slurm_run_array.sh
# =============================================================================

#SBATCH -N 1                            # 1 node
#SBATCH -c 4                            # 4 CPU cores
#SBATCH --mem=16gb                      # 16 GB RAM per job
#SBATCH --time=0-08:00:00               # 8 hours limit per job
#SBATCH --job-name=sc_erl_mujoco        # Job name
#SBATCH -p lem-gpu                      # GPU partition
#SBATCH --gres=gpu:hopper:1             # 1 GPU card
#SBATCH --output=logs/slurm-%A_%a.out   # Standard output log
#SBATCH --error=logs/slurm-%A_%a.err    # Error log
#SBATCH --mail-type=FAIL                # Email on failure

# ==========================================
# Dynamic environment selection
# ==========================================
ENV="${TARGET_ENV:-Ant-v5}"

# ==========================================
# Define algorithms and seeds (8 * 5 = 40 combinations)
# ==========================================
ALGORITHMS=(
  "sc_erl:dropout"    # 0..4
  "sc_erl:ensemble"   # 5..9
  "sc_erl:evidential" # 10..14
  "sc_erl:random"     # 15..19
  "td3:"              # 20..24
  "erl:"              # 25..29
  "ddpg:"             # 30..34
  "ppo:"              # 35..39
)

SEEDS=(0 1 2 3 4)
N_SEEDS=${#SEEDS[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

ALGO_IDX=$((TASK_ID / N_SEEDS))
SEED_IDX=$((TASK_ID % N_SEEDS))

ALGO_MODE="${ALGORITHMS[$ALGO_IDX]}"
ALGO="${ALGO_MODE%%:*}"
SURROGATE_MODE="${ALGO_MODE##*:}"

SEED="${SEEDS[$SEED_IDX]}"

# ==========================================
# Build run name and WandB tags
# ==========================================
if [[ -n "$SURROGATE_MODE" ]]; then
  RUN_NAME="${ALGO}_${SURROGATE_MODE}_${ENV}_seed${SEED}"
  WANDB_TAGS="[MuJoCo,${ALGO},${SURROGATE_MODE}]"
  SURROGATE_ARG="surrogate.mode=${SURROGATE_MODE}"
else
  RUN_NAME="${ALGO}_${ENV}_seed${SEED}"
  WANDB_TAGS="[MuJoCo,${ALGO},baseline]"
  SURROGATE_ARG=""
fi

# ==========================================
# Execution environment setup
# ==========================================
echo "=========================================="
echo " SC-ERL SLURM Array Job (.venv mode)"
echo "=========================================="
echo " Task ID   : ${TASK_ID}"
echo " Algorithm : ${ALGO}"
echo " Mode      : ${SURROGATE_MODE:-N/A}"
echo " Env       : ${ENV}"
echo " Seed      : ${SEED}"
echo " Run Name  : ${RUN_NAME}"
echo "=========================================="

# Load system modules
source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0

# Navigate to project directory
PROJECT_DIR="/home/jakgil6519/workspace/ue_evo_rl"
cd "${PROJECT_DIR}" || {
  echo "ERROR: Cannot navigate to ${PROJECT_DIR}"
  exit 1
}

# Activate local .venv environment
if [ -d ".venv" ]; then
  source .venv/bin/activate
else
  echo "ERROR: .venv directory does not exist! Please build the environment."
  exit 1
fi

# ==========================================
# WandB Configuration (Offline mode)
# ==========================================
export WANDB_API_KEY="INSERT_YOUR_WANDB_API_KEY_HERE"
export WANDB_MODE="offline"

# Local directory for WandB logs to avoid NFS issues
export WANDB_DIR="${PROJECT_DIR}/wandb_logs"

# Create necessary directory structures
mkdir -p logs
mkdir -p "${WANDB_DIR}"

# ==========================================
# Run the experiment
# ==========================================
echo "Running python entry_point.py..."

python entry_point.py \
  algorithm="${ALGO}" \
  "${SURROGATE_ARG}" \
  seed="${SEED}" \
  env.id="${ENV}" \
  eval_env.id="${ENV}" \
  n_steps=1000000 \
  wandb.enabled=true \
  "wandb.name=${RUN_NAME}" \
  "wandb.tags=${WANDB_TAGS}" \
  hydra.run.dir="outputs/${RUN_NAME}"

EXIT_CODE=$?

# ==========================================
# Sync WandB logs after training
# ==========================================
if [[ $EXIT_CODE -eq 0 ]]; then
  echo "Training completed. Syncing WandB logs..."
  for d in "${WANDB_DIR}/wandb/offline-run-"*; do
    [[ -d "$d" ]] && wandb sync "$d"
  done
  echo "Job ${RUN_NAME} completed successfully."
else
  echo "ERROR: Job ${RUN_NAME} failed with exit code: ${EXIT_CODE}"
  exit ${EXIT_CODE}
fi
