#!/bin/bash

# Matrix for ONE environment: 8 algorithms × 5 seeds = 40 runs
#   TARGET_ENV=HalfCheetah-v5 sbatch --array=0-39 slurm_run_array.sh

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

ENV="${TARGET_ENV:-Ant-v5}"

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

if [[ -n "$SURROGATE_MODE" ]]; then
  RUN_NAME="${ALGO}_${SURROGATE_MODE}_${ENV}_seed${SEED}"
  WANDB_TAGS="[MuJoCo,${ALGO},${SURROGATE_MODE}]"
  SURROGATE_ARG="surrogate.mode=${SURROGATE_MODE}"
else
  RUN_NAME="${ALGO}_${ENV}_seed${SEED}"
  WANDB_TAGS="[MuJoCo,${ALGO},baseline]"
  SURROGATE_ARG=""
fi

echo "Task ${TASK_ID} | ${ALGO} ${SURROGATE_MODE:-N/A} | ${ENV} | seed ${SEED}"

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0

PROJECT_DIR="/home/jakgil6519/workspace/ue_evo_rl"
cd "${PROJECT_DIR}" || {
  echo "ERROR: Cannot navigate to ${PROJECT_DIR}"
  exit 1
}

if [ -d ".venv" ]; then
  source .venv/bin/activate
else
  echo "ERROR: .venv directory does not exist! Please build the environment."
  exit 1
fi

export WANDB_API_KEY="INSERT_YOUR_WANDB_API_KEY_HERE"
export WANDB_MODE="offline"
export WANDB_DIR="${PROJECT_DIR}/wandb_logs"
mkdir -p logs "${WANDB_DIR}"

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
