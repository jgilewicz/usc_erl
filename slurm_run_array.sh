#!/bin/bash

# Matrix run for ONE environment — backend and algo matrix auto-detected from TARGET_ENV.
#
# MuJoCo  (10 algos × 5 seeds = 50 tasks):
#   TARGET_ENV=HalfCheetah-v5          sbatch --array=0-49 slurm_run_array.sh
#   TARGET_ENV=Hopper-v5               sbatch --array=0-49 slurm_run_array.sh
#   TARGET_ENV=Walker2d-v5             sbatch --array=0-49 slurm_run_array.sh
#   TARGET_ENV=Ant-v5                  sbatch --array=0-49 slurm_run_array.sh
#   TARGET_ENV=Swimmer-v5              sbatch --array=0-49 slurm_run_array.sh
#
# DMC dog  (4 SC-ERL modes × 5 seeds = 20 tasks):
#   TARGET_ENV=dm_control/dog-stand-v0 sbatch --array=0-19 slurm_run_array.sh
#   TARGET_ENV=dm_control/dog-walk-v0  sbatch --array=0-19 slurm_run_array.sh
#   TARGET_ENV=dm_control/dog-trot-v0  sbatch --array=0-19 slurm_run_array.sh
#   TARGET_ENV=dm_control/dog-run-v0   sbatch --array=0-19 slurm_run_array.sh
#   TARGET_ENV=dm_control/dog-fetch-v0 sbatch --array=0-19 slurm_run_array.sh
#
# Optional overrides:
#   N_STEPS  Training steps per run (default: 1000000)

#SBATCH -N 1                            # 1 node
#SBATCH -c 4                            # 4 CPU cores
#SBATCH --mem=16gb                      # 16 GB RAM per job
#SBATCH --time=0-12:00:00               # 12 hours per job
#SBATCH --job-name=sc_erl               # Job name
#SBATCH -p lem-gpu                      # GPU partition
#SBATCH --gres=gpu:hopper:1             # 1 GPU card
#SBATCH --output=logs/slurm-%A_%a.out   # Standard output log
#SBATCH --error=logs/slurm-%A_%a.err    # Error log
#SBATCH --mail-type=FAIL                # Email on failure

ENV="${TARGET_ENV:-HalfCheetah-v5}"
N_STEPS="${N_STEPS:-1000000}"

# ---- Backend + algorithm matrix: auto-detected from env ID prefix ----
if [[ "$ENV" == dm_control/* || "$ENV" == fancy/* || "$ENV" == metaworld/* ]]; then
  BACKEND="fancy_gym"
  ENV_TAG="DMC"
  ALGORITHMS=(
    "sc_erl:dropout"    # 0..4
    "sc_erl:ensemble"   # 5..9
    "sc_erl:evidential" # 10..14
    "sc_erl:random"     # 15..19
  )
else
  BACKEND="mujoco"
  ENV_TAG="MuJoCo"
  ALGORITHMS=(
    "sc_erl:dropout"    # 0..4
    "sc_erl:ensemble"   # 5..9
    "sc_erl:evidential" # 10..14
    "sc_erl:random"     # 15..19
    "td3:"              # 20..24
    "erl:"              # 25..29
    "ddpg:"             # 30..34
    "ppo:"              # 35..39
    "sac:"              # 40..44
    "crossq:"           # 45..49
  )
fi

SEEDS=(0 1 2 3 4)
N_SEEDS=${#SEEDS[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
ALGO_IDX=$((TASK_ID / N_SEEDS))
SEED_IDX=$((TASK_ID % N_SEEDS))

ALGO_MODE="${ALGORITHMS[$ALGO_IDX]}"
ALGO="${ALGO_MODE%%:*}"
SURROGATE_MODE="${ALGO_MODE##*:}"
SEED="${SEEDS[$SEED_IDX]}"

# Sanitize env id (dm_control/dog-stand-v0 → dm_control_dog-stand-v0)
ENV_SLUG=$(echo "${ENV}" | tr '/' '_' | tr ':' '_')

if [[ -n "$SURROGATE_MODE" ]]; then
  RUN_NAME="${ALGO}_${SURROGATE_MODE}_${ENV_SLUG}_seed${SEED}"
  WANDB_TAGS="[${ENV_TAG},${ALGO},${SURROGATE_MODE}]"
else
  RUN_NAME="${ALGO}_${ENV_SLUG}_seed${SEED}"
  WANDB_TAGS="[${ENV_TAG},${ALGO},baseline]"
fi

echo "Task ${TASK_ID} | ${ALGO} ${SURROGATE_MODE:-N/A} | ${ENV} | seed ${SEED} | backend ${BACKEND}"

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

PROJECT_DIR="/home/jakgil6519/workspace/ue_sc_erl"
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
export LD_LIBRARY_PATH="${PROJECT_DIR}/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
mkdir -p logs "${WANDB_DIR}"

# Build optional args as arrays to avoid empty-string pitfalls
SURROGATE_ARGS=()
[[ -n "$SURROGATE_MODE" ]] && SURROGATE_ARGS=("surrogate.mode=${SURROGATE_MODE}")

BACKEND_ARGS=()
[[ "$BACKEND" == "fancy_gym" ]] && BACKEND_ARGS=("env.backend=fancy_gym" "eval_env.backend=fancy_gym")

python entry_point.py \
  algorithm="${ALGO}" \
  "${SURROGATE_ARGS[@]}" \
  "${BACKEND_ARGS[@]}" \
  seed="${SEED}" \
  env.id="${ENV}" \
  eval_env.id="${ENV}" \
  n_steps="${N_STEPS}" \
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
