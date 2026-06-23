#!/bin/bash

# Stage 1 (single job — tune shared + backbone + dropout params):
#   TARGET_ENV=HalfCheetah-v5 sbatch optim/slurm_tune.sh
#
# Stage 2 (array job — tune evidential + ensemble, one task each):
#   TARGET_ENV=HalfCheetah-v5 BASE_STUDY=optuna_dropout_HalfCheetah-v5.db sbatch --array=0-1 optim/slurm_tune.sh
#
# Optional env var overrides:
#   TARGET_ENV   MuJoCo env id (default: HalfCheetah-v5)
#   N_TRIALS     Optuna trials per mode (default: 30)
#   N_STEPS      Training steps per trial (default: 200000)
#   SAMPLER      tpe or random (default: tpe)
#   BACKBONE     ddpg or td3 (default: td3)
#   BASE_STUDY   path to Stage 1 .db file (triggers Stage 2 when set)

#SBATCH -N 1                            # 1 node
#SBATCH -c 4                            # 4 CPU cores
#SBATCH --mem=16gb                      # 16 GB RAM per job
#SBATCH --time=1-00:00:00               # 24 hours per task
#SBATCH --job-name=sc_erl_tune          # Job name
#SBATCH -p lem-gpu                      # GPU partition
#SBATCH --gres=gpu:hopper:1             # 1 GPU card
#SBATCH --output=logs/tune-%A_%a.out    # Standard output log
#SBATCH --error=logs/tune-%A_%a.err     # Error log
#SBATCH --mail-type=FAIL                # Email on failure

ENV="${TARGET_ENV:-HalfCheetah-v5}"
N_TRIALS="${N_TRIALS:-30}"
N_STEPS="${N_STEPS:-200000}"
SAMPLER="${SAMPLER:-tpe}"
BACKBONE="${BACKBONE:-td3}"
BASE_STUDY="${BASE_STUDY:-}"

if [[ -z "$BASE_STUDY" ]]; then
  # Stage 1: single job, mode=dropout
  MODE="dropout"
  STAGE="1 (shared+backbone+dropout)"
else
  # Stage 2: array job 0-1, map task id → mode
  MODES=("evidential" "ensemble")
  TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
  MODE="${MODES[$TASK_ID]}"
  if [[ -z "$MODE" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range (0-1)"
    exit 1
  fi
  STAGE="2 (${MODE}-specific)"
fi

STORAGE="sqlite:///optuna_${MODE}_${ENV}.db"
STUDY_NAME="sc_erl_${MODE}_${ENV}"

echo "Stage: ${STAGE} | Mode: ${MODE} | Env: ${ENV} | Trials: ${N_TRIALS} | Steps: ${N_STEPS}"
[[ -n "$BASE_STUDY" ]] && echo "Base study: ${BASE_STUDY}"

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

PROJECT_DIR="/home/jakgil6519/workspace/ue_evo_rl"
cd "${PROJECT_DIR}" || {
  echo "ERROR: Cannot navigate to ${PROJECT_DIR}"
  exit 1
}

if [ -d ".venv" ]; then
  source .venv/bin/activate
else
  echo "ERROR: .venv does not exist. Run: python -m venv .venv && pip install -e ."
  exit 1
fi

export WANDB_MODE="disabled"
export LD_LIBRARY_PATH=$(find ${PROJECT_DIR}/.venv/lib/python3.12/site-packages/nvidia -name "lib" -type d | tr '\n' ':')${LD_LIBRARY_PATH:-}
mkdir -p logs outputs/optuna

BASE_STUDY_ARG=""
if [[ -n "$BASE_STUDY" ]]; then
  BASE_STUDY_ARG="--base-study ${BASE_STUDY}"
fi

python optim/tune_sc_erl.py \
  --env "${ENV}" \
  --mode "${MODE}" \
  --backbone "${BACKBONE}" \
  --n-trials "${N_TRIALS}" \
  --n-steps "${N_STEPS}" \
  --sampler "${SAMPLER}" \
  --storage "${STORAGE}" \
  --study-name "${STUDY_NAME}" \
  ${BASE_STUDY_ARG}

EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
  echo "Tuning stage=${STAGE} mode=${MODE} completed successfully."
else
  echo "ERROR: Tuning mode=${MODE} failed with exit code: ${EXIT_CODE}"
  exit ${EXIT_CODE}
fi
