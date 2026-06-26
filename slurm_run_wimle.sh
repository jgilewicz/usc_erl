#!/bin/bash

# WIMLE training matrix: 10 environments × 5 seeds = 50 tasks.
#
# MuJoCo v5 (5 envs × 5 seeds, tasks 0-24):
#   HalfCheetah-v5, Hopper-v5, Walker2d-v5, Ant-v5, Swimmer-v5
#
# DMC dog via dm_control (5 envs × 5 seeds, tasks 25-49):
#   dog-stand, dog-walk, dog-trot, dog-run, dog-fetch
#
# Usage:
#   sbatch --array=0-49 slurm_run_wimle.sh
#
# Optional overrides (env vars):
#   N_STEPS   Training steps per run (default: 1000000)
#   NUM_SEEDS Number of parallel envs within each wimle run (default: 4)

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16gb
#SBATCH --time=0-12:00:00
#SBATCH --job-name=wimle
#SBATCH -p lem-gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --mail-type=FAIL

# ---- Environment matrix: "benchmark:canonical_env_id" pairs ----
# canonical_env_id matches the project-wide convention used by download_results.py
ENVS=(
    "gym:HalfCheetah-v5"             # 0..4
    "gym:Hopper-v5"                  # 5..9
    "gym:Walker2d-v5"                # 10..14
    "gym:Ant-v5"                     # 15..19
    "gym:Swimmer-v5"                 # 20..24
    "dmc:dm_control/dog-stand-v0"    # 25..29
    "dmc:dm_control/dog-walk-v0"     # 30..34
    "dmc:dm_control/dog-trot-v0"     # 35..39
    "dmc:dm_control/dog-run-v0"      # 40..44
    "dmc:dm_control/dog-fetch-v0"    # 45..49
)

SEEDS=(0 1 2 3 4)
N_SEEDS=${#SEEDS[@]}

N_STEPS="${N_STEPS:-1000000}"
NUM_SEEDS="${NUM_SEEDS:-4}"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
ENV_IDX=$((TASK_ID / N_SEEDS))
SEED_IDX=$((TASK_ID % N_SEEDS))

ENV_PAIR="${ENVS[$ENV_IDX]}"
BENCHMARK="${ENV_PAIR%%:*}"
CANONICAL_ENV="${ENV_PAIR##*:}"
SEED="${SEEDS[$SEED_IDX]}"

# Run name uses the project-wide slug so download_results.py can parse it
ENV_SLUG=$(echo "${CANONICAL_ENV}" | tr '/' '_' | tr ':' '_')
RUN_NAME="wimle_${ENV_SLUG}_seed${SEED}"

# WIMLE's --env_name flag uses dm_control.suite format (no prefix, no version suffix)
if [[ "$BENCHMARK" == "dmc" ]]; then
    WIMLE_ENV=$(echo "$CANONICAL_ENV" | sed 's|dm_control/||' | sed 's|-v0$||')
else
    WIMLE_ENV="$CANONICAL_ENV"
fi

echo "Task ${TASK_ID} | WIMLE | benchmark=${BENCHMARK} env=${CANONICAL_ENV} | seed ${SEED}"

source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0


PROJECT_DIR="/home/jakgil6519/workspace/ue_sc_erl"
WIMLE_DIR="${PROJECT_DIR}/wimle"

cd "${WIMLE_DIR}" || {
    echo "ERROR: Cannot navigate to ${WIMLE_DIR}"
    exit 1
}

if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "ERROR: ${WIMLE_DIR}/.venv does not exist! Please build the environment."
    exit 1
fi

CUDNN_PATH=$(python -c "import nvidia.cudnn; import os; print(os.path.dirname(nvidia.cudnn.__file__))")
export LD_LIBRARY_PATH="${CUDNN_PATH}/lib:${LD_LIBRARY_PATH:-}"

export WANDB_API_KEY="INSERT_YOUR_WANDB_API_KEY_HERE"
export WANDB_MODE="offline"
export WANDB_DIR="${PROJECT_DIR}/wandb_logs"
mkdir -p "${PROJECT_DIR}/logs" "${WANDB_DIR}"

python train_parallel.py \
    --benchmark="${BENCHMARK}" \
    --env_name="${WIMLE_ENV}" \
    --seed="${SEED}" \
    --num_seeds="${NUM_SEEDS}" \
    --max_steps="${N_STEPS}" \
    --batch_size=256 \
    --start_training=25000 \
    --eval_interval=5000 \
    --run_name="${RUN_NAME}" \
    --wandb_project="ue_evo_rl_3" \
    --wandb_mode="offline" \
    --save_dir="${PROJECT_DIR}/wimle_outputs"

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
