#!/bin/bash

# =============================================================================
# SC-ERL MuJoCo Experiment Matrix — SLURM Array Job
# =============================================================================
# Macierz: 8 algorytmów × 5 środowisk × 5 seeds = 200 przebiegów
#
# Uruchamianie (pełna macierz):
#   sbatch --array=0-199 slurm_run_array.sh
#
# Uruchamianie (tylko SC-ERL, pierwsze 75 zadań):
#   sbatch --array=0-74 slurm_run_array.sh
#
# Uruchamianie (test: jedno zadanie):
#   sbatch --array=0 slurm_run_array.sh
# =============================================================================

# ==========================================
# Konfiguracja SLURM
# ==========================================
#SBATCH -N 1                            # 1 węzeł obliczeniowy
#SBATCH -c 4                            # 4 rdzenie CPU (MuJoCo + PyTorch loader)
#SBATCH --mem=16gb                      # 16 GB RAM na zadanie (replay buffer + modele)
#SBATCH --time=0-08:00:00              # 8 godzin na jedno zadanie (z zapasem dla Ant)
#SBATCH --job-name=sc_erl_mujoco        # Nazwa zadania widoczna w squeue
#SBATCH -p lem-gpu                      # Partycja GPU
#SBATCH --gres=gpu:1                    # 1 karta GPU (CUDA)
#SBATCH --output=logs/slurm-%A_%a.out  # Log: %A = job_id, %a = array_id
#SBATCH --error=logs/slurm-%A_%a.err   # Plik błędów
#SBATCH --mail-type=FAIL                # Email tylko przy awarii zadania

# ==========================================
# Definicja macierzy eksperymentów
# ==========================================
# Kolejność: algorytm, środowisko, seed
# Całkowita liczba kombinacji: 8 * 5 * 5 = 200

ALGORITHMS=(
    "sc_erl:dropout"       # 0..24
    "sc_erl:ensemble"      # 25..49
    "sc_erl:evidential"    # 50..74
    "sc_erl:random"        # 75..99
    "td3:"                 # 100..124
    "erl:"                 # 125..149
    "ddpg:"                # 150..174
    "ppo:"                 # 175..199
)

ENVS=(
    "HalfCheetah-v5"
    "Hopper-v5"
    "Walker2d-v5"
    "Ant-v5"
    "Swimmer-v5"
)

SEEDS=(0 1 2 3 4)

N_ENVS=${#ENVS[@]}    # 5
N_SEEDS=${#SEEDS[@]}  # 5

# Mapowanie SLURM_ARRAY_TASK_ID -> (algo_idx, env_idx, seed_idx)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

ALGO_IDX=$(( TASK_ID / (N_ENVS * N_SEEDS) ))
REMAINDER=$(( TASK_ID % (N_ENVS * N_SEEDS) ))
ENV_IDX=$(( REMAINDER / N_SEEDS ))
SEED_IDX=$(( REMAINDER % N_SEEDS ))

# Parsowanie algorytmu i trybu (format "algo:mode")
ALGO_MODE="${ALGORITHMS[$ALGO_IDX]}"
ALGO="${ALGO_MODE%%:*}"          # część przed ":"
SURROGATE_MODE="${ALGO_MODE##*:}" # część po ":"

ENV="${ENVS[$ENV_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

# ==========================================
# Budowanie nazwy i tagów WandB
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
# Środowisko
# ==========================================
echo "=========================================="
echo " SC-ERL SLURM Array Job"
echo "=========================================="
echo " Task ID   : ${TASK_ID}"
echo " Algorithm : ${ALGO}"
echo " Mode      : ${SURROGATE_MODE:-N/A}"
echo " Env       : ${ENV}"
echo " Seed      : ${SEED}"
echo " Run Name  : ${RUN_NAME}"
echo " Node      : $(hostname)"
echo " GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

# Ładowanie modułów systemowych
source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0

# Przejście do katalogu projektu (ZMIEŃ NA SWOJĄ ŚCIEŻKĘ)
PROJECT_DIR="/home/jakgil6519/workspace/SC-ERL-UE"
cd "${PROJECT_DIR}" || { echo "ERROR: Nie można przejść do ${PROJECT_DIR}"; exit 1; }

# Tworzenie katalogu na logi jeśli nie istnieje
mkdir -p logs

# ==========================================
# Instalacja zależności przez uv
# ==========================================
# Instalujemy uv jeśli nie jest dostępne globalnie
if ! command -v uv &> /dev/null; then
    echo "Instaluję uv..."
    pip install uv --quiet
fi

echo "Synchronizuję środowisko uv..."
uv sync --frozen

# ==========================================
# Uruchomienie eksperymentu
# ==========================================
echo "Uruchamiam: ${RUN_NAME}..."

uv run python entry_point.py \
    algorithm="${ALGO}" \
    ${SURROGATE_ARG} \
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
# Synchronizacja WandB po zakończeniu
# ==========================================
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "Trening zakończony sukcesem. Synchronizuję WandB offline..."
    uv run wandb sync --sync-all 2>/dev/null || echo "WandB sync pominięty (tryb online)."
    echo "Zadanie ${RUN_NAME} ukończone pomyślnie."
else
    echo "ERROR: Zadanie ${RUN_NAME} zakończyło się błędem (exit code: ${EXIT_CODE})"
    exit ${EXIT_CODE}
fi
