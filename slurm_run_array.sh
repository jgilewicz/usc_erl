#!/bin/bash

# =============================================================================
# SC-ERL MuJoCo Experiment Matrix — SLURM Array Job (Single Env Mode)
# =============================================================================
# Macierz dla JEDNEGO środowiska: 8 algorytmów × 5 seeds = 40 przebiegów
#
# Uruchamianie (domyślnie Ant-v5):
#   sbatch --array=0-39 slurm_run_array.sh
# =============================================================================

#SBATCH -N 1                            # 1 węzeł obliczeniowy
#SBATCH -c 4                            # 4 rdzenie CPU
#SBATCH --mem=16gb                      # 16 GB RAM na zadanie
#SBATCH --time=0-08:00:00               # 8 godzin na jedno zadanie
#SBATCH --job-name=sc_erl_mujoco        # Nazwa zadania
#SBATCH -p lem-gpu                      # Partycja GPU
#SBATCH --gres=gpu:hopper:1             # 1 karta GPU
#SBATCH --output=logs/slurm-%A_%a.out   # Log standardowy
#SBATCH --error=logs/slurm-%A_%a.err    # Plik błędów
#SBATCH --mail-type=FAIL                # Email przy awarii

# ==========================================
# Dynamiczny wybór środowiska
# ==========================================
ENV="${TARGET_ENV:-Ant-v5}"

# ==========================================
# Definicja algorytmów i seedów (8 * 5 = 40 kombinacji)
# ==========================================
ALGORITHMS=(
    "sc_erl:dropout"       # 0..4
    "sc_erl:ensemble"      # 5..9
    "sc_erl:evidential"    # 10..14
    "sc_erl:random"        # 15..19
    "td3:"                 # 20..24
    "erl:"                 # 25..29
    "ddpg:"                # 30..34
    "ppo:"                 # 35..39
)

SEEDS=(0 1 2 3 4)
N_SEEDS=${#SEEDS[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

ALGO_IDX=$(( TASK_ID / N_SEEDS ))
SEED_IDX=$(( TASK_ID % N_SEEDS ))

ALGO_MODE="${ALGORITHMS[$ALGO_IDX]}"
ALGO="${ALGO_MODE%%:*}"          
SURROGATE_MODE="${ALGO_MODE##*:}" 

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
# Konfiguracja środowiska uruchomieniowego
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

# Ładowanie modułów systemowych
source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0

# Przejście do katalogu projektu
PROJECT_DIR="/home/jakgil6519/workspace/ue_evo_rl"
cd "${PROJECT_DIR}" || { echo "ERROR: Nie można przejść do ${PROJECT_DIR}"; exit 1; }

# Aktywacja lokalnego środowiska .venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "ERROR: Katalog .venv nie istnieje! Zbuduj środowisko na UI."
    exit 1
fi

# ==========================================
# Bezpieczna konfiguracja WandB (Pełny tryb OFFLINE)
# ==========================================
export WANDB_API_KEY="TUTAJ_WKLEJ_SWOJ_KLUCZ_WANDB"
export WANDB_MODE="offline"

# Definiujemy czysty podkatalog na logi lokalne, by uniknąć problemów z NFS
export WANDB_DIR="${PROJECT_DIR}/wandb_logs"

# Tworzenie niezbędnych struktur katalogów
mkdir -p logs
mkdir -p "${WANDB_DIR}"

# ==========================================
# Uruchomienie eksperymentu
# ==========================================
echo "Uruchamiam python entry_point.py..."

python entry_point.py \
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
# Sync wandb po zakończeniu treningu
# ==========================================
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "Trening ukończony. Syncuję logi wandb..."
    for d in "${WANDB_DIR}/wandb/offline-run-"*; do
        [[ -d "$d" ]] && wandb sync "$d"
    done
    echo "Zadanie ${RUN_NAME} ukończone pomyślnie."
else
    echo "ERROR: Zadanie ${RUN_NAME} zakończyło się błędem (exit code: ${EXIT_CODE})"
    exit ${EXIT_CODE}
fi