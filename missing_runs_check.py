from datetime import datetime
import wandb
from itertools import product

api = wandb.Api()

seeds = [0, 1, 2, 3, 4]
mujoco_envs = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5", "Ant-v5", "Swimmer-v5"]
dmc_envs = [
    "dm_control/dog-stand-v0",
    "dm_control/dog-walk-v0",
    "dm_control/dog-trot-v0",
    "dm_control/dog-run-v0",
    "dm_control/dog-fetch-v0",
]
all_envs = mujoco_envs + dmc_envs

sc_erl_modes = ["evidential", "dropout", "ensemble", "random"]
baselines = ["td3", "erl", "ppo", "ddpg", "sac", "crossq", "wimle"]

runs = api.runs("evo_rl/ue_evo_rl_3")
print(f"Runów na wandb: {len(runs)}")

existing_baselines = set()
existing_sc_erl = set()

for run in runs:
    algo = run.config.get("name")
    env = (run.config.get("env") or {}).get("id")
    seed = run.config.get("seed")
    if not (algo and env and seed is not None):
        continue
    seed = int(seed)

    if algo == "sc_erl":
        surrogate = (run.config.get("surrogate") or {}).get("mode")
        if surrogate:
            existing_sc_erl.add((surrogate, env, seed))
    else:
        existing_baselines.add((algo, env, seed))

missing = []

for mode, env, seed in product(sc_erl_modes, all_envs, seeds):
    if (mode, env, seed) not in existing_sc_erl:
        missing.append((f"sc_erl_{mode}", env, seed))

for algo, env, seed in product(baselines, all_envs, seeds):
    if (algo, env, seed) not in existing_baselines:
        missing.append((algo, env, seed))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"missing_runs_{timestamp}.txt"

with open(output_file, "w") as f:
    f.write(f"Sprawdzono: {timestamp}\n")
    f.write(f"Runów na wandb: {len(runs)}\n")
    f.write(f"Brakuje {len(missing)} / 550 runów\n")
    f.write("=" * 50 + "\n")

    current_algo = None
    for algo, env, seed in sorted(missing):
        if algo != current_algo:
            f.write(f"\n[{algo}]\n")
            current_algo = algo
        f.write(f"  {env} | seed={seed}\n")

print(f"Brakuje {len(missing)} / 550 runów")
print(f"Zapisano do: {output_file}")
