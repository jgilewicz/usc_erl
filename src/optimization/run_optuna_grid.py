import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def resolve_algorithm_override(algo: str, env_id: str) -> str:
    env_slug = env_id.replace("/", "_").replace(":", "_")
    env_specific_path = os.path.join(
        PROJECT_ROOT,
        "configs",
        "algorithm",
        algo,
        f"{algo}_{env_slug}.yaml",
    )

    if os.path.exists(env_specific_path):
        return f"{algo}/{algo}_{env_slug}"

    return algo


def main():
    algorithms = [
        "sc_erl",
        "sc_erl_dropout",
        "sc_erl_ensemble",
        "sc_erl_evidential",
    ]
    environments = [
        "FetchPush-v4",
        "FetchSlide-v4",
        "FetchPickAndPlace-v4",
    ]

    # Capture any additional Hydra parameters or overrides passed via command line
    extra_args = sys.argv[1:]

    for algo in algorithms:
        for env in environments:
            print("=" * 60)
            print(f"STARTING OPTUNA TUNING: {algo.upper()} on {env}")
            print("=" * 60)

            base_algo = "sc_erl" if algo.startswith("sc_erl") else algo
            algorithm_override = resolve_algorithm_override(base_algo, env)

            cmd = [
                "uv",
                "run",
                "python",
                "src/optimization/optuna_tune.py",
                f"algorithm={algorithm_override}",
                f"env.id={env}",
                f"eval_env.id={env}",
                f"name={algo}",
            ] + extra_args

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Tuning run failed for {algo} on {env}: {e}")
            except KeyboardInterrupt:
                print("\nTuning grid interrupted by user. Exiting.")
                return


if __name__ == "__main__":
    main()
