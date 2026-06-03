import os
import re
import pandas as pd
import wandb
from omegaconf import OmegaConf


def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def determine_method_from_run(run):
    cfg = run.config
    name = cfg.get("name")
    if name == "sc_erl":
        surr_mode = cfg.get("surrogate", {}).get("mode")
        if not surr_mode:
            surr_mode = cfg.get("surrogate.mode")
        if not surr_mode:
            surr_mode = cfg.get("surrogate/mode")
        if surr_mode:
            return f"sc_erl_{surr_mode.lower()}"
        return "sc_erl_random"
    elif name:
        return name.lower()
    run_name = run.name
    for prefix in [
        "sc_erl_evidential",
        "sc_erl_dropout",
        "sc_erl_ensemble",
        "sc_erl_random",
        "td3",
        "erl",
        "ppo",
        "ddpg",
    ]:
        if run_name.startswith(prefix):
            return prefix
    tags = [t.lower() for t in run.tags]
    if "sc_erl" in tags:
        if "dropout" in tags:
            return "sc_erl_dropout"
        if "ensemble" in tags:
            return "sc_erl_ensemble"
        if "random" in tags:
            return "sc_erl_random"
        if "sc_erl_evidential" in tags:
            return "sc_erl_evidential"
        return "sc_erl_random"
    for baseline in ["td3", "erl", "ppo", "ddpg"]:
        if baseline in tags:
            return baseline
    return None


def determine_env_and_seed(run):
    cfg = run.config
    env_id = cfg.get("env", {}).get("id")
    if not env_id:
        env_id = cfg.get("env.id")
    if not env_id:
        env_id = cfg.get("env/id")
    seed = cfg.get("seed")
    run_name = run.name
    if not env_id or seed is None:
        match = re.match(
            "^(?:sc_erl_evidential|sc_erl_dropout|sc_erl_ensemble|sc_erl_random|td3|erl|ppo|ddpg)_([A-Za-z0-9\\-_]+)_seed(\\d+)",
            run_name,
        )
        if match:
            if not env_id:
                env_id = match.group(1)
            if seed is None:
                seed = int(match.group(2))
    return (env_id, seed)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(script_dir), "configs", "download.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    config = OmegaConf.load(config_path)
    entity = config.wandb.entity
    project = config.wandb.project
    allowed_algorithms = set(config.algorithms)
    allowed_environments = set(config.environments)
    print("Initializing WandB API...")
    api = wandb.Api()
    project_path = f"{entity}/{project}"
    print(f"Fetching runs from project '{project_path}'...")
    runs = api.runs(project_path)
    print(f"Total runs found: {len(runs)}")
    METRICS = [
        "eval_reward",
        "best_population_fitness",
        "avg_population_fitness",
        "uncertainty_mean",
        "uncertainty_max",
        "uncertainty_threshold",
        "surrogate_ratio",
        "critic_loss",
        "total_steps",
        "generation",
        "raw_sigma_mean",
        "raw_sigma_max",
    ]
    for metric in METRICS + ["summary"]:
        os.makedirs(os.path.join(script_dir, metric), exist_ok=True)
    env_groups = {}
    for run in runs:
        env_id, seed = determine_env_and_seed(run)
        method = determine_method_from_run(run)
        if not env_id or seed is None or (not method):
            print(
                f"Skipping run '{run.name}' (Unable to determine Env ID, Seed or Method)."
            )
            continue
        if env_id not in allowed_environments:
            continue
        if method not in allowed_algorithms:
            continue
        if env_id not in env_groups:
            env_groups[env_id] = []
        env_groups[env_id].append(
            {"run": run, "seed": seed, "method": method, "env_id": env_id}
        )
    print(
        f"Grouped runs into {len(env_groups)} environment(s): {list(env_groups.keys())}"
    )
    for env_id, group in env_groups.items():
        print(f"\nProcessing environment: {env_id} ({len(group)} runs)...")
        metric_dfs = {metric: [] for metric in METRICS}
        for item in group:
            run = item["run"]
            seed = item["seed"]
            method = item["method"]
            print(f"  Downloading full history for run '{run.name}'...")
            try:
                run_df = run.history(samples=10000)
                if run_df.empty or "_step" not in run_df.columns:
                    continue
                for metric in METRICS:
                    if metric not in run_df.columns:
                        continue
                    history = run_df[["_step", metric]].dropna().copy()
                    history["_step"] = pd.to_numeric(history["_step"], errors="coerce")
                    history[metric] = pd.to_numeric(history[metric], errors="coerce")
                    history = history.dropna(subset=["_step", metric])
                    if history.empty:
                        continue
                    history = history.groupby("_step", as_index=False)[metric].mean()
                    col_name = f"{method}_{env_id}_seed{seed} - {metric}"
                    history = history.rename(
                        columns={"_step": "Step", metric: col_name}
                    )
                    metric_dfs[metric].append(history)
            except Exception as e:
                print(f"    Error downloading history for run {run.name}: {e}")
        for metric in METRICS:
            data_frames = metric_dfs[metric]
            if data_frames:
                print(f"  Merging and saving CSV for '{metric}'...")
                merged_df = data_frames[0]
                for next_df in data_frames[1:]:
                    merged_df = pd.merge(merged_df, next_df, on="Step", how="outer")
                merged_df = merged_df.sort_values("Step").reset_index(drop=True)
                csv_path = os.path.join(script_dir, metric, f"{env_id}.csv")
                merged_df.to_csv(csv_path, index=False)
                print(f"    Saved {csv_path}")
            else:
                print(f"  No data found for metric '{metric}' in {env_id}.")
        print("  Compiling summary table...")
        summary_rows = []
        all_config_keys = set()
        all_summary_keys = set()
        preprocessed_runs = []
        for item in group:
            run = item["run"]
            flat_config = flatten_dict(
                {k: v for k, v in run.config.items() if not k.startswith("_")}
            )
            all_config_keys.update(flat_config.keys())
            flat_summary = {
                k: v for k, v in run.summary._json_dict.items() if not k.startswith("_")
            }
            all_summary_keys.update(flat_summary.keys())
            preprocessed_runs.append(
                {"run": run, "flat_config": flat_config, "flat_summary": flat_summary}
            )
        sorted_config_keys = sorted(list(all_config_keys))
        sorted_summary_keys = sorted(list(all_summary_keys))
        for item in preprocessed_runs:
            run = item["run"]
            flat_cfg = item["flat_config"]
            flat_sum = item["flat_summary"]
            row = {
                "Name": run.name,
                "State": run.state,
                "Notes": run.notes if run.notes else "-",
                "User": run.user.username if run.user else "",
                "Tags": str(run.tags),
                "Created": run.created_at,
                "Runtime": str(run.summary.get("_runtime", "")),
                "Sweep": run.sweep.id if run.sweep else "",
            }
            for k in sorted_config_keys:
                row[k] = flat_cfg.get(k, "")
            for k in sorted_summary_keys:
                row[k] = flat_sum.get(k, "")
            summary_rows.append(row)
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(script_dir, "summary", f"{env_id}.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"    Saved {summary_path}")
    print("\nAll data downloads and formatting successfully completed!")


if __name__ == "__main__":
    main()
