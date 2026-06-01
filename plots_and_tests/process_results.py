import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure Seaborn and Matplotlib parameters to match strict scientific/academic publication standards
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)

# Globally consistent visualization parameters
METHOD_COLORS = {
    "sc_erl_dropout": "#0173b2",
    "sc_erl_ensemble": "#029e73",
    "erl": "#de8f05",
    "ppo": "#d55e00",
    "td3": "#cc78bc",
    "ddpg": "#56b4e9",
    "sc_erl_random": "#949494",
}

METHOD_LABELS = {
    "ppo": "PPO (Baseline)",
    "td3": "TD3 (Baseline)",
    "ddpg": "DDPG (Baseline)",
    "erl": "ERL (Baseline)",
    "sc_erl_ensemble": "SC-ERL (Ensemble) [Ours]",
    "sc_erl_dropout": "SC-ERL (Dropout) [Ours]",
    "sc_erl_random": "SC-ERL (Random)",
}


def parse_column_header(col_name, env_id):
    """
    Parses complex telemetry column names to extract method name, seed, and metric type.
    Example input: 'sc_erl_ensemble_FetchReach-v4_seed2 - eval_reward'
    Example output: ('sc_erl_ensemble', 2, 'eval_reward')
    """
    pattern = f"^([a-z_0-9]+)_{re.escape(env_id)}_seed(\\d+)\\s*-\\s*([a-z_]+)$"
    match = re.match(pattern, col_name)
    if match:
        return match.group(1), int(match.group(2)), match.group(3)
    return None, None, None


def smooth_series(series, window=7):
    """
    Applies a rolling-average window smoothing to clean up visual noise
    in telemetry charts while preserving historical trends.
    """
    s = pd.Series(series)
    return s.rolling(window=window, min_periods=1).mean().values


def load_environment_data(env_id, base_dir="results"):
    """
    Loads, cleans, and merges CSV telemetry outputs collected during training.

    Processing Steps:
    1. Reads telemetry metrics from their respective subfolders.
    2. Automatically replaces infinity strings ('Infinity', 'inf') with NaNs.
    3. Groups the metrics dynamically by method and random seed.
    4. Handles timeline alignment:
       - Since non-evolutionary algorithms (PPO, TD3, DDPG) record metrics on a per-step basis,
         their 'total_steps' maps directly to the step index.
       - Evolutionary methods (ERL variants) might report steps differently depending on generation cycles.
         We compute a robust average step-to-frame alignment across active seeds using linear interpolation.
    """
    metrics = [
        "total_steps",
        "eval_reward",
        "best_population_fitness",
        "avg_population_fitness",
        "uncertainty_mean",
        "uncertainty_max",
        "uncertainty_threshold",
        "generation",
    ]
    run_data = {}

    # Step 1: Parse and load files for each recorded metric
    for metric in metrics:
        file_path = os.path.join(base_dir, metric, f"{env_id}.csv")
        if not os.path.exists(file_path):
            continue
        df = pd.read_csv(file_path)
        df = df.replace(["Infinity", "inf", "inf.0"], np.nan)

        for col in df.columns:
            if col == "Step" or col.endswith("__MIN") or col.endswith("__MAX"):
                continue

            method, seed, parsed_metric = parse_column_header(col, env_id)
            if method is None or parsed_metric != metric:
                continue

            if method not in run_data:
                run_data[method] = {}
            if seed not in run_data[method]:
                run_data[method][seed] = []

            sub_df = df[["Step", col]].copy()
            sub_df[col] = pd.to_numeric(sub_df[col], errors="coerce")
            sub_df = sub_df.dropna()
            sub_df = sub_df.rename(columns={col: metric})
            run_data[method][seed].append(sub_df)

    # Step 2: Merge distinct metric frames on Step index
    merged_data = {}
    for method in run_data:
        merged_data[method] = {}
        for seed in run_data[method]:
            dfs = run_data[method][seed]
            if not dfs:
                continue
            merged_df = dfs[0]
            for next_df in dfs[1:]:
                merged_df = pd.merge(merged_df, next_df, on="Step", how="outer")
            merged_df = merged_df.sort_values("Step").reset_index(drop=True)
            merged_data[method][seed] = merged_df

    # Step 3: Align timeline indices (total_steps mapping)
    for method in list(merged_data.keys()):
        is_evo = method in ["erl", "sc_erl_ensemble", "sc_erl_dropout", "sc_erl_random"]
        if not is_evo:
            # Direct mapping for standard step-based RL agents
            for seed in merged_data[method]:
                df = merged_data[method][seed]
                if "total_steps" not in df.columns or df["total_steps"].isna().all():
                    df["total_steps"] = df["Step"]
            continue

        # Aligns steps for evolutionary algorithms via linear interpolation across all active seeds
        seeds_with_steps = [
            s
            for s, df in merged_data[method].items()
            if "total_steps" in df.columns and (not df["total_steps"].isna().all())
        ]
        if seeds_with_steps:
            all_pairs = []
            for s in seeds_with_steps:
                temp = merged_data[method][s][["Step", "total_steps"]].dropna()
                all_pairs.append(temp)
            if all_pairs:
                combined_steps = (
                    pd.concat(all_pairs).groupby("Step")["total_steps"].mean().to_dict()
                )
                for seed in merged_data[method]:
                    df = merged_data[method][seed]
                    if (
                        "total_steps" not in df.columns
                        or df["total_steps"].isna().all()
                    ):
                        df["total_steps"] = df["Step"].map(combined_steps)
                        df["total_steps"] = (
                            df["total_steps"]
                            .interpolate(method="linear")
                            .ffill()
                            .bfill()
                        )
    return merged_data


def generate_sample_efficiency_plot(env_id, merged_data, out_path):
    """
    Generates a high-quality sample efficiency comparison plot.
    Plots the true evaluation reward as a function of environmental interaction steps.
    Includes shaded area representing standard deviation across seeds.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.grid(True, which="both", color="#f2f2f2", linestyle="-", linewidth=0.5)

    # Detect absolute maximum horizontal step index across all active configurations
    max_steps_all = 0
    for method in merged_data:
        for seed, df in merged_data[method].items():
            if "total_steps" in df.columns:
                m = df["total_steps"].max()
                if m > max_steps_all:
                    max_steps_all = m
    if max_steps_all == 0:
        max_steps_all = 2000000

    step_grid = np.linspace(0, max_steps_all, 200)

    for method in sorted(merged_data.keys()):
        color = METHOD_COLORS.get(method, "#333333")
        label = METHOD_LABELS.get(method, method)
        is_proposed = method in ["sc_erl_ensemble", "sc_erl_dropout"]

        # Design aesthetics: thick solid lines for ours, thin dotted lines for baselines
        linewidth = 1.5 if is_proposed else 1.0
        line_alpha = 1.0 if is_proposed else 0.6
        linestyle = "-" if is_proposed else ":"
        fill_alpha = 0.12 if is_proposed else 0.05

        interpolated_ys = []
        for seed, df in merged_data[method].items():
            if "total_steps" not in df.columns:
                continue

            # Reward selection logic: use the unified evaluation reward for all methods.
            if "eval_reward" in df.columns and (not df["eval_reward"].isna().all()):
                y_metric = "eval_reward"
            elif "best_population_fitness" in df.columns and (
                not df["best_population_fitness"].isna().all()
            ):
                y_metric = "best_population_fitness"
            else:
                continue

            temp_df = df[["total_steps", y_metric]].dropna()
            if temp_df.empty:
                continue
            temp_df = temp_df.sort_values("total_steps")

            # Map original metrics to standard horizontal steps grid using linear interpolation
            interp_y = np.interp(
                step_grid, temp_df["total_steps"].values, temp_df[y_metric].values
            )
            interpolated_ys.append(interp_y)

        if not interpolated_ys:
            continue

        interpolated_ys = np.array(interpolated_ys)
        mean_y_smooth = smooth_series(np.mean(interpolated_ys, axis=0), window=5)
        std_y_smooth = smooth_series(np.std(interpolated_ys, axis=0), window=5)

        ax.plot(
            step_grid,
            mean_y_smooth,
            label=label,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=line_alpha,
        )
        ax.fill_between(
            step_grid,
            mean_y_smooth - std_y_smooth,
            mean_y_smooth + std_y_smooth,
            color=color,
            alpha=fill_alpha,
        )

    ax.set_title(
        f"Sample Efficiency Comparison - {env_id}",
        fontsize=13,
        pad=15,
        fontweight="bold",
    )
    ax.set_xlabel("Environmental Interaction Steps", labelpad=10)
    ax.set_ylabel("True Evaluation Reward", labelpad=10)

    # Format labels (e.g., 1.0M, 500k)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, p: f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x / 1e3:.0f}k"
        )
    )
    sns.despine(ax=ax, top=True, right=True)
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.3,
        edgecolor="#f2f2f2",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def generate_surrogate_analysis_plot(env_id, merged_data, out_path):
    """
    Generates a dual-axis subplotted chart analyzing the Surrogate Controller metrics.
    Compares epistemic uncertainty estimations (left axis) with average population fitness (right axis)
    across the active generations of training.

    Crucial fix implemented: Computes consistent absolute Y-limits for the population fitness axis (Y2)
    across both methods, allowing direct visual comparison without misleading scale offsets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)
    surrogate_methods = ["sc_erl_ensemble", "sc_erl_dropout"]
    titles = ["SC-ERL (Ensemble Method) [Ours]", "SC-ERL (Dropout Method) [Ours]"]

    # Determine globally synchronized Y2 axes boundaries across both ensemble and dropout plots
    all_fit_vals = []
    for method in surrogate_methods:
        if method in merged_data:
            for seed, df in merged_data[method].items():
                if "avg_population_fitness" in df.columns:
                    all_fit_vals.extend(df["avg_population_fitness"].dropna().values)
    y2_min = min(all_fit_vals) if all_fit_vals else -100
    y2_max = max(all_fit_vals) if all_fit_vals else 100

    for i, method in enumerate(surrogate_methods):
        ax = axes[i]
        ax.grid(True, which="both", color="#f2f2f2", linestyle="-", linewidth=0.5)
        if method not in merged_data:
            ax.text(0.5, 0.5, f"No data for {method}", ha="center", va="center")
            continue

        color = METHOD_COLORS[method]
        title = titles[i]
        gen_grid = np.linspace(0, 1400, 141)
        mean_uncertainties = []
        avg_fitnesses = []

        for seed, df in merged_data[method].items():
            required = ["generation", "uncertainty_mean", "avg_population_fitness"]
            if not all((col in df.columns for col in required)):
                continue
            temp_df = (
                df[required]
                .dropna(subset=["generation", "uncertainty_mean"])
                .sort_values("generation")
            )
            if temp_df.empty:
                continue
            mean_uncertainties.append(
                np.interp(
                    gen_grid,
                    temp_df["generation"].values,
                    temp_df["uncertainty_mean"].values,
                )
            )
            avg_fitnesses.append(
                np.interp(
                    gen_grid,
                    temp_df["generation"].values,
                    temp_df["avg_population_fitness"].values,
                )
            )

        if not mean_uncertainties:
            ax.text(0.5, 0.5, f"Empty dataset for {method}", ha="center", va="center")
            continue

        avg_mean_u_smooth = smooth_series(np.mean(mean_uncertainties, axis=0), window=7)
        avg_fit_smooth = smooth_series(np.mean(avg_fitnesses, axis=0), window=7)

        # Left axis configuration: Epistemic Uncertainty Estimation
        ax.set_xlabel("Generations (Training Progress)", labelpad=10)
        ax.set_ylabel("Uncertainty Estimation Value", color=color, labelpad=10)
        ax.tick_params(axis="y", labelcolor=color)
        (line_mean,) = ax.plot(
            gen_grid,
            avg_mean_u_smooth,
            color=color,
            linewidth=1.2,
            label="Uncertainty Mean",
        )

        # Right axis configuration (twinx): Standardized Average Population Fitness
        ax2 = ax.twinx()
        ax2.set_ylabel("Average Population Fitness", color="#de8f05", labelpad=10)
        ax2.tick_params(axis="y", labelcolor="#de8f05")
        ax2.set_ylim(y2_min - 0.05 * abs(y2_min), y2_max + 0.05 * abs(y2_max))
        ax2.grid(False)
        (line_fit,) = ax2.plot(
            gen_grid,
            avg_fit_smooth,
            color="#de8f05",
            linewidth=1.0,
            linestyle="-.",
            label="Avg Pop Fitness",
        )

        lines = [line_mean, line_fit]
        labels = [line.get_label() for line in lines]
        ax.legend(
            lines,
            labels,
            loc="upper left",
            frameon=True,
            facecolor="white",
            framealpha=0.3,
            edgecolor="#f2f2f2",
        )
        ax.set_title(title, fontsize=12, pad=12, fontweight="bold")
        sns.despine(ax=ax, top=True, left=False, right=False)

    plt.suptitle(
        f"Surrogate Controller Uncertainty & Fitness Analysis - {env_id}",
        fontsize=14,
        y=0.98,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def generate_summary_table(env_id, base_dir="results", out_base_dir=None):
    """
    Compiles key statistics (mean and standard deviation) for true evaluation reward,
    best population fitness, average population fitness, training time [h], and average surrogate uncertainty.
    Exports the compiled data using modern Pandas 2.x styling mechanics to LaTeX table format.
    """
    summary_path = os.path.join(base_dir, "summary", f"{env_id}.csv")
    if not os.path.exists(summary_path):
        return
    df = pd.read_csv(summary_path)
    parsed_rows = []

    for idx, row in df.iterrows():
        name = row["Name"]
        pattern = f"^([a-z_0-9]+)_{re.escape(env_id)}_seed(\\d+)$"
        match = re.match(pattern, name)
        if match:
            method = match.group(1)
            if "evidential" in method:
                continue
            seed = int(match.group(2))

            # 1. Test Reward
            test_reward = row.get("eval_reward", np.nan)

            # 2. Best Fitness
            best_pop = row.get("best_population_fitness", np.nan)

            # 3. Average Population Fitness
            avg_pop = row.get("avg_population_fitness", np.nan)

            # 4. Training Time [h]
            runtime_raw = row.get("Runtime", np.nan)
            try:
                runtime_val = (
                    float(runtime_raw)
                    if pd.notna(runtime_raw) and runtime_raw != ""
                    else np.nan
                )
            except ValueError:
                runtime_val = np.nan
            runtime_h = runtime_val / 3600.0 if pd.notna(runtime_val) else np.nan

            # 5. Avg Surrogate Uncertainty
            avg_uncertainty = row.get("uncertainty_mean", np.nan)

            parsed_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "test_reward": pd.to_numeric(test_reward, errors="coerce"),
                    "best_pop": pd.to_numeric(best_pop, errors="coerce"),
                    "avg_pop": pd.to_numeric(avg_pop, errors="coerce"),
                    "runtime_h": pd.to_numeric(runtime_h, errors="coerce"),
                    "avg_uncertainty": pd.to_numeric(avg_uncertainty, errors="coerce"),
                }
            )

    parsed_df = pd.DataFrame(parsed_rows)
    if parsed_df.empty:
        return

    # Group by method and calculate descriptive statistics
    summary_stats = parsed_df.groupby("method").agg(
        {
            "test_reward": ["mean", "std"],
            "best_pop": ["mean", "std"],
            "avg_pop": ["mean", "std"],
            "runtime_h": ["mean", "std"],
            "avg_uncertainty": ["mean", "std"],
        }
    )
    summary_stats.columns = [f"{col[0]}_{col[1]}" for col in summary_stats.columns]
    summary_stats = summary_stats.reset_index()

    # Format helper
    def fmt(mean, std):
        if pd.isna(mean):
            return "-"
        if pd.isna(std):
            return f"${mean:.2f} \\pm 0.00$"
        return f"${mean:.2f} \\pm {std:.2f}$"

    polish_labels = {
        "ppo": "PPO (Baseline)",
        "td3": "TD3 (Baseline)",
        "ddpg": "DDPG (Baseline)",
        "erl": "ERL (Bez surogata)",
        "sc_erl_random": "SC-ERL (Random)",
        "sc_erl_ensemble": "SC-ERL (Ensemble)",
        "sc_erl_dropout": "SC-ERL (Dropout)",
    }

    csv_rows = []
    method_order = [
        "ppo",
        "td3",
        "ddpg",
        "erl",
        "sc_erl_random",
        "sc_erl_ensemble",
        "sc_erl_dropout",
    ]
    for method in method_order:
        row_data = summary_stats[summary_stats["method"] == method]
        if row_data.empty:
            continue
        row_data = row_data.iloc[0]
        label = polish_labels.get(method, method)

        csv_rows.append(
            {
                "Architektura": label,
                "Nagroda testowa (RL)": fmt(
                    row_data["test_reward_mean"], row_data["test_reward_std"]
                ),
                "Najlepsze przystosowanie": fmt(
                    row_data["best_pop_mean"], row_data["best_pop_std"]
                ),
                "Średnie przystosowanie populacji": fmt(
                    row_data["avg_pop_mean"], row_data["avg_pop_std"]
                ),
                "Czas treningu [h]": fmt(
                    row_data["runtime_h_mean"], row_data["runtime_h_std"]
                ),
                "Średnia niepewność surogata": fmt(
                    row_data["avg_uncertainty_mean"], row_data["avg_uncertainty_std"]
                ),
            }
        )

    csv_table = pd.DataFrame(csv_rows)
    out_dir = os.path.join(
        out_base_dir if out_base_dir is not None else base_dir, env_id
    )
    os.makedirs(out_dir, exist_ok=True)
    latex_out = os.path.join(out_dir, f"{env_id}_summary_table.tex")

    with open(latex_out, "w") as f:
        f.write(
            csv_table.to_latex(
                index=False,
                column_format="lccccc",
                caption=f"Nagrody ewaluacyjne, przystosowanie populacji, czas obliczeń oraz niepewność surogata dla środowiska {env_id}.",
                label=f"tab:summary_{env_id}",
                escape=False,
            )
        )


def generate_critic_correlation_plot(env_id, base_dir="results", out_base_dir=None):
    """
    Computes and visualizes Pearson and Spearman rank correlation coefficients
    between Critic Temporal Difference (TD) Loss and Critic Epistemic Uncertainty.
    This analysis validates whether the surrogate uncertainty metrics act as an accurate
    proxy for predictive actor-critic model inaccuracy.
    """
    df_loss_path = os.path.join(base_dir, "critic_loss", f"{env_id}.csv")
    df_mean_path = os.path.join(base_dir, "uncertainty_mean", f"{env_id}.csv")
    if not os.path.exists(df_loss_path) or not os.path.exists(df_mean_path):
        return

    df_loss = pd.read_csv(df_loss_path)
    df_mean = pd.read_csv(df_mean_path)
    methods = ["sc_erl_ensemble", "sc_erl_dropout"]
    valid_methods_data = {}

    for method in methods:
        seeds = []
        for col in df_loss.columns:
            match = re.match(
                rf"^{method}_{re.escape(env_id)}_seed(\d+) - critic_loss$", col
            )
            if match:
                seeds.append(int(match.group(1)))

        all_x = []
        all_y = []
        for seed in seeds:
            col_loss = f"{method}_{env_id}_seed{seed} - critic_loss"
            col_mean = f"{method}_{env_id}_seed{seed} - uncertainty_mean"
            if col_loss not in df_loss.columns or col_mean not in df_mean.columns:
                continue

            sub_loss = df_loss[["Step", col_loss]].copy()
            sub_mean = df_mean[["Step", col_mean]].copy()
            sub_loss[col_loss] = pd.to_numeric(
                sub_loss[col_loss].replace("Infinity", np.nan), errors="coerce"
            )
            sub_mean[col_mean] = pd.to_numeric(
                sub_mean[col_mean].replace("Infinity", np.nan), errors="coerce"
            )
            sub_loss = sub_loss.dropna()
            sub_mean = sub_mean.dropna()

            merged = pd.merge(sub_loss, sub_mean, on="Step", how="inner")
            if not merged.empty:
                all_x.extend(merged[col_loss].values)
                all_y.extend(merged[col_mean].values)

        if all_x and all_y:
            valid_methods_data[method] = (np.array(all_x), np.array(all_y))

    if not valid_methods_data:
        return

    n_plots = len(valid_methods_data)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5), squeeze=False)
    correlations = []

    for i, (method, (x, y)) in enumerate(valid_methods_data.items()):
        ax = axes[0][i]
        ax.grid(True, which="both", color="#f2f2f2", linestyle="-", linewidth=0.5)

        # Calculate Pearson and Spearman correlation statistics
        p_corr = np.corrcoef(x, y)[0, 1]
        x_rank = pd.Series(x).rank()
        y_rank = pd.Series(y).rank()
        s_corr = x_rank.corr(y_rank, method="pearson")

        label = "SC-ERL (Ensemble)" if "ensemble" in method else "SC-ERL (Dropout)"
        correlations.append(
            {
                "Algorithm/Method": label,
                "Pearson Correlation": f"{p_corr:.4f}",
                "Spearman Correlation": f"{s_corr:.4f}",
                "Sample Size (N)": len(x),
            }
        )

        color = METHOD_COLORS.get(method, "#000000")
        ax.scatter(
            x, y, color=color, alpha=0.4, s=20, edgecolor="none", label="Step Metrics"
        )

        # Draw linear trendline fit
        slope, intercept = np.polyfit(x, y, 1)
        x_grid = np.linspace(min(x), max(x), 100)
        ax.plot(
            x_grid,
            slope * x_grid + intercept,
            color="#333333",
            linestyle="--",
            linewidth=1.0,
            label="Trendline",
        )

        ax.set_title(
            f"{label}\n(Pearson r = {p_corr:.3f})",
            fontsize=11,
            pad=10,
            fontweight="bold",
        )
        ax.set_xlabel("Critic TD Loss", labelpad=8)
        ax.set_ylabel("Critic Epistemic Uncertainty", labelpad=8)
        ax.legend(
            loc="upper right",
            frameon=True,
            facecolor="white",
            framealpha=0.3,
            edgecolor="#f2f2f2",
        )
        sns.despine(ax=ax, top=True, right=True)

    plt.suptitle(
        f"Critic Epistemic Uncertainty vs. TD Loss Correlation - {env_id}",
        fontsize=13,
        y=0.98,
        fontweight="bold",
    )
    plt.tight_layout()
    out_dir = os.path.join(
        out_base_dir if out_base_dir is not None else base_dir, env_id
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{env_id}_critic_correlation.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    corr_df = pd.DataFrame(correlations)
    latex_out = os.path.join(out_dir, f"{env_id}_critic_correlation.tex")
    with open(latex_out, "w") as f:
        f.write(
            corr_df.style.hide(axis="index").to_latex(
                column_format="lccc",
                caption=f"Correlation analysis between Critic TD Loss and Critic Epistemic Uncertainty on {env_id}.",
                label=f"tab:critic_corr_{env_id}",
            )
        )


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = script_dir
    project_root = os.path.dirname(script_dir)
    output_base_dir = os.path.join(project_root, "results")

    eval_reward_dir = os.path.join(base_dir, "eval_reward")
    if not os.path.exists(eval_reward_dir):
        return

    env_files = glob.glob(os.path.join(eval_reward_dir, "*.csv"))
    if not env_files:
        return

    environments = [os.path.basename(f).replace(".csv", "") for f in env_files]
    for env_id in environments:
        # Load and align telemetry dataset across active seeds and runs
        merged_data = load_environment_data(env_id, base_dir)
        out_dir = os.path.join(output_base_dir, env_id)
        os.makedirs(out_dir, exist_ok=True)

        # Plot evaluation results, sample efficiencies, surrogate analysis subplots, and LaTeX sheets
        se_plot_path = os.path.join(out_dir, f"{env_id}_sample_efficiency.png")
        generate_sample_efficiency_plot(env_id, merged_data, se_plot_path)
        sa_plot_path = os.path.join(out_dir, f"{env_id}_surrogate_analysis.png")
        generate_surrogate_analysis_plot(env_id, merged_data, sa_plot_path)
        generate_summary_table(env_id, base_dir, out_base_dir=output_base_dir)
        generate_critic_correlation_plot(env_id, base_dir, out_base_dir=output_base_dir)


if __name__ == "__main__":
    main()
