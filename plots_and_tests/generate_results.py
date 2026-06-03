import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats

# ==========================================
# CONSTANTS & STYLING CONFIGURATION
# ==========================================
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

METHOD_COLORS = {
    "sc_erl_dropout": "#0173b2",
    "sc_erl_ensemble": "#029e73",
    "sc_erl_evidential": "#8c564b",
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
    "sc_erl_evidential": "SC-ERL (Evidential) [Ours]",
    "sc_erl_random": "SC-ERL (Random)",
}

PROPOSED_METHODS = ["sc_erl_ensemble", "sc_erl_dropout", "sc_erl_evidential"]


# ==========================================
# DATA LOADING & CLEANING
# ==========================================
def parse_column_header(col_name, env_id):
    pattern = f"^([a-z_0-9]+)_{re.escape(env_id)}_seed(\\d+)\\s*-\\s*([a-z_]+)$"
    match = re.match(pattern, col_name)
    if match:
        return match.group(1), int(match.group(2)), match.group(3)
    return None, None, None


def smooth_series(series, window=7):
    s = pd.Series(series)
    return s.rolling(window=window, min_periods=1).mean().values


def load_environment_data(env_id, base_dir="."):
    metrics = [
        "total_steps",
        "eval_reward",
        "best_population_fitness",
        "avg_population_fitness",
        "uncertainty_mean",
        "uncertainty_max",
        "uncertainty_threshold",
        "surrogate_ratio",
        "generation",
        "raw_sigma_mean",
        "raw_sigma_max",
    ]
    run_data = {}

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

    for method in list(merged_data.keys()):
        is_evo = method in ["erl", "sc_erl_ensemble", "sc_erl_dropout", "sc_erl_random", "sc_erl_evidential"]
        if not is_evo:
            for seed in merged_data[method]:
                df = merged_data[method][seed]
                if "total_steps" not in df.columns or df["total_steps"].isna().all():
                    df["total_steps"] = df["Step"]
            continue

        seeds_with_steps = [
            s
            for s, df in merged_data[method].items()
            if "total_steps" in df.columns and not df["total_steps"].isna().all()
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


def get_stable_final_values(merged_data):
    stable_values = {}
    for method in merged_data:
        stable_values[method] = []
        for seed, df in merged_data[method].items():
            if "total_steps" not in df.columns:
                continue
            y_metric = (
                "eval_reward"
                if "eval_reward" in df.columns and not df["eval_reward"].isna().all()
                else None
            )
            if not y_metric:
                continue
            max_steps = df["total_steps"].max()
            df_last_10 = df[df["total_steps"] >= max_steps * 0.9].dropna(
                subset=[y_metric]
            )
            if not df_last_10.empty:
                stable_values[method].append(df_last_10[y_metric].mean())
    return stable_values


# ==========================================
# PLOT GENERATION FUNCTIONS (WITH SAFE GUARDS)
# ==========================================
def generate_sample_efficiency_plot(env_id, merged_data, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.grid(True, which="both", color="#f2f2f2", linestyle="-", linewidth=0.5)

    # Bezpieczne obliczanie maksymalnego kroku
    step_maxes = []
    for m in merged_data:
        for s, df in merged_data[m].items():
            if "total_steps" in df.columns and not df["total_steps"].empty:
                step_maxes.append(df["total_steps"].max())

    max_steps_all = max(step_maxes) if step_maxes else 2000000
    step_grid = np.linspace(0, max_steps_all, 200)

    for method in sorted(merged_data.keys()):
        color = METHOD_COLORS.get(method, "#333333")
        label = METHOD_LABELS.get(method, method)
        is_proposed = method in PROPOSED_METHODS

        linewidth = 1.8 if is_proposed else 1.2
        line_alpha = 1.0 if is_proposed else 0.6
        linestyle = "-" if is_proposed else ":"
        fill_alpha = 0.15 if is_proposed else 0.05

        interpolated_ys = []
        for seed, df in merged_data[method].items():
            y_metric = (
                "eval_reward"
                if "eval_reward" in df.columns and not df["eval_reward"].isna().all()
                else (
                    "best_population_fitness"
                    if "best_population_fitness" in df.columns
                    else None
                )
            )
            if not y_metric:
                continue
            temp_df = df[["total_steps", y_metric]].dropna().sort_values("total_steps")
            if temp_df.empty:
                continue
            interpolated_ys.append(
                np.interp(
                    step_grid, temp_df["total_steps"].values, temp_df[y_metric].values
                )
            )

        if not interpolated_ys:
            continue
        interpolated_ys = np.array(interpolated_ys)
        mean_y = smooth_series(np.mean(interpolated_ys, axis=0), window=5)
        std_y = smooth_series(np.std(interpolated_ys, axis=0), window=5)

        ax.plot(
            step_grid,
            mean_y,
            label=label,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=line_alpha,
        )
        ax.fill_between(
            step_grid, mean_y - std_y, mean_y + std_y, color=color, alpha=fill_alpha
        )

    ax.set_title(
        f"Sample Efficiency Comparison - {env_id}",
        fontsize=13,
        pad=15,
        fontweight="bold",
    )
    ax.set_xlabel("Environmental Interaction Steps", labelpad=10)
    ax.set_ylabel("True Evaluation Reward", labelpad=10)
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
        framealpha=0.8,
        edgecolor="#f2f2f2",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def generate_surrogate_analysis_plot(env_id, merged_data, out_path):
    surrogate_methods = [m for m in PROPOSED_METHODS if m in merged_data]
    if not surrogate_methods:
        return

    fig, axes = plt.subplots(
        1,
        len(surrogate_methods),
        figsize=(5 * len(surrogate_methods), 5.5),
        sharey=False,
    )
    axes = np.atleast_1d(axes)

    all_fit_vals = [
        v
        for m in surrogate_methods
        for s, df in merged_data[m].items()
        if "avg_population_fitness" in df.columns
        for v in df["avg_population_fitness"].dropna().values
    ]
    y2_min, y2_max = (
        (min(all_fit_vals) if all_fit_vals else -100),
        (max(all_fit_vals) if all_fit_vals else 100),
    )

    for i, method in enumerate(surrogate_methods):
        ax = axes[i]
        ax.grid(True, which="both", color="#f2f2f2", linestyle="-", linewidth=0.5)
        color = METHOD_COLORS.get(method, "#000000")

        gen_grid = np.linspace(0, 1400, 141)
        mean_uncertainties, avg_fitnesses = [], []

        for seed, df in merged_data[method].items():
            required = ["generation", "uncertainty_mean", "avg_population_fitness"]
            if not all(col in df.columns for col in required):
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
            continue
        u_smooth = smooth_series(np.mean(mean_uncertainties, axis=0), window=7)
        f_smooth = smooth_series(np.mean(avg_fitnesses, axis=0), window=7)

        ax.set_xlabel("Generations", labelpad=10)
        ax.set_ylabel("Uncertainty Estimation", color=color, labelpad=10)
        ax.tick_params(axis="y", labelcolor=color)
        (l1,) = ax.plot(
            gen_grid, u_smooth, color=color, linewidth=1.5, label="Uncertainty Mean"
        )

        ax2 = ax.twinx()
        ax2.set_ylabel("Average Population Fitness", color="#de8f05", labelpad=10)
        ax2.tick_params(axis="y", labelcolor="#de8f05")
        ax2.set_ylim(y2_min - 0.05 * abs(y2_min), y2_max + 0.05 * abs(y2_max))
        ax2.grid(False)
        (l2,) = ax2.plot(
            gen_grid,
            f_smooth,
            color="#de8f05",
            linewidth=1.2,
            linestyle="-.",
            label="Avg Pop Fitness",
        )

        ax.legend(
            [l1, l2],
            [l.get_label() for l in [l1, l2]],
            loc="upper left",
            frameon=True,
            framealpha=0.8,
        )
        ax.set_title(
            METHOD_LABELS.get(method, method), fontsize=12, pad=12, fontweight="bold"
        )
        sns.despine(ax=ax, top=True, left=False, right=False)

    plt.suptitle(
        f"Surrogate Uncertainty & Fitness Analysis - {env_id}",
        fontsize=14,
        y=0.98,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def generate_critic_correlation_plot(env_id, base_dir, out_path):
    df_loss_path = os.path.join(base_dir, "critic_loss", f"{env_id}.csv")
    df_mean_path = os.path.join(base_dir, "uncertainty_mean", f"{env_id}.csv")
    if not os.path.exists(df_loss_path) or not os.path.exists(df_mean_path):
        return None

    df_loss, df_mean = pd.read_csv(df_loss_path), pd.read_csv(df_mean_path)
    valid_methods_data = {}

    for method in PROPOSED_METHODS:
        seeds = [
            int(match.group(1))
            for col in df_loss.columns
            if (
                match := re.match(
                    rf"^{method}_{re.escape(env_id)}_seed(\d+) - critic_loss$", col
                )
            )
        ]
        all_x, all_y = [], []
        for seed in seeds:
            col_loss, col_mean = (
                f"{method}_{env_id}_seed{seed} - critic_loss",
                f"{method}_{env_id}_seed{seed} - uncertainty_mean",
            )
            if col_loss not in df_loss.columns or col_mean not in df_mean.columns:
                continue
            merged = pd.merge(
                df_loss[["Step", col_loss]].dropna(),
                df_mean[["Step", col_mean]].dropna(),
                on="Step",
                how="inner",
            )
            if not merged.empty:
                all_x.extend(
                    pd.to_numeric(merged[col_loss], errors="coerce").fillna(0).values
                )
                all_y.extend(
                    pd.to_numeric(merged[col_mean], errors="coerce").fillna(0).values
                )
        if all_x and all_y:
            valid_methods_data[method] = (np.array(all_x), np.array(all_y))

    if not valid_methods_data:
        return None

    n_plots = len(valid_methods_data)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5), squeeze=False)
    correlations = []

    for i, (method, (x, y)) in enumerate(valid_methods_data.items()):
        ax = axes[0][i]
        ax.grid(True, which="both", color="#f2f2f2", linestyle="-", linewidth=0.5)

        p_corr = (
            np.corrcoef(x, y)[0, 1]
            if len(x) > 1 and np.var(x) > 0 and np.var(y) > 0
            else 0.0
        )
        s_corr = pd.Series(x).rank().corr(pd.Series(y).rank(), method="pearson")
        label = METHOD_LABELS.get(method, method)
        correlations.append(
            {"Method": label, "Pearson": p_corr, "Spearman": s_corr, "N": len(x)}
        )

        color = METHOD_COLORS.get(method, "#000000")
        ax.scatter(
            x, y, color=color, alpha=0.3, s=15, edgecolor="none", label="Steps Metrics"
        )

        if len(x) > 1 and np.var(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            x_grid = np.linspace(min(x), max(x), 100)
            ax.plot(
                x_grid,
                slope * x_grid + intercept,
                color="#333333",
                linestyle="--",
                linewidth=1.2,
                label="Trendline",
            )

        ax.set_title(
            f"{label}\n(Pearson r = {p_corr:.3f})",
            fontsize=11,
            pad=10,
            fontweight="bold",
        )
        ax.set_xlabel("Critic TD Loss", labelpad=8)
        ax.set_ylabel("Epistemic Uncertainty", labelpad=8)
        ax.legend(loc="upper right", frameon=True)
        sns.despine(ax=ax, top=True, right=True)

    plt.suptitle(
        f"Critic Epistemic Uncertainty vs. TD Loss - {env_id}",
        fontsize=13,
        y=0.98,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return correlations


def generate_speedup_plot(env_id, merged_data, out_path):
    """Generations reached vs. environmental steps — proves surrogate compression."""
    evo_methods = [
        m for m in merged_data
        if m in ["erl", "sc_erl_random", "sc_erl_ensemble", "sc_erl_dropout", "sc_erl_evidential"]
    ]
    if not evo_methods:
        return

    step_maxes = [
        df["total_steps"].max()
        for m in evo_methods
        for s, df in merged_data[m].items()
        if "total_steps" in df.columns and not df["total_steps"].isna().all()
    ]
    if not step_maxes:
        return
    step_grid = np.linspace(0, max(step_maxes), 200)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.grid(True, which="both", color="#f2f2f2", linestyle="-", linewidth=0.5)

    for method in evo_methods:
        color = METHOD_COLORS.get(method, "#333333")
        label = METHOD_LABELS.get(method, method)
        linewidth = 2.0 if method in PROPOSED_METHODS else 1.5
        linestyle = "-" if method in PROPOSED_METHODS else "--"

        interpolated_gens = []
        for seed, df in merged_data[method].items():
            if "total_steps" not in df.columns or "generation" not in df.columns:
                continue
            temp_df = df[["total_steps", "generation"]].dropna().sort_values("total_steps")
            if temp_df.empty:
                continue
            interpolated_gens.append(
                np.interp(step_grid, temp_df["total_steps"].values, temp_df["generation"].values)
            )

        if not interpolated_gens:
            continue
        interpolated_gens = np.array(interpolated_gens)
        mean_gen = np.mean(interpolated_gens, axis=0)
        std_gen = np.std(interpolated_gens, axis=0)

        ax.plot(step_grid, mean_gen, label=label, color=color, linewidth=linewidth, linestyle=linestyle)
        ax.fill_between(step_grid, mean_gen - std_gen, mean_gen + std_gen, color=color, alpha=0.1)

    ax.set_title(f"Evolutionary Speedup (Sample Efficiency) - {env_id}", fontsize=13, pad=15, fontweight="bold")
    ax.set_xlabel("Environmental Interaction Steps", labelpad=10)
    ax.set_ylabel("Generations Reached", labelpad=10)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k")
    )
    sns.despine(ax=ax, top=True, right=True)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.8, edgecolor="#f2f2f2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def generate_ratio_plot(env_id, merged_data, out_path):
    """Surrogate ratio over generations — shows epistemic breathing vs. flat random."""
    ratio_methods = [m for m in ["sc_erl_random", "sc_erl_ensemble", "sc_erl_dropout", "sc_erl_evidential"] if m in merged_data]
    if not ratio_methods:
        return

    gen_maxes = [
        df["generation"].max()
        for m in ratio_methods
        for s, df in merged_data[m].items()
        if "generation" in df.columns and not df["generation"].isna().all()
    ]
    gen_end = min(max(gen_maxes), 600) if gen_maxes else 600
    gen_grid = np.linspace(0, gen_end, 200)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.grid(True, which="both", color="#f2f2f2", linestyle="-", linewidth=0.5)

    for method in ratio_methods:
        color = METHOD_COLORS.get(method, "#000000")
        label = METHOD_LABELS.get(method, method)
        linewidth = 2.0 if method in PROPOSED_METHODS else 1.5
        linestyle = "-" if method in PROPOSED_METHODS else "--"

        interpolated_ratios = []
        for seed, df in merged_data[method].items():
            if "generation" not in df.columns or "surrogate_ratio" not in df.columns:
                continue
            temp_df = df[["generation", "surrogate_ratio"]].dropna().sort_values("generation")
            if temp_df.empty:
                continue
            interpolated_ratios.append(
                np.interp(gen_grid, temp_df["generation"].values, temp_df["surrogate_ratio"].values)
            )

        if not interpolated_ratios:
            continue
        mean_ratio = smooth_series(np.mean(interpolated_ratios, axis=0), window=5)
        std_ratio = smooth_series(np.std(interpolated_ratios, axis=0), window=5)

        ax.plot(gen_grid, mean_ratio, label=label, color=color, linewidth=linewidth, linestyle=linestyle)
        ax.fill_between(gen_grid, mean_ratio - std_ratio, mean_ratio + std_ratio, color=color, alpha=0.15)

    ax.set_title(f"Surrogate Utilization Dynamics - {env_id}", fontsize=13, pad=15, fontweight="bold")
    ax.set_xlabel("Generations", labelpad=10)
    ax.set_ylabel("Surrogate Ratio", labelpad=10)
    ax.set_ylim(-0.05, 1.05)
    sns.despine(ax=ax, top=True, right=True)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.8, edgecolor="#f2f2f2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ==========================================
# LATEX GENERATION STRATEGIES
# ==========================================
def build_summary_table_latex(env_id, base_dir="."):
    summary_path = os.path.join(base_dir, "summary", f"{env_id}.csv")
    if not os.path.exists(summary_path):
        return "% Summary data file missing\n"

    df = pd.read_csv(summary_path)
    parsed_rows = []
    for _, row in df.iterrows():
        match = re.match(
            f"^([a-z_0-9]+)_{re.escape(env_id)}_seed(\\d+)$", str(row.get("Name", ""))
        )
        if match:
            runtime_h = (
                pd.to_numeric(row.get("Runtime", np.nan), errors="coerce") / 3600.0
            )
            parsed_rows.append(
                {
                    "method": match.group(1),
                    "test_reward": pd.to_numeric(
                        row.get("eval_reward", np.nan), errors="coerce"
                    ),
                    "best_pop": pd.to_numeric(
                        row.get("best_population_fitness", np.nan), errors="coerce"
                    ),
                    "runtime_h": runtime_h,
                }
            )

    if not parsed_rows:
        return "% No parseable summary entries\n"
    parsed_df = pd.DataFrame(parsed_rows)
    stats_df = parsed_df.groupby("method").agg(["mean", "std"])

    tex = "\\begin{table}[htbp]\n\\centering\n"
    tex += f"\\caption{{Performance metrics and computational overhead for \\texttt{{{env_id}}}.}}\n"
    tex += f"\\label{{tab:summary_{env_id}}}\n"
    tex += "\\begin{tabular}{lccc}\n\\toprule\n"
    tex += "\\textbf{Method / Algorithm} & \\textbf{Mean Eval Reward} & \\textbf{Best Pop. Fitness} & \\textbf{Training Time [h]} \\\\\n\\midrule\n"

    method_order = [
        "ppo",
        "td3",
        "ddpg",
        "erl",
        "sc_erl_random",
        "sc_erl_ensemble",
        "sc_erl_dropout",
        "sc_erl_evidential",
    ]
    for m in method_order:
        if m not in stats_df.index:
            continue
        label = METHOD_LABELS.get(m, m)
        r_m, r_s = (
            stats_df.loc[m, ("test_reward", "mean")],
            stats_df.loc[m, ("test_reward", "std")],
        )
        b_m, b_s = (
            stats_df.loc[m, ("best_pop", "mean")],
            stats_df.loc[m, ("best_pop", "std")],
        )
        t_m, t_s = (
            stats_df.loc[m, ("runtime_h", "mean")],
            stats_df.loc[m, ("runtime_h", "std")],
        )

        # Zabezpieczenie przed NaN w odchyleniu standardowym (std)
        r_s_str = f"{r_s:.2f}" if pd.notna(r_s) else "0.00"
        b_s_str = f"{b_s:.2f}" if pd.notna(b_s) else "0.00"
        t_s_str = f"{t_s:.2f}" if pd.notna(t_s) else "0.00"

        tex += f"{label} & ${r_m:.2f} \\pm {r_s_str}$ & ${b_m:.2f} \\pm {b_s_str}$ & ${t_m:.2f} \\pm {t_s_str}$ \\\\\n"

    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return tex


def build_significance_table_latex(env_id, stable_values):
    baselines = ["ppo", "td3", "ddpg", "erl", "sc_erl_random"]

    tex = "\\begin{table}[htbp]\n\\centering\n"
    tex += f"\\caption{{Statistical Significance Testing for \\texttt{{{env_id}}} (Proposed vs Baselines).}}\n"
    tex += f"\\label{{tab:sig_{env_id}}}\n"
    tex += "\\begin{tabular}{llccc}\n\\toprule\n"
    tex += "\\textbf{Proposed Method} & \\textbf{Baseline} & \\textbf{Test Type} & \\textbf{$p$-value} & \\textbf{Sig.} \\\\\n\\midrule\n"

    has_rows = False
    for ours in PROPOSED_METHODS:
        group_A = np.array(stable_values.get(ours, []))
        group_A = group_A[~np.isnan(group_A)]
        if len(group_A) < 2:
            continue

        for base in baselines:
            group_B = np.array(stable_values.get(base, []))
            group_B = group_B[~np.isnan(group_B)]
            if len(group_B) < 2:
                continue

            p_shapiro_A = (
                stats.shapiro(group_A)[1]
                if len(group_A) >= 3 and np.var(group_A) > 0
                else 0.0
            )
            p_shapiro_B = (
                stats.shapiro(group_B)[1]
                if len(group_B) >= 3 and np.var(group_B) > 0
                else 0.0
            )

            try:
                if p_shapiro_A >= 0.05 and p_shapiro_B >= 0.05:
                    test_name = "Welch t-test"
                    p_val = stats.ttest_ind(group_A, group_B, equal_var=False)[1]
                else:
                    test_name = "Mann-Whitney"
                    p_val = stats.mannwhitneyu(
                        group_A, group_B, alternative="two-sided"
                    )[1]

                sig = (
                    "***"
                    if p_val < 0.001
                    else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
                )
                p_str = f"{p_val:.4e}" if p_val < 0.001 else f"{p_val:.4f}"

                tex += f"{METHOD_LABELS.get(ours, ours)} & {METHOD_LABELS.get(base, base)} & {test_name} & {p_str} & \\textbf{{{sig}}} \\\\\n"
                has_rows = True
            except Exception:
                continue
        tex += "\\hline\n"

    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return tex if has_rows else "% Insufficient data for statistical testing\n"


def build_correlation_table_latex(env_id, corr_data):
    if not corr_data:
        return "% No critic correlation data available\n"
    tex = "\\begin{table}[htbp]\n\\centering\n"
    tex += f"\\caption{{Correlation Analysis between Critic TD Loss and Epistemic Uncertainty (\\texttt{{{env_id}}}).}}\n"
    tex += f"\\label{{tab:corr_{env_id}}}\n"
    tex += "\\begin{tabular}{llcc}\n\\toprule\n"
    tex += "\\textbf{Algorithm / Method} & \\textbf{Pearson $r$} & \\textbf{Spearman $\\rho$} & \\textbf{Sample Size ($N$)} \\\\\n\\midrule\n"
    for row in corr_data:
        tex += f"{row['Method']} & {row['Pearson']:.4f} & {row['Spearman']:.4f} & {row['N']} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return tex


# ==========================================
# NEMENYI RANKING ANALYSIS
# ==========================================

# Critical values q_alpha for Nemenyi test, alpha=0.05 (two-tailed)
_NEMENYI_Q = {
    2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850,
    7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}


def compute_rankings_and_nemenyi(all_stable_values, environments):
    """Rank methods within each environment and run Friedman + Nemenyi tests.

    Parameters
    ----------
    all_stable_values : dict[env_id, dict[method, list[float]]]
    environments      : list[str]

    Returns
    -------
    rank_matrix  : dict[method, dict[env_id, int]]   1 = best
    avg_ranks    : dict[method, float]
    cd           : float | None    Nemenyi critical difference at alpha=0.05
    friedman_p   : float | None
    """
    methods = sorted({m for sv in all_stable_values.values() for m in sv})
    rank_matrix = {m: {} for m in methods}

    for env_id in environments:
        sv = all_stable_values.get(env_id, {})
        method_perf = {}
        for m in methods:
            vals = [v for v in sv.get(m, []) if not np.isnan(v)]
            if vals:
                method_perf[m] = np.mean(vals)
        if len(method_perf) < 2:
            continue
        for rank, m in enumerate(
            sorted(method_perf, key=method_perf.__getitem__, reverse=True), 1
        ):
            rank_matrix[m][env_id] = rank

    avg_ranks = {}
    for m in methods:
        ranks = [rank_matrix[m][e] for e in environments if e in rank_matrix[m]]
        avg_ranks[m] = float(np.mean(ranks)) if ranks else float("nan")

    # Friedman test uses only environments where every method has a rank
    common_envs = [
        e for e in environments
        if all(e in rank_matrix[m] for m in methods)
    ]
    friedman_p = cd = None
    if len(common_envs) >= 2 and len(methods) >= 2:
        rank_lists = [[rank_matrix[m][e] for e in common_envs] for m in methods]
        try:
            _, friedman_p = stats.friedmanchisquare(*rank_lists)
        except Exception:
            pass
        k, N = len(methods), len(common_envs)
        q = _NEMENYI_Q.get(k) or _NEMENYI_Q[max(k2 for k2 in _NEMENYI_Q if k2 <= min(k, 10))]
        cd = q * np.sqrt(k * (k + 1) / (6 * N))

    return rank_matrix, avg_ranks, cd, friedman_p


def build_nemenyi_ranking_table_latex(all_stable_values, environments):
    rank_matrix, avg_ranks, cd, friedman_p = compute_rankings_and_nemenyi(
        all_stable_values, environments
    )

    method_order = [
        "ppo", "td3", "ddpg", "erl", "sc_erl_random",
        "sc_erl_ensemble", "sc_erl_dropout", "sc_erl_evidential",
    ]
    present = [m for m in method_order if m in avg_ranks]
    sorted_methods = sorted(present, key=lambda m: avg_ranks.get(m, float("inf")))
    best_method = sorted_methods[0] if sorted_methods else None
    best_avg = avg_ranks.get(best_method, float("nan")) if best_method else float("nan")

    def short_env(e):
        return e.replace("-v5", "").replace("-v4", "").replace("-v3", "")

    env_cols = " & ".join(f"\\textbf{{{short_env(e)}}}" for e in environments)

    caption = (
        "Per-environment method rankings (1~=~best) and average rank. "
        "Bold rows are within Nemenyi CD of the best average rank (not significantly different). "
    )
    if friedman_p is not None:
        caption += f"Friedman test: $p = {friedman_p:.4f}$. "
    if cd is not None:
        caption += f"Nemenyi CD~$= {cd:.3f}$ ($\\alpha = 0.05$)."

    tex = "\\begin{table}[htbp]\n\\centering\n"
    tex += f"\\caption{{{caption}}}\n"
    tex += "\\label{tab:nemenyi_rankings}\n"
    tex += f"\\begin{{tabular}}{{l{'c' * (len(environments) + 1)}}}\n\\toprule\n"
    tex += f"\\textbf{{Method}} & {env_cols} & \\textbf{{Avg.~Rank}} \\\\\n\\midrule\n"

    for m in sorted_methods:
        label = METHOD_LABELS.get(m, m)
        cells = [
            "--" if rank_matrix[m].get(env_id) is None else str(rank_matrix[m][env_id])
            for env_id in environments
        ]
        avg = avg_ranks.get(m, float("nan"))
        avg_str = f"{avg:.2f}" if not np.isnan(avg) else "--"

        within_cd = (
            cd is not None
            and not np.isnan(avg)
            and not np.isnan(best_avg)
            and abs(avg - best_avg) <= cd
        )
        if within_cd:
            tex += f"\\textbf{{{label}}} & {' & '.join(cells)} & \\textbf{{{avg_str}}} \\\\\n"
        else:
            tex += f"{label} & {' & '.join(cells)} & {avg_str} \\\\\n"

    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return tex


def generate_nemenyi_cd_plot(avg_ranks, cd, out_path):
    """Horizontal CD diagram: methods ranked left-to-right, CD bracket shown."""
    present = {m: r for m, r in avg_ranks.items() if not np.isnan(r)}
    if not present:
        return

    sorted_methods = sorted(present, key=present.__getitem__)
    n = len(sorted_methods)
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.55 * n) + 1.2))

    y_pos = {m: i for i, m in enumerate(sorted_methods)}
    best_r = present[sorted_methods[0]]

    for m in sorted_methods:
        r = present[m]
        color = METHOD_COLORS.get(m, "#333333")
        label = METHOD_LABELS.get(m, m)
        y = y_pos[m]
        ax.scatter(r, y, color=color, s=70, zorder=4)
        ax.text(r + 0.05, y, f"{r:.2f}", va="center", fontsize=8.5, color=color)
        ax.text(-0.1, y, label, ha="right", va="center", fontsize=9)

    # CD bracket above the plot
    if cd is not None:
        y_top = n + 0.3
        ax.annotate(
            "",
            xy=(best_r + cd, y_top),
            xytext=(best_r, y_top),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.5),
        )
        ax.text(
            best_r + cd / 2, y_top + 0.25,
            f"CD = {cd:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
        # Underline methods within CD of best (not significantly different)
        within = [m for m in sorted_methods if abs(present[m] - best_r) <= cd]
        if len(within) > 1:
            xs = [present[m] for m in within]
            ax.plot(
                [min(xs) - 0.12, max(xs) + 0.12],
                [-0.6, -0.6],
                color="#555555", lw=3.5, alpha=0.35, solid_capstyle="round",
            )
            ax.text(
                np.mean(xs), -0.95,
                "no significant difference",
                ha="center", fontsize=8, color="#555555", style="italic",
            )

    ax.set_xlabel("Average Rank (lower = better)", labelpad=10)
    ax.set_title("Nemenyi Critical Difference Diagram", fontweight="bold", pad=14)
    ax.set_ylim(-1.4, n + 0.9)
    ax.set_xlim(0.5, max(present.values()) + 1.0)
    ax.set_yticks([])
    ax.grid(axis="x", color="#eeeeee", linestyle="-")
    sns.despine(ax=ax, left=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
def main():
    base_dir = "."
    eval_reward_dir = os.path.join(base_dir, "eval_reward")

    if not os.path.exists(eval_reward_dir):
        print(f"Error: Database structure directory '{eval_reward_dir}' not found.")
        return

    env_files = glob.glob(os.path.join(eval_reward_dir, "*.csv"))
    environments = [os.path.basename(f).replace(".csv", "") for f in env_files]

    output_dir = os.path.join(base_dir, "results_output")
    os.makedirs(output_dir, exist_ok=True)

    latex_document = (
        "\\documentclass[10pt]{article}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{geometry}\n"
        "\\usepackage{float}\n"
        "\\geometry{a4paper, margin=1in}\n"
        "\\title{Evolutionary Reinforcement Learning - Comprehensive Report}\n"
        "\\author{Automated Statistical Pipeline}\n"
        "\\date{\\today}\n"
        "\\begin{document}\n\\maketitle\n\\tableofcontents\n\\newpage\n"
    )

    all_stable_values = {}  # env_id -> {method -> [final vals]}

    for env_id in environments:
        print(f"Processing environment: {env_id}...")
        merged_data = load_environment_data(env_id, base_dir)
        stable_values = get_stable_final_values(merged_data)
        all_stable_values[env_id] = stable_values

        se_path  = os.path.join(output_dir, f"{env_id}_sample_efficiency.png")
        sa_path  = os.path.join(output_dir, f"{env_id}_surrogate_analysis.png")
        cc_path  = os.path.join(output_dir, f"{env_id}_critic_correlation.png")
        spd_path = os.path.join(output_dir, f"{env_id}_speedup.png")
        rat_path = os.path.join(output_dir, f"{env_id}_ratio.png")

        # Generowanie wykresów otoczone blokami try-except dla bezpieczeństwa .tex
        try:
            generate_sample_efficiency_plot(env_id, merged_data, se_path)
            has_se = True
        except Exception as e:
            print(f"Warning: Could not generate sample efficiency plot for {env_id}: {e}")
            has_se = False

        try:
            generate_surrogate_analysis_plot(env_id, merged_data, sa_path)
            has_sa = True
        except Exception as e:
            print(f"Warning: Could not generate surrogate analysis plot for {env_id}: {e}")
            has_sa = False

        try:
            corr_results = generate_critic_correlation_plot(env_id, base_dir, cc_path)
        except Exception as e:
            print(f"Warning: Could not generate critic correlation plot for {env_id}: {e}")
            corr_results = None

        try:
            generate_speedup_plot(env_id, merged_data, spd_path)
            has_spd = True
        except Exception as e:
            print(f"Warning: Could not generate speedup plot for {env_id}: {e}")
            has_spd = False

        try:
            generate_ratio_plot(env_id, merged_data, rat_path)
            has_rat = True
        except Exception as e:
            print(f"Warning: Could not generate ratio plot for {env_id}: {e}")
            has_rat = False

        latex_document += f"\\section{{Environment Results: \\texttt{{{env_id}}}}}\n"

        if has_se:
            latex_document += (
                f"\\begin{{figure}}[H]\n\\centering\n"
                f"  \\includegraphics[width=0.85\\textwidth]{{{os.path.basename(se_path)}}}\n"
                f"  \\caption{{Sample Efficiency comparison across evaluation metrics on {env_id}.}}\n"
                f"\\end{{figure}}\n\n"
            )

        latex_document += build_summary_table_latex(env_id, base_dir)
        latex_document += build_significance_table_latex(env_id, stable_values)

        if corr_results:
            latex_document += build_correlation_table_latex(env_id, corr_results)
            latex_document += (
                f"\\begin{{figure}}[H]\n\\centering\n"
                f"  \\includegraphics[width=0.95\\textwidth]{{{os.path.basename(cc_path)}}}\n"
                f"  \\caption{{Scatter plots mapping surrogate epistemic uncertainty against critic TD loss values.}}\n"
                f"\\end{{figure}}\n"
            )

        if has_sa:
            latex_document += (
                f"\\begin{{figure}}[H]\n\\centering\n"
                f"  \\includegraphics[width=0.85\\textwidth]{{{os.path.basename(sa_path)}}}\n"
                f"  \\caption{{Surrogate controller uncertainty trends compared to average population fitness across generations.}}\n"
                f"\\end{{figure}}\n"
            )

        if has_spd:
            latex_document += (
                f"\\begin{{figure}}[H]\n\\centering\n"
                f"  \\includegraphics[width=0.85\\textwidth]{{{os.path.basename(spd_path)}}}\n"
                f"  \\caption{{Evolutionary speedup: number of generations completed per environmental step on {env_id}. "
                f"A steeper slope indicates more surrogate-driven generations within the same interaction budget.}}\n"
                f"\\end{{figure}}\n\n"
            )

        if has_rat:
            latex_document += (
                f"\\begin{{figure}}[H]\n\\centering\n"
                f"  \\includegraphics[width=0.85\\textwidth]{{{os.path.basename(rat_path)}}}\n"
                f"  \\caption{{Surrogate utilization dynamics on {env_id}. "
                f"Uncertainty-driven methods (Ensemble, Dropout, Evidential) exhibit epistemic breathing --- "
                f"deep drops in surrogate ratio coincide with high-uncertainty discovery phases, "
                f"whereas SC-ERL Random maintains a flat, uninformed utilization profile.}}\n"
                f"\\end{{figure}}\n\n"
            )

        latex_document += "\\newpage\n"

    # ---- Global Nemenyi ranking section ----
    if len(environments) >= 2:
        print("\nComputing global Nemenyi ranking table...")
        _, avg_ranks, cd, friedman_p = compute_rankings_and_nemenyi(
            all_stable_values, environments
        )
        cd_path = os.path.join(output_dir, "nemenyi_cd_diagram.png")
        try:
            generate_nemenyi_cd_plot(avg_ranks, cd, cd_path)
            has_cd = True
        except Exception as e:
            print(f"Warning: Could not generate CD diagram: {e}")
            has_cd = False

        latex_document += "\\section{Global Ranking Analysis (Nemenyi)}\n"
        latex_document += build_nemenyi_ranking_table_latex(all_stable_values, environments)
        if has_cd:
            latex_document += (
                "\\begin{figure}[H]\n\\centering\n"
                f"  \\includegraphics[width=0.72\\textwidth]{{nemenyi_cd_diagram.png}}\n"
                "  \\caption{Critical Difference diagram. "
                "Methods connected by the grey bar are not significantly different "
                "from the best-ranked method at $\\alpha = 0.05$.}\n"
                "\\end{figure}\n"
            )
        latex_document += "\\newpage\n"

    latex_document += "\\end{document}\n"

    tex_output_path = os.path.join(output_dir, "full_report.tex")
    with open(tex_output_path, "w", encoding="utf-8") as f:
        f.write(latex_document)

    print(f"\n[SUCCESS] Pipeline completed safely. Target TeX: {tex_output_path}")


if __name__ == "__main__":
    main()
