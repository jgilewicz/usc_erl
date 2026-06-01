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
        "generation",
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
        is_evo = method in ["erl", "sc_erl_ensemble", "sc_erl_dropout", "sc_erl_random"]
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

    for env_id in environments:
        print(f"Processing environment: {env_id}...")
        merged_data = load_environment_data(env_id, base_dir)
        stable_values = get_stable_final_values(merged_data)

        se_path = os.path.join(output_dir, f"{env_id}_sample_efficiency.png")
        sa_path = os.path.join(output_dir, f"{env_id}_surrogate_analysis.png")
        cc_path = os.path.join(output_dir, f"{env_id}_critic_correlation.png")

        # Generowanie wykresów otoczone blokami try-except dla bezpieczeństwa .tex
        try:
            generate_sample_efficiency_plot(env_id, merged_data, se_path)
            has_se = True
        except Exception as e:
            print(
                f"Warning: Could not generate sample efficiency plot for {env_id}: {e}"
            )
            has_se = False

        try:
            generate_surrogate_analysis_plot(env_id, merged_data, sa_path)
            has_sa = True
        except Exception as e:
            print(
                f"Warning: Could not generate surrogate analysis plot for {env_id}: {e}"
            )
            has_sa = False

        try:
            corr_results = generate_critic_correlation_plot(env_id, base_dir, cc_path)
        except Exception as e:
            print(
                f"Warning: Could not generate critic correlation plot for {env_id}: {e}"
            )
            corr_results = None

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

        latex_document += "\\newpage\n"

    latex_document += "\\end{document}\n"

    tex_output_path = os.path.join(output_dir, "full_report.tex")
    with open(tex_output_path, "w", encoding="utf-8") as f:
        f.write(latex_document)

    print(f"\n[SUCCESS] Pipeline completed safely. Target TeX: {tex_output_path}")


if __name__ == "__main__":
    main()
