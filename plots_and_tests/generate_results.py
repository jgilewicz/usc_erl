import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

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


def normalize_env_id(env_id):
    if not env_id:
        return env_id
    return re.sub(r"(?i)walker(?:2d)?", "Walker2d", env_id)


def display_env_id(env_id: str) -> str:
    """Human-readable env name for plot titles and table captions."""
    e = re.sub(r"^dm_control_", "DMC/", env_id)
    e = re.sub(r"-v\d+$", "", e)
    return e


def get_env_file_variants(env_id):
    if not re.search(r"(?i)walker", env_id):
        return [env_id]
    canonical = re.sub(r"(?i)walker(?:2d)?", "Walker2d", env_id)
    short = re.sub(r"(?i)walker(?:2d)?", "Walker", env_id)
    return list(dict.fromkeys([canonical, short]))


METHOD_COLORS = {
    "sc_erl_dropout": "#0173b2",
    "sc_erl_ensemble": "#029e73",
    "sc_erl_evidential": "#8c564b",
    "erl": "#de8f05",
    "ppo": "#d55e00",
    "td3": "#cc78bc",
    "ddpg": "#56b4e9",
    "sc_erl_random": "#949494",
    "sac": "#e377c2",
    "crossq": "#bcbd22",
}

METHOD_LABELS = {
    "ppo": "PPO (Baseline)",
    "td3": "TD3 (Baseline)",
    "ddpg": "DDPG (Baseline)",
    "sac": "SAC (Baseline)",
    "crossq": "CrossQ (Baseline)",
    "erl": "ERL (Baseline)",
    "sc_erl_ensemble": "SC-ERL (Ensemble) [Ours]",
    "sc_erl_dropout": "SC-ERL (Dropout) [Ours]",
    "sc_erl_evidential": "SC-ERL (Evidential) [Ours]",
    "sc_erl_random": "SC-ERL (Random)",
}

PROPOSED_METHODS = ["sc_erl_ensemble", "sc_erl_dropout", "sc_erl_evidential"]

SC_ERL_VARIANTS = [
    "sc_erl_random",
    "sc_erl_dropout",
    "sc_erl_ensemble",
    "sc_erl_evidential",
]

METHOD_ORDER_BARS = [
    "ppo",
    "ddpg",
    "td3",
    "sac",
    "crossq",
    "erl",
    "sc_erl_random",
    "sc_erl_dropout",
    "sc_erl_ensemble",
    "sc_erl_evidential",
]

BUDGET_CUTOFFS = [200_000, 500_000, 1_000_000]
BUDGET_LABELS = ["200k", "500k", "1M"]
# Alpha levels for 200k / 500k / 1M — lighter shade = smaller budget
BUDGET_ALPHAS = [0.35, 0.65, 1.0]


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
        "n_real",
        "generation",
        "raw_sigma_mean",
        "raw_sigma_max",
        "raw_sigma_cv",
        "rho",
        "e_hat_mean",
        "gate_auc_u",
        "gate_auc_d",
        "gate_spearman",
        "gate_n_pool",
        "behavioral_distance_mean",
        "d_cv",
    ]
    run_data = {}

    for metric in metrics:
        for file_env_id in get_env_file_variants(env_id):
            file_path = os.path.join(base_dir, metric, f"{file_env_id}.csv")
            if not os.path.exists(file_path):
                continue
            df = pd.read_csv(file_path)
            df = df.replace(["Infinity", "inf", "inf.0"], np.nan)

            for col in df.columns:
                if col == "Step" or col.endswith(("__MIN", "__MAX")):
                    continue

                method, seed, parsed_metric = None, None, None
                for variant in get_env_file_variants(file_env_id):
                    method, seed, parsed_metric = parse_column_header(col, variant)
                    if method is not None:
                        break
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
    for method, seed_data in run_data.items():
        merged_data[method] = {}
        for seed, dfs in seed_data.items():
            if not dfs:
                continue
            merged_df = dfs[0]
            for next_df in dfs[1:]:
                merged_df = pd.merge(merged_df, next_df, on="Step", how="outer")
            merged_df = merged_df.sort_values("Step").reset_index(drop=True)
            merged_data[method][seed] = merged_df

    for method in list(merged_data.keys()):
        is_evo = method in [
            "erl",
            "sc_erl_ensemble",
            "sc_erl_dropout",
            "sc_erl_random",
            "sc_erl_evidential",
        ]
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

    # surrogate_ratio isn't logged directly (rho adapts it dynamically, so the
    # realized fraction is an observation, not a config identity) — derive it
    # from the logged n_real and the run's population size.
    pop_size = _get_population_size(env_id, base_dir)
    if pop_size:
        for seed_data in merged_data.values():
            for df in seed_data.values():
                if "n_real" in df.columns:
                    df["surrogate_ratio"] = 1.0 - df["n_real"] / pop_size

    return merged_data


def get_stable_final_values(merged_data):
    stable_values = {}
    for method in merged_data:
        stable_values[method] = []
        for df in merged_data[method].values():
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


def generate_sample_efficiency_plot(env_id, merged_data, out_path):
    _fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.grid(True, which="both", color="#f2f2f2", linestyle="-", linewidth=0.5)

    # Bezpieczne obliczanie maksymalnego kroku
    step_maxes = []
    for m in merged_data:
        for df in merged_data[m].values():
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
        for df in merged_data[method].values():
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


def generate_gate_dynamics_plot(env_id, merged_data, out_path):
    """Fig. F-B: rho and e_hat on a dual axis vs. steps, one panel per gated mode."""
    gated_methods = [m for m in PROPOSED_METHODS if m in merged_data]
    if not gated_methods:
        return

    _fig, axes = plt.subplots(
        1, len(gated_methods), figsize=(5 * len(gated_methods), 5.0), sharey=False
    )
    axes = np.atleast_1d(axes)

    for ax, method in zip(axes, gated_methods):
        ax.grid(True, which="both", color="#f2f2f2", linestyle="-", linewidth=0.5)
        color = METHOD_COLORS.get(method, "#000000")

        step_maxes = [
            df["total_steps"].max()
            for s, df in merged_data[method].items()
            if "total_steps" in df.columns and not df["total_steps"].isna().all()
        ]
        if not step_maxes:
            continue
        step_grid = np.linspace(0, max(step_maxes), 200)

        rho_curves, e_hat_curves = [], []
        for df in merged_data[method].values():
            if not all(c in df.columns for c in ["total_steps", "rho", "e_hat_mean"]):
                continue
            temp_df = (
                df[["total_steps", "rho", "e_hat_mean"]]
                .dropna()
                .sort_values("total_steps")
            )
            if temp_df.empty:
                continue
            rho_curves.append(
                np.interp(
                    step_grid, temp_df["total_steps"].values, temp_df["rho"].values
                )
            )
            e_hat_curves.append(
                np.interp(
                    step_grid,
                    temp_df["total_steps"].values,
                    temp_df["e_hat_mean"].values,
                )
            )

        if not rho_curves:
            continue
        rho_smooth = smooth_series(np.mean(rho_curves, axis=0), window=7)
        e_hat_smooth = smooth_series(np.mean(e_hat_curves, axis=0), window=7)

        ax.set_xlabel("Environmental Interaction Steps", labelpad=10)
        ax.set_ylabel(r"$\rho$ (gated fraction)", color=color, labelpad=10)
        ax.tick_params(axis="y", labelcolor=color)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(
                lambda x, p: f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x / 1e3:.0f}k"
            )
        )
        (l1,) = ax.plot(
            step_grid, rho_smooth, color=color, linewidth=1.6, label=r"$\rho$"
        )

        ax2 = ax.twinx()
        ax2.set_ylabel(r"$\hat{e}$ (surrogate error)", color="#de8f05", labelpad=10)
        ax2.tick_params(axis="y", labelcolor="#de8f05")
        ax2.grid(False)
        (l2,) = ax2.plot(
            step_grid,
            e_hat_smooth,
            color="#de8f05",
            linewidth=1.4,
            linestyle="-.",
            label=r"$\hat{e}$",
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
        f"Gate Controller Dynamics - {env_id}", fontsize=14, y=0.98, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def generate_gate_scatter_plot(env_id, merged_data, out_path):
    """Fig. F-C: mean uncertainty vs. surrogate error, aggregated over the whole run.

    Points are per-generation aggregates (mean uncertainty of the epsilon-explored
    individuals that generation vs. the windowed surrogate error e_hat) rather than
    raw per-individual pairs — the logged history only retains the scalar
    per-generation gate summary, not each epsilon-sampled individual's (u, e).
    """
    gated_methods = [m for m in PROPOSED_METHODS if m in merged_data]
    if not gated_methods:
        return

    _fig, axes = plt.subplots(
        1, len(gated_methods), figsize=(5 * len(gated_methods), 4.5), squeeze=False
    )

    for i, method in enumerate(gated_methods):
        ax = axes[0][i]
        ax.grid(True, which="both", color="#f2f2f2", linestyle="-", linewidth=0.5)
        color = METHOD_COLORS.get(method, "#000000")

        u_vals, e_vals = [], []
        for df in merged_data[method].values():
            if not all(c in df.columns for c in ["uncertainty_mean", "e_hat_mean"]):
                continue
            temp_df = df[["uncertainty_mean", "e_hat_mean"]].dropna()
            u_vals.extend(temp_df["uncertainty_mean"].values)
            e_vals.extend(temp_df["e_hat_mean"].values)

        label = METHOD_LABELS.get(method, method)
        if len(u_vals) < 2:
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        u_arr, e_arr = np.array(u_vals), np.array(e_vals)
        rho_corr = (
            pd.Series(u_arr).corr(pd.Series(e_arr), method="spearman")
            if np.var(u_arr) > 0 and np.var(e_arr) > 0
            else np.nan
        )
        ax.scatter(u_arr, e_arr, color=color, alpha=0.3, s=15, edgecolor="none")

        ax.set_title(
            f"{label}\n"
            + (f"($\\rho$ = {rho_corr:.3f})" if pd.notna(rho_corr) else ""),
            fontsize=11,
            pad=10,
            fontweight="bold",
        )
        ax.set_xlabel("Mean Uncertainty (per generation)", labelpad=8)
        ax.set_ylabel("Surrogate Error $\\hat{e}$", labelpad=8)
        sns.despine(ax=ax, top=True, right=True)

    plt.suptitle(
        f"Uncertainty vs. Surrogate Error, $\\epsilon$-sampled Individuals - {env_id}",
        fontsize=13,
        y=0.98,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def generate_behavioral_uncertainty_scatter_plot(env_id, merged_data, out_path):
    """Fig. F-D: per-generation mean epistemic uncertainty (of the pi_j policies)
    vs. mean behavioural distance between each pi_j and the RL actor, aggregated
    over the whole run. Pearson r (scipy) is annotated per subplot.
    """
    gated_methods = [m for m in PROPOSED_METHODS if m in merged_data]
    if not gated_methods:
        return

    _fig, axes = plt.subplots(
        1, len(gated_methods), figsize=(5 * len(gated_methods), 4.5), squeeze=False
    )

    for i, method in enumerate(gated_methods):
        ax = axes[0][i]
        ax.grid(True, which="both", color="#f2f2f2", linestyle="-", linewidth=0.5)
        color = METHOD_COLORS.get(method, "#000000")

        u_vals, d_vals = [], []
        for df in merged_data[method].values():
            required = ["uncertainty_mean", "behavioral_distance_mean"]
            if not all(c in df.columns for c in required):
                continue
            temp_df = df[required].dropna()
            u_vals.extend(temp_df["uncertainty_mean"].values)
            d_vals.extend(temp_df["behavioral_distance_mean"].values)

        label = METHOD_LABELS.get(method, method)
        if len(u_vals) < 2:
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        u_arr, d_arr = np.array(u_vals), np.array(d_vals)
        if np.var(u_arr) > 0 and np.var(d_arr) > 0:
            pearson_r, pearson_p = stats.pearsonr(u_arr, d_arr)
        else:
            pearson_r, pearson_p = np.nan, np.nan
        ax.scatter(u_arr, d_arr, color=color, alpha=0.3, s=15, edgecolor="none")

        title_stats = (
            f"($r$ = {pearson_r:.3f}, $p$ = {pearson_p:.2e})"
            if pd.notna(pearson_r)
            else ""
        )
        ax.set_title(f"{label}\n{title_stats}", fontsize=11, pad=10, fontweight="bold")
        ax.set_xlabel("Mean Uncertainty (per generation)", labelpad=8)
        ax.set_ylabel("Mean Behavioural Distance $\\pi_j$ vs. RL", labelpad=8)
        sns.despine(ax=ax, top=True, right=True)

    plt.suptitle(
        f"Policy Uncertainty vs. Behavioural Distance from RL Actor - {env_id}",
        fontsize=13,
        y=0.98,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _final_window_mean(df, col, frac=0.1):
    """Mean of ``col`` over the last ``frac`` of the run (by total_steps)."""
    if "total_steps" not in df.columns or col not in df.columns:
        return np.nan
    sub = df[["total_steps", col]].dropna()
    if sub.empty:
        return np.nan
    max_steps = sub["total_steps"].max()
    window = sub[sub["total_steps"] >= max_steps * (1 - frac)]
    window = window if not window.empty else sub
    return float(window[col].mean())


def _overall_mean(df, col):
    if col not in df.columns:
        return np.nan
    vals = df[col].dropna()
    return float(vals.mean()) if not vals.empty else np.nan


def get_gate_quality_values(merged_data):
    """Per-method, per-seed gate-quality summary (final-window mean of each metric).

    Only meaningful for the uncertainty-gated SC-ERL variants — the underlying
    ``gate_*``/``rho``/``e_hat_mean`` metrics are only logged from
    ``SurrogateController._gated_evaluation`` (dropout/ensemble/evidential).
    """
    out = {}
    for method in PROPOSED_METHODS:
        if method not in merged_data:
            continue
        rows = []
        for df in merged_data[method].values():
            auc_u = _final_window_mean(df, "gate_auc_u")
            auc_d = _final_window_mean(df, "gate_auc_d")
            spearman = _final_window_mean(df, "gate_spearman")
            n_pool = _final_window_mean(df, "gate_n_pool")
            if np.isnan(auc_u) and np.isnan(spearman):
                continue
            rows.append(
                {
                    "auc_u": auc_u,
                    "auc_d": auc_d,
                    "spearman": spearman,
                    "n_pool": n_pool,
                }
            )
        if rows:
            out[method] = rows
    return out


def get_gating_cost_values(env_id, merged_data, base_dir):
    """Per-method budget/cost summary: final rho, real evals/gen, generations, surrogate ratio."""
    pop_size = _get_population_size(env_id, base_dir)
    out = {}
    for method in SC_ERL_VARIANTS + ["erl"]:
        if method not in merged_data:
            continue
        rows = []
        for df in merged_data[method].values():
            gens = df["generation"].max() if "generation" in df.columns else np.nan
            ratio = _overall_mean(df, "surrogate_ratio")
            rho_final = _final_window_mean(df, "rho")
            real_evals = (
                pop_size * (1 - ratio)
                if pop_size is not None and not np.isnan(ratio)
                else np.nan
            )
            if np.isnan(gens) and np.isnan(ratio):
                continue
            rows.append(
                {
                    "rho_final": rho_final,
                    "real_evals_gen": real_evals,
                    "generations": gens,
                    "surrogate_ratio": ratio,
                }
            )
        if rows:
            out[method] = rows
    return out


def _get_population_size(env_id, base_dir):
    summary_path = os.path.join(base_dir, "summary", f"{env_id}.csv")
    if not os.path.exists(summary_path):
        return None
    df = pd.read_csv(summary_path)
    col = "evolution.population_size"
    if col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.mean()) if not vals.empty else None


def generate_speedup_plot(env_id, merged_data, out_path):
    evo_methods = [
        m
        for m in merged_data
        if m
        in [
            "erl",
            "sc_erl_random",
            "sc_erl_ensemble",
            "sc_erl_dropout",
            "sc_erl_evidential",
        ]
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

    _fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.grid(True, which="both", color="#f2f2f2", linestyle="-", linewidth=0.5)

    for method in evo_methods:
        color = METHOD_COLORS.get(method, "#333333")
        label = METHOD_LABELS.get(method, method)
        linewidth = 2.0 if method in PROPOSED_METHODS else 1.5
        linestyle = "-" if method in PROPOSED_METHODS else "--"

        interpolated_gens = []
        for df in merged_data[method].values():
            if "total_steps" not in df.columns or "generation" not in df.columns:
                continue
            temp_df = (
                df[["total_steps", "generation"]].dropna().sort_values("total_steps")
            )
            if temp_df.empty:
                continue
            interpolated_gens.append(
                np.interp(
                    step_grid,
                    temp_df["total_steps"].values,
                    temp_df["generation"].values,
                )
            )

        if not interpolated_gens:
            continue
        interpolated_gens = np.array(interpolated_gens)
        mean_gen = np.mean(interpolated_gens, axis=0)
        std_gen = np.std(interpolated_gens, axis=0)

        ax.plot(
            step_grid,
            mean_gen,
            label=label,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        ax.fill_between(
            step_grid, mean_gen - std_gen, mean_gen + std_gen, color=color, alpha=0.1
        )

    ax.set_title(
        f"Evolutionary Speedup (Sample Efficiency) - {env_id}",
        fontsize=13,
        pad=15,
        fontweight="bold",
    )
    ax.set_xlabel("Environmental Interaction Steps", labelpad=10)
    ax.set_ylabel("Generations Reached", labelpad=10)
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


def generate_ratio_plot(env_id, merged_data, out_path):
    ratio_methods = [
        m
        for m in [
            "sc_erl_random",
            "sc_erl_ensemble",
            "sc_erl_dropout",
            "sc_erl_evidential",
        ]
        if m in merged_data
    ]
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

    _fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.grid(True, which="both", color="#f2f2f2", linestyle="-", linewidth=0.5)

    any_seed_end_marked = False
    for method in ratio_methods:
        color = METHOD_COLORS.get(method, "#000000")
        label = METHOD_LABELS.get(method, method)
        linewidth = 2.0 if method in PROPOSED_METHODS else 1.5
        linestyle = "-" if method in PROPOSED_METHODS else "--"

        interpolated_ratios = []
        seed_end_gens = []
        for df in merged_data[method].values():
            if "generation" not in df.columns or "surrogate_ratio" not in df.columns:
                continue
            temp_df = (
                df[["generation", "surrogate_ratio"]].dropna().sort_values("generation")
            )
            if temp_df.empty:
                continue
            interpolated_ratios.append(
                np.interp(
                    gen_grid,
                    temp_df["generation"].values,
                    temp_df["surrogate_ratio"].values,
                )
            )
            seed_end_gens.append(float(temp_df["generation"].max()))

        if not interpolated_ratios:
            continue
        mean_ratio = smooth_series(np.mean(interpolated_ratios, axis=0), window=5)
        std_ratio = smooth_series(np.std(interpolated_ratios, axis=0), window=5)

        ax.plot(
            gen_grid,
            mean_ratio,
            label=label,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        ax.fill_between(
            gen_grid,
            mean_ratio - std_ratio,
            mean_ratio + std_ratio,
            color=color,
            alpha=0.15,
        )

        # Beyond a seed's last recorded generation, np.interp holds the value
        # flat — the mean then steps whenever a seed drops out. Mark those
        # generations so the steps in the curve are legible as an artefact,
        # not a real change in surrogate utilization.
        for end_gen in sorted(set(seed_end_gens)):
            if end_gen < gen_end - 1e-6:
                ax.axvline(
                    end_gen, color=color, linestyle=":", linewidth=0.8, alpha=0.5
                )
                any_seed_end_marked = True

    ax.set_title(
        f"Surrogate Utilization Dynamics - {env_id}",
        fontsize=13,
        pad=15,
        fontweight="bold",
    )
    ax.set_xlabel("Generations", labelpad=10)
    ax.set_ylabel("Surrogate Ratio", labelpad=10)
    ax.set_ylim(-0.05, 1.05)
    sns.despine(ax=ax, top=True, right=True)
    handles, labels = ax.get_legend_handles_labels()
    if any_seed_end_marked:
        handles.append(
            plt.Line2D([0], [0], color="#555555", linestyle=":", linewidth=0.8)
        )
        labels.append("seed ends (fewer seeds beyond)")
    ax.legend(
        handles,
        labels,
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.8,
        edgecolor="#f2f2f2",
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def generate_raw_sigma_cv_plot(env_id, merged_data, out_path):
    """Raw ensemble-disagreement CV across the population, one line per gated mode.

    Explains why gate_mode=relative's fixed MAD threshold failed: population
    spread in raw_sigma sits at 0.5-1% for most of training, too small a
    signal for a static cutoff to separate individuals on.
    """
    gated_methods = [m for m in PROPOSED_METHODS if m in merged_data]
    if not gated_methods:
        return

    step_maxes = [
        df["total_steps"].max()
        for m in gated_methods
        for s, df in merged_data[m].items()
        if "total_steps" in df.columns and not df["total_steps"].isna().all()
    ]
    if not step_maxes:
        return
    step_grid = np.linspace(0, max(step_maxes), 200)

    _fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.grid(True, which="both", color="#f2f2f2", linestyle="-", linewidth=0.5)

    for method in gated_methods:
        color = METHOD_COLORS.get(method, "#333333")
        label = METHOD_LABELS.get(method, method)

        cv_curves = []
        for df in merged_data[method].values():
            required = ["total_steps", "raw_sigma_cv"]
            if not all(col in df.columns for col in required):
                continue
            temp_df = df[required].dropna().sort_values("total_steps")
            if temp_df.empty:
                continue
            cv_curves.append(
                np.interp(
                    step_grid,
                    temp_df["total_steps"].values,
                    temp_df["raw_sigma_cv"].values,
                )
            )

        if not cv_curves:
            continue
        mean_cv = smooth_series(np.mean(cv_curves, axis=0), window=7)
        std_cv = smooth_series(np.std(cv_curves, axis=0), window=7)

        ax.plot(step_grid, mean_cv, label=label, color=color, linewidth=1.8)
        ax.fill_between(
            step_grid, mean_cv - std_cv, mean_cv + std_cv, color=color, alpha=0.15
        )

    ax.set_title(
        f"Ensemble Disagreement Spread Across Population - {env_id}",
        fontsize=13,
        pad=15,
        fontweight="bold",
    )
    ax.set_xlabel("Environmental Interaction Steps", labelpad=10)
    ax.set_ylabel(r"raw $\sigma$ CV (population)", labelpad=10)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, p: f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x / 1e3:.0f}k"
        )
    )
    sns.despine(ax=ax, top=True, right=True)
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.8,
        edgecolor="#f2f2f2",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


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
        "sac",
        "crossq",
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
        t_s_str = f"{t_s:.2f}" if pd.notna(t_s) else "0.00"

        # Baselines without a population (PPO/TD3/DDPG/SAC/CrossQ) never log
        # best_population_fitness — show em-dash instead of "nan +/- 0.00".
        best_pop_str = f"${b_m:.2f} \\pm {b_s:.2f}$" if pd.notna(b_m) else "---"

        tex += f"{label} & ${r_m:.2f} \\pm {r_s_str}$ & {best_pop_str} & ${
            t_m:.2f} \\pm {t_s_str}$ \\\\\n"

    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return tex


def holm_bonferroni(pvalues, alpha=0.05):
    """Holm-Bonferroni step-down correction for multiple comparisons.

    Given ``m`` raw p-values, returns ``(adjusted_pvalues, reject)`` in the SAME
    order as the input. Procedure: sort ascending ``p_(1) <= ... <= p_(m)``; the
    ``i``-th smallest is compared against ``alpha / (m - i + 1)``. Equivalently the
    monotone-enforced adjusted p-value is
    ``p_adj_(i) = max_{j<=i} min((m - j + 1) * p_(j), 1)`` and hypothesis ``i`` is
    rejected at family-wise error rate ``alpha`` iff ``p_adj_i < alpha``.

    Holm-Bonferroni is uniformly more powerful than plain Bonferroni while still
    controlling the family-wise error rate, which matters here because each
    environment's significance table runs up to |PROPOSED| x |baselines| pairwise
    tests as a single family.
    """
    m = len(pvalues)
    if m == 0:
        return [], []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, min((m - rank) * pvalues[idx], 1.0))
        adj[idx] = running_max
    reject = [adj[i] < alpha for i in range(m)]
    return adj, reject


def build_significance_table_latex(env_id, stable_values, alpha=0.05):
    baselines = ["ppo", "td3", "ddpg", "sac", "crossq", "erl", "sc_erl_random"]

    # --- Pass 1: collect every valid pairwise comparison as one family ---
    comparisons = []  # (ours, base, test_name, raw_p)
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
                comparisons.append((ours, base, test_name, float(p_val)))
            except Exception as e:  # noqa: BLE001 -- scipy can raise on degenerate samples; skip this pair, keep testing the rest
                print(f"    Warning: significance test {ours} vs {base} failed: {e}")
                continue

    if not comparisons:
        return "% Insufficient data for statistical testing\n"

    # --- Family-wise Holm-Bonferroni correction over all comparisons ---
    raw_ps = [c[3] for c in comparisons]
    adj_ps, rejects = holm_bonferroni(raw_ps, alpha=alpha)
    m_tests = len(comparisons)

    def _fmt_p(p):
        return f"{p:.4e}" if p < 0.001 else f"{p:.4f}"

    def _stars(padj, reject):
        if not reject:
            return "ns"
        return "***" if padj < 0.001 else ("**" if padj < 0.01 else "*")

    tex = "\\begin{table}[htbp]\n\\centering\n"
    tex += (
        f"\\caption{{Statistical Significance Testing for \\texttt{{{env_id}}} "
        "(Proposed vs Baselines). $p$-values are adjusted with the "
        f"Holm--Bonferroni step-down correction over the family of $m = {m_tests}$ "
        "pairwise comparisons in this environment; significance (Sig.) is decided "
        f"on $p_{{\\mathrm{{adj}}}}$ at family-wise $\\alpha = {alpha}$ "
        "($^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$, ns~=~not significant).}}\n"
    )
    tex += f"\\label{{tab:sig_{env_id}}}\n"
    tex += "\\begin{tabular}{llcccc}\n\\toprule\n"
    tex += (
        "\\textbf{Proposed Method} & \\textbf{Baseline} & \\textbf{Test Type} & "
        "\\textbf{Raw $p$} & \\textbf{Holm $p_{\\mathrm{adj}}$} & \\textbf{Sig.} "
        "\\\\\n\\midrule\n"
    )

    for ours in PROPOSED_METHODS:
        rows = [i for i, c in enumerate(comparisons) if c[0] == ours]
        if not rows:
            continue
        for i in rows:
            _, base, test_name, raw_p = comparisons[i]
            sig = _stars(adj_ps[i], rejects[i])
            tex += (
                f"{METHOD_LABELS.get(ours, ours)} & {METHOD_LABELS.get(base, base)} "
                f"& {test_name} & {_fmt_p(raw_p)} & {_fmt_p(adj_ps[i])} "
                f"& \\textbf{{{sig}}} \\\\\n"
            )
        tex += "\\hline\n"

    present = [
        b
        for b in baselines
        if len([v for v in stable_values.get(b, []) if not np.isnan(v)]) >= 2
    ]
    missing = [b for b in baselines if b not in present]
    if missing:
        missing_labels = ", ".join(METHOD_LABELS.get(m, m) for m in missing)
        tex += f"\\multicolumn{{6}}{{l}}{{\\small\\textit{{Omitted (no data): {missing_labels}}}}} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return tex


def build_gate_quality_table_latex(env_id, gate_quality_values):
    if not gate_quality_values:
        return "% No gate-quality data available\n"

    tex = "\\begin{table}[htbp]\n\\centering\n"
    tex += (
        "\\caption{Gate quality for \\texttt{"
        f"{env_id}"
        "}: how well the gate's uncertainty signal (\\textbf{AUC-u}) and, "
        "separately, behavioural distance (\\textbf{AUC-d}) discriminate "
        "high-error individuals, plus their rank correlation with error "
        "($\\boldsymbol{\\rho(u,e)}$) and the epsilon-pool size the estimate "
        "rests on.}\n"
    )
    tex += f"\\label{{tab:gate_quality_{env_id}}}\n"
    tex += "\\begin{tabular}{lcccc}\n\\toprule\n"
    tex += (
        "\\textbf{Method} & \\textbf{AUC-u} & \\textbf{AUC-d} & "
        "$\\boldsymbol{\\rho(u,e)}$ & \\textbf{n pool} \\\\\n\\midrule\n"
    )
    for method in PROPOSED_METHODS:
        rows = gate_quality_values.get(method)
        if not rows:
            continue
        label = METHOD_LABELS.get(method, method)
        auc_u = np.nanmean([r["auc_u"] for r in rows])
        auc_d = np.nanmean([r["auc_d"] for r in rows])
        spearman = np.nanmean([r["spearman"] for r in rows])
        n_pool = np.nanmean([r["n_pool"] for r in rows])
        auc_d_str = f"{auc_d:.3f}" if pd.notna(auc_d) else "---"
        tex += (
            f"{label} & {auc_u:.3f} & {auc_d_str} & "
            f"{spearman:.3f} & {n_pool:.0f} \\\\\n"
        )
    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return tex


def build_gating_cost_table_latex(env_id, gating_cost_values):
    if not gating_cost_values:
        return "% No gating-cost data available\n"

    tex = "\\begin{table}[htbp]\n\\centering\n"
    tex += (
        "\\caption{Gating cost for \\texttt{"
        f"{env_id}"
        "}: the real-environment sample budget each method actually spent, "
        "so that reward comparisons can be read against equal (or unequal) cost.}\n"
    )
    tex += f"\\label{{tab:gating_cost_{env_id}}}\n"
    tex += "\\begin{tabular}{lcccc}\n\\toprule\n"
    tex += (
        "\\textbf{Method} & $\\boldsymbol{\\rho}$ \\textbf{(final)} & "
        "\\textbf{Real evals/gen} & \\textbf{Generations} & "
        "\\textbf{Surrogate ratio} \\\\\n\\midrule\n"
    )
    for method in SC_ERL_VARIANTS + ["erl"]:
        rows = gating_cost_values.get(method)
        if not rows:
            continue
        label = METHOD_LABELS.get(method, method)
        rho_final = np.nanmean([r["rho_final"] for r in rows])
        real_evals = np.nanmean([r["real_evals_gen"] for r in rows])
        gens = np.nanmean([r["generations"] for r in rows])
        ratio = np.nanmean([r["surrogate_ratio"] for r in rows])

        rho_str = f"{rho_final:.3f}" if pd.notna(rho_final) else "---"
        evals_str = f"{real_evals:.1f}" if pd.notna(real_evals) else "---"
        gens_str = f"{gens:.0f}" if pd.notna(gens) else "---"
        ratio_str = f"{ratio:.3f}" if pd.notna(ratio) else "---"

        tex += f"{label} & {rho_str} & {evals_str} & {gens_str} & {ratio_str} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return tex


# Critical values q_alpha for Nemenyi test, alpha=0.05 (two-tailed)
_NEMENYI_Q = {
    2: 1.960,
    3: 2.344,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
}


def compute_rankings_and_nemenyi(all_stable_values, environments):
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
    common_envs = [e for e in environments if all(e in rank_matrix[m] for m in methods)]
    friedman_p = cd = None
    if len(common_envs) >= 2 and len(methods) >= 2:
        rank_lists = [[rank_matrix[m][e] for e in common_envs] for m in methods]
        try:
            _, friedman_p = stats.friedmanchisquare(*rank_lists)
        except Exception as e:  # noqa: BLE001 -- scipy can raise on degenerate rank data; leave friedman_p as None
            print(f"    Warning: Friedman test failed: {e}")
        k, N = len(methods), len(common_envs)
        q = (
            _NEMENYI_Q.get(k)
            or _NEMENYI_Q[max(k2 for k2 in _NEMENYI_Q if k2 <= min(k, 10))]
        )
        cd = q * np.sqrt(k * (k + 1) / (6 * N))

    return rank_matrix, avg_ranks, cd, friedman_p


def build_nemenyi_ranking_table_latex(all_stable_values, environments):
    rank_matrix, avg_ranks, cd, friedman_p = compute_rankings_and_nemenyi(
        all_stable_values, environments
    )

    method_order = [
        "ppo",
        "td3",
        "ddpg",
        "sac",
        "crossq",
        "erl",
        "sc_erl_random",
        "sc_erl_ensemble",
        "sc_erl_dropout",
        "sc_erl_evidential",
    ]
    present = [m for m in method_order if m in avg_ranks]
    sorted_methods = sorted(present, key=lambda m: avg_ranks.get(m, float("inf")))
    best_method = sorted_methods[0] if sorted_methods else None
    best_avg = avg_ranks.get(best_method, float("nan")) if best_method else float("nan")

    def short_env(e):
        return display_env_id(e)

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
    tex += "\\resizebox{\\textwidth}{!}{%\n"
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
            tex += f"\\textbf{{{label}}} & {' & '.join(cells)} & \\textbf{{{
                avg_str
            }}} \\\\\n"
        else:
            tex += f"{label} & {' & '.join(cells)} & {avg_str} \\\\\n"

    tex += "\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n"
    return tex


def generate_nemenyi_cd_plot(avg_ranks, cd, out_path):
    present = {m: r for m, r in avg_ranks.items() if not np.isnan(r)}
    if not present:
        return

    sorted_methods = sorted(present, key=present.__getitem__)
    n = len(sorted_methods)
    _fig, ax = plt.subplots(figsize=(8, max(2.5, 0.55 * n) + 1.2))

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
            arrowprops={"arrowstyle": "<->", "color": "black", "lw": 1.5},
        )
        ax.text(
            best_r + cd / 2,
            y_top + 0.25,
            f"CD = {cd:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
        # Underline methods within CD of best (not significantly different)
        within = [m for m in sorted_methods if abs(present[m] - best_r) <= cd]
        if len(within) > 1:
            xs = [present[m] for m in within]
            ax.plot(
                [min(xs) - 0.12, max(xs) + 0.12],
                [-0.6, -0.6],
                color="#555555",
                lw=3.5,
                alpha=0.35,
                solid_capstyle="round",
            )
            ax.text(
                np.mean(xs),
                -0.95,
                "no significant difference",
                ha="center",
                fontsize=8,
                color="#555555",
                style="italic",
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


def _interpolate_seed_to_grid(df, step_grid):
    y_col = (
        "eval_reward"
        if "eval_reward" in df.columns and not df["eval_reward"].isna().all()
        else None
    )
    if y_col is None or "total_steps" not in df.columns:
        return None
    tmp = df[["total_steps", y_col]].dropna().sort_values("total_steps")
    if len(tmp) < 2:
        return None
    return np.interp(step_grid, tmp["total_steps"].values, tmp[y_col].values)


def generate_auc_bar_chart(all_merged_data, environments, out_path):
    """Plot A: normalized AUC grouped by budget cutoff, one subplot per env."""
    from matplotlib.patches import Patch

    n_envs = len(environments)
    n_methods = len(METHOD_ORDER_BARS)
    n_budgets = len(BUDGET_CUTOFFS)
    group_width = 0.8
    bar_w = group_width / n_methods
    group_pos = np.arange(n_budgets)

    with plt.rc_context({"font.family": "serif", "text.usetex": False}):
        fig, axes = plt.subplots(1, n_envs, figsize=(20, 4), sharey=False)
        axes = np.atleast_1d(axes)

        for ax, env_id in zip(axes, environments):
            merged_data = all_merged_data[env_id]

            for m_idx, method in enumerate(METHOD_ORDER_BARS):
                color = METHOD_COLORS.get(method, "#555555")
                label = METHOD_LABELS.get(method, method)

                for b_idx, cutoff in enumerate(BUDGET_CUTOFFS):
                    step_grid = np.linspace(0, cutoff, 500)
                    aucs = []
                    if method in merged_data:
                        for df in merged_data[method].values():
                            curve = _interpolate_seed_to_grid(df, step_grid)
                            if curve is not None:
                                span = step_grid[-1] - step_grid[0]
                                aucs.append(
                                    float(np.trapezoid(curve, step_grid) / span)
                                )
                    mean_auc = float(np.mean(aucs)) if aucs else np.nan
                    std_auc = float(np.std(aucs)) if len(aucs) > 1 else 0.0

                    x = group_pos[b_idx] - group_width / 2 + (m_idx + 0.5) * bar_w
                    ax.bar(
                        x,
                        mean_auc if not np.isnan(mean_auc) else 0,
                        width=bar_w * 0.92,
                        color=color,
                        alpha=BUDGET_ALPHAS[b_idx],
                        edgecolor="none",
                        label=label if b_idx == 0 else None,
                    )
                    if not np.isnan(mean_auc) and std_auc > 0:
                        ax.errorbar(
                            x,
                            mean_auc,
                            yerr=std_auc,
                            fmt="none",
                            ecolor="#333333",
                            elinewidth=0.8,
                            capsize=2,
                            capthick=0.8,
                        )

            ax.set_xticks(group_pos)
            ax.set_xticklabels(BUDGET_LABELS)
            ax.set_xlabel("Budget Cutoff")
            ax.set_ylabel("Normalized AUC")
            ax.set_title(display_env_id(env_id), fontweight="bold")
            ax.grid(axis="y", color="#eeeeee", linestyle="-", linewidth=0.5, zorder=0)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Legend: method colors only — budget distinction is already encoded in
        # x-axis group labels (200k / 500k / 1M) and alpha shading
        handles, labels = axes[0].get_legend_handles_labels()
        alpha_handles = [
            Patch(
                facecolor="#888888",
                alpha=BUDGET_ALPHAS[i],
                edgecolor="none",
                label=BUDGET_LABELS[i],
            )
            for i in range(n_budgets)
        ]
        fig.legend(
            handles + alpha_handles,
            labels + BUDGET_LABELS,
            loc="lower center",
            ncol=n_methods + n_budgets,
            frameon=True,
            framealpha=0.9,
            edgecolor="#cccccc",
            fontsize=7.5,
            bbox_to_anchor=(0.5, -0.22),
        )
        plt.suptitle(
            "Normalized Area Under Reward Curve at Three Step Budgets",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()


def generate_relative_improvement_plot(all_merged_data, environments, out_path):
    """Plot C: per-step relative improvement of SC-ERL variants over ERL mean."""
    n_envs = len(environments)
    cap = 1_000_000
    step_grid = np.linspace(0, cap, 500)

    with plt.rc_context({"font.family": "serif", "text.usetex": False}):
        fig, axes = plt.subplots(1, n_envs, figsize=(20, 4), sharey=False)
        axes = np.atleast_1d(axes)

        for ax, env_id in zip(axes, environments):
            merged_data = all_merged_data[env_id]

            erl_curves = [
                _interpolate_seed_to_grid(df, step_grid)
                for df in merged_data.get("erl", {}).values()
            ]
            erl_curves = [c for c in erl_curves if c is not None]
            if not erl_curves:
                ax.set_title(display_env_id(env_id), fontweight="bold")
                ax.text(0.5, 0.5, "No ERL data", transform=ax.transAxes, ha="center")
                continue
            erl_mean = np.mean(erl_curves, axis=0)

            # Floor denominator at the 70th percentile of |ERL| to avoid zero-crossing spikes
            erl_abs = np.abs(erl_mean)
            pos_vals = erl_abs[erl_abs > 0]
            denom_floor = float(np.percentile(pos_vals, 70)) if len(pos_vals) else 1.0
            denom = np.maximum(erl_abs, denom_floor)

            ax.axhline(
                0,
                color="#444444",
                linestyle="--",
                linewidth=1.0,
                zorder=3,
                label="ERL (reference)",
            )

            all_mean_rels = []
            plot_items = []
            for method in SC_ERL_VARIANTS:
                if method not in merged_data:
                    continue
                rel_curves = [
                    (c - erl_mean) / denom
                    for df in merged_data[method].values()
                    if (c := _interpolate_seed_to_grid(df, step_grid)) is not None
                ]
                if not rel_curves:
                    continue
                rel_arr = np.array(rel_curves)
                mean_rel = smooth_series(np.mean(rel_arr, axis=0), window=7)
                std_rel = smooth_series(np.std(rel_arr, axis=0), window=7)
                all_mean_rels.append(mean_rel)
                plot_items.append((method, mean_rel, std_rel))

            if all_mean_rels:
                combined = np.concatenate(all_mean_rels)
                y_lo = np.percentile(combined, 5)
                y_hi = np.percentile(combined, 95)
                pad = max((y_hi - y_lo) * 0.12, 0.05)
                ax.set_ylim(y_lo - pad, y_hi + pad)

            for method, mean_rel, std_rel in plot_items:
                color = METHOD_COLORS.get(method, "#333333")
                label = METHOD_LABELS.get(method, method)
                is_proposed = method in PROPOSED_METHODS
                ax.plot(
                    step_grid,
                    mean_rel,
                    color=color,
                    linewidth=1.8 if is_proposed else 1.2,
                    linestyle="-" if is_proposed else "--",
                    label=label,
                    zorder=4,
                )
                ax.fill_between(
                    step_grid,
                    mean_rel - std_rel,
                    mean_rel + std_rel,
                    color=color,
                    alpha=0.12,
                    zorder=2,
                )

            ax.set_title(display_env_id(env_id), fontweight="bold")
            ax.set_xlabel("Interaction Steps")
            ax.set_ylabel("Relative Improvement over ERL")
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(
                    lambda x, _: f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x / 1e3:.0f}k"
                )
            )
            ax.set_xlim(0, cap)
            ax.grid(color="#eeeeee", linestyle="-", linewidth=0.5, zorder=0)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for ax in reversed(axes):
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(SC_ERL_VARIANTS) + 1,
            frameon=True,
            framealpha=0.9,
            edgecolor="#cccccc",
            fontsize=8.5,
            bbox_to_anchor=(0.5, -0.22),
        )
        plt.suptitle(
            "Relative Improvement over ERL Baseline",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()


def _build_group_section(
    group_label, group_tag, group_envs, all_merged_data, all_stable_values, output_dir
):
    """Generate Nemenyi ranking + CD diagram + AUC + relative-improvement for one env group."""
    if not group_envs:
        return ""
    tex = f"\\section{{{group_label} — Ranking and Cross-Environment Analysis}}\n"

    sv_subset = {e: all_stable_values[e] for e in group_envs if e in all_stable_values}
    md_subset = {e: all_merged_data[e] for e in group_envs if e in all_merged_data}

    if len(group_envs) >= 2:
        _, avg_ranks, cd, _ = compute_rankings_and_nemenyi(sv_subset, group_envs)
        cd_path = os.path.join(output_dir, f"nemenyi_cd_diagram_{group_tag}.png")
        try:
            generate_nemenyi_cd_plot(avg_ranks, cd, cd_path)
            has_cd = True
        except Exception as e:  # noqa: BLE001 -- best-effort report generation: skip this plot/table on failure, keep the rest
            print(f"Warning: CD diagram [{group_label}]: {e}")
            has_cd = False

        tex += build_nemenyi_ranking_table_latex(sv_subset, group_envs)
        if has_cd:
            tex += (
                "\\begin{figure}[H]\n\\centering\n"
                f"  \\includegraphics[width=0.72\\textwidth]{{{os.path.basename(cd_path)}}}\n"
                f"  \\caption{{Critical Difference diagram ({group_label}). "
                "Methods connected by the grey bar are not significantly different "
                "from the best-ranked method at $\\alpha = 0.05$.}}\n"
                "\\end{figure}\n"
            )
    else:
        tex += "% Only one environment in this group — ranking skipped.\n"

    if len(group_envs) >= 2:
        auc_path = os.path.join(output_dir, f"auc_bar_chart_{group_tag}.png")
        try:
            generate_auc_bar_chart(md_subset, group_envs, auc_path)
            has_auc = True
        except Exception as e:  # noqa: BLE001 -- best-effort report generation: skip this plot/table on failure, keep the rest
            print(f"Warning: AUC chart [{group_label}]: {e}")
            has_auc = False

        rel_path = os.path.join(output_dir, f"relative_improvement_{group_tag}.png")
        try:
            generate_relative_improvement_plot(md_subset, group_envs, rel_path)
            has_rel = True
        except Exception as e:  # noqa: BLE001 -- best-effort report generation: skip this plot/table on failure, keep the rest
            print(f"Warning: Relative improvement [{group_label}]: {e}")
            has_rel = False

        if has_auc:
            tex += (
                "\\subsection{Normalized Area Under Reward Curve}\n"
                "\\begin{figure}[H]\n\\centering\n"
                f"  \\includegraphics[width=\\textwidth]{{{os.path.basename(auc_path)}}}\n"
                "  \\caption{Normalized AUC (trapezoidal, divided by step budget) at three "
                "environmental interaction budgets: 200k, 500k, and 1M steps. "
                "Error bars show $\\pm 1$ std across seeds.}\n"
                "\\end{figure}\n\n"
            )
        if has_rel:
            tex += (
                "\\subsection{Relative Improvement over ERL Baseline}\n"
                "\\begin{figure}[H]\n\\centering\n"
                f"  \\includegraphics[width=\\textwidth]{{{os.path.basename(rel_path)}}}\n"
                "  \\caption{Per-step relative improvement of SC-ERL variants over the "
                "mean ERL reward curve. "
                "Denominator floored at 70th percentile of $|r_{\\text{ERL}}|$ to suppress "
                "zero-crossing artefacts. Shaded band: $\\pm 1$ std across seeds.}\n"
                "\\end{figure}\n"
            )

    tex += "\\newpage\n"
    return tex


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    eval_reward_dir = os.path.join(base_dir, "eval_reward")

    if not os.path.exists(eval_reward_dir):
        print(f"Error: Database structure directory '{eval_reward_dir}' not found.")
        return

    env_files = glob.glob(os.path.join(eval_reward_dir, "*.csv"))
    seen_envs: set = set()
    environments = []
    for f in sorted(env_files):
        norm = normalize_env_id(os.path.basename(f).replace(".csv", ""))
        if norm not in seen_envs:
            seen_envs.add(norm)
            environments.append(norm)

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
    all_merged_data = {}  # env_id -> merged_data (kept for cross-env plots)

    for env_id in environments:
        print(f"Processing environment: {env_id}...")
        merged_data = load_environment_data(env_id, base_dir)
        all_merged_data[env_id] = merged_data
        stable_values = get_stable_final_values(merged_data)
        all_stable_values[env_id] = stable_values

        gate_quality_values = get_gate_quality_values(merged_data)
        gating_cost_values = get_gating_cost_values(env_id, merged_data, base_dir)

        se_path = os.path.join(output_dir, f"{env_id}_sample_efficiency.png")
        gd_path = os.path.join(output_dir, f"{env_id}_gate_dynamics.png")
        gs_path = os.path.join(output_dir, f"{env_id}_gate_scatter.png")
        bu_path = os.path.join(output_dir, f"{env_id}_behavioral_uncertainty.png")
        spd_path = os.path.join(output_dir, f"{env_id}_speedup.png")
        rat_path = os.path.join(output_dir, f"{env_id}_ratio.png")
        sc_path = os.path.join(output_dir, f"{env_id}_raw_sigma_cv.png")

        # Generowanie wykresów otoczone blokami try-except dla bezpieczeństwa .tex
        try:
            generate_sample_efficiency_plot(env_id, merged_data, se_path)
            has_se = True
        except Exception as e:  # noqa: BLE001 -- best-effort report generation: skip this plot/table on failure, keep the rest
            print(
                f"Warning: Could not generate sample efficiency plot for {env_id}: {e}"
            )
            has_se = False

        try:
            generate_gate_dynamics_plot(env_id, merged_data, gd_path)
            has_gd = True
        except Exception as e:  # noqa: BLE001 -- best-effort report generation: skip this plot/table on failure, keep the rest
            print(f"Warning: Could not generate gate dynamics plot for {env_id}: {e}")
            has_gd = False

        try:
            generate_gate_scatter_plot(env_id, merged_data, gs_path)
            has_gs = True
        except Exception as e:  # noqa: BLE001 -- best-effort report generation: skip this plot/table on failure, keep the rest
            print(f"Warning: Could not generate gate scatter plot for {env_id}: {e}")
            has_gs = False

        try:
            generate_behavioral_uncertainty_scatter_plot(env_id, merged_data, bu_path)
            has_bu = True
        except Exception as e:  # noqa: BLE001 -- best-effort report generation: skip this plot/table on failure, keep the rest
            print(
                f"Warning: Could not generate behavioral/uncertainty scatter plot "
                f"for {env_id}: {e}"
            )
            has_bu = False

        try:
            generate_speedup_plot(env_id, merged_data, spd_path)
            has_spd = True
        except Exception as e:  # noqa: BLE001 -- best-effort report generation: skip this plot/table on failure, keep the rest
            print(f"Warning: Could not generate speedup plot for {env_id}: {e}")
            has_spd = False

        try:
            generate_ratio_plot(env_id, merged_data, rat_path)
            has_rat = True
        except Exception as e:  # noqa: BLE001 -- best-effort report generation: skip this plot/table on failure, keep the rest
            print(f"Warning: Could not generate ratio plot for {env_id}: {e}")
            has_rat = False

        try:
            generate_raw_sigma_cv_plot(env_id, merged_data, sc_path)
            has_sc = True
        except Exception as e:  # noqa: BLE001 -- best-effort report generation: skip this plot/table on failure, keep the rest
            print(f"Warning: Could not generate raw sigma CV plot for {env_id}: {e}")
            has_sc = False

        latex_document += (
            f"\\section{{Environment Results: \\texttt{{{display_env_id(env_id)}}}}}\n"
        )

        if has_se:
            latex_document += (
                f"\\begin{{figure}}[H]\n\\centering\n"
                f"  \\includegraphics[width=0.85\\textwidth]{{{
                    os.path.basename(se_path)
                }}}\n"
                f"  \\caption{{Sample Efficiency comparison across evaluation metrics on {
                    env_id
                }.}}\n"
                f"\\end{{figure}}\n\n"
            )

        latex_document += build_summary_table_latex(env_id, base_dir)
        latex_document += build_significance_table_latex(env_id, stable_values)
        latex_document += build_gate_quality_table_latex(env_id, gate_quality_values)
        latex_document += build_gating_cost_table_latex(env_id, gating_cost_values)

        if has_gd:
            latex_document += (
                f"\\begin{{figure}}[H]\n\\centering\n"
                f"  \\includegraphics[width=0.95\\textwidth]{{{
                    os.path.basename(gd_path)
                }}}\n"
                f"  \\caption{{Gate controller dynamics on {env_id}: adaptive gated fraction "
                f"$\\rho$ against the windowed surrogate error $\\hat{{e}}$, showing the "
                f"controller reacting to its own error estimate.}}\n"
                f"\\end{{figure}}\n\n"
            )

        if has_gs:
            latex_document += (
                f"\\begin{{figure}}[H]\n\\centering\n"
                f"  \\includegraphics[width=0.95\\textwidth]{{{
                    os.path.basename(gs_path)
                }}}\n"
                f"  \\caption{{Mean uncertainty vs. surrogate error on {env_id}, "
                f"aggregated per generation over the full run.}}\n"
                f"\\end{{figure}}\n"
            )

        if has_bu:
            latex_document += (
                f"\\begin{{figure}}[H]\n\\centering\n"
                f"  \\includegraphics[width=0.95\\textwidth]{{{
                    os.path.basename(bu_path)
                }}}\n"
                f"  \\caption{{Mean epistemic uncertainty of the population policies "
                f"$\\pi_j$ vs. their mean behavioural distance from the RL actor on "
                f"{env_id}, aggregated per generation over the full run "
                f"(Pearson $r$, scipy).}}\n"
                f"\\end{{figure}}\n\n"
            )

        if has_spd:
            latex_document += (
                f"\\begin{{figure}}[H]\n\\centering\n"
                f"  \\includegraphics[width=0.85\\textwidth]{{{
                    os.path.basename(spd_path)
                }}}\n"
                f"  \\caption{{Evolutionary speedup: number of generations completed per environmental step on {
                    env_id
                }. "
                f"A steeper slope indicates more surrogate-driven generations within the same interaction budget.}}\n"
                f"\\end{{figure}}\n\n"
            )

        if has_rat:
            latex_document += (
                f"\\begin{{figure}}[H]\n\\centering\n"
                f"  \\includegraphics[width=0.85\\textwidth]{{{
                    os.path.basename(rat_path)
                }}}\n"
                f"  \\caption{{Surrogate utilization dynamics on {env_id}. "
                f"Dotted vertical lines mark generations where a seed's recorded history ends; "
                f"steps in the mean curve beyond that point are an interpolation artefact from "
                f"fewer contributing seeds, not a real change in utilization.}}\n"
                f"\\end{{figure}}\n\n"
            )

        if has_sc:
            latex_document += (
                f"\\begin{{figure}}[H]\n\\centering\n"
                f"  \\includegraphics[width=0.85\\textwidth]{{{
                    os.path.basename(sc_path)
                }}}\n"
                f"  \\caption{{Raw ensemble disagreement, coefficient of variation "
                f"across the population ($\\sigma$/mean of \\texttt{{raw\\_sigma}}) on "
                f"{env_id}. Explains why the fixed MAD-based threshold "
                f"(\\texttt{{gate\\_mode=relative}}) failed to separate individuals: "
                f"population spread stays in the 0.5--1\\% range.}}\n"
                f"\\end{{figure}}\n\n"
            )

        latex_document += "\\newpage\n"

    # ---- Per-benchmark Nemenyi + cross-env plots ----
    mujoco_envs = [
        e
        for e in environments
        if not e.startswith("dm_control_") and not e.lower().startswith("myo")
    ]
    dmc_envs = [e for e in environments if e.startswith("dm_control_")]
    myo_envs = [e for e in environments if e.lower().startswith("myo")]

    if mujoco_envs:
        print(f"\nBuilding MuJoCo ranking section ({len(mujoco_envs)} envs)...")
        latex_document += _build_group_section(
            "MuJoCo Environments",
            "mujoco",
            mujoco_envs,
            all_merged_data,
            all_stable_values,
            output_dir,
        )

    if dmc_envs:
        print(f"\nBuilding DMC ranking section ({len(dmc_envs)} envs)...")
        latex_document += _build_group_section(
            "DMC Dog Environments",
            "dmc",
            dmc_envs,
            all_merged_data,
            all_stable_values,
            output_dir,
        )

    if myo_envs:
        print(f"\nBuilding MyoSuite ranking section ({len(myo_envs)} envs)...")
        latex_document += _build_group_section(
            "MyoSuite Environments",
            "myosuite",
            myo_envs,
            all_merged_data,
            all_stable_values,
            output_dir,
        )

    # ---- Combined Nemenyi over all environments ----
    if len(environments) >= 2:
        print("\nComputing combined (all-environment) Nemenyi ranking...")
        _, avg_ranks, cd, _ = compute_rankings_and_nemenyi(
            all_stable_values, environments
        )
        cd_path = os.path.join(output_dir, "nemenyi_cd_diagram_all.png")
        try:
            generate_nemenyi_cd_plot(avg_ranks, cd, cd_path)
            has_cd = True
        except Exception as e:  # noqa: BLE001 -- best-effort report generation: skip this plot/table on failure, keep the rest
            print(f"Warning: Could not generate combined CD diagram: {e}")
            has_cd = False

        latex_document += (
            "\\section{Global Ranking Analysis — All Environments (Nemenyi)}\n"
        )
        latex_document += build_nemenyi_ranking_table_latex(
            all_stable_values, environments
        )
        if has_cd:
            latex_document += (
                "\\begin{figure}[H]\n\\centering\n"
                f"  \\includegraphics[width=0.72\\textwidth]{{{os.path.basename(cd_path)}}}\n"
                "  \\caption{Critical Difference diagram across all environments (MuJoCo + DMC + MyoSuite). "
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
