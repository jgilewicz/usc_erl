import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Use professional paper style
plt.style.use('seaborn-v0_8-paper')

# Set professional plotting style with light grid and clean background
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# Define colorblind-friendly colors (derived from seaborn colorblind palette)
METHOD_COLORS = {
    'sc_erl_dropout': '#0173b2',   # Blue
    'sc_erl_ensemble': '#029e73',  # Green
    'erl': '#de8f05',              # Orange
    'ppo': '#d55e00',              # Red/Vermillion
    'td3': '#cc78bc',              # Purple
    'sc_erl_random': '#949494',    # Gray
}

METHOD_LABELS = {
    'ppo': 'PPO (Baseline)',
    'td3': 'TD3 (Baseline)',
    'erl': 'ERL (Baseline)',
    'sc_erl_ensemble': 'SC-ERL (Ensemble) [Ours]',
    'sc_erl_dropout': 'SC-ERL (Dropout) [Ours]',
    'sc_erl_random': 'SC-ERL (Random)',
}

def parse_column_header(col_name, env_id):
    """
    Parses column header format: 'method_env_seedX - metric'
    Returns: (method, seed, metric) or (None, None, None)
    """
    pattern = rf"^([a-z_0-9]+)_{re.escape(env_id)}_seed(\d+)\s*-\s*([a-z_]+)$"
    match = re.match(pattern, col_name)
    if match:
        method = match.group(1)
        seed = int(match.group(2))
        metric = match.group(3)
        return method, seed, metric
    return None, None, None

def smooth_series(series, window=7):
    """
    Smooth a 1D array using a rolling mean to eliminate noisy spikes.
    """
    s = pd.Series(series)
    return s.rolling(window=window, min_periods=1).mean().values

def load_environment_data(env_id, base_dir="results"):
    """
    Loads all metric CSVs for a given environment and merges them by run (method, seed).
    Imputes missing total_steps for evolutionary runs using seed mapping.
    Converts string 'Infinity' values to NaN for robust numeric parsing.
    """
    metrics = [
        'total_steps', 'eval_reward', 'best_population_fitness', 
        'avg_population_fitness', 'uncertainty_mean', 'uncertainty_max', 
        'uncertainty_threshold', 'generation'
    ]
    
    run_data = {}
    
    for metric in metrics:
        file_path = os.path.join(base_dir, metric, f"{env_id}.csv")
        if not os.path.exists(file_path):
            continue
            
        df = pd.read_csv(file_path)
        
        # Robustly clean data: replace string 'Infinity' and 'inf' with NaN
        df = df.replace(['Infinity', 'inf', 'inf.0'], np.nan)
        
        for col in df.columns:
            if col == 'Step' or col.endswith('__MIN') or col.endswith('__MAX'):
                continue
                
            method, seed, parsed_metric = parse_column_header(col, env_id)
            if method is None or parsed_metric != metric:
                continue
                
            if method not in run_data:
                run_data[method] = {}
            if seed not in run_data[method]:
                run_data[method][seed] = []
                
            # Convert values to float numeric types
            sub_df = df[['Step', col]].copy()
            sub_df[col] = pd.to_numeric(sub_df[col], errors='coerce')
            sub_df = sub_df.dropna()
            
            sub_df = sub_df.rename(columns={col: metric})
            run_data[method][seed].append(sub_df)
            
    # Merge metrics for each run
    merged_data = {}
    for method in run_data:
        merged_data[method] = {}
        for seed in run_data[method]:
            dfs = run_data[method][seed]
            if not dfs:
                continue
            merged_df = dfs[0]
            for next_df in dfs[1:]:
                merged_df = pd.merge(merged_df, next_df, on='Step', how='outer')
            merged_df = merged_df.sort_values('Step').reset_index(drop=True)
            merged_data[method][seed] = merged_df
            
    # IMPUTATION LOGIC:
    # 1. For PPO/TD3, Step column is exactly equivalent to total_steps
    # 2. For ERL/SC_ERL, we map Step (generation) to average total_steps of other seeds
    for method in list(merged_data.keys()):
        is_evo = method in ['erl', 'sc_erl_ensemble', 'sc_erl_dropout', 'sc_erl_random']
        if not is_evo:
            for seed in merged_data[method]:
                df = merged_data[method][seed]
                if 'total_steps' not in df.columns or df['total_steps'].isna().all():
                    df['total_steps'] = df['Step']
            continue
            
        seeds_with_steps = [s for s, df in merged_data[method].items() if 'total_steps' in df.columns and not df['total_steps'].isna().all()]
        if seeds_with_steps:
            all_pairs = []
            for s in seeds_with_steps:
                temp = merged_data[method][s][['Step', 'total_steps']].dropna()
                all_pairs.append(temp)
            if all_pairs:
                combined_steps = pd.concat(all_pairs).groupby('Step')['total_steps'].mean().to_dict()
                for seed in merged_data[method]:
                    df = merged_data[method][seed]
                    if 'total_steps' not in df.columns or df['total_steps'].isna().all():
                        df['total_steps'] = df['Step'].map(combined_steps)
                        df['total_steps'] = df['total_steps'].interpolate(method='linear').ffill().bfill()
                        
    return merged_data

def generate_sample_efficiency_plot(env_id, merged_data, out_path):
    """
    Plots eval_reward/best_population_fitness vs total_steps.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.grid(True, which='both', color='#f2f2f2', linestyle='-', linewidth=0.5)
    
    step_grid = np.linspace(0, 1000000, 200)
    
    for method in sorted(merged_data.keys()):
        color = METHOD_COLORS.get(method, '#333333')
        label = METHOD_LABELS.get(method, method)
        
        is_evo = method in ['erl', 'sc_erl_ensemble', 'sc_erl_dropout', 'sc_erl_random']
        y_metric = 'best_population_fitness' if is_evo else 'eval_reward'
        
        is_proposed = method in ['sc_erl_ensemble', 'sc_erl_dropout']
        linewidth = 2.5 if is_proposed else 1.5
        line_alpha = 1.0 if is_proposed else 0.6
        linestyle = '-' if is_proposed else ':'
        fill_alpha = 0.15 if is_proposed else 0.08
        
        interpolated_ys = []
        
        for seed, df in merged_data[method].items():
            if 'total_steps' not in df.columns or y_metric not in df.columns:
                continue
            temp_df = df[['total_steps', y_metric]].dropna()
            if temp_df.empty:
                continue
            temp_df = temp_df.sort_values('total_steps')
            
            steps = temp_df['total_steps'].values
            ys = temp_df[y_metric].values
            
            interp_y = np.interp(step_grid, steps, ys)
            interpolated_ys.append(interp_y)
            
        if not interpolated_ys:
            continue
            
        interpolated_ys = np.array(interpolated_ys)
        mean_y = np.mean(interpolated_ys, axis=0)
        std_y = np.std(interpolated_ys, axis=0)
        
        # Apply smoothing to the mean curves to make them look publication-ready
        mean_y_smooth = smooth_series(mean_y, window=5)
        std_y_smooth = smooth_series(std_y, window=5)
        
        ax.plot(step_grid, mean_y_smooth, label=label, color=color, linewidth=linewidth, linestyle=linestyle, alpha=line_alpha)
        ax.fill_between(step_grid, mean_y_smooth - std_y_smooth, mean_y_smooth + std_y_smooth, color=color, alpha=fill_alpha)
        
    ax.set_title(f"Sample Efficiency Comparison - {env_id}", fontsize=13, pad=15, fontweight='bold')
    ax.set_xlabel("Environmental Interaction Steps", labelpad=10)
    ax.set_ylabel("Evaluation Reward / Best Population Fitness", labelpad=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}k'))
    
    sns.despine(ax=ax, top=True, right=True)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#f2f2f2')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def generate_surrogate_analysis_plot(env_id, merged_data, out_path):
    """
    Plots uncertainty_mean, uncertainty_max, uncertainty_threshold and avg_population_fitness vs generation.
    Incorporates curve smoothing (window=7), clean upper-shading, and ignores early infinite threshold values.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)
    
    surrogate_methods = ['sc_erl_ensemble', 'sc_erl_dropout']
    titles = ['SC-ERL (Ensemble Method) [Ours]', 'SC-ERL (Dropout Method) [Ours]']
    
    for i, method in enumerate(surrogate_methods):
        ax = axes[i]
        
        # Soft, clean grid
        ax.grid(True, which='both', color='#f2f2f2', linestyle='-', linewidth=0.5)
        
        if method not in merged_data:
            ax.text(0.5, 0.5, f"No data for {method}", ha='center', va='center')
            continue
            
        color = METHOD_COLORS[method]
        title = titles[i]
        
        gen_grid = np.linspace(0, 1400, 141)
        
        mean_uncertainties = []
        max_uncertainties = []
        threshold_uncertainties = []
        avg_fitnesses = []
        
        for seed, df in merged_data[method].items():
            required = ['generation', 'uncertainty_mean', 'uncertainty_max', 'uncertainty_threshold', 'avg_population_fitness']
            if not all(col in df.columns for col in required):
                continue
            
            temp_df = df[required].copy()
            # Drop rows with NaN in key plotting metrics
            temp_df = temp_df.dropna(subset=['generation', 'uncertainty_mean'])
            if temp_df.empty:
                continue
            temp_df = temp_df.sort_values('generation')
            
            gens = temp_df['generation'].values
            
            # Interpolate seeds separately
            mean_uncertainties.append(np.interp(gen_grid, gens, temp_df['uncertainty_mean'].values))
            
            # Handle max uncertainty (mask any NaNs)
            max_u_vals = temp_df['uncertainty_max'].values
            valid_max = pd.Series(max_u_vals).interpolate(method='linear').ffill().bfill().values
            max_uncertainties.append(np.interp(gen_grid, gens, valid_max))
            
            # Handle threshold uncertainty (mask any infinite/NaN values during warmup)
            thresh_vals = temp_df['uncertainty_threshold'].values
            valid_thresh = pd.Series(thresh_vals).interpolate(method='linear').ffill().bfill().values
            # If still all NaN (no threshold active), fallback to NaN
            if np.isnan(valid_thresh).all():
                threshold_uncertainties.append(np.full_like(gen_grid, np.nan))
            else:
                threshold_uncertainties.append(np.interp(gen_grid, gens, valid_thresh))
                
            avg_fitnesses.append(np.interp(gen_grid, gens, temp_df['avg_population_fitness'].values))
            
        if not mean_uncertainties:
            ax.text(0.5, 0.5, f"Empty dataset for {method}", ha='center', va='center')
            continue
            
        # Average across seeds
        avg_mean_u = np.mean(mean_uncertainties, axis=0)
        avg_max_u = np.mean(max_uncertainties, axis=0)
        
        # For threshold, average only over valid non-nan entries across seeds at each index
        avg_thresh_u = np.nanmean(threshold_uncertainties, axis=0)
        avg_fit = np.mean(avg_fitnesses, axis=0)
        
        # Apply smoothing (window=7) to eliminate noisy spikes
        avg_mean_u_smooth = smooth_series(avg_mean_u, window=7)
        avg_max_u_smooth = smooth_series(avg_max_u, window=7)
        avg_thresh_u_smooth = smooth_series(avg_thresh_u, window=7)
        avg_fit_smooth = smooth_series(avg_fit, window=7)
        
        # Left Y Axis: Uncertainty Values
        ax.set_xlabel("Generations (Training Progress)", labelpad=10)
        ax.set_ylabel("Uncertainty Estimation Value", color=color, labelpad=10)
        ax.tick_params(axis='y', labelcolor=color)
        
        # Solid bold line for the mean uncertainty
        line_mean, = ax.plot(gen_grid, avg_mean_u_smooth, color=color, linewidth=2.5, label="Uncertainty Mean")
        
        # Dashed line for the threshold
        line_thresh, = ax.plot(gen_grid, avg_thresh_u_smooth, color=color, linestyle="--", linewidth=1.5, label="Uncertainty Threshold")
        
        # Shaded area only above the mean (between mean and max) to keep it clean and neat
        shade_max = ax.fill_between(gen_grid, avg_mean_u_smooth, avg_max_u_smooth, color=color, alpha=0.10, label="Uncertainty Max")
        
        # Right Y Axis: Average Population Fitness (orange)
        ax2 = ax.twinx()
        ax2.set_ylabel("Average Population Fitness", color="#de8f05", labelpad=10)
        ax2.tick_params(axis='y', labelcolor="#de8f05")
        ax2.grid(False)
        
        line_fit, = ax2.plot(gen_grid, avg_fit_smooth, color="#de8f05", linewidth=2.0, linestyle="-.", label="Avg Pop Fitness")
        
        # Combine legends
        lines = [line_mean, line_thresh, line_fit]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='lower left', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#f2f2f2')
        
        ax.set_title(title, fontsize=12, pad=12, fontweight='bold')
        sns.despine(ax=ax, top=True, left=False, right=False)
        
    plt.suptitle(f"Surrogate Controller Uncertainty & Fitness Analysis - {env_id}", fontsize=14, y=0.98, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def generate_summary_table(env_id, base_dir="results"):
    """
    Saves strictly raw float numerical values to summary_table.csv (no formatted string prints).
    """
    summary_path = os.path.join(base_dir, "summary", f"{env_id}.csv")
    if not os.path.exists(summary_path):
        return
        
    df = pd.read_csv(summary_path)
    
    parsed_rows = []
    for idx, row in df.iterrows():
        name = row['Name']
        pattern = rf"^([a-z_0-9]+)_{re.escape(env_id)}_seed(\d+)$"
        match = re.match(pattern, name)
        if match:
            method = match.group(1)
            seed = int(match.group(2))
            
            eval_reward = row['eval_reward']
            rl_reward = row['rl_reward']
            best_pop = row['best_population_fitness']
            
            parsed_rows.append({
                'method': method,
                'seed': seed,
                'eval_reward': rl_reward if pd.notna(rl_reward) else eval_reward,
                'best_population_fitness': best_pop
            })
            
    parsed_df = pd.DataFrame(parsed_rows)
    if parsed_df.empty:
        return
        
    summary_stats = parsed_df.groupby('method').agg({
        'eval_reward': ['mean', 'std'],
        'best_population_fitness': ['mean', 'std']
    })
    
    summary_stats.columns = [f"{col[0]}_{col[1]}" for col in summary_stats.columns]
    summary_stats = summary_stats.reset_index()
    
    csv_rows = []
    method_order = ['ppo', 'td3', 'erl', 'sc_erl_random', 'sc_erl_ensemble', 'sc_erl_dropout']
    
    for method in method_order:
        row_data = summary_stats[summary_stats['method'] == method]
        if row_data.empty:
            continue
            
        row_data = row_data.iloc[0]
        label = METHOD_LABELS.get(method, method)
        
        csv_rows.append({
            'Algorithm/Method': label,
            'Eval_Reward_Mean': row_data['eval_reward_mean'],
            'Eval_Reward_Std': row_data['eval_reward_std'],
            'Best_Pop_Fitness_Mean': row_data['best_population_fitness_mean'],
            'Best_Pop_Fitness_Std': row_data['best_population_fitness_std']
        })
        
    csv_table = pd.DataFrame(csv_rows)
    
    csv_out = os.path.join(base_dir, f"{env_id}_summary_table.csv")
    csv_table.to_csv(csv_out, index=False)

def generate_critic_correlation_plot(env_id, base_dir="results"):
    """
    Plots Scatter plots and trendlines showing the correlation between Critic Loss and Uncertainty Mean.
    """
    df_loss_path = os.path.join(base_dir, "critic_loss", f"{env_id}.csv")
    df_mean_path = os.path.join(base_dir, "uncertainty_mean", f"{env_id}.csv")
    
    if not os.path.exists(df_loss_path) or not os.path.exists(df_mean_path):
        return
        
    df_loss = pd.read_csv(df_loss_path)
    df_mean = pd.read_csv(df_mean_path)
    
    # We will plot the three runs side-by-side
    runs = [
        ('sc_erl_ensemble', 4, 'Ensemble Seed 4'),
        ('sc_erl_ensemble', 3, 'Ensemble Seed 3'),
        ('sc_erl_dropout', 4, 'Dropout Seed 4')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    correlations = []
    
    for i, (method, seed, label) in enumerate(runs):
        ax = axes[i]
        ax.grid(True, which='both', color='#f2f2f2', linestyle='-', linewidth=0.5)
        
        col_loss = f"{method}_{env_id}_seed{seed} - critic_loss"
        col_mean = f"{method}_{env_id}_seed{seed} - uncertainty_mean"
        
        if col_loss not in df_loss.columns or col_mean not in df_mean.columns:
            ax.text(0.5, 0.5, f"Data missing for {label}", ha='center', va='center')
            continue
            
        sub_loss = df_loss[['Step', col_loss]].copy()
        sub_mean = df_mean[['Step', col_mean]].copy()
        
        sub_loss[col_loss] = pd.to_numeric(sub_loss[col_loss].replace('Infinity', np.nan), errors='coerce')
        sub_mean[col_mean] = pd.to_numeric(sub_mean[col_mean].replace('Infinity', np.nan), errors='coerce')
        
        merged = pd.merge(sub_loss, sub_mean, on='Step', how='inner').dropna()
        if merged.empty:
            ax.text(0.5, 0.5, f"No overlap for {label}", ha='center', va='center')
            continue
            
        x = merged[col_loss].values
        y = merged[col_mean].values
        
        p_corr = merged[col_loss].corr(merged[col_mean], method='pearson')
        s_corr = merged[col_loss].rank().corr(merged[col_mean].rank(), method='pearson')
        
        correlations.append({
            'Algorithm/Method': label,
            'Pearson Correlation': f"{p_corr:.4f}",
            'Spearman Correlation': f"{s_corr:.4f}",
            'Sample Size (N)': len(merged)
        })
        
        color = METHOD_COLORS[method]
        
        # Plot Scatter
        ax.scatter(x, y, color=color, alpha=0.4, s=20, edgecolor='none', label='Step Metrics')
        
        # Fit linear regression trendline
        slope, intercept = np.polyfit(x, y, 1)
        x_grid = np.linspace(min(x), max(x), 100)
        ax.plot(x_grid, slope * x_grid + intercept, color='#333333', linestyle='--', linewidth=1.5, label='Trendline')
        
        ax.set_title(f"{label}\n(Pearson r = {p_corr:.3f})", fontsize=11, pad=10, fontweight='bold')
        ax.set_xlabel("Critic TD Loss", labelpad=8)
        ax.set_ylabel("Critic Epistemic Uncertainty", labelpad=8)
        ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#f2f2f2')
        sns.despine(ax=ax, top=True, right=True)
        
    plt.suptitle(f"Critic Epistemic Uncertainty vs. TD Loss Correlation - {env_id}", fontsize=13, y=0.98, fontweight='bold')
    plt.tight_layout()
    
    out_path = os.path.join(base_dir, f"{env_id}_critic_correlation.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    # Save correlation table as CSV
    corr_df = pd.DataFrame(correlations)
    csv_out = os.path.join(base_dir, f"{env_id}_critic_correlation.csv")
    corr_df.to_csv(csv_out, index=False)

def main():
    base_dir = "results"
    
    eval_reward_dir = os.path.join(base_dir, "eval_reward")
    if not os.path.exists(eval_reward_dir):
        return
        
    env_files = glob.glob(os.path.join(eval_reward_dir, "*.csv"))
    if not env_files:
        return
        
    environments = [os.path.basename(f).replace(".csv", "") for f in env_files]
    
    for env_id in environments:
        merged_data = load_environment_data(env_id, base_dir)
        
        se_plot_path = os.path.join(base_dir, f"{env_id}_sample_efficiency.png")
        generate_sample_efficiency_plot(env_id, merged_data, se_plot_path)
        
        sa_plot_path = os.path.join(base_dir, f"{env_id}_surrogate_analysis.png")
        generate_surrogate_analysis_plot(env_id, merged_data, sa_plot_path)
        
        generate_summary_table(env_id, base_dir)
        generate_critic_correlation_plot(env_id, base_dir)

if __name__ == "__main__":
    main()
