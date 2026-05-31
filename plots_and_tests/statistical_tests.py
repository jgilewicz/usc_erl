import glob
import os
import re
import numpy as np
import pandas as pd
import scipy.stats as stats
from process_results import METHOD_LABELS, load_environment_data

def get_stable_final_values(env_id, merged_data):
    """
    Extracts stable final run values by averaging the selected metric (rl_reward or eval_reward)
    over the final 10% of environmental interaction steps for each method and seed.
    This ensures that statistical significance tests compare converged performance.
    """
    stable_values = {}
    for method in merged_data:
        stable_values[method] = []
        for seed, df in merged_data[method].items():
            if 'total_steps' not in df.columns:
                continue
            
            # Select first available metric containing valid non-null rewards
            if 'rl_reward' in df.columns and (not df['rl_reward'].isna().all()):
                y_metric = 'rl_reward'
            elif 'eval_reward' in df.columns and (not df['eval_reward'].isna().all()):
                y_metric = 'eval_reward'
            else:
                continue
            
            # Compute reward average over the last 10% steps of the run
            max_steps = df['total_steps'].max()
            df_last_10 = df[df['total_steps'] >= max_steps * 0.9].dropna(subset=[y_metric])
            if not df_last_10.empty:
                mean_val = df_last_10[y_metric].mean()
                stable_values[method].append(mean_val)
    return stable_values

def run_significance_tests(env_id, stable_values, base_dir='results'):
    """
    Executes automated statistical significance testing comparing the proposed SC-ERL variants
    (Ensemble & Dropout) against established baselines (PPO, TD3, DDPG, ERL, ERL-Random).
    
    The testing pipeline follows a rigorous methodology:
    1. Normality verification: Shapiro-Wilk test is performed on each group.
       - Safeguards are active to prevent SciPy errors: requires N >= 3 and variance > 0.
    2. Decision engine:
       - If BOTH groups are normal (p >= 0.05), perform Welch's t-test (handles unequal variances, two-sided).
       - Otherwise, fall back to the non-parametric Mann-Whitney U test (two-sided).
    3. Output: Compiles significance markers (*, **, ***, ns) and exports results to CSV and LaTeX.
    """
    proposed_methods = ['sc_erl_ensemble', 'sc_erl_dropout']
    baselines = ['ppo', 'td3', 'ddpg', 'erl', 'sc_erl_random']
    rows = []
    
    for ours in proposed_methods:
        # Pre-process group data: convert to numpy array and remove any NaNs
        group_A = np.array(stable_values.get(ours, []))
        group_A = group_A[~np.isnan(group_A)]
        if len(group_A) < 2:
            continue
        label_A = METHOD_LABELS.get(ours, ours)
        
        for base in baselines:
            group_B = np.array(stable_values.get(base, []))
            group_B = group_B[~np.isnan(group_B)]
            if len(group_B) < 2:
                continue
            label_B = METHOD_LABELS.get(base, base)
            comparison = f'{label_A} vs {label_B}'
            
            p_shapiro_A = np.nan
            p_shapiro_B = np.nan
            
            # Shapiro-Wilk Normality test with robust numerical safety checks
            # Avoids ValueError: 'mu and sigma must be finite' or 'sample size too small'
            if len(group_A) >= 3 and np.var(group_A) > 0:
                _, p_shapiro_A = stats.shapiro(group_A)
            if len(group_B) >= 3 and np.var(group_B) > 0:
                _, p_shapiro_B = stats.shapiro(group_B)
                
            is_normal_A = pd.notna(p_shapiro_A) and p_shapiro_A >= 0.05
            is_normal_B = pd.notna(p_shapiro_B) and p_shapiro_B >= 0.05
            
            # Perform parametric Welch's t-test or non-parametric Mann-Whitney U test
            if is_normal_A and is_normal_B:
                test_name = "Welch's t-test"
                _, p_val = stats.ttest_ind(group_A, group_B, equal_var=False, alternative='two-sided')
            else:
                test_name = 'Mann-Whitney U'
                _, p_val = stats.mannwhitneyu(group_A, group_B, alternative='two-sided')
                
            # Classify significance stars based on scientific publishing standards
            if p_val < 0.001:
                sig = '***'
            elif p_val < 0.01:
                sig = '**'
            elif p_val < 0.05:
                sig = '*'
            else:
                sig = 'ns'
                
            shapiro_A_str = f'{p_shapiro_A:.4f}' if pd.notna(p_shapiro_A) else 'N/A'
            shapiro_B_str = f'{p_shapiro_B:.4f}' if pd.notna(p_shapiro_B) else 'N/A'
            shapiro_str = f'({shapiro_A_str}, {shapiro_B_str})'
            
            rows.append({
                'Comparison': comparison,
                'Shapiro p-value (A, B)': shapiro_str,
                'Test (Welch / M-W)': test_name,
                'p-value': p_val,
                'Significance': sig
            })
            
    df_tests = pd.DataFrame(rows)
    if df_tests.empty:
        return
        
    out_dir = os.path.join(base_dir, env_id)
    os.makedirs(out_dir, exist_ok=True)
    
    # Save raw CSV significance records
    csv_out = os.path.join(out_dir, f'{env_id}_significance_table.csv')
    df_tests.to_csv(csv_out, index=False)
    
    # Format p-values beautifully (scientific notation for small numbers, decimals for larger)
    latex_out = os.path.join(out_dir, f'{env_id}_significance_table.tex')
    df_latex = df_tests.copy()
    df_latex['p-value'] = df_latex['p-value'].apply(lambda x: f'{x:.4e}' if x < 0.001 else f'{x:.4f}')
    
    # Export LaTeX code using modern Pandas 2.x Styler interface to avoid deprecation warnings
    with open(latex_out, 'w') as f:
        f.write(df_latex.style.hide(axis="index").to_latex(
            column_format='lcccc', 
            caption=f"Statistical Significance testing for {env_id} based on Shapiro-Wilk normality checking and subsequent Welch's t-test / Mann-Whitney U test.", 
            label=f'tab:sig_{env_id}'
        ))

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = script_dir
    project_root = os.path.dirname(script_dir)
    output_base_dir = os.path.join(project_root, 'results')
    
    eval_reward_dir = os.path.join(base_dir, 'eval_reward')
    if not os.path.exists(eval_reward_dir):
        return
        
    env_files = glob.glob(os.path.join(eval_reward_dir, '*.csv'))
    if not env_files:
        return
        
    environments = [os.path.basename(f).replace('.csv', '') for f in env_files]
    for env_id in environments:
        # Load telemetry records, compute converged metrics, run the significance tests
        merged_data = load_environment_data(env_id, base_dir)
        stable_values = get_stable_final_values(env_id, merged_data)
        run_significance_tests(env_id, stable_values, output_base_dir)

if __name__ == '__main__':
    main()