import os
import re
import shutil

def parse_tex_table(filepath):
    if not os.path.exists(filepath):
        print(f'Warning: {filepath} does not exist.')
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    tabular_match = re.search('\\\\begin{tabular}{.*?}(.*?)\\\\end{tabular}', content, re.DOTALL)
    if not tabular_match:
        print(f'Warning: Could not parse tabular block in {filepath}')
        return None
    tabular_content = tabular_match.group(1).strip()
    lines = tabular_content.split('\n')
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('\\toprule') or line.startswith('\\midrule') or line.startswith('\\bottomrule') or line.startswith('\\hline'):
            continue
        clean_line = re.sub('\\\\\\\\(\\s*\\[.*?\\])?', '', line).strip()
        clean_line = clean_line.replace('\\%', '%').replace('\\_', '_').replace('\\&', '&').replace('\\#', '#')
        cols = [c.strip() for c in clean_line.split('&')]
        if len(cols) > 0 and any(cols):
            rows.append(cols)
    if not rows:
        return ''
    headers = rows[0]
    data_rows = rows[1:]
    md_table = '| ' + ' | '.join(headers) + ' |\n'
    md_table += '| ' + ' | '.join(['---'] * len(headers)) + ' |\n'
    for r in data_rows:
        md_table += '| ' + ' | '.join(r) + ' |\n'
    return md_table

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, 'results')

    from omegaconf import OmegaConf
    config_path = os.path.join(project_root, 'configs', 'download.yaml')
    if os.path.exists(config_path):
        config = OmegaConf.load(config_path)
        environments = list(config.environments)
    else:
        environments = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

    workspace_report_path = os.path.join(results_dir, 'full_report.md')
    brain_dir = '/Users/kuba/.gemini/antigravity-ide/brain/7050f68b-db2d-4da6-a245-3612cfba39bf'
    brain_results_dir = os.path.join(brain_dir, 'results')
    os.makedirs(brain_results_dir, exist_ok=True)
    brain_report_path = os.path.join(brain_dir, 'full_report.md')
    
    env_list_str = ', '.join(environments)
    md_content = f'# Evolutionary Reinforcement Learning - Statistical & Performance Report\nThis report contains performance results, statistical significance tests, critic correlation tables, and experimental plots for the evaluated environments: {env_list_str}.\n\n---\n'
    for env in environments:
        env_dir = os.path.join(results_dir, env)
        print(f'Processing {env}...')
        md_content += f'\n## {env}\n\n'
        summary_tex = os.path.join(env_dir, f'{env}_summary_table.tex')
        if os.path.exists(summary_tex):
            md_content += '### Performance Summary Table\n\n'
            summary_md = parse_tex_table(summary_tex)
            if summary_md:
                md_content += summary_md + '\n'
        sig_tex = os.path.join(env_dir, f'{env}_significance_table.tex')
        if os.path.exists(sig_tex):
            md_content += '### Statistical Significance Table\n\n'
            sig_md = parse_tex_table(sig_tex)
            if sig_md:
                md_content += sig_md + '\n'
        corr_tex = os.path.join(env_dir, f'{env}_critic_correlation.tex')
        if os.path.exists(corr_tex):
            md_content += '### Critic Correlation Analysis\n\n'
            corr_md = parse_tex_table(corr_tex)
            if corr_md:
                md_content += corr_md + '\n'
        md_content += '### Performance & Analysis Plots\n\n'
        plot_names = [('Sample Efficiency', f'{env}_sample_efficiency.png'), ('Surrogate Analysis', f'{env}_surrogate_analysis.png'), ('Critic Correlation', f'{env}_critic_correlation.png')]
        brain_env_dir = os.path.join(brain_results_dir, env)
        os.makedirs(brain_env_dir, exist_ok=True)
        for caption, filename in plot_names:
            src_img = os.path.join(env_dir, filename)
            if os.path.exists(src_img):
                dst_img = os.path.join(brain_env_dir, filename)
                shutil.copy2(src_img, dst_img)
    workspace_md = md_content
    for env in environments:
        pass
    ws_content = md_content
    for env in environments:
        ws_content += f'\n### Plots - {env}\n'
        ws_content += f'![Sample Efficiency - {env}](./{env}/{env}_sample_efficiency.png)\n'
        ws_content += f'![Surrogate Analysis - {env}](./{env}/{env}_surrogate_analysis.png)\n'
        ws_content += f'![Critic Correlation - {env}](./{env}/{env}_critic_correlation.png)\n'
        ws_content += '\n---\n'
    with open(workspace_report_path, 'w', encoding='utf-8') as f:
        f.write(ws_content)
    print(f'Saved workspace report to {workspace_report_path}')
    brain_content = md_content
    for env in environments:
        brain_content += f'\n### Plots - {env}\n'
        brain_content += f'![Sample Efficiency - {env}](/Users/kuba/.gemini/antigravity-ide/brain/7050f68b-db2d-4da6-a245-3612cfba39bf/results/{env}/{env}_sample_efficiency.png)\n'
        brain_content += f'![Surrogate Analysis - {env}](/Users/kuba/.gemini/antigravity-ide/brain/7050f68b-db2d-4da6-a245-3612cfba39bf/results/{env}/{env}_surrogate_analysis.png)\n'
        brain_content += f'![Critic Correlation - {env}](/Users/kuba/.gemini/antigravity-ide/brain/7050f68b-db2d-4da6-a245-3612cfba39bf/results/{env}/{env}_critic_correlation.png)\n'
        brain_content += '\n---\n'
    with open(brain_report_path, 'w', encoding='utf-8') as f:
        f.write(brain_content)
    print(f'Saved brain report to {brain_report_path}')
if __name__ == '__main__':
    main()