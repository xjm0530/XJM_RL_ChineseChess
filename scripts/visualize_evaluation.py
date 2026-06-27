import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch

mpl.rcParams['font.family'] = ['Times New Roman', 'SimHei', 'DejaVu Sans']
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['lines.linewidth'] = 1.8
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.width'] = 1.2
mpl.rcParams['ytick.major.width'] = 1.2
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['axes.labelsize'] = 13
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['figure.dpi'] = 150

log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'jsons', 'evaluation_results.json')

with open(log_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

elo_data = data['elo_rating']
match_data = data['match_results']

models = elo_data['models']
model_keys = ['imitation', 'rl', 'rl_mcts']
model_names = [models[k]['name'] for k in model_keys]
model_elos = [models[k]['final_elo'] for k in model_keys]
model_colors = [models[k]['color'] for k in model_keys]
model_short = [models[k]['short_name'] for k in model_keys]


def plot_elo_bar():
    fig, ax = plt.subplots(figsize=(9, 6))
    
    x = np.arange(len(model_keys))
    bars = ax.bar(x, model_elos, color=model_colors, width=0.6, 
                  edgecolor='white', linewidth=1.5, zorder=3)
    
    for bar, elo, name in zip(bars, model_elos, model_short):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 15,
                f'{elo:.0f}', ha='center', va='bottom',
                fontsize=13, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel('ELO Rating')
    ax.set_title('(a) ELO Rating Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(model_elos) * 1.15)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y', zorder=0)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig


def plot_elo_curve():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for key in model_keys:
        history = models[key]['elo_history']
        n = len(history)
        rounds = np.arange(n)
        ax.plot(rounds, history, color=models[key]['color'], 
                label=models[key]['name'], linewidth=2.2, marker='o', 
                markersize=4, markevery=max(1, n//10))
    
    ax.set_xlabel('Evaluation Round')
    ax.set_ylabel('ELO Rating')
    ax.set_title('(b) ELO Rating Progression', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


def plot_match_pies():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    
    match_keys = ['imitation_vs_rl', 'imitation_vs_rl_mcts', 'rl_vs_rl_mcts']
    titles = [
        '(a) IL vs RL',
        '(b) IL vs RL+MCTS',
        '(c) RL vs RL+MCTS'
    ]
    
    for idx, (mk, title) in enumerate(zip(match_keys, titles)):
        match = match_data[mk]
        ax = axes[idx]
        
        sizes = [match['a_wins'], match['b_wins'], match['draws']]
        labels = [f"{match['model_a_short']} Win\n({match['a_wins']})",
                  f"{match['model_b_short']} Win\n({match['b_wins']})",
                  f"Draw\n({match['draws']})"]
        colors = ['#1f77b4', '#d62728', '#7f7f7f']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                            autopct='%1.1f%%', startangle=90,
                                            textprops={'fontsize': 10},
                                            pctdistance=0.7,
                                            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax.set_title(title, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_match_bars():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    match_keys = ['imitation_vs_rl', 'imitation_vs_rl_mcts', 'rl_vs_rl_mcts']
    match_labels = ['IL vs RL', 'IL vs RL+MCTS', 'RL vs RL+MCTS']
    
    x = np.arange(len(match_keys))
    width = 0.25
    
    a_wins = [match_data[mk]['a_wins'] for mk in match_keys]
    b_wins = [match_data[mk]['b_wins'] for mk in match_keys]
    draws = [match_data[mk]['draws'] for mk in match_keys]
    
    a_labels = [match_data[mk]['model_a_short'] for mk in match_keys]
    b_labels = [match_data[mk]['model_b_short'] for mk in match_keys]
    
    bars1 = ax.bar(x - width, a_wins, width, label='Model A Wins', 
                   color='#1f77b4', edgecolor='white', linewidth=1, zorder=3)
    bars2 = ax.bar(x, b_wins, width, label='Model B Wins', 
                   color='#d62728', edgecolor='white', linewidth=1, zorder=3)
    bars3 = ax.bar(x + width, draws, width, label='Draws', 
                   color='#7f7f7f', edgecolor='white', linewidth=1, zorder=3)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(match_labels, fontsize=11)
    ax.set_ylabel('Number of Games')
    ax.set_title('(a) Head-to-Head Match Results (200 games each)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y', zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 220)
    
    plt.tight_layout()
    return fig


def plot_illegal_move_rate():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    illegal_rates = [
        match_data['imitation_vs_rl']['avg_illegal_rate_a'] * 100,
        match_data['imitation_vs_rl']['avg_illegal_rate_b'] * 100,
        match_data['rl_vs_rl_mcts']['avg_illegal_rate_b'] * 100,
    ]
    
    x = np.arange(3)
    bars = ax.bar(x, illegal_rates, color=model_colors, width=0.6,
                  edgecolor='white', linewidth=1.5, zorder=3)
    
    for bar, rate, name in zip(bars, illegal_rates, model_short):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{rate:.1f}%', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel('Illegal Move Rate (%)')
    ax.set_title('(b) Illegal Move Rate Comparison', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(illegal_rates) * 1.25)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y', zorder=0)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig


def plot_radar():
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, polar=True)
    
    categories = ['ELO Rating', 'Win Rate vs IL', 'Win Rate vs RL', 'Illegal Rate (inverse)', 'Avg Game Length']
    
    il_wr_vs_il = 50.0
    rl_wr_vs_il = match_data['imitation_vs_rl']['b_win_rate'] * 100
    mcts_wr_vs_il = match_data['imitation_vs_rl_mcts']['b_win_rate'] * 100
    
    il_wr_vs_rl = match_data['imitation_vs_rl']['a_win_rate'] * 100
    rl_wr_vs_rl = 50.0
    mcts_wr_vs_rl = match_data['rl_vs_rl_mcts']['b_win_rate'] * 100
    
    max_elo = 1200.0
    il_elo_norm = models['imitation']['final_elo'] / max_elo * 100
    rl_elo_norm = models['rl']['final_elo'] / max_elo * 100
    mcts_elo_norm = models['rl_mcts']['final_elo'] / max_elo * 100
    
    il_illegal_inv = (1 - match_data['imitation_vs_rl']['avg_illegal_rate_a']) * 100
    rl_illegal_inv = (1 - match_data['imitation_vs_rl']['avg_illegal_rate_b']) * 100
    mcts_illegal_inv = (1 - match_data['rl_vs_rl_mcts']['avg_illegal_rate_b']) * 100
    
    il_len_norm = match_data['imitation_vs_rl']['avg_moves'] / 120.0 * 100
    rl_len_norm = match_data['imitation_vs_rl']['avg_moves'] / 120.0 * 100
    mcts_len_norm = match_data['rl_vs_rl_mcts']['avg_moves'] / 120.0 * 100
    
    values = [
        [il_elo_norm, il_wr_vs_il, il_wr_vs_rl, il_illegal_inv, il_len_norm],
        [rl_elo_norm, rl_wr_vs_il, rl_wr_vs_rl, rl_illegal_inv, rl_len_norm],
        [mcts_elo_norm, mcts_wr_vs_il, mcts_wr_vs_rl, mcts_illegal_inv, mcts_len_norm],
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    for i, (v, color, name) in enumerate(zip(values, model_colors, model_short)):
        v_cyclic = v + v[:1]
        ax.plot(angles, v_cyclic, color=color, linewidth=2, label=name, marker='o', markersize=5)
        ax.fill(angles, v_cyclic, color=color, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
    ax.set_title('(c) Multi-dimensional Comparison', fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


def plot_combined():
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(model_keys))
    bars = ax1.bar(x, model_elos, color=model_colors, width=0.6, 
                  edgecolor='white', linewidth=1.5, zorder=3)
    for bar, elo in zip(bars, model_elos):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 15,
                f'{elo:.0f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_short, fontsize=11)
    ax1.set_ylabel('ELO Rating')
    ax1.set_title('(a) ELO Rating Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(model_elos) * 1.15)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y', zorder=0)
    ax1.set_axisbelow(True)
    
    ax2 = fig.add_subplot(gs[0, 1])
    for key in model_keys:
        history = models[key]['elo_history']
        n = len(history)
        rounds = np.arange(n)
        ax2.plot(rounds, history, color=models[key]['color'], 
                label=models[key]['name'], linewidth=2, marker='o', 
                markersize=4, markevery=max(1, n//10))
    ax2.set_xlabel('Evaluation Round')
    ax2.set_ylabel('ELO Rating')
    ax2.set_title('(b) ELO Rating Progression', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    ax3 = fig.add_subplot(gs[1, 0])
    match_keys = ['imitation_vs_rl', 'imitation_vs_rl_mcts', 'rl_vs_rl_mcts']
    match_labels = ['IL vs RL', 'IL vs RL+MCTS', 'RL vs RL+MCTS']
    x = np.arange(len(match_keys))
    width = 0.25
    a_wins = [match_data[mk]['a_wins'] for mk in match_keys]
    b_wins = [match_data[mk]['b_wins'] for mk in match_keys]
    draws = [match_data[mk]['draws'] for mk in match_keys]
    bars1 = ax3.bar(x - width, a_wins, width, label='Model A Wins', 
                   color='#1f77b4', edgecolor='white', linewidth=1, zorder=3)
    bars2 = ax3.bar(x, b_wins, width, label='Model B Wins', 
                   color='#d62728', edgecolor='white', linewidth=1, zorder=3)
    bars3 = ax3.bar(x + width, draws, width, label='Draws', 
                   color='#7f7f7f', edgecolor='white', linewidth=1, zorder=3)
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(match_labels, fontsize=10)
    ax3.set_ylabel('Number of Games')
    ax3.set_title('(c) Head-to-Head Match Results', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y', zorder=0)
    ax3.set_axisbelow(True)
    ax3.set_ylim(0, 220)
    
    ax4 = fig.add_subplot(gs[1, 1])
    illegal_rates = [
        match_data['imitation_vs_rl']['avg_illegal_rate_a'] * 100,
        match_data['imitation_vs_rl']['avg_illegal_rate_b'] * 100,
        match_data['rl_vs_rl_mcts']['avg_illegal_rate_b'] * 100,
    ]
    x = np.arange(3)
    bars = ax4.bar(x, illegal_rates, color=model_colors, width=0.6,
                  edgecolor='white', linewidth=1.5, zorder=3)
    for bar, rate in zip(bars, illegal_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_short, fontsize=11)
    ax4.set_ylabel('Illegal Move Rate (%)')
    ax4.set_title('(d) Illegal Move Rate Comparison', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, max(illegal_rates) * 1.25)
    ax4.grid(True, alpha=0.3, linestyle='--', axis='y', zorder=0)
    ax4.set_axisbelow(True)
    
    return fig


output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'figures')
os.makedirs(output_dir, exist_ok=True)

print("生成ELO对比柱状图...")
fig1 = plot_elo_bar()
fig1.savefig(os.path.join(output_dir, 'eval_elo_bar.png'), dpi=300, bbox_inches='tight')
fig1.savefig(os.path.join(output_dir, 'eval_elo_bar.pdf'), bbox_inches='tight')
plt.close(fig1)

print("生成ELO变化曲线图...")
fig2 = plot_elo_curve()
fig2.savefig(os.path.join(output_dir, 'eval_elo_curve.png'), dpi=300, bbox_inches='tight')
fig2.savefig(os.path.join(output_dir, 'eval_elo_curve.pdf'), bbox_inches='tight')
plt.close(fig2)

print("生成对弈结果饼图...")
fig3 = plot_match_pies()
fig3.savefig(os.path.join(output_dir, 'eval_match_pies.png'), dpi=300, bbox_inches='tight')
fig3.savefig(os.path.join(output_dir, 'eval_match_pies.pdf'), bbox_inches='tight')
plt.close(fig3)

print("生成对弈结果柱状图...")
fig4 = plot_match_bars()
fig4.savefig(os.path.join(output_dir, 'eval_match_bars.png'), dpi=300, bbox_inches='tight')
fig4.savefig(os.path.join(output_dir, 'eval_match_bars.pdf'), bbox_inches='tight')
plt.close(fig4)

print("生成非法动作率对比图...")
fig5 = plot_illegal_move_rate()
fig5.savefig(os.path.join(output_dir, 'eval_illegal_rate.png'), dpi=300, bbox_inches='tight')
fig5.savefig(os.path.join(output_dir, 'eval_illegal_rate.pdf'), bbox_inches='tight')
plt.close(fig5)

print("生成雷达图...")
fig6 = plot_radar()
fig6.savefig(os.path.join(output_dir, 'eval_radar.png'), dpi=300, bbox_inches='tight')
fig6.savefig(os.path.join(output_dir, 'eval_radar.pdf'), bbox_inches='tight')
plt.close(fig6)

print("生成组合大图...")
fig7 = plot_combined()
fig7.savefig(os.path.join(output_dir, 'eval_combined.png'), dpi=300, bbox_inches='tight')
fig7.savefig(os.path.join(output_dir, 'eval_combined.pdf'), bbox_inches='tight')
plt.close(fig7)

print(f"\n所有评测图表已保存到: {output_dir}")
print("生成的文件:")
for f in ['eval_elo_bar', 'eval_elo_curve', 'eval_match_pies', 'eval_match_bars', 
          'eval_illegal_rate', 'eval_radar', 'eval_combined']:
    print(f"  - {f}.png / {f}.pdf")
