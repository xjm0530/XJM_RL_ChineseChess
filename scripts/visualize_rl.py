import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

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
                         'data', 'jsons', 'rl_training_log.json')

with open(log_path, 'r', encoding='utf-8') as f:
    log_data = json.load(f)

timesteps = np.array([d['timestep'] for d in log_data])
ep_rew_mean = np.array([d['rollout/ep_rew_mean'] for d in log_data])
ep_len_mean = np.array([d['rollout/ep_len_mean'] for d in log_data])
value_loss = np.array([d['train/value_loss'] for d in log_data])
pg_loss = np.array([d['train/policy_gradient_loss'] for d in log_data])
entropy_loss = np.array([d['train/entropy_loss'] for d in log_data])
approx_kl = np.array([d['train/approx_kl'] for d in log_data])
clip_fraction = np.array([d['train/clip_fraction'] for d in log_data])
explained_variance = np.array([d['train/explained_variance'] for d in log_data])
learning_rate = np.array([d['train/learning_rate'] for d in log_data])


def smooth_curve(data, window=5):
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='same')


def format_timestep(x, pos):
    if x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}k'
    return f'{x:.0f}'


def plot_reward_and_len():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    ax = axes[0]
    ax.plot(timesteps, ep_rew_mean, color='#1f77b4', alpha=0.35, linewidth=0.8)
    ax.plot(timesteps, smooth_curve(ep_rew_mean, 7), color='#1f77b4', 
            label='Mean Reward', linewidth=2.2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    best_idx = np.argmax(ep_rew_mean)
    best_ts = timesteps[best_idx]
    best_rew = ep_rew_mean[best_idx]
    ax.scatter(best_ts, best_rew, color='red', s=60, zorder=5, marker='*')
    ax.annotate(f'Best: {best_rew:.2f}\n({best_ts/1000:.0f}k steps)', 
                xy=(best_ts, best_rew), xytext=(best_ts - 80000, best_rew - 0.3),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', alpha=0.9))
    
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('(a) Training Reward Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_formatter(FuncFormatter(format_timestep))
    
    ax = axes[1]
    ax.plot(timesteps, ep_len_mean, color='#2ca02c', alpha=0.35, linewidth=0.8)
    ax.plot(timesteps, smooth_curve(ep_len_mean, 7), color='#2ca02c', 
            label='Mean Episode Length', linewidth=2.2)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Mean Episode Length')
    ax.set_title('(b) Episode Length Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_formatter(FuncFormatter(format_timestep))
    
    plt.tight_layout()
    return fig


def plot_losses():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.plot(timesteps, value_loss, color='#d62728', alpha=0.3, linewidth=0.8)
    ax.plot(timesteps, smooth_curve(value_loss, 7), color='#d62728', 
            label='Value Loss', linewidth=2)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Value Loss')
    ax.set_title('(a) Value Network Loss', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_formatter(FuncFormatter(format_timestep))
    
    ax = axes[0, 1]
    ax.plot(timesteps, pg_loss, color='#ff7f0e', alpha=0.3, linewidth=0.8)
    ax.plot(timesteps, smooth_curve(pg_loss, 7), color='#ff7f0e', 
            label='Policy Gradient Loss', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Policy Gradient Loss')
    ax.set_title('(b) Policy Gradient Loss', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_formatter(FuncFormatter(format_timestep))
    
    ax = axes[1, 0]
    ax.plot(timesteps, np.abs(entropy_loss), color='#9467bd', alpha=0.3, linewidth=0.8)
    ax.plot(timesteps, smooth_curve(np.abs(entropy_loss), 7), color='#9467bd', 
            label='|Entropy|', linewidth=2)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Policy Entropy (abs)')
    ax.set_title('(c) Policy Entropy (Exploration)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_formatter(FuncFormatter(format_timestep))
    
    ax = axes[1, 1]
    ax.plot(timesteps, explained_variance, color='#17becf', alpha=0.3, linewidth=0.8)
    ax.plot(timesteps, smooth_curve(explained_variance, 7), color='#17becf', 
            label='Explained Variance', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Explained Variance')
    ax.set_title('(d) Value Function Explained Variance', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_formatter(FuncFormatter(format_timestep))
    
    plt.tight_layout()
    return fig


def plot_ppo_metrics():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    ax = axes[0]
    ax.plot(timesteps, approx_kl, color='#e377c2', alpha=0.3, linewidth=0.8)
    ax.plot(timesteps, smooth_curve(approx_kl, 7), color='#e377c2', 
            label='Approx KL Divergence', linewidth=2.2)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Approx KL Divergence')
    ax.set_title('(a) Policy KL Divergence', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_formatter(FuncFormatter(format_timestep))
    
    ax = axes[1]
    ax.plot(timesteps, clip_fraction, color='#7f7f7f', alpha=0.3, linewidth=0.8)
    ax.plot(timesteps, smooth_curve(clip_fraction, 7), color='#7f7f7f', 
            label='Clip Fraction', linewidth=2.2)
    ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, linewidth=1, label='clip_range=0.2')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Clip Fraction')
    ax.set_title('(b) Clipping Fraction', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_formatter(FuncFormatter(format_timestep))
    
    plt.tight_layout()
    return fig


def plot_combined():
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(timesteps, ep_rew_mean, color='#1f77b4', alpha=0.3, linewidth=0.8)
    ax1.plot(timesteps, smooth_curve(ep_rew_mean, 7), color='#1f77b4', 
             label='Mean Reward', linewidth=2.2)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    best_idx = np.argmax(ep_rew_mean)
    best_ts = timesteps[best_idx]
    best_rew = ep_rew_mean[best_idx]
    ax1.scatter(best_ts, best_rew, color='red', s=70, zorder=5, marker='*',
                edgecolors='darkred', linewidth=0.5)
    ax1.annotate(f'Best: {best_rew:.2f}\n({best_ts/1000:.0f}k steps)', 
                xy=(best_ts, best_rew), xytext=(best_ts - 70000, best_rew - 0.35),
                fontsize=9.5, arrowprops=dict(arrowstyle='->', color='darkred', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='darkred', alpha=0.95))
    
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Mean Episode Reward')
    ax1.set_title('(a) Training Reward', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.xaxis.set_major_formatter(FuncFormatter(format_timestep))
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(timesteps, value_loss, color='#d62728', alpha=0.3, linewidth=0.8)
    ax2.plot(timesteps, smooth_curve(value_loss, 7), color='#d62728', 
             label='Value Loss', linewidth=2.2)
    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Value Loss')
    ax2.set_title('(b) Value Network Loss', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.xaxis.set_major_formatter(FuncFormatter(format_timestep))
    
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(timesteps, pg_loss, color='#ff7f0e', alpha=0.3, linewidth=0.8)
    ax3.plot(timesteps, smooth_curve(pg_loss, 7), color='#ff7f0e', 
             label='Policy Gradient Loss', linewidth=2.2)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax3.set_xlabel('Timesteps')
    ax3.set_ylabel('Policy Gradient Loss')
    ax3.set_title('(c) Policy Gradient Loss', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.xaxis.set_major_formatter(FuncFormatter(format_timestep))
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(timesteps, np.abs(entropy_loss), color='#9467bd', alpha=0.3, linewidth=0.8)
    ax4.plot(timesteps, smooth_curve(np.abs(entropy_loss), 7), color='#9467bd', 
             label='|Entropy|', linewidth=2.2)
    ax4.set_xlabel('Timesteps')
    ax4.set_ylabel('Policy Entropy (abs)')
    ax4.set_title('(d) Policy Entropy', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.xaxis.set_major_formatter(FuncFormatter(format_timestep))
    
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(timesteps, approx_kl, color='#e377c2', alpha=0.3, linewidth=0.8)
    ax5.plot(timesteps, smooth_curve(approx_kl, 7), color='#e377c2', 
             label='Approx KL', linewidth=2.2)
    ax5.set_xlabel('Timesteps')
    ax5.set_ylabel('Approx KL Divergence')
    ax5.set_title('(e) KL Divergence', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right', framealpha=0.9)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.xaxis.set_major_formatter(FuncFormatter(format_timestep))
    
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(timesteps, clip_fraction, color='#7f7f7f', alpha=0.3, linewidth=0.8)
    ax6.plot(timesteps, smooth_curve(clip_fraction, 7), color='#7f7f7f', 
             label='Clip Fraction', linewidth=2.2)
    ax6.axhline(y=0.2, color='red', linestyle='--', alpha=0.4, linewidth=1)
    ax6.text(timesteps[-1] * 0.95, 0.21, 'clip_range=0.2', color='red', 
             fontsize=9, ha='right')
    ax6.set_xlabel('Timesteps')
    ax6.set_ylabel('Clip Fraction')
    ax6.set_title('(f) Clipping Fraction', fontsize=12, fontweight='bold')
    ax6.legend(loc='upper right', framealpha=0.9)
    ax6.grid(True, alpha=0.3, linestyle='--')
    ax6.xaxis.set_major_formatter(FuncFormatter(format_timestep))
    
    return fig


output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'figures')
os.makedirs(output_dir, exist_ok=True)

print("生成奖励和回合长度图...")
fig1 = plot_reward_and_len()
fig1.savefig(os.path.join(output_dir, 'rl_reward_length.png'), dpi=300, bbox_inches='tight')
fig1.savefig(os.path.join(output_dir, 'rl_reward_length.pdf'), bbox_inches='tight')
plt.close(fig1)

print("生成损失函数图...")
fig2 = plot_losses()
fig2.savefig(os.path.join(output_dir, 'rl_losses.png'), dpi=300, bbox_inches='tight')
fig2.savefig(os.path.join(output_dir, 'rl_losses.pdf'), bbox_inches='tight')
plt.close(fig2)

print("生成PPO指标图...")
fig3 = plot_ppo_metrics()
fig3.savefig(os.path.join(output_dir, 'rl_ppo_metrics.png'), dpi=300, bbox_inches='tight')
fig3.savefig(os.path.join(output_dir, 'rl_ppo_metrics.pdf'), bbox_inches='tight')
plt.close(fig3)

print("生成组合大图...")
fig4 = plot_combined()
fig4.savefig(os.path.join(output_dir, 'rl_combined.png'), dpi=300, bbox_inches='tight')
fig4.savefig(os.path.join(output_dir, 'rl_combined.pdf'), bbox_inches='tight')
plt.close(fig4)

print(f"\n所有RL图表已保存到: {output_dir}")
print("生成的文件:")
for f in ['rl_reward_length.png/pdf', 'rl_losses.png/pdf', 
          'rl_ppo_metrics.png/pdf', 'rl_combined.png/pdf']:
    print(f"  - {f}")

print(f"\n统计信息:")
print(f"  总时间步: {timesteps[-1]:,}")
print(f"  初始奖励: {ep_rew_mean[0]:.3f}")
print(f"  最终奖励: {ep_rew_mean[-1]:.3f}")
best_idx = np.argmax(ep_rew_mean)
print(f"  最高奖励: {ep_rew_mean[best_idx]:.3f} ({timesteps[best_idx]:,} 步)")
print(f"  初始价值损失: {value_loss[0]:.4f}")
print(f"  最终价值损失: {value_loss[-1]:.4f}")
print(f"  初始解释方差: {explained_variance[0]:.4f}")
print(f"  最终解释方差: {explained_variance[-1]:.4f}")
print(f"  初始熵: {abs(entropy_loss[0]):.2f}")
print(f"  最终熵: {abs(entropy_loss[-1]):.2f}")
