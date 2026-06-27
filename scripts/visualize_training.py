import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

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
                         'data', 'jsons', 'imitation_training_log.json')

with open(log_path, 'r', encoding='utf-8') as f:
    log = json.load(f)

epochs = np.array(log['epochs'])
train_loss = np.array(log['train_loss'])
train_acc = np.array(log['train_acc']) * 100
val_loss = np.array(log['val_loss'])
val_acc = np.array(log['val_acc']) * 100
lr = np.array(log['lr'])
stages = np.array(log['stage'])


def smooth_curve(data, window=5):
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='same')


def plot_overview():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    colors = {'train': '#1f77b4', 'val': '#d62728'}
    
    ax = axes[0]
    ax.plot(epochs, train_loss, color=colors['train'], alpha=0.3, linewidth=0.8)
    ax.plot(epochs, smooth_curve(train_loss, 7), color=colors['train'], 
            label='Training Loss', linewidth=2.2)
    ax.plot(epochs, val_loss, color=colors['val'], alpha=0.3, linewidth=0.8)
    ax.plot(epochs, smooth_curve(val_loss, 7), color=colors['val'], 
            label='Validation Loss', linewidth=2.2)
    
    for s in [80, 160]:
        ax.axvline(x=s, color='gray', linestyle='--', alpha=0.6, linewidth=1)
    
    stage_info = [(40, 'Stage 1\n(Normal)', 6.5), 
                  (120, 'Stage 2\n(Normal)', 4.5), 
                  (200, 'Stage 3\n(High Quality)', 2.5)]
    for s, name, y in stage_info:
        ax.text(s, y, name, ha='center', va='center', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           edgecolor='lightgray', alpha=0.9))
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('(a) Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, shadow=False)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 240)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax = axes[1]
    ax.plot(epochs, train_acc, color=colors['train'], alpha=0.3, linewidth=0.8)
    ax.plot(epochs, smooth_curve(train_acc, 7), color=colors['train'], 
            label='Training Accuracy', linewidth=2.2)
    ax.plot(epochs, val_acc, color=colors['val'], alpha=0.3, linewidth=0.8)
    ax.plot(epochs, smooth_curve(val_acc, 7), color=colors['val'], 
            label='Validation Accuracy', linewidth=2.2)
    
    best_idx = np.argmax(val_acc)
    best_epoch = epochs[best_idx]
    best_val = val_acc[best_idx]
    ax.scatter(best_epoch, best_val, color='red', s=60, zorder=5, marker='*')
    ax.annotate(f'Best: {best_val:.1f}%\n(Epoch {best_epoch})', 
                xy=(best_epoch, best_val), xytext=(best_epoch - 25, best_val - 8),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', alpha=0.9))
    
    for s in [80, 160]:
        ax.axvline(x=s, color='gray', linestyle='--', alpha=0.6, linewidth=1)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('(b) Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, shadow=False)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 240)
    ax.set_ylim(0, 90)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    return fig


def plot_stages_separate():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    stage_names = ['Stage 1: Normal Data (Early Rounds)', 
                    'Stage 2: Normal Data (More Rounds)', 
                    'Stage 3: High Quality Data']
    
    for s in range(3):
        mask = stages == (s + 1)
        ep_local = np.arange(1, np.sum(mask) + 1)
        
        ax = axes[0, s]
        ax.plot(ep_local, train_loss[mask], color='#1f77b4', alpha=0.4, linewidth=0.8)
        ax.plot(ep_local, smooth_curve(train_loss[mask], 5), color='#1f77b4', 
                label='Train Loss', linewidth=2)
        ax.plot(ep_local, val_loss[mask], color='#d62728', alpha=0.4, linewidth=0.8)
        ax.plot(ep_local, smooth_curve(val_loss[mask], 5), color='#d62728', 
                label='Val Loss', linewidth=2)
        ax.set_title(stage_names[s], fontsize=11, fontweight='bold')
        ax.set_ylabel('Loss' if s == 0 else '')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        ax = axes[1, s]
        ax.plot(ep_local, train_acc[mask], color='#1f77b4', alpha=0.4, linewidth=0.8)
        ax.plot(ep_local, smooth_curve(train_acc[mask], 5), color='#1f77b4', 
                label='Train Acc', linewidth=2)
        ax.plot(ep_local, val_acc[mask], color='#d62728', alpha=0.4, linewidth=0.8)
        ax.plot(ep_local, smooth_curve(val_acc[mask], 5), color='#d62728', 
                label='Val Acc', linewidth=2)
        ax.set_xlabel('Epoch (within Stage)')
        ax.set_ylabel('Accuracy (%)' if s == 0 else '')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


def plot_lr():
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.step(epochs, lr, where='post', color='#9467bd', linewidth=2)
    ax.fill_between(epochs, lr, step='post', alpha=0.15, color='#9467bd')
    
    for s in [80, 160]:
        ax.axvline(x=s, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    lr_point_epochs = [1, 21, 41, 61, 81, 101, 121, 141, 161, 181, 201, 221]
    for p in lr_point_epochs:
        if p <= len(lr):
            v = lr[p - 1]
            ax.text(p, v * 1.8, f'{v:.1e}', ha='center', va='bottom', 
                    fontsize=8.5, color='#9467bd')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule Across 240 Epochs', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 240)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    return fig


def plot_combined():
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.35, wspace=0.3)
    
    colors = {'train': '#1f77b4', 'val': '#d62728'}
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_loss, color=colors['train'], alpha=0.25, linewidth=0.8)
    ax1.plot(epochs, smooth_curve(train_loss, 7), color=colors['train'], 
             label='Training Loss', linewidth=2.2)
    ax1.plot(epochs, val_loss, color=colors['val'], alpha=0.25, linewidth=0.8)
    ax1.plot(epochs, smooth_curve(val_loss, 7), color=colors['val'], 
             label='Validation Loss', linewidth=2.2)
    for s in [80, 160]:
        ax1.axvline(x=s, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.text(40, 7.5, 'Stage 1', ha='center', fontsize=10, fontweight='bold', color='#2ca02c')
    ax1.text(120, 7.5, 'Stage 2', ha='center', fontsize=10, fontweight='bold', color='#ff7f0e')
    ax1.text(200, 7.5, 'Stage 3', ha='center', fontsize=10, fontweight='bold', color='#d62728')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('(a) Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 240)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, train_acc, color=colors['train'], alpha=0.25, linewidth=0.8)
    ax2.plot(epochs, smooth_curve(train_acc, 7), color=colors['train'], 
             label='Training Accuracy', linewidth=2.2)
    ax2.plot(epochs, val_acc, color=colors['val'], alpha=0.25, linewidth=0.8)
    ax2.plot(epochs, smooth_curve(val_acc, 7), color=colors['val'], 
             label='Validation Accuracy', linewidth=2.2)
    
    best_idx = np.argmax(val_acc)
    best_epoch = epochs[best_idx]
    best_val = val_acc[best_idx]
    ax2.scatter(best_epoch, best_val, color='red', s=80, zorder=5, marker='*', 
                edgecolors='darkred', linewidth=0.5)
    ax2.annotate(f'Best Val Acc: {best_val:.1f}%\n(Epoch {best_epoch})', 
                xy=(best_epoch, best_val), xytext=(best_epoch - 20, best_val - 7),
                fontsize=9.5, arrowprops=dict(arrowstyle='->', color='darkred', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='darkred', alpha=0.95))
    
    for s in [80, 160]:
        ax2.axvline(x=s, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('(b) Training & Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 240)
    ax2.set_ylim(0, 90)
    
    ax3 = fig.add_subplot(gs[1, :])
    ax3.step(epochs, lr, where='post', color='#9467bd', linewidth=2)
    ax3.fill_between(epochs, lr, step='post', alpha=0.15, color='#9467bd')
    for s in [80, 160]:
        ax3.axvline(x=s, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    lr_point_epochs = [1, 21, 41, 61, 81, 101, 121, 141, 161, 181, 201, 221]
    for p in lr_point_epochs:
        if p <= len(lr):
            v = lr[p - 1]
            ax3.text(p, v * 1.8, f'{v:.1e}', ha='center', va='bottom', 
                     fontsize=8.5, color='#9467bd')
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('(c) Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(0, 240)
    
    return fig


output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'figures')
os.makedirs(output_dir, exist_ok=True)

print("生成概览图...")
fig1 = plot_overview()
fig1.savefig(os.path.join(output_dir, 'training_overview.png'), dpi=300, bbox_inches='tight')
fig1.savefig(os.path.join(output_dir, 'training_overview.pdf'), bbox_inches='tight')
plt.close(fig1)

print("生成分阶段图...")
fig2 = plot_stages_separate()
fig2.savefig(os.path.join(output_dir, 'training_by_stage.png'), dpi=300, bbox_inches='tight')
fig2.savefig(os.path.join(output_dir, 'training_by_stage.pdf'), bbox_inches='tight')
plt.close(fig2)

print("生成学习率图...")
fig3 = plot_lr()
fig3.savefig(os.path.join(output_dir, 'learning_rate_schedule.png'), dpi=300, bbox_inches='tight')
fig3.savefig(os.path.join(output_dir, 'learning_rate_schedule.pdf'), bbox_inches='tight')
plt.close(fig3)

print("生成组合图...")
fig4 = plot_combined()
fig4.savefig(os.path.join(output_dir, 'training_combined.png'), dpi=300, bbox_inches='tight')
fig4.savefig(os.path.join(output_dir, 'training_combined.pdf'), bbox_inches='tight')
plt.close(fig4)

print(f"\n所有图表已保存到: {output_dir}")
print("生成的文件:")
for f in ['training_overview.png/pdf', 'training_by_stage.png/pdf', 
          'learning_rate_schedule.png/pdf', 'training_combined.png/pdf']:
    print(f"  - {f}")

print(f"\n统计信息:")
print(f"  总 Epochs: {len(epochs)}")
best_idx = np.argmax(val_acc)
print(f"  最佳验证准确率: {np.max(val_acc):.2f}% (Epoch {epochs[best_idx]})")
print(f"  最终训练准确率: {train_acc[-1]:.2f}%")
print(f"  最终验证准确率: {val_acc[-1]:.2f}%")
print(f"  最终训练损失: {train_loss[-1]:.4f}")
print(f"  最终验证损失: {val_loss[-1]:.4f}")
print(f"  测试准确率: {log['final_test']['test_acc']*100:.2f}%")
print(f"  测试损失: {log['final_test']['test_loss']:.4f}")
