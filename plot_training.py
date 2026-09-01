"""
Search-R1 训练结果可视化
读取training_log.json，生成一系列图表
"""
import json
import os

# 检查日志文件
log_path = 'D:/Search-R1/training_log.json'
if not os.path.exists(log_path):
    print('training_log.json not found. Please run train_full_fixed.py first.')
    exit(1)

with open(log_path, 'r') as f:
    data = json.load(f)

steps = data['steps']
config = data['config']
summary = data.get('summary', {})
test_results = data.get('test_results', [])

# 提取数据
step_nums = [s['step'] for s in steps]
rewards = [s['reward'] for s in steps]
step_accs = [s['step_acc'] for s in steps]
overall_accs = [s['overall_acc'] for s in steps]
times = [s['time'] for s in steps]

import matplotlib
matplotlib.use('Agg')  # 不需要GUI
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

output_dir = 'D:/Search-R1/plots'
os.makedirs(output_dir, exist_ok=True)

# ==================== 图1: 奖励曲线 ====================
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(step_nums, rewards, 'b-', alpha=0.3, linewidth=0.8, label='Step Reward')
# 滑动平均
window = min(10, len(rewards))
if len(rewards) >= window:
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax.plot(step_nums[window-1:], moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window})')
ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Average Reward', fontsize=12)
ax.set_title(f'Search-R1 GRPO Training - Reward Curve\n'
             f'Model: {config["model"]} | Data: {config["data"]} NQ questions | Steps: {config["steps"]}',
             fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(step_nums) + 1)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig(f'{output_dir}/1_reward_curve.png', dpi=150)
plt.close()
print(f'[OK] 1_reward_curve.png')

# ==================== 图2: 准确率曲线 ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 左图：每步准确率
ax1.plot(step_nums, step_accs, 'g-', alpha=0.3, linewidth=0.8, label='Step Accuracy')
window = min(10, len(step_accs))
if len(step_accs) >= window:
    ma = np.convolve(step_accs, np.ones(window)/window, mode='valid')
    ax1.plot(step_nums[window-1:], ma, 'g-', linewidth=2, label=f'Moving Avg ({window})')
ax1.set_xlabel('Training Step', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Step Accuracy (per batch)', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.05, 1.05)

# 右图：累计准确率
ax2.plot(step_nums, overall_accs, 'm-', linewidth=2, label='Overall Accuracy')
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
ax2.set_xlabel('Training Step', fontsize=12)
ax2.set_ylabel('Cumulative Accuracy', fontsize=12)
ax2.set_title('Overall Accuracy (cumulative)', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.05, 1.05)

plt.suptitle(f'Search-R1 Training - Accuracy (Final: {overall_accs[-1]:.1%})', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/2_accuracy_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'[OK] 2_accuracy_curve.png')

# ==================== 图3: 训练速度 ====================
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(step_nums, times, color='steelblue', alpha=0.6, width=0.8)
ax.axhline(y=np.mean(times), color='red', linestyle='--', linewidth=2, label=f'Avg: {np.mean(times):.0f}s')
ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Time per Step (seconds)', fontsize=12)
ax.set_title(f'Training Speed | Total time: {sum(times)/3600:.1f} hours | Avg: {np.mean(times):.0f}s/step',
             fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{output_dir}/3_training_speed.png', dpi=150)
plt.close()
print(f'[OK] 3_training_speed.png')

# ==================== 图4: 测试结果 ====================
if test_results:
    fig, ax = plt.subplots(figsize=(14, 8))
    questions = [t['question'][:40] for t in test_results]
    correct = [1 if t['correct'] else 0 for t in test_results]
    answers = [t['answer'] if t['answer'] else 'None' for t in test_results]
    goldens = [t['golden'] for t in test_results]

    colors = ['#2ecc71' if c else '#e74c3c' for c in correct]
    bars = ax.barh(range(len(questions)), correct, color=colors, height=0.6)

    # 标注答案
    for i, (ans, gold, c) in enumerate(zip(answers, goldens, correct)):
        label = f'{ans} (correct: {gold})' if c else f'{ans} (answer: {gold})'
        ax.text(0.02, i, label, va='center', fontsize=9, color='white' if c else 'black')

    ax.set_yticks(range(len(questions)))
    ax.set_yticklabels(questions, fontsize=10)
    ax.set_xlabel('Correct (1) / Wrong (0)', fontsize=12)
    acc = sum(correct) / len(correct)
    ax.set_title(f'Post-Training Test Results ({sum(correct)}/{len(correct)} = {acc:.0%})', fontsize=14)
    ax.set_xlim(-0.1, 1.3)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4_test_results.png', dpi=150)
    plt.close()
    print(f'[OK] 4_test_results.png')

# ==================== 图5: 综合仪表盘 ====================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 左上：奖励曲线
ax = axes[0][0]
ax.plot(step_nums, rewards, 'b-', alpha=0.3, linewidth=0.8)
if len(rewards) >= 10:
    ma = np.convolve(rewards, np.ones(10)/10, mode='valid')
    ax.plot(step_nums[9:], ma, 'r-', linewidth=2, label='Avg (10)')
ax.set_title('GRPO Reward Curve', fontsize=13)
ax.set_xlabel('Step')
ax.set_ylabel('Reward')
ax.legend()
ax.grid(True, alpha=0.3)

# 右上：累计准确率
ax = axes[0][1]
ax.plot(step_nums, overall_accs, 'm-', linewidth=2)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Cumulative Accuracy', fontsize=13)
ax.set_xlabel('Step')
ax.set_ylabel('Accuracy')
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

# 左下：训练速度
ax = axes[1][0]
ax.bar(step_nums, times, color='steelblue', alpha=0.5, width=0.8)
ax.axhline(y=np.mean(times), color='red', linestyle='--', linewidth=2)
ax.set_title(f'Training Speed (avg {np.mean(times):.0f}s/step)', fontsize=13)
ax.set_xlabel('Step')
ax.set_ylabel('Time (s)')
ax.grid(True, alpha=0.3, axis='y')

# 右下：统计信息
ax = axes[1][1]
ax.axis('off')
info_text = (
    f"===== Search-R1 Training Summary =====\n\n"
    f"Model: {config['model']}\n"
    f"GPU: {config['gpu']}\n"
    f"Training Steps: {config['steps']}\n"
    f"Batch Size: {config['batch']}\n"
    f"N-Agent: {config['n_agent']}\n"
    f"Learning Rate: {config['lr']}\n\n"
    f"Dataset: {config['data']} NQ questions\n\n"
    f"--- Results ---\n"
    f"Final Reward: {rewards[-1]:.3f}\n"
    f"Best Reward: {summary.get('best_reward', max(rewards)):.3f}\n"
    f"Training Accuracy: {summary.get('training_acc', overall_accs[-1]):.1%}\n"
    f"Test Accuracy: {summary.get('test_acc', 0):.0%}\n"
    f"Total Time: {sum(times)/3600:.1f} hours\n"
)
ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Search-R1 GRPO Training Dashboard', fontsize=18, y=0.98)
plt.tight_layout()
plt.savefig(f'{output_dir}/5_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'[OK] 5_dashboard.png')

print(f'\nAll plots saved to: {output_dir}/')
print('Files:')
for f in sorted(os.listdir(output_dir)):
    if f.endswith('.png'):
        size = os.path.getsize(f'{output_dir}/{f}') / 1024
        print(f'  {f} ({size:.0f}KB)')
