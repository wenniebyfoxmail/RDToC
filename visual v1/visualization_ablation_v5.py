#!/usr/bin/env python3
"""
RMTwin 消融实验可视化 v5.0
===========================
为合并版消融实验生成论文级图表

输出:
- fig8_ablation_complete.pdf/png: 2x2 完整消融图
- fig8_ablation_validity.pdf/png: 单独的 Validity 图
- fig8_ablation_quality.pdf/png: 单独的 Quality 图

Author: RMTwin Research Team
Version: 5.0
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# 样式配置
# =============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# 颜色方案
COLORS = {
    'full_ontology': '#2ecc71',  # 绿色 - 基准
    'no_type_inference': '#e74c3c',  # 红色 - 最大影响
    'no_compatibility': '#3498db',  # 蓝色
    'noise_30': '#f39c12',  # 橙色
    'combined_degraded': '#9b59b6',  # 紫色 - 组合
}

COLOR_LIST = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']


# =============================================================================
# 图表生成
# =============================================================================

def fig8_complete(df: pd.DataFrame, output_dir: Path):
    """
    生成完整的 2x2 消融实验图

    (a) Validity Rate - 随机采样有效率
    (b) False Feasible - 误判数量
    (c) Pareto Solutions - 优化解数量
    (d) Hypervolume - 优化质量
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 获取数据
    if 'short_name' in df.columns:
        labels = df['short_name'].tolist()
    elif 'mode_name' in df.columns:
        labels = [name.replace(' ', '\n') for name in df['mode_name'].tolist()]
    else:
        labels = df['variant'].tolist()

    x = np.arange(len(labels))
    colors = COLOR_LIST[:len(labels)]

    # === (a) Validity Rate ===
    ax1 = axes[0, 0]
    validity = df['validity_rate'].values
    bars1 = ax1.bar(x, validity, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Validity Rate', fontsize=12)
    ax1.set_title('(a) Configuration Validity (Random Sampling)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(0, 1.15)
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, linewidth=1)

    # 添加数值标签
    for i, v in enumerate(validity):
        color = 'green' if v >= 0.95 else ('orange' if v >= 0.7 else 'red')
        ax1.text(i, v + 0.03, f'{v:.0%}', ha='center', fontsize=11, fontweight='bold', color=color)

    # 添加下降箭头
    baseline_v = validity[0]
    for i in range(1, len(validity)):
        if validity[i] < baseline_v - 0.05:
            delta = (baseline_v - validity[i]) * 100
            ax1.annotate('', xy=(i, validity[i] + 0.08), xytext=(i, baseline_v - 0.02),
                         arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
            ax1.text(i + 0.15, (validity[i] + baseline_v) / 2, f'-{delta:.0f}pp',
                     fontsize=9, color='red', fontweight='bold')

    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # === (b) False Feasible ===
    ax2 = axes[0, 1]
    if 'n_false_feasible' in df.columns:
        false_feas = df['n_false_feasible'].values
    else:
        # 如果没有这列，用 (1-validity) * feasible_ablated 估算
        false_feas = ((1 - df['validity_rate']) * 200).astype(int).values

    bars2 = ax2.bar(x, false_feas, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('False Feasible Count', fontsize=12)
    ax2.set_title('(b) Invalid Configs Wrongly Accepted', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)

    for i, v in enumerate(false_feas):
        ax2.text(i, v + max(false_feas) * 0.02, f'{v}', ha='center', fontsize=11, fontweight='bold')

    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # === (c) Pareto Solutions ===
    ax3 = axes[1, 0]
    if 'n_pareto' in df.columns and 'n_valid' in df.columns:
        n_pareto = df['n_pareto'].values
        n_valid = df['n_valid'].values

        width = 0.35
        bars3a = ax3.bar(x - width / 2, n_pareto, width, label='Pareto (Ablated)',
                         color=[c + '80' for c in colors], edgecolor='black', linewidth=0.5)
        bars3b = ax3.bar(x + width / 2, n_valid, width, label='Valid (Full Check)',
                         color=colors, edgecolor='black', linewidth=0.5)
        ax3.legend(loc='upper right')

        for i, (p, v) in enumerate(zip(n_pareto, n_valid)):
            ax3.text(i - width / 2, p + 0.5, f'{p}', ha='center', fontsize=9)
            ax3.text(i + width / 2, v + 0.5, f'{v}', ha='center', fontsize=9, fontweight='bold')
    else:
        # 只有 feasible_rate
        feas = df['feasible_rate'].values if 'feasible_rate' in df.columns else validity
        bars3 = ax3.bar(x, feas, color=colors, edgecolor='black', linewidth=0.5)
        for i, v in enumerate(feas):
            ax3.text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=11)

    ax3.set_ylabel('Number of Solutions', fontsize=12)
    ax3.set_title('(c) Solution Count (Optimization)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=10)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # === (d) Hypervolume ===
    ax4 = axes[1, 1]
    if 'hv_valid' in df.columns:
        hv = df['hv_valid'].values
    elif 'hv_6d' in df.columns:
        hv = df['hv_6d'].values
    else:
        hv = np.zeros(len(df))

    bars4 = ax4.bar(x, hv, color=colors, edgecolor='black', linewidth=0.5)
    ax4.set_ylabel('Hypervolume (6D)', fontsize=12)
    ax4.set_title('(d) Solution Quality (Optimization)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, fontsize=10)
    ax4.set_ylim(0, max(hv) * 1.25 if max(hv) > 0 else 1)

    # 数值标签和变化
    baseline_hv = hv[0]
    for i, v in enumerate(hv):
        ax4.text(i, v + max(hv) * 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
        if i > 0 and baseline_hv > 0:
            delta_pct = (v - baseline_hv) / baseline_hv * 100
            color = 'green' if delta_pct >= 0 else 'red'
            ax4.text(i, v + max(hv) * 0.08, f'{delta_pct:+.0f}%', ha='center', fontsize=9, color=color)

    ax4.grid(axis='y', alpha=0.3, linestyle='--')

    # 总标题
    plt.suptitle('Ontology Ablation Study', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # 保存
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig8_ablation_complete.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ fig8_ablation_complete saved")


def fig8_validity_only(df: pd.DataFrame, output_dir: Path):
    """
    单独的 Validity 图 - 强调本体防止无效配置的能力
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if 'short_name' in df.columns:
        labels = df['short_name'].tolist()
    elif 'mode_name' in df.columns:
        labels = [name.replace(' ', '\n') for name in df['mode_name'].tolist()]
    else:
        labels = df['variant'].tolist()

    x = np.arange(len(labels))
    validity = df['validity_rate'].values
    colors = COLOR_LIST[:len(labels)]

    bars = ax.bar(x, validity, color=colors, edgecolor='black', linewidth=1, width=0.6)

    # 基准线
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Perfect Validity')
    ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, linewidth=1, label='50% Threshold')

    # 数值标签
    baseline_v = validity[0]
    for i, v in enumerate(validity):
        # 主标签
        color = 'darkgreen' if v >= 0.95 else ('darkorange' if v >= 0.7 else 'darkred')
        ax.text(i, v + 0.03, f'{v:.0%}', ha='center', fontsize=14, fontweight='bold', color=color)

        # 差异标签
        if i > 0:
            delta = (v - baseline_v) * 100
            if abs(delta) > 1:
                y_pos = v - 0.08 if delta < 0 else v + 0.10
                ax.text(i, y_pos, f'{delta:+.0f}pp', ha='center', fontsize=11,
                        color='red' if delta < 0 else 'green', fontweight='bold')

    ax.set_ylabel('Validity Rate', fontsize=14)
    ax.set_xlabel('Ablation Mode', fontsize=14)
    ax.set_title('Ontology Ablation: Configuration Validity', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加解释文本
    ax.text(0.02, 0.98, 'Higher = Better\n(100% = all configs valid)', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', style='italic', alpha=0.7)

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig8_ablation_validity.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ fig8_ablation_validity saved")


def fig8_quality_only(df: pd.DataFrame, output_dir: Path):
    """
    单独的 Quality 图 - 强调本体提升优化质量
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if 'short_name' in df.columns:
        labels = df['short_name'].tolist()
    elif 'mode_name' in df.columns:
        labels = [name.replace(' ', '\n') for name in df['mode_name'].tolist()]
    else:
        labels = df['variant'].tolist()

    x = np.arange(len(labels))

    if 'hv_valid' in df.columns:
        hv = df['hv_valid'].values
    elif 'hv_6d' in df.columns:
        hv = df['hv_6d'].values
    else:
        logger.warning("No HV data found")
        return

    colors = COLOR_LIST[:len(labels)]

    bars = ax.bar(x, hv, color=colors, edgecolor='black', linewidth=1, width=0.6)

    # 基准线
    baseline_hv = hv[0]
    ax.axhline(y=baseline_hv, color='green', linestyle='--', alpha=0.7, linewidth=2,
               label=f'Full Ontology: {baseline_hv:.3f}')

    # 数值标签
    for i, v in enumerate(hv):
        ax.text(i, v + max(hv) * 0.02, f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')

        if i > 0 and baseline_hv > 0:
            delta_pct = (v - baseline_hv) / baseline_hv * 100
            color = 'green' if delta_pct >= 0 else 'red'
            y_pos = v + max(hv) * 0.08
            ax.text(i, y_pos, f'{delta_pct:+.1f}%', ha='center', fontsize=11,
                    color=color, fontweight='bold')

    ax.set_ylabel('Hypervolume (6D)', fontsize=14)
    ax.set_xlabel('Ablation Mode', fontsize=14)
    ax.set_title('Ontology Ablation: Optimization Quality', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, max(hv) * 1.30 if max(hv) > 0 else 1)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加解释文本
    ax.text(0.02, 0.98, 'Higher = Better\n(larger Pareto front coverage)', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', style='italic', alpha=0.7)

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig8_ablation_quality.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ fig8_ablation_quality saved")


def fig8_summary_horizontal(df: pd.DataFrame, output_dir: Path):
    """
    横向对比图 - Validity 和 HV 并排
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if 'short_name' in df.columns:
        labels = df['short_name'].tolist()
    elif 'mode_name' in df.columns:
        labels = df['mode_name'].tolist()
    else:
        labels = df['variant'].tolist()

    colors = COLOR_LIST[:len(labels)]

    # === Left: Validity ===
    ax1 = axes[0]
    validity = df['validity_rate'].values
    y = np.arange(len(labels))

    bars1 = ax1.barh(y, validity, color=colors, edgecolor='black', linewidth=0.5, height=0.6)
    ax1.axvline(x=1.0, color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax1.axvline(x=0.5, color='red', linestyle=':', alpha=0.5, linewidth=1)

    for i, v in enumerate(validity):
        color = 'darkgreen' if v >= 0.95 else ('darkorange' if v >= 0.7 else 'darkred')
        ax1.text(v + 0.02, i, f'{v:.0%}', va='center', fontsize=12, fontweight='bold', color=color)

    ax1.set_xlabel('Validity Rate', fontsize=12)
    ax1.set_title('(a) Configuration Validity', fontsize=12, fontweight='bold')
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=11)
    ax1.set_xlim(0, 1.25)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # === Right: HV ===
    ax2 = axes[1]
    if 'hv_valid' in df.columns:
        hv = df['hv_valid'].values
    elif 'hv_6d' in df.columns:
        hv = df['hv_6d'].values
    else:
        hv = np.zeros(len(df))

    bars2 = ax2.barh(y, hv, color=colors, edgecolor='black', linewidth=0.5, height=0.6)

    baseline_hv = hv[0]
    ax2.axvline(x=baseline_hv, color='green', linestyle='--', alpha=0.7, linewidth=2)

    for i, v in enumerate(hv):
        ax2.text(v + max(hv) * 0.02, i, f'{v:.3f}', va='center', fontsize=12, fontweight='bold')

    ax2.set_xlabel('Hypervolume (6D)', fontsize=12)
    ax2.set_title('(b) Optimization Quality', fontsize=12, fontweight='bold')
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=11)
    ax2.set_xlim(0, max(hv) * 1.25 if max(hv) > 0 else 1)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    plt.suptitle('Ontology Ablation Study', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig8_ablation_summary.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ fig8_ablation_summary saved")


# =============================================================================
# Main
# =============================================================================

def generate_all_figures(data_path: str, output_dir: str = None):
    """生成所有消融实验图表"""
    data_path = Path(data_path)

    if output_dir is None:
        output_dir = data_path.parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取数据
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data: {len(df)} rows")
    logger.info(f"Columns: {list(df.columns)}")

    # 生成图表
    print("\n" + "=" * 60)
    print("Generating Ablation Figures")
    print("=" * 60)

    fig8_complete(df, output_dir)
    fig8_validity_only(df, output_dir)
    fig8_quality_only(df, output_dir)
    fig8_summary_horizontal(df, output_dir)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Ablation Visualization v5.0')
    parser.add_argument('--data', default='./results/ablation_v5/ablation_complete_v5.csv',
                        help='Path to ablation results CSV')
    parser.add_argument('--output', default=None, help='Output directory (default: same as data)')
    args = parser.parse_args()

    generate_all_figures(args.data, args.output)


if __name__ == '__main__':
    main()