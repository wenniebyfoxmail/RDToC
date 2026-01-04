#!/usr/bin/env python3
"""
RMTwin Publication-Ready Visualization (Rigorous Version)
==========================================================
修复审稿级问题：
1. 支配区域使用正确的最小化形式 (cost vs 1-recall)
2. 去除饼图，改用条形图
3. 图内标题精简，由caption承担
4. Coverage双向正确展示

Author: RMTwin Research Team
Version: 2.0 (Publication-Ready)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from typing import Dict
import warnings

warnings.filterwarnings('ignore')

# 配色方案（简洁）
COLORS = {
    'NSGA-III': '#1f77b4',
    'random': '#7f7f7f',
    'weighted': '#ff7f0e',
    'grid': '#2ca02c',
    'expert': '#d62728',
}

# IEEE/顶刊风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': ':',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_data(pareto_path: str):
    """加载数据"""
    pareto_df = pd.read_csv(pareto_path)
    pareto_dir = Path(pareto_path).parent
    baseline_dfs = {}
    for f in pareto_dir.glob('baseline_*.csv'):
        name = f.stem.replace('baseline_', '')
        df = pd.read_csv(f)
        if 'is_feasible' in df.columns:
            df = df[df['is_feasible']]
        baseline_dfs[name] = df
    return pareto_df, baseline_dfs


def fig_pareto_minimization(pareto_df, baseline_dfs, output_dir):
    """
    Pareto前沿图 - 使用正确的最小化形式

    X轴: Cost (minimize)
    Y轴: 1-Recall (minimize)

    这样支配区域的填充才是严格正确的
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # 转换为最小化形式
    pareto_df = pareto_df.copy()
    pareto_df['one_minus_recall'] = 1 - pareto_df['detection_recall']

    # Baselines（背景）
    for name, df in baseline_dfs.items():
        if len(df) == 0:
            continue
        df = df.copy()
        df['one_minus_recall'] = 1 - df['detection_recall']
        ax.scatter(df['f1_total_cost_USD'] / 1e6, df['one_minus_recall'],
                   alpha=0.2, s=15, c=COLORS.get(name, 'gray'),
                   label=f'{name.title()} (n={len(df)})')

    # NSGA-III Pareto前沿（排序后）
    pareto_sorted = pareto_df.sort_values('f1_total_cost_USD')
    x_pareto = pareto_sorted['f1_total_cost_USD'].values / 1e6
    y_pareto = pareto_sorted['one_minus_recall'].values

    # 阶梯状前沿线（对于最小化问题，这是严格正确的）
    x_step = []
    y_step = []
    for i in range(len(x_pareto)):
        if i > 0:
            x_step.append(x_pareto[i])
            y_step.append(y_pareto[i - 1])
        x_step.append(x_pareto[i])
        y_step.append(y_pareto[i])

    # 填充被支配区域（严格正确：向右上方填充）
    x_fill = list(x_step) + [max(x_step) * 1.2, max(x_step) * 1.2, x_step[0]]
    y_fill = list(y_step) + [y_step[-1], max(y_step) * 1.2, max(y_step) * 1.2]
    ax.fill(x_fill, y_fill, alpha=0.1, color=COLORS['NSGA-III'])

    # Pareto前沿线
    ax.plot(x_step, y_step, color=COLORS['NSGA-III'], linewidth=2.5,
            linestyle='-', zorder=8, label='NSGA-III Pareto Front')

    # Pareto解点
    ax.scatter(x_pareto, y_pareto, s=120, c=COLORS['NSGA-III'],
               marker='*', zorder=10, edgecolors='white', linewidths=1,
               label=f'NSGA-III Solutions (n={len(pareto_df)})')

    ax.set_xlabel('Total Cost (Million USD)', fontsize=12)
    ax.set_ylabel('1 - Detection Recall (minimize)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.set_xlim(0, max(x_pareto) * 1.3)
    ax.set_ylim(0, 0.5)

    # 添加recall刻度参考
    ax2 = ax.secondary_yaxis('right', functions=(lambda x: 1 - x, lambda x: 1 - x))
    ax2.set_ylabel('Detection Recall', fontsize=12)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig_pareto_minimization.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Pareto Front (minimization form)")


def fig_pareto_tradeoff(pareto_df, baseline_dfs, output_dir):
    """
    Pareto前沿图 - 直观展示（cost vs recall）
    不填充支配区域，只画前沿包络线
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Baselines
    for name, df in baseline_dfs.items():
        if len(df) == 0:
            continue
        ax.scatter(df['f1_total_cost_USD'] / 1e6, df['detection_recall'],
                   alpha=0.2, s=15, c=COLORS.get(name, 'gray'),
                   label=f'{name.title()} (n={len(df)})')

    # NSGA-III
    pareto_sorted = pareto_df.sort_values('f1_total_cost_USD')
    x = pareto_sorted['f1_total_cost_USD'].values / 1e6
    y = pareto_sorted['detection_recall'].values

    # 简单连线（不填充）
    ax.plot(x, y, color=COLORS['NSGA-III'], linewidth=2,
            linestyle='-', marker='*', markersize=12, zorder=10,
            markerfacecolor=COLORS['NSGA-III'], markeredgecolor='white',
            label=f'NSGA-III (n={len(pareto_df)})')

    # 参考线
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(0.5, 0.955, 'Target: 0.95', fontsize=9, color='red', alpha=0.7)

    ax.set_xlabel('Total Cost (Million USD)', fontsize=12)
    ax.set_ylabel('Detection Recall', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, max(x) * 1.3)
    ax.set_ylim(0.6, 1.0)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig_pareto_tradeoff.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Pareto Front (trade-off view)")


def fig_coverage_bidirectional(metrics_dir, output_dir):
    """
    Coverage双向对比图 - 正确展示双向关系
    """
    try:
        coverage_df = pd.read_csv(metrics_dir / 'coverage_6d.csv')
    except:
        try:
            coverage_df = pd.read_csv(metrics_dir / 'coverage_metrics.csv')
            # 重命名列以匹配
            coverage_df = coverage_df.rename(columns={
                'NSGA_Dominates_%': 'C(NSGA,Baseline)',
                'Baseline_Dominates_%': 'C(Baseline,NSGA)'
            })
        except:
            print("  ⚠ Coverage metrics not found")
            return

    fig, ax = plt.subplots(figsize=(10, 6))

    # 提取baseline名称
    if 'Comparison' in coverage_df.columns:
        names = [c.replace('NSGA-III vs ', '') for c in coverage_df['Comparison']]
    else:
        names = coverage_df['Baseline'].tolist()

    x = np.arange(len(names))
    width = 0.35

    c_nsga = coverage_df['C(NSGA,Baseline)'].values
    c_base = coverage_df['C(Baseline,NSGA)'].values

    bars1 = ax.bar(x - width / 2, c_nsga, width, label='C(NSGA→Baseline)',
                   color=COLORS['NSGA-III'], alpha=0.8)
    bars2 = ax.bar(x + width / 2, c_base, width, label='C(Baseline→NSGA)',
                   color='gray', alpha=0.6)

    # 数值标签
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f'{h:.1f}%',
                ha='center', fontsize=9, color=COLORS['NSGA-III'])
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f'{h:.1f}%',
                ha='center', fontsize=9, color='gray')

    # Net advantage线
    net = c_nsga - c_base
    for i, n in enumerate(net):
        color = 'green' if n > 0 else 'red'
        ax.annotate(f'Net: {n:+.1f}%', xy=(i, max(c_nsga[i], c_base[i]) + 8),
                    ha='center', fontsize=9, color=color, fontweight='bold')

    ax.set_ylabel('Dominated Solutions (%)', fontsize=12)
    ax.set_xlabel('Baseline Method', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([n.title() for n in names], fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0, 120)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig_coverage_bidirectional.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Coverage (bidirectional)")


def fig_contribution_bar(metrics_dir, output_dir):
    """
    贡献度条形图 - 替代饼图（顶刊更偏好）
    """
    try:
        contrib_df = pd.read_csv(metrics_dir / 'contribution_6d.csv')
    except:
        try:
            contrib_df = pd.read_csv(metrics_dir / 'contribution_metrics.csv')
        except:
            print("  ⚠ Contribution metrics not found")
            return

    # 过滤零贡献
    contrib_df = contrib_df[contrib_df['Contribution_%'] > 0].sort_values('Contribution_%', ascending=True)

    if len(contrib_df) == 0:
        print("  ⚠ No contribution data")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    methods = contrib_df['Method'].tolist()
    contribs = contrib_df['Contribution_%'].values
    colors = [COLORS.get(m, 'gray') if m != 'NSGA-III' else COLORS['NSGA-III'] for m in methods]

    bars = ax.barh(methods, contribs, color=colors, alpha=0.8, height=0.6)

    # 数值标签
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1, bar.get_y() + bar.get_height() / 2, f'{w:.1f}%',
                va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Contribution to Combined Pareto Front (%)', fontsize=12)
    ax.set_xlim(0, max(contribs) * 1.2)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig_contribution_bar.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Contribution (bar chart)")


def fig_hypervolume_comparison(metrics_dir, output_dir):
    """
    Hypervolume对比图 - 同时显示6D和2D
    """
    try:
        hv_df = pd.read_csv(metrics_dir / 'hypervolume_6d.csv')
    except:
        try:
            hv_df = pd.read_csv(metrics_dir / 'quality_metrics.csv')
            hv_df = hv_df.rename(columns={'HV': 'HV_2D'})
            hv_df['HV_6D'] = hv_df['HV_2D']  # 如果只有2D
        except:
            print("  ⚠ HV metrics not found")
            return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    methods = hv_df['Method'].tolist()
    colors = [COLORS.get(m, 'gray') if m != 'NSGA-III' else COLORS['NSGA-III'] for m in methods]

    # 6D HV
    ax1 = axes[0]
    if 'HV_6D' in hv_df.columns:
        bars = ax1.bar(methods, hv_df['HV_6D'], color=colors, alpha=0.8)
        ax1.set_ylabel('Hypervolume (6D)', fontsize=12)
        ax1.set_title('(a) 6D Hypervolume', fontsize=12)
        # 标注最高
        max_idx = hv_df['HV_6D'].idxmax()
        bars[max_idx].set_edgecolor('gold')
        bars[max_idx].set_linewidth(2)
    ax1.tick_params(axis='x', rotation=45)

    # 2D HV
    ax2 = axes[1]
    if 'HV_2D' in hv_df.columns:
        bars = ax2.bar(methods, hv_df['HV_2D'], color=colors, alpha=0.8)
        ax2.set_ylabel('Hypervolume (2D: Cost-Recall)', fontsize=12)
        ax2.set_title('(b) 2D Hypervolume (for visualization)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig_hypervolume.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Hypervolume comparison")


def fig_efficiency_comparison(pareto_df, baseline_dfs, metrics_dir, output_dir):
    """
    效率对比图 - 贡献度/解数量
    """
    try:
        contrib_df = pd.read_csv(metrics_dir / 'contribution_6d.csv')
    except:
        try:
            contrib_df = pd.read_csv(metrics_dir / 'contribution_metrics.csv')
        except:
            print("  ⚠ Contribution metrics not found")
            return

    fig, ax = plt.subplots(figsize=(10, 6))

    # 计算效率
    efficiency_data = []
    for _, row in contrib_df.iterrows():
        method = row['Method']
        contrib = row['Contribution_%']

        if method == 'NSGA-III':
            n = len(pareto_df)
        elif method in baseline_dfs:
            n = len(baseline_dfs[method])
        else:
            n = row.get('N_Solutions', 1)

        if n > 0 and contrib > 0:
            efficiency_data.append({
                'Method': method,
                'Contribution': contrib,
                'N_Solutions': n,
                'Efficiency': contrib / n
            })

    if not efficiency_data:
        print("  ⚠ No efficiency data")
        return

    eff_df = pd.DataFrame(efficiency_data).sort_values('Efficiency', ascending=True)

    colors = [COLORS.get(m, 'gray') if m != 'NSGA-III' else COLORS['NSGA-III'] for m in eff_df['Method']]

    bars = ax.barh(eff_df['Method'], eff_df['Efficiency'], color=colors, alpha=0.8, height=0.6)

    # 标签
    for bar, (_, row) in zip(bars, eff_df.iterrows()):
        w = bar.get_width()
        ax.text(w + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{w:.3f}%/sol\n({row["Contribution"]:.1f}%/{row["N_Solutions"]})',
                va='center', fontsize=9)

    ax.set_xlabel('Efficiency (Contribution % per Solution)', fontsize=12)
    ax.set_xlim(0, eff_df['Efficiency'].max() * 1.5)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'fig_efficiency.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Efficiency comparison")


def generate_all_figures(pareto_path: str, metrics_dir: str = None,
                         output_dir: str = './results/figures_rigorous'):
    """生成所有严谨版图表"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if metrics_dir is None:
        metrics_dir = Path(pareto_path).parent.parent / 'metrics_6d'
        if not metrics_dir.exists():
            metrics_dir = Path(pareto_path).parent.parent / 'metrics'
    else:
        metrics_dir = Path(metrics_dir)

    print("=" * 60)
    print("Generating Rigorous Publication Figures")
    print("=" * 60)

    pareto_df, baseline_dfs = load_data(pareto_path)
    print(f"Loaded: NSGA-III ({len(pareto_df)}), Baselines: {list(baseline_dfs.keys())}")
    print(f"Metrics dir: {metrics_dir}")
    print()

    # 生成图表
    fig_pareto_minimization(pareto_df, baseline_dfs, output_dir)
    fig_pareto_tradeoff(pareto_df, baseline_dfs, output_dir)
    fig_coverage_bidirectional(metrics_dir, output_dir)
    fig_contribution_bar(metrics_dir, output_dir)
    fig_hypervolume_comparison(metrics_dir, output_dir)
    fig_efficiency_comparison(pareto_df, baseline_dfs, metrics_dir, output_dir)

    print()
    print("=" * 60)
    print(f"✓ All figures saved to: {output_dir}")
    print("=" * 60)
    print("\nGenerated figures:")
    print("  1. fig_pareto_minimization.pdf  - Pareto front (correct minimization form)")
    print("  2. fig_pareto_tradeoff.pdf      - Pareto front (intuitive cost-recall)")
    print("  3. fig_coverage_bidirectional.pdf - Coverage (both directions)")
    print("  4. fig_contribution_bar.pdf     - Contribution (bar chart, not pie)")
    print("  5. fig_hypervolume.pdf          - HV comparison (6D + 2D)")
    print("  6. fig_efficiency.pdf           - Solution efficiency")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualization_rigorous.py <pareto_csv> [metrics_dir] [output_dir]")
        sys.exit(1)

    pareto_path = sys.argv[1]
    metrics_dir = sys.argv[2] if len(sys.argv) > 2 else None
    output_dir = sys.argv[3] if len(sys.argv) > 3 else './results/figures_rigorous'

    generate_all_figures(pareto_path, metrics_dir, output_dir)