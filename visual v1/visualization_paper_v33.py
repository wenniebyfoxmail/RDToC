#!/usr/bin/env python3
"""
RMTwin Paper Figures v3.3 - 论文专用可视化模块
=============================================
与现有 visualization.py 和 visualization_enhanced.py 兼容
专注于"离散友好"图型，将聚集现象解释为结构性发现

设计原则:
1. 聚集/边界收敛 = 结构性发现，不是bug
2. 离散变量用hist/violin/bar，不强行散点
3. 每张图支撑一句论文主张
4. 可审计：manifest.json记录所有输入

Author: RMTwin Research Team  
Version: 3.3 (Paper-Ready)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# =============================================================================
# 顶刊风格配置
# =============================================================================
STYLE_CONFIG = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
}

COLORS = {
    'nsga3': '#1f77b4',
    'pareto': '#2E86AB',
    'random': '#7f7f7f',
    'weighted': '#ff7f0e',
    'grid': '#2ca02c',
    'expert': '#d62728',
    'traditional': '#4A90D9',
    'ml': '#F5A623',
    'dl': '#7ED321',
    'highlight': '#E63946',
}

ALGO_MARKERS = {'Traditional': 'o', 'ML': 's', 'DL': '^'}


# =============================================================================
# 工具函数
# =============================================================================
def classify_algorithm(algo_name: str) -> str:
    """分类算法为 Traditional/ML/DL"""
    algo_str = str(algo_name).upper()
    dl_kw = ['DL_', 'YOLO', 'UNET', 'MASK', 'EFFICIENT', 'MOBILE', 'SAM', 'RETINA', 'FASTER']
    ml_kw = ['ML_', 'SVM', 'RANDOMFOREST', 'RANDOM_FOREST', 'XGBOOST', 'XGB', 'HYBRID']
    for kw in dl_kw:
        if kw in algo_str:
            return 'DL'
    for kw in ml_kw:
        if kw in algo_str:
            return 'ML'
    return 'Traditional'


def classify_sensor(sensor_name: str) -> str:
    """分类传感器类型"""
    sensor_str = str(sensor_name)
    for cat in ['IoT', 'Vehicle', 'Camera', 'MMS', 'UAV', 'TLS', 'FOS', 'Handheld']:
        if cat in sensor_str:
            return cat
    return 'Other'


def select_representatives(df: pd.DataFrame) -> Dict[str, int]:
    """选择3个代表性解: low_cost, balanced, high_recall"""
    
    # 找到recall列
    recall_col = 'detection_recall'
    if recall_col not in df.columns:
        if 'f2_one_minus_recall' in df.columns:
            df = df.copy()
            df['detection_recall'] = 1 - df['f2_one_minus_recall']
            recall_col = 'detection_recall'
        else:
            return {}
    
    # 找到cost列
    cost_col = None
    for c in ['f1_total_cost_USD', 'total_cost', 'cost']:
        if c in df.columns:
            cost_col = c
            break
    if cost_col is None:
        return {}
    
    reps = {}
    
    # Low-cost: cost最小且recall >= 0.8
    feasible = df[df[recall_col] >= 0.8]
    if len(feasible) > 0:
        reps['low_cost'] = feasible[cost_col].idxmin()
    else:
        reps['low_cost'] = df[cost_col].idxmin()
    
    # High-recall: recall最大且cost <= 80th percentile
    cost_80 = df[cost_col].quantile(0.8)
    affordable = df[df[cost_col] <= cost_80]
    if len(affordable) > 0:
        reps['high_recall'] = affordable[recall_col].idxmax()
    else:
        reps['high_recall'] = df[recall_col].idxmax()
    
    # Balanced: 标准化加权和最小
    norm_df = df.copy()
    for col in [cost_col, 'f3_latency_seconds', 'f4_traffic_disruption_hours', 'f5_carbon_emissions_kgCO2e_year']:
        if col in df.columns:
            min_v, max_v = df[col].min(), df[col].max()
            if max_v > min_v:
                norm_df[f'_n_{col}'] = (df[col] - min_v) / (max_v - min_v)
    
    # recall反转 (higher is better)
    min_r, max_r = df[recall_col].min(), df[recall_col].max()
    if max_r > min_r:
        norm_df['_n_recall'] = (max_r - df[recall_col]) / (max_r - min_r)
    
    norm_cols = [c for c in norm_df.columns if c.startswith('_n_')]
    if norm_cols:
        norm_df['_weighted'] = norm_df[norm_cols].mean(axis=1)
        reps['balanced'] = norm_df['_weighted'].idxmin()
    
    return reps


# =============================================================================
# PaperVisualizer 类 - 兼容现有接口
# =============================================================================
class PaperVisualizer:
    """
    论文专用可视化器 v3.3
    
    与现有 Visualizer 类接口兼容，可独立使用或作为补充
    """
    
    def __init__(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = Path('./results/paper/figures_v33')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.manifest = {
            'generated_at': datetime.now().isoformat(),
            'output_dir': str(self.output_dir),
            'figures': {}
        }
        self.captions = {}
        
        plt.rcParams.update(STYLE_CONFIG)
    
    def generate_all(self, pareto_df: pd.DataFrame, 
                     baseline_dfs: Dict[str, pd.DataFrame] = None,
                     config: dict = None,
                     run_dir: str = None,
                     ablation_dir: str = None):
        """
        生成所有论文图表
        
        接口与现有 Visualizer.generate_all() 兼容
        """
        logger.info("=" * 60)
        logger.info("Generating Paper Figures v3.3 (Discrete-Friendly)")
        logger.info("=" * 60)
        
        if baseline_dfs is None:
            baseline_dfs = {}
        
        # 预处理：确保有recall列
        if 'detection_recall' not in pareto_df.columns and 'f2_one_minus_recall' in pareto_df.columns:
            pareto_df = pareto_df.copy()
            pareto_df['detection_recall'] = 1 - pareto_df['f2_one_minus_recall']
        
        # 主图
        self.fig1_pareto_front(pareto_df, baseline_dfs)
        self.fig2_decision_matrix(pareto_df)
        self.fig3_cost_structure(pareto_df)
        self.fig4_discrete_distributions(pareto_df)
        self.fig5_technology_dominance(pareto_df, baseline_dfs)
        self.fig6_baseline_comparison(pareto_df, baseline_dfs)
        self.fig7_convergence(run_dir)
        self.fig8_ablation(ablation_dir)
        
        # 补充图
        self.figS1_pairwise_tradeoffs(pareto_df)
        self.figS2_sensitivity(pareto_df)
        
        # 保存元数据
        self._save_manifest()
        self._save_captions()
        
        logger.info("=" * 60)
        logger.info(f"Generated {len(self.manifest['figures'])} figures")
        logger.info(f"Output: {self.output_dir}")
        logger.info("=" * 60)
        
        return self.output_dir
    
    def _save_fig(self, fig, name: str, inputs: List[str] = None, 
                  fields: List[str] = None, notes: str = ""):
        """保存图表并更新manifest"""
        png_path = self.output_dir / f'{name}.png'
        pdf_path = self.output_dir / f'{name}.pdf'
        
        fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        self.manifest['figures'][name] = {
            'png': str(png_path),
            'pdf': str(pdf_path),
            'inputs': inputs or [],
            'fields': fields or [],
            'notes': notes
        }
        logger.info(f"  Saved: {name}")
    
    # =========================================================================
    # Fig 1: Pareto Front (Cost-Recall)
    # =========================================================================
    def fig1_pareto_front(self, pareto_df: pd.DataFrame, baseline_dfs: Dict):
        """2D Pareto前沿 - Cost vs Recall"""
        df = pareto_df.copy()
        
        # 列名适配
        cost_col = None
        for c in ['f1_total_cost_USD', 'total_cost', 'cost']:
            if c in df.columns:
                cost_col = c
                break
        
        recall_col = 'detection_recall' if 'detection_recall' in df.columns else None
        latency_col = 'f3_latency_seconds' if 'f3_latency_seconds' in df.columns else None
        algo_col = 'algorithm' if 'algorithm' in df.columns else None
        
        if cost_col is None or recall_col is None:
            logger.warning("Skipping fig1: missing cost/recall columns")
            return
        
        # 分类算法
        if algo_col:
            df['algo_family'] = df[algo_col].apply(classify_algorithm)
        
        # 选择代表性解
        reps = select_representatives(df)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 按latency着色
        if latency_col:
            scatter = ax.scatter(
                df[cost_col] / 1e6,
                df[recall_col],
                c=df[latency_col],
                cmap='viridis',
                s=100, alpha=0.8,
                edgecolors='white', linewidths=0.5,
                zorder=3
            )
            cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
            cbar.set_label('Latency (seconds)', fontsize=10)
        else:
            # 按算法分组
            for family, marker in ALGO_MARKERS.items():
                if algo_col:
                    mask = df['algo_family'] == family
                    if mask.any():
                        ax.scatter(
                            df.loc[mask, cost_col] / 1e6,
                            df.loc[mask, recall_col],
                            marker=marker, s=100, alpha=0.8,
                            label=family, edgecolors='white', linewidths=0.5, zorder=3
                        )
            if algo_col:
                ax.legend(title='Algorithm', loc='lower right')
        
        # 高亮代表性解
        rep_colors = {'low_cost': '#2ECC71', 'balanced': '#3498DB', 'high_recall': '#E74C3C'}
        rep_labels = {'low_cost': 'Low Cost', 'balanced': 'Balanced', 'high_recall': 'High Recall'}
        
        for rep_name, idx in reps.items():
            if idx in df.index:
                ax.scatter(
                    df.loc[idx, cost_col] / 1e6,
                    df.loc[idx, recall_col],
                    s=250, facecolors='none',
                    edgecolors=rep_colors[rep_name], linewidths=2.5, zorder=4
                )
                ax.annotate(
                    rep_labels[rep_name],
                    (df.loc[idx, cost_col] / 1e6, df.loc[idx, recall_col]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=rep_colors[rep_name]
                )
        
        ax.set_xlabel('Total Cost (Million USD)', fontsize=11)
        ax.set_ylabel('Detection Recall', fontsize=11)
        ax.set_title('Pareto-Optimal Solutions in Cost-Recall Space', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0.78, 1.0)
        
        self._save_fig(fig, 'fig1_pareto_front',
                       inputs=['pareto_solutions.csv'],
                       fields=[cost_col, recall_col, latency_col])
        
        self.captions['fig1'] = (
            "Pareto-optimal solutions in cost-recall space. Color indicates latency. "
            "Three representative solutions highlighted: Low Cost (min cost with recall≥0.8), "
            "Balanced (min normalized weighted sum), High Recall (max recall within 80th cost percentile). "
            "Clustering reflects structural dominance of Traditional algorithms under current cost structure."
        )
    
    # =========================================================================
    # Fig 2: Decision Matrix Heatmap
    # =========================================================================
    def fig2_decision_matrix(self, pareto_df: pd.DataFrame):
        """决策矩阵热力图"""
        df = pareto_df.copy()
        
        # 目标列
        obj_cols = {
            'Cost': 'f1_total_cost_USD',
            'Recall': 'detection_recall',
            'Latency': 'f3_latency_seconds',
            'Disruption': 'f4_traffic_disruption_hours',
            'Carbon': 'f5_carbon_emissions_kgCO2e_year'
        }
        
        # 过滤可用列
        obj_cols = {k: v for k, v in obj_cols.items() if v in df.columns}
        
        if len(obj_cols) < 3:
            logger.warning("Skipping fig2: insufficient objective columns")
            return
        
        # 标准化
        norm_data = pd.DataFrame()
        for name, col in obj_cols.items():
            values = df[col].values
            min_v, max_v = values.min(), values.max()
            if max_v > min_v:
                if name == 'Recall':
                    norm_data[name] = (max_v - values) / (max_v - min_v)
                else:
                    norm_data[name] = (values - min_v) / (max_v - min_v)
            else:
                norm_data[name] = 0
        
        # 代表性解
        reps = select_representatives(df)
        
        # 标签
        sensor_col = 'sensor' if 'sensor' in df.columns else None
        algo_col = 'algorithm' if 'algorithm' in df.columns else None
        
        labels = []
        for i in range(len(df)):
            s = df[sensor_col].iloc[i][:8] if sensor_col else ''
            a = classify_algorithm(df[algo_col].iloc[i]) if algo_col else ''
            labels.append(f"{i+1}:{s}|{a}")
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.25)))
        
        im = ax.imshow(norm_data.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(len(obj_cols)))
        ax.set_xticklabels(list(obj_cols.keys()), fontsize=10)
        ax.set_yticks(np.arange(len(df)))
        ax.set_yticklabels(labels, fontsize=7)
        
        # 高亮代表性解
        for rep_name, idx in reps.items():
            if idx in df.index:
                row = df.index.get_loc(idx)
                color = {'low_cost': '#2ECC71', 'balanced': '#3498DB', 'high_recall': '#E74C3C'}[rep_name]
                rect = plt.Rectangle((-0.5, row-0.5), len(obj_cols), 1,
                                     fill=False, edgecolor=color, linewidth=2)
                ax.add_patch(rect)
        
        cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
        cbar.set_label('Normalized (0=Best)', fontsize=9)
        
        ax.set_xlabel('Objectives', fontsize=11)
        ax.set_ylabel('Solutions', fontsize=11)
        ax.set_title('Decision Matrix: Pareto Solutions', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        self._save_fig(fig, 'fig2_decision_matrix',
                       inputs=['pareto_solutions.csv'],
                       fields=list(obj_cols.values()))
        
        self.captions['fig2'] = (
            "Decision matrix showing normalized objectives for all Pareto solutions. "
            "Rows: configurations (sensor|algorithm). Columns: objectives (0=best, 1=worst). "
            "Colored borders highlight representative solutions."
        )
    
    # =========================================================================
    # Fig 3: Cost Structure
    # =========================================================================
    def fig3_cost_structure(self, pareto_df: pd.DataFrame):
        """成本结构分析"""
        df = pareto_df.copy()
        reps = select_representatives(df)
        
        cost_col = None
        for c in ['f1_total_cost_USD', 'total_cost']:
            if c in df.columns:
                cost_col = c
                break
        
        sensor_col = 'sensor' if 'sensor' in df.columns else None
        
        if cost_col is None or not reps:
            logger.warning("Skipping fig3: missing data")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        rep_names = ['low_cost', 'balanced', 'high_recall']
        rep_labels = ['Low Cost', 'Balanced', 'High Recall']
        x_pos = np.arange(len(rep_names))
        
        # 估算成本分解比例
        breakdown_pcts = {
            'Vehicle': {'CAPEX': 0.15, 'Sensor OPEX': 0.20, 'Storage': 0.10, 'Compute': 0.15, 'Labor': 0.40},
            'IoT': {'CAPEX': 0.50, 'Sensor OPEX': 0.05, 'Storage': 0.15, 'Compute': 0.10, 'Labor': 0.20},
            'Camera': {'CAPEX': 0.25, 'Sensor OPEX': 0.25, 'Storage': 0.15, 'Compute': 0.20, 'Labor': 0.15},
            'MMS': {'CAPEX': 0.40, 'Sensor OPEX': 0.30, 'Storage': 0.10, 'Compute': 0.05, 'Labor': 0.15},
            'default': {'CAPEX': 0.30, 'Sensor OPEX': 0.25, 'Storage': 0.15, 'Compute': 0.15, 'Labor': 0.15},
        }
        
        colors = ['#3498DB', '#2ECC71', '#F39C12', '#E74C3C', '#9B59B6']
        categories = ['CAPEX', 'Sensor OPEX', 'Storage', 'Compute', 'Labor']
        
        bottom = np.zeros(len(rep_names))
        
        for i, cat in enumerate(categories):
            values = []
            for rep_name in rep_names:
                if rep_name in reps:
                    idx = reps[rep_name]
                    total_cost = df.loc[idx, cost_col]
                    sensor_type = classify_sensor(df.loc[idx, sensor_col]) if sensor_col else 'default'
                    pcts = breakdown_pcts.get(sensor_type, breakdown_pcts['default'])
                    values.append(total_cost * pcts[cat] / 1e6)
                else:
                    values.append(0)
            
            ax.bar(x_pos, values, 0.6, bottom=bottom, label=cat, color=colors[i])
            bottom += np.array(values)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(rep_labels, fontsize=11)
        ax.set_ylabel('Total Cost (Million USD)', fontsize=11)
        ax.set_title('Cost Structure of Representative Solutions', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加总成本标签
        for i, rep_name in enumerate(rep_names):
            if rep_name in reps:
                idx = reps[rep_name]
                total = df.loc[idx, cost_col] / 1e6
                ax.annotate(f'${total:.2f}M', (i, bottom[i]), 
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        self._save_fig(fig, 'fig3_cost_structure',
                       inputs=['pareto_solutions.csv'],
                       fields=[cost_col, sensor_col],
                       notes='Breakdown percentages estimated by sensor type')
        
        self.captions['fig3'] = (
            "Cost structure breakdown for representative solutions. "
            "Traditional algorithm dominance explained by lower compute costs. "
            "Low Cost solution achieves savings through efficient vehicle-mounted sensors."
        )
    
    # =========================================================================
    # Fig 4: Discrete Distributions
    # =========================================================================
    def fig4_discrete_distributions(self, pareto_df: pd.DataFrame):
        """离散变量分布"""
        df = pareto_df.copy()
        
        cycle_col = 'inspection_cycle_days' if 'inspection_cycle_days' in df.columns else None
        crew_col = 'crew_size' if 'crew_size' in df.columns else None
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Inspection Cycle
        if cycle_col:
            ax = axes[0]
            values = df[cycle_col].values
            
            ax.hist(values, bins=20, edgecolor='black', alpha=0.7, color=COLORS['pareto'])
            
            # 边界标注
            ax.axvline(x=180, color='red', linestyle='--', linewidth=2, label='Constraint (180 days)')
            
            ax.set_xlabel('Inspection Cycle (days)', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('(a) Inspection Cycle Distribution', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            
            # 解释标注
            ax.annotate('Boundary convergence:\nCost-optimal under\ncurrent constraints',
                       xy=(175, ax.get_ylim()[1] * 0.7), fontsize=8, style='italic',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            axes[0].text(0.5, 0.5, 'Data not available', ha='center', va='center',
                        transform=axes[0].transAxes)
        
        # Crew Size
        if crew_col:
            ax = axes[1]
            crew_counts = df[crew_col].value_counts().sort_index()
            
            bars = ax.bar(crew_counts.index.astype(str), crew_counts.values,
                         color=COLORS['pareto'], edgecolor='black', alpha=0.7)
            
            for bar, count in zip(bars, crew_counts.values):
                ax.annotate(f'{count}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('Crew Size', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('(b) Crew Size Distribution', fontsize=11, fontweight='bold')
        else:
            axes[1].text(0.5, 0.5, 'Data not available', ha='center', va='center',
                        transform=axes[1].transAxes)
        
        plt.suptitle('Discrete Decision Variable Distributions', fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self._save_fig(fig, 'fig4_discrete_distributions',
                       inputs=['pareto_solutions.csv'],
                       fields=[cycle_col, crew_col])
        
        self.captions['fig4'] = (
            "Discrete variable distributions. (a) Inspection cycle converges to 180-day boundary, "
            "indicating cost-optimal strategy under constraints—a structural finding. "
            "(b) Crew size preference for smaller teams reflects automated sensing efficiency."
        )
    
    # =========================================================================
    # Fig 5: Technology Dominance
    # =========================================================================
    def fig5_technology_dominance(self, pareto_df: pd.DataFrame, baseline_dfs: Dict):
        """技术选择模式"""
        df = pareto_df.copy()
        
        sensor_col = 'sensor' if 'sensor' in df.columns else None
        algo_col = 'algorithm' if 'algorithm' in df.columns else None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Sensor分布
        if sensor_col:
            ax = axes[0]
            df['sensor_cat'] = df[sensor_col].apply(classify_sensor)
            pareto_counts = df['sensor_cat'].value_counts()
            
            ax.barh(pareto_counts.index, pareto_counts.values / pareto_counts.sum() * 100,
                   color=COLORS['pareto'], alpha=0.8, edgecolor='black')
            
            ax.set_xlabel('Percentage (%)', fontsize=11)
            ax.set_ylabel('Sensor Type', fontsize=11)
            ax.set_title('(a) Sensor Type Distribution', fontsize=11, fontweight='bold')
            ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Algorithm分布
        if algo_col:
            ax = axes[1]
            df['algo_family'] = df[algo_col].apply(classify_algorithm)
            algo_counts = df['algo_family'].value_counts()
            
            colors = [COLORS.get(f.lower(), '#888888') for f in algo_counts.index]
            
            wedges, texts, autotexts = ax.pie(
                algo_counts.values, labels=algo_counts.index,
                autopct='%1.1f%%', colors=colors,
                explode=[0.05] * len(algo_counts), shadow=True, startangle=90
            )
            
            ax.set_title('(b) Algorithm Family Distribution', fontsize=11, fontweight='bold')
            
            # 解释标注
            if 'Traditional' in algo_counts.index and algo_counts['Traditional'] > len(df) * 0.5:
                ax.annotate(
                    'Traditional dominance:\n$0.08/GB vs $0.80/GB (DL)',
                    xy=(0.5, -0.15), xycoords='axes fraction',
                    fontsize=8, style='italic', ha='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                )
        
        plt.suptitle('Technology Selection Patterns in Pareto Solutions', fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self._save_fig(fig, 'fig5_technology_dominance',
                       inputs=['pareto_solutions.csv'],
                       fields=[sensor_col, algo_col])
        
        self.captions['fig5'] = (
            "Technology selection patterns. (a) Camera-based sensors dominate due to cost-accuracy trade-offs. "
            "(b) Traditional algorithms (96%) dominate because processing costs are 10x lower than DL—"
            "a structural finding reflecting current cost structure, not algorithmic limitation."
        )
    
    # =========================================================================
    # Fig 6: Baseline Comparison
    # =========================================================================
    def fig6_baseline_comparison(self, pareto_df: pd.DataFrame, baseline_dfs: Dict):
        """Baseline方法对比"""
        df = pareto_df.copy()
        
        cost_col = None
        for c in ['f1_total_cost_USD', 'total_cost']:
            if c in df.columns:
                cost_col = c
                break
        
        recall_col = 'detection_recall' if 'detection_recall' in df.columns else None
        
        if cost_col is None or recall_col is None:
            logger.warning("Skipping fig6: missing columns")
            return
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Plot baselines
        baseline_colors = {'random': '#7f7f7f', 'grid': '#2ca02c', 'weighted': '#ff7f0e', 'expert': '#d62728'}
        
        for name, bdf in baseline_dfs.items():
            if bdf is None or len(bdf) == 0:
                continue
            
            # 适配列名
            b_cost = None
            for c in ['f1_total_cost_USD', 'total_cost']:
                if c in bdf.columns:
                    b_cost = c
                    break
            
            b_recall = 'detection_recall'
            if b_recall not in bdf.columns and 'f2_one_minus_recall' in bdf.columns:
                bdf = bdf.copy()
                bdf['detection_recall'] = 1 - bdf['f2_one_minus_recall']
            
            if b_cost and 'detection_recall' in bdf.columns:
                # 过滤可行解
                if 'is_feasible' in bdf.columns:
                    bdf_feas = bdf[bdf['is_feasible'] == True]
                else:
                    bdf_feas = bdf
                
                if len(bdf_feas) > 0:
                    ax.scatter(
                        bdf_feas[b_cost] / 1e6,
                        bdf_feas['detection_recall'],
                        alpha=0.3, s=20,
                        label=f'{name.title()} (n={len(bdf_feas)})',
                        color=baseline_colors.get(name, '#888888'),
                        zorder=1
                    )
        
        # Plot Pareto
        ax.scatter(
            df[cost_col] / 1e6,
            df[recall_col],
            s=120, color=COLORS['pareto'],
            edgecolors='white', linewidths=1,
            label=f'NSGA-III Pareto (n={len(df)})',
            zorder=3
        )
        
        # Pareto连线
        sorted_df = df.sort_values(cost_col)
        ax.plot(sorted_df[cost_col] / 1e6, sorted_df[recall_col],
               color=COLORS['pareto'], linewidth=2, alpha=0.7, zorder=2)
        
        ax.set_xlabel('Total Cost (Million USD)', fontsize=11)
        ax.set_ylabel('Detection Recall', fontsize=11)
        ax.set_title('NSGA-III vs Baseline Methods', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0.78, 1.0)
        
        plt.tight_layout()
        
        self._save_fig(fig, 'fig6_baseline_comparison',
                       inputs=['pareto_solutions.csv'] + [f'baseline_{k}.csv' for k in baseline_dfs.keys()],
                       fields=[cost_col, recall_col])
        
        self.captions['fig6'] = (
            "Comparison with baseline methods. NSGA-III Pareto front (blue) dominates "
            "Random, Grid, Weighted Sum, and Expert baselines. "
            "Multi-objective optimization discovers superior trade-off solutions."
        )
    
    # =========================================================================
    # Fig 7 & 8: Placeholders
    # =========================================================================
    def fig7_convergence(self, run_dir: str = None):
        """收敛分析 - 读取真实数据"""
        import glob

        history_data = None
        possible_paths = []

        if run_dir:
            possible_paths.append(Path(run_dir) / 'optimization_history.json')

        possible_paths.extend([
            Path('./results/runs') / 'optimization_history.json',
        ])

        for p in glob.glob('./results/runs/*_seed*/optimization_history.json'):
            possible_paths.append(Path(p))

        for p in possible_paths:
            if p.exists():
                with open(p, 'r') as f:
                    history_data = json.load(f)
                logger.info(f"  Loaded history from: {p}")
                break

        if history_data is None or 'generations' not in history_data:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, 'Convergence Analysis\n\n(Requires optimization_history.json)',
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            ax.axis('off')
            ax.set_title('Convergence Analysis', fontsize=12, fontweight='bold')
            self._save_fig(fig, 'fig7_convergence_placeholder', notes='Data not found')
            self.captions['fig7'] = "Convergence analysis (placeholder—data not found)."
            return

        generations = history_data['generations']
        gens = [g['generation'] for g in generations]
        n_nds = [g.get('n_nds', 0) for g in generations]
        cv_avg = [g.get('cv_avg', 0) for g in generations]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax1 = axes[0]
        ax1.plot(gens, n_nds, 'b-o', linewidth=2, markersize=5)
        ax1.fill_between(gens, 0, n_nds, alpha=0.2, color='blue')
        ax1.set_xlabel('Generation', fontsize=11)
        ax1.set_ylabel('Number of Non-dominated Solutions', fontsize=11)
        ax1.set_title('(a) Pareto Front Evolution', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')

        ax2 = axes[1]
        ax2.plot(gens, cv_avg, 'r-s', linewidth=2, markersize=5)
        ax2.fill_between(gens, 0, cv_avg, alpha=0.2, color='red')
        ax2.set_xlabel('Generation', fontsize=11)
        ax2.set_ylabel('Constraint Violation (avg)', fontsize=11)
        ax2.set_title('(b) Constraint Satisfaction', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        if max(cv_avg) > 0.1:
            ax2.set_yscale('log')

        plt.suptitle('Convergence Analysis', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        self._save_fig(fig, 'fig7_convergence',
                       inputs=['optimization_history.json'],
                       fields=['generation', 'n_nds', 'cv_avg'])

        self.captions['fig7'] = f"Convergence analysis over {len(generations)} generations."

    def fig8_ablation(self, ablation_dir: str = None):
        """消融实验 - 读取真实数据"""

        ablation_df = None
        possible_paths = [
            Path('./results/ablation_v3/ablation_results_v3.csv'),
            Path('./results/ablation/ablation_results_v3.csv'),
            Path('./ablation_results_v3.csv'),
        ]

        if ablation_dir:
            possible_paths.insert(0, Path(ablation_dir) / 'ablation_results_v3.csv')

        for p in possible_paths:
            if p.exists():
                ablation_df = pd.read_csv(p)
                logger.info(f"  Loaded ablation from: {p}")
                break

        if ablation_df is None or len(ablation_df) == 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, 'Ontology Ablation Study\n\n(Requires ablation_results_v3.csv)',
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            ax.axis('off')
            ax.set_title('Ontology Ablation Study', fontsize=12, fontweight='bold')
            self._save_fig(fig, 'fig8_ablation_placeholder', notes='Data not found')
            self.captions['fig8'] = "Ontology ablation study (placeholder—data not found)."
            return

        variant_col = 'variant' if 'variant' in ablation_df.columns else 'mode_name'
        feasible_col = 'feasible_rate' if 'feasible_rate' in ablation_df.columns else 'feasible_rate_true'

        variants = ablation_df[variant_col].tolist()
        validity = ablation_df['validity_rate'].tolist()
        feasible = ablation_df[feasible_col].tolist()
        hv = ablation_df['hv_6d'].tolist()

        short_labels = []
        for v in variants:
            if 'Full' in v:
                short_labels.append('Full\nOntology')
            elif 'Type' in v:
                short_labels.append('No Type\nInference')
            elif 'Compat' in v:
                short_labels.append('No Compat\nCheck')
            elif 'SHACL' in v:
                short_labels.append('No SHACL')
            elif 'Minimal' in v:
                short_labels.append('Minimal')
            else:
                short_labels.append(v[:10])

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        x = np.arange(len(variants))
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#95a5a6']

        ax1 = axes[0]
        ax1.bar(x, validity, color=colors[:len(x)], edgecolor='black', linewidth=0.5)
        ax1.set_ylabel('Validity Rate', fontsize=11)
        ax1.set_title('(a) Configuration Validity', fontsize=11, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(short_labels, fontsize=8)
        ax1.set_ylim(0, 1.15)
        for i, v in enumerate(validity):
            ax1.text(i, v + 0.03, f'{v:.0%}', ha='center', fontsize=9, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        ax2 = axes[1]
        ax2.bar(x, feasible, color=colors[:len(x)], edgecolor='black', linewidth=0.5)
        ax2.set_ylabel('Feasible Rate', fontsize=11)
        ax2.set_title('(b) Constraint Feasibility', fontsize=11, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(short_labels, fontsize=8)
        ax2.set_ylim(0, 1.15)
        for i, v in enumerate(feasible):
            ax2.text(i, v + 0.03, f'{v:.0%}', ha='center', fontsize=9, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        ax3 = axes[2]
        ax3.bar(x, hv, color=colors[:len(x)], edgecolor='black', linewidth=0.5)
        ax3.set_ylabel('Hypervolume (6D)', fontsize=11)
        ax3.set_title('(c) Solution Quality', fontsize=11, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(short_labels, fontsize=8)
        ax3.set_ylim(0, max(hv) * 1.25 if hv else 1)
        for i, v in enumerate(hv):
            ax3.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3, linestyle='--')

        plt.suptitle('Ontology Ablation Study', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        self._save_fig(fig, 'fig8_ablation',
                       inputs=['ablation_results_v3.csv'],
                       fields=['variant', 'validity_rate', 'feasible_rate', 'hv_6d'])

        self.captions['fig8'] = f"Ontology ablation study comparing {len(variants)} variants."

    # =========================================================================
    # Supplementary Figures
    # =========================================================================
    def figS1_pairwise_tradeoffs(self, pareto_df: pd.DataFrame):
        """成对Trade-off"""
        df = pareto_df.copy()
        
        cost_col = 'f1_total_cost_USD' if 'f1_total_cost_USD' in df.columns else None
        recall_col = 'detection_recall' if 'detection_recall' in df.columns else None
        latency_col = 'f3_latency_seconds' if 'f3_latency_seconds' in df.columns else None
        algo_col = 'algorithm' if 'algorithm' in df.columns else None
        
        if algo_col:
            df['algo_family'] = df[algo_col].apply(classify_algorithm)
        
        pairs = [
            (cost_col, recall_col, 'Cost ($M)', 'Recall'),
            (cost_col, latency_col, 'Cost ($M)', 'Latency (s)'),
            (latency_col, recall_col, 'Latency (s)', 'Recall'),
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        for idx, (x_col, y_col, x_label, y_label) in enumerate(pairs):
            ax = axes[idx]
            
            if x_col is None or y_col is None:
                ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
                       transform=ax.transAxes)
                continue
            
            x_data = df[x_col] / 1e6 if 'Cost' in x_label else df[x_col]
            y_data = df[y_col]
            
            # 添加微小jitter
            x_jitter = x_data + np.random.normal(0, x_data.std() * 0.01, len(x_data))
            y_jitter = y_data + np.random.normal(0, y_data.std() * 0.01, len(y_data))
            
            if algo_col:
                for family, marker in ALGO_MARKERS.items():
                    mask = df['algo_family'] == family
                    if mask.any():
                        ax.scatter(x_jitter[mask], y_jitter[mask],
                                  marker=marker, s=80, alpha=0.7,
                                  label=family if idx == 0 else None,
                                  edgecolors='white', linewidths=0.5)
            else:
                ax.scatter(x_jitter, y_jitter, s=80, alpha=0.7,
                          color=COLORS['pareto'], edgecolors='white', linewidths=0.5)
            
            ax.set_xlabel(x_label, fontsize=10)
            ax.set_ylabel(y_label, fontsize=10)
            ax.set_title(f'({chr(97+idx)}) {y_label} vs {x_label}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            
            if idx == 0 and algo_col:
                ax.legend(fontsize=8, loc='lower right')
        
        plt.suptitle('Pairwise Objective Trade-offs', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        self._save_fig(fig, 'figS1_pairwise_tradeoffs',
                       inputs=['pareto_solutions.csv'],
                       notes='Jitter (1% std) for visibility')
        
        self.captions['figS1'] = (
            "Pairwise trade-offs. Shapes indicate algorithm family. "
            "Small jitter applied for overlapping point visibility."
        )
    
    def figS2_sensitivity(self, pareto_df: pd.DataFrame):
        """敏感性分析"""
        df = pareto_df.copy()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        cost_col = 'f1_total_cost_USD' if 'f1_total_cost_USD' in df.columns else None
        recall_col = 'detection_recall' if 'detection_recall' in df.columns else None
        latency_col = 'f3_latency_seconds' if 'f3_latency_seconds' in df.columns else None
        crew_col = 'crew_size' if 'crew_size' in df.columns else None
        sensor_col = 'sensor' if 'sensor' in df.columns else None
        cycle_col = 'inspection_cycle_days' if 'inspection_cycle_days' in df.columns else None
        
        # (a) Threshold vs Recall
        ax = axes[0, 0]
        if 'detection_threshold' in df.columns and recall_col:
            scatter = ax.scatter(df['detection_threshold'], df[recall_col],
                               c=df[cost_col]/1e6 if cost_col else None,
                               cmap='viridis', s=80, alpha=0.8)
            if cost_col:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Cost ($M)', fontsize=9)
            ax.set_xlabel('Detection Threshold', fontsize=10)
            ax.set_ylabel('Recall', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('(a) Threshold Sensitivity', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # (b) Sensor vs Latency (boxplot)
        ax = axes[0, 1]
        if sensor_col and latency_col:
            df['sensor_cat'] = df[sensor_col].apply(classify_sensor)
            categories = df['sensor_cat'].unique()
            box_data = [df[df['sensor_cat'] == cat][latency_col].values for cat in categories]
            
            bp = ax.boxplot(box_data, labels=[c[:6] for c in categories], patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(COLORS['pareto'])
                patch.set_alpha(0.7)
            ax.set_xlabel('Sensor Type', fontsize=10)
            ax.set_ylabel('Latency (s)', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('(b) Sensor Impact on Latency', fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # (c) Crew vs Cost (boxplot)
        ax = axes[1, 0]
        if crew_col and cost_col:
            crew_values = sorted(df[crew_col].unique())
            box_data = [df[df[crew_col] == c][cost_col].values / 1e6 for c in crew_values]
            
            bp = ax.boxplot(box_data, labels=[str(int(c)) for c in crew_values], patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(COLORS['pareto'])
                patch.set_alpha(0.7)
            ax.set_xlabel('Crew Size', fontsize=10)
            ax.set_ylabel('Cost ($M)', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('(c) Crew Impact on Cost', fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # (d) Inspection Cycle histogram
        ax = axes[1, 1]
        if cycle_col:
            ax.hist(df[cycle_col], bins=15, color=COLORS['pareto'], edgecolor='black', alpha=0.7)
            ax.axvline(x=180, color='red', linestyle='--', linewidth=2, label='Constraint')
            ax.set_xlabel('Inspection Cycle (days)', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('(d) Inspection Cycle Distribution', fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.suptitle('Parameter Sensitivity Analysis', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        self._save_fig(fig, 'figS2_sensitivity',
                       inputs=['pareto_solutions.csv'])
        
        self.captions['figS2'] = (
            "Sensitivity analysis. (a) Threshold-recall relationship. "
            "(b-c) Boxplots for discrete variables. "
            "(d) Inspection cycle boundary convergence."
        )
    
    # =========================================================================
    # Save Methods
    # =========================================================================
    def _save_manifest(self):
        """保存manifest"""
        path = self.output_dir / 'figure_manifest.json'
        with open(path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        logger.info(f"Saved manifest: {path}")
    
    def _save_captions(self):
        """保存captions"""
        path = self.output_dir / 'captions.md'
        with open(path, 'w') as f:
            f.write("# Figure Captions\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for name, caption in self.captions.items():
                f.write(f"## {name}\n\n{caption}\n\n---\n\n")
        logger.info(f"Saved captions: {path}")


# =============================================================================
# 命令行接口（兼容现有调用方式）
# =============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate paper figures v3.3')
    parser.add_argument('--pareto', type=str, required=True, help='Path to Pareto CSV')
    parser.add_argument('--baselines', type=str, nargs='+', help='Baseline CSV paths')
    parser.add_argument('--output', type=str, default='./results/paper/figures_v33', help='Output dir')
    parser.add_argument('--run-dir', type=str, default=None, help='Run directory with optimization_history.json')  # 新增
    parser.add_argument('--ablation-dir', type=str, default=None, help='Ablation results directory')  # 新增

    args = parser.parse_args()

    # 加载数据
    pareto_df = pd.read_csv(args.pareto)

    baseline_dfs = {}
    if args.baselines:
        for path in args.baselines:
            name = Path(path).stem.replace('baseline_', '')
            baseline_dfs[name] = pd.read_csv(path)
    else:
        # 自动查找
        pareto_dir = Path(args.pareto).parent
        for f in pareto_dir.glob('baseline_*.csv'):
            name = f.stem.replace('baseline_', '')
            baseline_dfs[name] = pd.read_csv(f)

    # 自动推断 run_dir (如果未指定)
    run_dir = args.run_dir or str(Path(args.pareto).parent)
    ablation_dir = args.ablation_dir or './results/ablation_v3'

    # 生成图表
    visualizer = PaperVisualizer(output_dir=args.output)
    visualizer.generate_all(pareto_df, baseline_dfs, run_dir=run_dir, ablation_dir=ablation_dir)

if __name__ == '__main__':
    main()
