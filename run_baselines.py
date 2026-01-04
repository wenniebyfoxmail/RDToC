#!/usr/bin/env python3
"""
Baseline Methods Runner for RMTwin
===================================
独立运行baseline方法并生成比较结果

Usage:
    python run_baselines.py --config config.json --seed 42 --output ./results/baselines/
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run Baseline Methods for RMTwin')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='./results/baselines/', help='Output directory')
    parser.add_argument('--pareto', type=str, default=None, help='Path to pareto_solutions.csv for comparison')
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("RMTwin Baseline Methods Runner")
    logger.info("=" * 70)
    logger.info(f"Config: {args.config}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output: {output_dir}")

    # 导入模块
    try:
        from config_manager import ConfigManager
        from ontology_manager import OntologyManager
        from baseline_methods import BaselineRunner
        logger.info("✓ All modules imported successfully")
    except ImportError as e:
        logger.error(f"Import error: {e}")
        sys.exit(1)

    # 加载配置
    logger.info("\n[1/4] Loading configuration...")
    config = ConfigManager.from_json(args.config)
    
    # 打印约束
    logger.info("\nConstraints:")
    logger.info(f"  Budget Cap: ${config.budget_cap_usd:,}")
    logger.info(f"  Min Recall: {config.min_recall_threshold}")
    logger.info(f"  Max Latency: {config.max_latency_seconds}s")
    logger.info(f"  Min MTBF: {config.min_mtbf_hours}h")
    logger.info(f"  Max Carbon: {config.max_carbon_emissions_kgCO2e_year:,} kgCO2e/year")

    # 加载本体
    logger.info("\n[2/4] Loading ontology...")
    ontology = OntologyManager()
    
    # 构建本体
    txt_files = {
        'sensor_csv': 'sensors_data.txt',
        'algorithm_csv': 'algorithms_data.txt',
        'infrastructure_csv': 'infrastructure_data.txt',
        'cost_benefit_csv': 'cost_benefit_data.txt',
    }
    
    if all(Path(f).exists() for f in txt_files.values()):
        ontology.populate_from_csv_files(**txt_files)
        logger.info("✓ Ontology populated from TXT files")
    else:
        logger.warning("Data files not found, using default ontology")

    # 运行baseline
    logger.info("\n[3/4] Running baseline methods...")
    start_time = time.time()
    
    runner = BaselineRunner(
        ontology_graph=ontology.g,
        config=config,
        seed=args.seed
    )
    
    # 使用正确的方法名运行所有baseline
    baseline_results = runner.run_all_methods()
    
    elapsed = time.time() - start_time
    logger.info(f"✓ Baseline methods completed in {elapsed:.1f}s")

    # 保存结果
    logger.info("\n[4/4] Saving results...")
    
    summary_data = []
    
    for method_name, df in baseline_results.items():
        # 保存原始结果
        csv_path = output_dir / f'baseline_{method_name.lower()}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"  Saved: {csv_path}")
        
        # 统计
        total = len(df)
        feasible = df['is_feasible'].sum() if 'is_feasible' in df.columns else 0
        
        if feasible > 0:
            feasible_df = df[df['is_feasible']]
            min_cost = feasible_df['f1_total_cost_USD'].min()
            max_recall = feasible_df['detection_recall'].max()
        else:
            min_cost = np.nan
            max_recall = np.nan
        
        summary_data.append({
            'Method': method_name,
            'Total': total,
            'Feasible': feasible,
            'Feasibility_Rate': f"{100*feasible/total:.1f}%" if total > 0 else "0%",
            'Min_Cost': f"${min_cost:,.0f}" if not np.isnan(min_cost) else "N/A",
            'Max_Recall': f"{max_recall:.4f}" if not np.isnan(max_recall) else "N/A",
        })
        
        logger.info(f"  {method_name}: {total} total, {feasible} feasible ({100*feasible/total:.1f}%)")

    # 加载Pareto结果进行比较
    if args.pareto and Path(args.pareto).exists():
        pareto_df = pd.read_csv(args.pareto)
        pareto_feasible = len(pareto_df)  # Pareto解都是可行的
        
        summary_data.insert(0, {
            'Method': 'NSGA-III',
            'Total': pareto_feasible,
            'Feasible': pareto_feasible,
            'Feasibility_Rate': "100%",
            'Min_Cost': f"${pareto_df['f1_total_cost_USD'].min():,.0f}",
            'Max_Recall': f"{pareto_df['detection_recall'].max():.4f}",
        })
        logger.info(f"  NSGA-III: {pareto_feasible} Pareto solutions")

    # 保存汇总表
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / 'baseline_comparison_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\n✓ Summary saved: {summary_path}")

    # 打印汇总
    logger.info("\n" + "=" * 70)
    logger.info("BASELINE COMPARISON SUMMARY")
    logger.info("=" * 70)
    print(summary_df.to_string(index=False))

    # 生成LaTeX表格
    latex_path = output_dir / 'baseline_comparison.tex'
    with open(latex_path, 'w') as f:
        f.write("% Baseline Method Comparison\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Baseline method comparison}\n")
        f.write("\\label{tab:baseline_comparison}\n")
        f.write("\\begin{tabular}{lrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Method & Total & Feasible & Rate & Min Cost & Max Recall \\\\\n")
        f.write("\\midrule\n")
        for _, row in summary_df.iterrows():
            f.write(f"{row['Method']} & {row['Total']} & {row['Feasible']} & "
                   f"{row['Feasibility_Rate']} & {row['Min_Cost']} & {row['Max_Recall']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    logger.info(f"✓ LaTeX table saved: {latex_path}")

    # 保存运行信息
    run_info = {
        'timestamp': datetime.now().isoformat(),
        'seed': args.seed,
        'config_file': args.config,
        'elapsed_seconds': elapsed,
        'results': {name: {'total': len(df), 'feasible': int(df['is_feasible'].sum())} 
                   for name, df in baseline_results.items()}
    }
    
    with open(output_dir / 'baseline_run_info.json', 'w') as f:
        json.dump(run_info, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("BASELINE RUN COMPLETE")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
