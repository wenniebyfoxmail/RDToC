#!/usr/bin/env python3
"""
Utility Functions for RMTwin Optimization
修复版 v2 - 修复报告生成逻辑

关键修复:
1. 修复 "No feasible solutions" 误触发的bug
2. 正确统计NSGA和baseline的可行解数量
3. 改进报告结构清晰度
"""

import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


def setup_logging(debug: bool = False, log_dir: Path = Path('./results/logs')) -> logging.Logger:
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'rmtwin_optimization_{timestamp}.log'

    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Setup handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Console always INFO or higher
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 清除已有handlers避免重复
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Log startup info
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info(f"Debug mode: {debug}")

    return logger


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types"""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)


def save_results_summary(results: Dict, config) -> None:
    """
    Save comprehensive results summary with empty DataFrame handling

    修复版: 正确判断可行解状态，避免矛盾报告
    """
    logger = logging.getLogger(__name__)

    summary_path = config.output_dir / 'optimization_summary.json'
    report_path = config.output_dir / 'optimization_report.txt'

    # Prepare summary data
    summary = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'road_network_km': config.road_network_length_km,
            'planning_years': config.planning_horizon_years,
            'budget_usd': config.budget_cap_usd,
            'objectives': config.n_objectives,
            'population_size': config.population_size,
            'generations': config.n_generations,
            'min_recall': config.min_recall_threshold,
            'max_latency': config.max_latency_seconds,
            'max_carbon': getattr(config, 'max_carbon_emissions_kgCO2e_year', 200000),
            'min_mtbf': config.min_mtbf_hours,
        },
        'results': {}
    }

    # 跟踪实际可行解数量
    nsga_feasible_count = 0
    baseline_feasible_count = 0

    # Process NSGA-III results if available
    if 'nsga3' in results:
        nsga3_df = results['nsga3']['dataframe']

        if nsga3_df is not None and len(nsga3_df) > 0 and 'f1_total_cost_USD' in nsga3_df.columns:
            # Pareto解默认都是可行的
            nsga_feasible_count = len(nsga3_df)

            summary['results']['nsga3'] = {
                'total_solutions': len(nsga3_df),
                'feasible_solutions': nsga_feasible_count,  # 显式添加feasible_solutions
                'computation_time': results['nsga3'].get('time', 0),
                'objectives': {
                    'cost': {
                        'min': float(nsga3_df['f1_total_cost_USD'].min()),
                        'max': float(nsga3_df['f1_total_cost_USD'].max()),
                        'mean': float(nsga3_df['f1_total_cost_USD'].mean())
                    },
                    'recall': {
                        'min': float(nsga3_df['detection_recall'].min()),
                        'max': float(nsga3_df['detection_recall'].max()),
                        'mean': float(nsga3_df['detection_recall'].mean())
                    }
                }
            }

            # Add other objectives if available
            if 'f5_carbon_emissions_kgCO2e_year' in nsga3_df.columns:
                summary['results']['nsga3']['objectives']['carbon'] = {
                    'min': float(nsga3_df['f5_carbon_emissions_kgCO2e_year'].min()),
                    'max': float(nsga3_df['f5_carbon_emissions_kgCO2e_year'].max()),
                    'mean': float(nsga3_df['f5_carbon_emissions_kgCO2e_year'].mean())
                }

            if 'f3_latency_seconds' in nsga3_df.columns:
                summary['results']['nsga3']['objectives']['latency'] = {
                    'min': float(nsga3_df['f3_latency_seconds'].min()),
                    'max': float(nsga3_df['f3_latency_seconds'].max()),
                    'mean': float(nsga3_df['f3_latency_seconds'].mean())
                }

            if 'f4_traffic_disruption_hours' in nsga3_df.columns:
                summary['results']['nsga3']['objectives']['disruption'] = {
                    'min': float(nsga3_df['f4_traffic_disruption_hours'].min()),
                    'max': float(nsga3_df['f4_traffic_disruption_hours'].max()),
                    'mean': float(nsga3_df['f4_traffic_disruption_hours'].mean())
                }
        else:
            summary['results']['nsga3'] = {
                'total_solutions': 0,
                'feasible_solutions': 0,
                'computation_time': results['nsga3'].get('time', 0),
                'message': 'No Pareto solutions found'
            }

    # Process baseline results if available
    if 'baselines' in results and results['baselines'].get('dataframes'):
        summary['results']['baselines'] = {}

        for method, df in results['baselines']['dataframes'].items():
            if df is not None and len(df) > 0:
                if 'is_feasible' in df.columns:
                    feasible = df[df['is_feasible']]
                    n_feasible = len(feasible)
                else:
                    feasible = df
                    n_feasible = len(df)

                baseline_feasible_count += n_feasible

                summary['results']['baselines'][method] = {
                    'total_solutions': len(df),
                    'feasible_solutions': n_feasible,
                    'best_cost': float(feasible[
                                           'f1_total_cost_USD'].min()) if n_feasible > 0 and 'f1_total_cost_USD' in feasible.columns else None,
                    'best_recall': float(feasible[
                                             'detection_recall'].max()) if n_feasible > 0 and 'detection_recall' in feasible.columns else None,
                    'min_carbon': float(feasible[
                                            'f5_carbon_emissions_kgCO2e_year'].min()) if n_feasible > 0 and 'f5_carbon_emissions_kgCO2e_year' in feasible.columns else None,
                }
            else:
                summary['results']['baselines'][method] = {
                    'total_solutions': 0,
                    'feasible_solutions': 0,
                    'message': 'No solutions generated'
                }

    # 添加汇总统计
    summary['summary'] = {
        'nsga_feasible': nsga_feasible_count,
        'baseline_feasible': baseline_feasible_count,
        'total_feasible': nsga_feasible_count + baseline_feasible_count,
        'has_feasible_solutions': (nsga_feasible_count + baseline_feasible_count) > 0
    }

    # Save JSON summary
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    # Generate text report (使用修复版)
    report = generate_text_report_fixed(summary, config, nsga_feasible_count, baseline_feasible_count)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"Saved optimization summary to {summary_path}")
    logger.info(f"Saved optimization report to {report_path}")


def generate_text_report_fixed(summary: Dict, config,
                               nsga_feasible: int,
                               baseline_feasible: int) -> str:
    """
    Generate human-readable text report

    修复版: 正确判断是否显示troubleshooting
    - 只有当NSGA和所有baseline都没有可行解时才显示
    """
    report = []
    report.append("=" * 80)
    report.append("RMTWIN MULTI-OBJECTIVE OPTIMIZATION REPORT")
    report.append("=" * 80)
    report.append("")
    report.append(f"Generated: {summary['timestamp']}")
    report.append(f"Output Directory: {config.output_dir}")
    report.append("")

    # Configuration section
    report.append("CONFIGURATION")
    report.append("-" * 40)
    conf = summary['configuration']
    report.append(f"Road Network Length: {conf['road_network_km']} km")
    report.append(f"Planning Horizon: {conf['planning_years']} years")
    report.append(f"Budget Cap: ${conf['budget_usd']:,.0f}")
    report.append(f"Number of Objectives: {conf['objectives']}")
    report.append(f"Population Size: {conf['population_size']}")
    report.append(f"Generations: {conf['generations']}")
    report.append("")

    # Constraints section
    report.append("CONSTRAINTS")
    report.append("-" * 40)
    report.append(f"Min Recall: {conf.get('min_recall', 0.6)}")
    report.append(f"Max Latency: {conf.get('max_latency', 400)} s")
    report.append(f"Max Carbon: {conf.get('max_carbon', 200000)} kgCO2e/year")
    report.append(f"Min MTBF: {conf.get('min_mtbf', 2000)} hours")
    report.append("")

    # Results section
    report.append("OPTIMIZATION RESULTS")
    report.append("-" * 40)

    # NSGA-III results
    if 'nsga3' in summary.get('results', {}):
        nsga3 = summary['results']['nsga3']
        report.append("NSGA-III Pareto Front:")
        report.append(f"  Total Solutions: {nsga3.get('total_solutions', 0)}")
        report.append(f"  Feasible Solutions: {nsga3.get('feasible_solutions', nsga3.get('total_solutions', 0))}")

        if 'objectives' in nsga3:
            report.append("  Objective Ranges:")
            obj = nsga3['objectives']

            if 'cost' in obj:
                report.append(f"    Cost: ${obj['cost']['min']:,.0f} - ${obj['cost']['max']:,.0f} "
                              f"(avg: ${obj['cost']['mean']:,.0f})")

            if 'recall' in obj:
                report.append(f"    Recall: {obj['recall']['min']:.4f} - {obj['recall']['max']:.4f} "
                              f"(avg: {obj['recall']['mean']:.4f})")

            if 'carbon' in obj:
                report.append(f"    Carbon: {obj['carbon']['min']:,.0f} - {obj['carbon']['max']:,.0f} "
                              f"kgCO2e/year (avg: {obj['carbon']['mean']:,.0f})")

            if 'latency' in obj:
                report.append(f"    Latency: {obj['latency']['min']:.1f} - {obj['latency']['max']:.1f} s "
                              f"(avg: {obj['latency']['mean']:.1f})")

            if 'disruption' in obj:
                report.append(f"    Disruption: {obj['disruption']['min']:.1f} - {obj['disruption']['max']:.1f} h "
                              f"(avg: {obj['disruption']['mean']:.1f})")

        if 'computation_time' in nsga3:
            report.append(f"  Computation Time: {nsga3['computation_time']:.2f} s")

        if 'message' in nsga3:
            report.append(f"  Status: {nsga3['message']}")

        report.append("")

    # Baseline results
    if 'baselines' in summary.get('results', {}):
        report.append("Baseline Methods:")
        for method, data in summary['results']['baselines'].items():
            report.append(f"  {method.title()}:")
            report.append(f"    Total: {data.get('total_solutions', 0)}, "
                          f"Feasible: {data.get('feasible_solutions', 0)}")

            if data.get('best_cost') is not None:
                report.append(f"    Best Cost: ${data['best_cost']:,.0f}")
            if data.get('best_recall') is not None:
                report.append(f"    Best Recall: {data['best_recall']:.4f}")
            if data.get('min_carbon') is not None:
                report.append(f"    Min Carbon: {data['min_carbon']:,.0f} kgCO2e/year")
            if 'message' in data:
                report.append(f"    Status: {data['message']}")
        report.append("")

    # Summary section
    report.append("SUMMARY")
    report.append("-" * 40)
    report.append(f"NSGA-III Feasible Solutions: {nsga_feasible}")
    report.append(f"Baseline Feasible Solutions: {baseline_feasible}")
    report.append(f"Total Feasible Solutions: {nsga_feasible + baseline_feasible}")
    report.append("")

    # 关键修复: 只有在真正没有可行解时才显示troubleshooting
    total_feasible = nsga_feasible + baseline_feasible

    if total_feasible == 0:
        report.append("TROUBLESHOOTING")
        report.append("-" * 40)
        report.append("WARNING: No feasible solutions were found in any method!")
        report.append("")
        report.append("Possible causes:")
        report.append("  1. Constraints are too strict")
        report.append("  2. Budget is insufficient for the road network length")
        report.append("  3. Invalid component combinations in data files")
        report.append("")
        report.append("Suggestions:")
        report.append("  1. Lower min_recall_threshold (e.g., 0.5 -> 0.4)")
        report.append("  2. Increase max_latency_seconds (e.g., 400 -> 600)")
        report.append("  3. Increase budget_cap_usd")
        report.append("  4. Increase max_carbon_emissions_kgCO2e_year")
        report.append("  5. Check data files for valid sensor/algorithm/deployment combinations")
        report.append("  6. Enable --debug flag to see constraint violation details")
        report.append("")
    else:
        # 有可行解时显示成功信息
        report.append("STATUS: Optimization completed successfully with feasible solutions.")
        report.append("")

    report.append("=" * 80)
    report.append("END OF REPORT")

    return '\n'.join(report)


# ============================================================================
# 原有工具函数 (保持不变)
# ============================================================================

def calculate_hypervolume(F: np.ndarray, ref_point: np.ndarray) -> float:
    """Calculate hypervolume indicator"""
    try:
        from pymoo.indicators.hv import HV
        ind = HV(ref_point=ref_point)
        return ind(F)
    except ImportError:
        logging.warning("pymoo not available for hypervolume calculation")
        return 0.0


def calculate_metrics(F: np.ndarray) -> Dict[str, float]:
    """Calculate various performance metrics for a Pareto front"""
    metrics = {}

    # Basic statistics
    metrics['n_solutions'] = len(F)

    # Spread metrics
    for i in range(F.shape[1]):
        metrics[f'obj{i + 1}_range'] = float(F[:, i].max() - F[:, i].min())
        metrics[f'obj{i + 1}_std'] = float(F[:, i].std())

    # Coverage metrics
    metrics['hypervolume'] = calculate_hypervolume(
        F,
        ref_point=np.array([2e7, 0.3, 200, 200, 50000, 1e-3])
    )

    return metrics


def create_optimization_info_file(config, output_dir: Path) -> None:
    """Create detailed optimization info file for debugging"""
    info_path = output_dir / 'optimization_info.txt'

    with open(info_path, 'w') as f:
        f.write("RMTWIN OPTIMIZATION DETAILED INFORMATION\n")
        f.write("=" * 60 + "\n\n")

        # Timestamp
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Configuration details
        f.write("CONFIGURATION PARAMETERS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Road Network Length: {config.road_network_length_km} km\n")
        f.write(f"Planning Horizon: {config.planning_horizon_years} years\n")
        f.write(f"Budget Cap: ${config.budget_cap_usd:,.0f}\n")
        f.write(f"Daily Wage per Person: ${config.daily_wage_per_person}\n")
        f.write(f"Carbon Intensity: {config.carbon_intensity_factor} kgCO2e/kWh\n")
        f.write(f"Scenario Type: {config.scenario_type}\n")
        f.write("\n")

        # Constraints
        f.write("CONSTRAINTS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Min Recall Threshold: {config.min_recall_threshold}\n")
        f.write(f"Max Latency: {config.max_latency_seconds} seconds\n")
        f.write(f"Max Disruption: {config.max_disruption_hours} hours\n")
        f.write(f"Max Carbon: {getattr(config, 'max_carbon_emissions_kgCO2e_year', 200000)} kgCO2e/year\n")
        f.write(f"Min MTBF: {config.min_mtbf_hours:,.0f} hours\n")
        f.write("\n")

    logging.info(f"Created optimization info file: {info_path}")


def compare_solutions(sol1: pd.Series, sol2: pd.Series) -> Dict[str, float]:
    """Compare two solutions across all objectives"""
    comparison = {}

    objectives = [
        ('Cost', 'f1_total_cost_USD', 'minimize'),
        ('Recall', 'detection_recall', 'maximize'),
        ('Latency', 'f3_latency_seconds', 'minimize'),
        ('Disruption', 'f4_traffic_disruption_hours', 'minimize'),
        ('Carbon', 'f5_carbon_emissions_kgCO2e_year', 'minimize'),
        ('MTBF', 'system_MTBF_hours', 'maximize')
    ]

    for name, col, direction in objectives:
        if col in sol1 and col in sol2:
            val1 = sol1[col]
            val2 = sol2[col]

            if direction == 'minimize':
                improvement = (val2 - val1) / val1 * 100 if val1 != 0 else 0
            else:
                improvement = (val1 - val2) / val2 * 100 if val2 != 0 else 0

            comparison[name] = {
                'sol1': val1,
                'sol2': val2,
                'improvement_pct': improvement,
                'better': 1 if improvement < 0 else 2 if improvement > 0 else 0
            }

    return comparison


def format_solution_summary(solution: pd.Series) -> str:
    """Format a solution for display"""
    summary = []
    summary.append(f"Solution ID: {solution.get('solution_id', 'N/A')}")
    summary.append(f"Configuration:")
    summary.append(f"  Sensor: {solution.get('sensor', 'N/A')}")
    summary.append(f"  Algorithm: {solution.get('algorithm', 'N/A')}")
    summary.append(f"  Deployment: {solution.get('deployment', 'N/A')}")
    summary.append(f"  Crew Size: {solution.get('crew_size', 'N/A')}")
    summary.append(f"  Inspection Cycle: {solution.get('inspection_cycle_days', 'N/A')} days")
    summary.append(f"Performance:")
    summary.append(f"  Cost: ${solution.get('f1_total_cost_USD', 0):,.0f}")
    summary.append(f"  Recall: {solution.get('detection_recall', 0):.4f}")
    summary.append(f"  Latency: {solution.get('f3_latency_seconds', 0):.1f}s")
    summary.append(f"  Carbon: {solution.get('f5_carbon_emissions_kgCO2e_year', 0):,.0f} kgCO2e/year")
    summary.append(f"  MTBF: {solution.get('system_MTBF_hours', 0):,.0f} hours")

    return '\n'.join(summary)


def timer(func):
    """Decorator to time function execution"""

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"{func.__name__} took {end - start:.2f} seconds")
        return result

    return wrapper


def validate_data_files(config) -> bool:
    """Validate that all required data files exist"""
    required_files = [
        config.sensor_csv,
        config.algorithm_csv,
        config.infrastructure_csv,
        config.cost_benefit_csv
    ]

    all_exist = True
    for filepath in required_files:
        if not Path(filepath).exists():
            logging.error(f"Required file not found: {filepath}")
            all_exist = False
        else:
            logging.info(f"Found data file: {filepath}")

    return all_exist


def create_experiment_id() -> str:
    """Create unique experiment ID"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def save_checkpoint(data: Dict, checkpoint_dir: Path, name: str):
    """Save optimization checkpoint"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{name}_checkpoint.pkl"

    import pickle
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)

    logging.info(f"Saved checkpoint: {checkpoint_path}")


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict]:
    """Load optimization checkpoint"""
    if not checkpoint_path.exists():
        return None

    import pickle
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)

    logging.info(f"Loaded checkpoint: {checkpoint_path}")
    return data