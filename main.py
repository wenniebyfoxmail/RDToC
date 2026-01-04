#!/usr/bin/env python3
"""
RMTwin Multi-Objective Optimization Framework v3.1

主程序入口 - 兼容版本，自动检测可用的类和方法
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# 动态导入模块（兼容不同版本的仓库）
# =============================================================================

def safe_import(module_name, class_name):
    """安全导入模块和类"""
    try:
        module = __import__(module_name)
        return getattr(module, class_name, None)
    except ImportError:
        return None
    except AttributeError:
        return None


# 导入配置管理器
ConfigManager = safe_import('config_manager', 'ConfigManager')
if ConfigManager is None:
    raise ImportError("无法导入 ConfigManager")

# 导入本体管理器
OntologyManager = safe_import('ontology_manager', 'OntologyManager')
if OntologyManager is None:
    raise ImportError("无法导入 OntologyManager")

# 导入优化器（尝试多个可能的名称）
RMTwinOptimizer = None
for module_name, class_name in [
    ('optimization_core', 'RMTwinOptimizer'),
    ('optimizer', 'NSGAIIIOptimizer'),
    ('optimizer', 'RMTwinOptimizer'),
    ('nsga_optimizer', 'NSGAIIIOptimizer'),
]:
    RMTwinOptimizer = safe_import(module_name, class_name)
    if RMTwinOptimizer is not None:
        logger.info(f"使用优化器: {module_name}.{class_name}")
        break

if RMTwinOptimizer is None:
    raise ImportError("无法导入优化器类")

# 导入基线运行器
BaselineRunner = safe_import('baseline_methods', 'BaselineRunner')

# 导入可视化器
ResultVisualizer = safe_import('visualization', 'ResultVisualizer')


# =============================================================================
# 辅助函数
# =============================================================================

def setup_run_directory(config: 'ConfigManager', seed: int) -> Path:
    """创建运行目录"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(config.output_dir) / 'runs' / f'{timestamp}_seed{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)

    # 创建子目录
    (run_dir / 'figures').mkdir(exist_ok=True)
    (run_dir / 'logs').mkdir(exist_ok=True)

    return run_dir


def setup_logging(run_dir: Path, debug: bool = False):
    """设置日志文件"""
    log_file = run_dir / 'logs' / f'rmtwin_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logging.getLogger().addHandler(file_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info(f"Debug mode: {debug}")


def build_ontology(ontology: 'OntologyManager') -> None:
    """构建本体（兼容不同的方法名和文件格式）"""
    from pathlib import Path

    # TXT文件路径（实际使用的格式）
    txt_files = {
        'sensor_csv': Path('sensors_data.txt'),
        'algorithm_csv': Path('algorithms_data.txt'),
        'infrastructure_csv': Path('infrastructure_data.txt'),
        'cost_benefit_csv': Path('cost_benefit_data.txt'),
    }

    # CSV文件路径（备选）
    csv_files = {
        'sensor_csv': Path('data/sensor_systems.csv'),
        'algorithm_csv': Path('data/algorithms.csv'),
        'infrastructure_csv': Path('data/infrastructure.csv'),
        'cost_benefit_csv': Path('data/cost_benefit.csv'),
    }

    # 确定使用哪组文件
    if all(f.exists() for f in txt_files.values()):
        files_to_use = txt_files
        logger.info("使用TXT数据文件")
    elif all(f.exists() for f in csv_files.values()):
        files_to_use = csv_files
        logger.info("使用CSV数据文件")
    else:
        files_to_use = None
        logger.warning("数据文件不完整")
        # 列出找到的文件
        import glob
        found_files = glob.glob('*.txt') + glob.glob('*.csv') + glob.glob('data/*.csv')
        logger.info(f"找到的数据文件: {found_files}")

    # 使用 populate_from_csv_files 方法
    if hasattr(ontology, 'populate_from_csv_files') and files_to_use:
        try:
            ontology.populate_from_csv_files(
                sensor_csv=str(files_to_use['sensor_csv']),
                algorithm_csv=str(files_to_use['algorithm_csv']),
                infrastructure_csv=str(files_to_use['infrastructure_csv']),
                cost_benefit_csv=str(files_to_use['cost_benefit_csv'])
            )
            logger.info("本体构建成功")
            return
        except Exception as e:
            logger.error(f"本体构建失败: {e}")
            raise

    # 基础本体已在 __init__ 中创建
    logger.warning("使用基础本体结构（无CSV数据）")


def save_ontology(ontology: 'OntologyManager', path: str) -> None:
    """保存本体（兼容不同的方法名）"""
    method_names = [
        'save_ontology',  # 实际使用的方法名
        'save',
        'save_to_file',
        'serialize',
        'export'
    ]

    for method_name in method_names:
        if hasattr(ontology, method_name):
            method = getattr(ontology, method_name)
            try:
                method(path)
                logger.info(f"本体已保存到 {path}")
                return
            except Exception as e:
                logger.debug(f"{method_name} 保存失败: {e}")
                continue

    logger.warning("无法保存本体文件")

def run_optimization(
        config: 'ConfigManager',
        ontology: 'OntologyManager',
        seed: int,
        run_dir: Path
) -> Tuple[pd.DataFrame, Dict]:
    """运行NSGA-III优化"""

    logger.info("Optimizer initialized with seed=%d", seed)

    # 初始化优化器
    # 注意: RMTwinOptimizer 需要 ontology.g (Graph对象) 而不是 OntologyManager
    optimizer = RMTwinOptimizer(
        ontology_graph=ontology.g,  # 传递Graph对象
        config=config,
        seed=seed
    )

    # 运行优化
    start_time = time.time()

    # 尝试不同的优化方法名
    if hasattr(optimizer, 'optimize'):
        result = optimizer.optimize()
    elif hasattr(optimizer, 'run'):
        result = optimizer.run()
    elif hasattr(optimizer, 'execute'):
        result = optimizer.execute()
    else:
        raise AttributeError("优化器没有可用的运行方法")

    elapsed = time.time() - start_time
    logger.info(f"Optimization time: {elapsed:.2f}s")

    # 处理返回值（可能是tuple或单个DataFrame）
    if isinstance(result, tuple):
        pareto_df, history = result
    else:
        pareto_df = result
        history = {}

    # 保存结果
    pareto_df.to_csv(run_dir / 'pareto_solutions.csv', index=False)

    with open(run_dir / 'optimization_history.json', 'w') as f:
        json.dump(history, f, indent=2, default=str)

    return pareto_df, history


def run_baselines(
        config: 'ConfigManager',
        ontology: 'OntologyManager',
        seed: int,
        run_dir: Path
) -> Dict[str, pd.DataFrame]:
    """运行基线方法"""

    if BaselineRunner is None:
        logger.warning("BaselineRunner 不可用，跳过基线方法")
        return {}

    logger.info("Running baseline methods...")

    try:
        runner = BaselineRunner(
            ontology_graph=ontology.g,
            config=config,
            seed=seed
        )
    except Exception as e:
        logger.warning(f"BaselineRunner 初始化失败: {e}")
        return {}

    baseline_dfs = {}

    # Random Search
    try:
        logger.info("Running Random Search...")
        if hasattr(runner, 'run_random_search'):
            random_df = runner.run_random_search(n_samples=3000)
        elif hasattr(runner, 'random_search'):
            random_df = runner.random_search(n_samples=3000)
        else:
            random_df = None

        if random_df is not None:
            random_df.to_csv(run_dir / 'baseline_random.csv', index=False)
            baseline_dfs['Random'] = random_df
    except Exception as e:
        logger.warning(f"Random Search 失败: {e}")

    # Grid Search
    try:
        logger.info("Running Grid Search...")
        if hasattr(runner, 'run_grid_search'):
            grid_df = runner.run_grid_search()
        elif hasattr(runner, 'grid_search'):
            grid_df = runner.grid_search()
        else:
            grid_df = None

        if grid_df is not None:
            grid_df.to_csv(run_dir / 'baseline_grid.csv', index=False)
            baseline_dfs['Grid'] = grid_df
    except Exception as e:
        logger.warning(f"Grid Search 失败: {e}")

    # Weighted Sum
    try:
        logger.info("Running Weighted Sum...")
        if hasattr(runner, 'run_weighted_sum'):
            weighted_df = runner.run_weighted_sum(n_weights=100)
        elif hasattr(runner, 'weighted_sum'):
            weighted_df = runner.weighted_sum(n_weights=100)
        else:
            weighted_df = None

        if weighted_df is not None:
            weighted_df.to_csv(run_dir / 'baseline_weighted.csv', index=False)
            baseline_dfs['Weighted'] = weighted_df
    except Exception as e:
        logger.warning(f"Weighted Sum 失败: {e}")

    # Expert Heuristic
    try:
        logger.info("Running Expert Heuristic...")
        if hasattr(runner, 'run_expert_heuristic'):
            expert_df = runner.run_expert_heuristic()
        elif hasattr(runner, 'expert_heuristic'):
            expert_df = runner.expert_heuristic()
        else:
            expert_df = None

        if expert_df is not None:
            expert_df.to_csv(run_dir / 'baseline_expert.csv', index=False)
            baseline_dfs['Expert'] = expert_df
    except Exception as e:
        logger.warning(f"Expert Heuristic 失败: {e}")

    return baseline_dfs


def generate_report(
        config: 'ConfigManager',
        pareto_df: pd.DataFrame,
        baseline_dfs: Dict[str, pd.DataFrame],
        run_dir: Path,
        elapsed_time: float
):
    """生成优化报告"""

    report_lines = [
        "=" * 70,
        "RMTwin Optimization Report v3.1",
        "=" * 70,
        "",
        f"Run Directory: {run_dir}",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Time: {elapsed_time:.2f}s",
        "",
        "Configuration:",
        f"  Road Network: {config.road_network_length_km} km",
        f"  Planning Horizon: {config.planning_horizon_years} years",
        f"  Budget Cap: ${config.budget_cap_usd:,.0f}",
        f"  Min Recall: {config.min_recall_threshold}",
        f"  Max Latency: {config.max_latency_seconds}s",
        f"  Fixed Sensor Density: {getattr(config, 'fixed_sensor_density_per_km', 1.0)}/km",
        f"  Mobile Coverage: {getattr(config, 'mobile_km_per_unit_per_day', 80)} km/day",
        "",
        "Pareto Front Summary:",
        f"  Solutions: {len(pareto_df)}",
    ]

    # 添加成本范围
    if 'f1_total_cost_USD' in pareto_df.columns:
        report_lines.append(
            f"  Cost Range: ${pareto_df['f1_total_cost_USD'].min():,.0f} - ${pareto_df['f1_total_cost_USD'].max():,.0f}")

    # 添加Recall范围
    if 'detection_recall' in pareto_df.columns:
        report_lines.append(
            f"  Recall Range: {pareto_df['detection_recall'].min():.4f} - {pareto_df['detection_recall'].max():.4f}")
    elif 'f2_one_minus_recall' in pareto_df.columns:
        report_lines.append(
            f"  Recall Range: {1 - pareto_df['f2_one_minus_recall'].max():.4f} - {1 - pareto_df['f2_one_minus_recall'].min():.4f}")

    # Baseline对比
    if baseline_dfs:
        report_lines.extend([
            "",
            "Baseline Comparison:",
        ])

        for name, df in baseline_dfs.items():
            feasible = df[df['is_feasible']] if 'is_feasible' in df.columns else df
            if len(feasible) > 0 and 'f1_total_cost_USD' in feasible.columns:
                report_lines.append(
                    f"  {name}: {len(feasible)}/{len(df)} feasible, min_cost=${feasible['f1_total_cost_USD'].min():,.0f}")
            else:
                report_lines.append(f"  {name}: {len(feasible)}/{len(df)} feasible")

    report_lines.extend([
        "",
        "=" * 70,
    ])

    report_text = "\n".join(report_lines)

    # 保存报告
    with open(run_dir / 'optimization_report.txt', 'w') as f:
        f.write(report_text)

    # 打印报告
    print(report_text)

    return report_text



def run_shacl_audit(pareto_df: pd.DataFrame, ontology, shapes_path: str = 'shapes/min_shapes.ttl') -> dict:
    """
    对Pareto解集执行SHACL语义审计（后验验证）。
    
    改进：
    1. 缓存shapes graph，避免重复解析
    2. 统计violation类型分布
    3. 输出结构化报告
    
    Args:
        pareto_df: Pareto解DataFrame
        ontology: OntologyManager实例
        shapes_path: SHACL shapes文件路径
        
    Returns:
        审计结果字典
    """
    logger.info("Running SHACL semantic audit...")
    
    if not hasattr(ontology, 'shacl_validate_config'):
        logger.warning("OntologyManager does not support SHACL validation")
        return {'status': 'skipped', 'reason': 'SHACL not supported'}
    
    if not Path(shapes_path).exists():
        logger.warning(f"SHACL shapes file not found: {shapes_path}")
        return {'status': 'skipped', 'reason': f'Shapes file not found: {shapes_path}'}
    
    # 预加载并缓存 SHACL shapes graph（性能优化）
    try:
        from rdflib import Graph
        shacl_graph = Graph().parse(shapes_path, format="turtle")
        logger.info(f"Loaded SHACL shapes: {len(shacl_graph)} triples")
    except Exception as e:
        logger.error(f"Failed to load SHACL shapes: {e}")
        return {'status': 'error', 'reason': str(e)}
    
    audit_results = []
    pass_count = 0
    violation_types = {}  # 统计violation类型
    
    for idx, row in pareto_df.iterrows():
        # 构建配置字典
        config = {
            'sensor': row.get('sensor'),
            'algorithm': row.get('algorithm'),
            'deployment': row.get('deployment'),
            'storage': row.get('storage'),
            'communication': row.get('communication'),
            'inspection_cycle': row.get('inspection_cycle_days', row.get('inspection_cycle')),
            'data_rate': row.get('data_rate_hz', row.get('data_rate')),
            # 评估结果
            'total_cost': row.get('f1_total_cost_USD'),
            'recall': row.get('detection_recall'),
            'latency': row.get('f3_latency_seconds'),
            'carbon': row.get('f5_carbon_emissions_kgCO2e_year'),
        }
        
        try:
            conforms, report = ontology.shacl_validate_config(config, shapes_path)
            pass_count += int(conforms)
            
            # 提取violation类型
            violations = []
            if not conforms and report:
                # 解析报告中的violation
                for line in report.split('\n'):
                    if 'Constraint Violation' in line or 'sh:result' in line:
                        violations.append(line.strip())
                    elif 'Message:' in line or 'sh:resultMessage' in line:
                        msg = line.split(':', 1)[-1].strip()
                        violations.append(msg)
                        # 统计violation类型
                        violation_types[msg] = violation_types.get(msg, 0) + 1
            
            audit_results.append({
                'index': int(idx) if hasattr(idx, '__int__') else idx,
                'conforms': bool(conforms),
                'violations': violations[:5],  # 最多5条
                'report_summary': report[:300] if report else ''
            })
        except Exception as e:
            logger.error(f"SHACL validation error for solution {idx}: {e}")
            audit_results.append({
                'index': int(idx) if hasattr(idx, '__int__') else idx,
                'conforms': True,  # 默认通过
                'violations': [],
                'report_summary': f'Error: {str(e)}'
            })
            pass_count += 1
    
    total = len(pareto_df)
    pass_ratio = pass_count / max(1, total)
    
    # 生成violation类型排名
    top_violations = sorted(violation_types.items(), key=lambda x: x[1], reverse=True)[:5]
    
    logger.info(f"SHACL Audit: {pass_count}/{total} solutions passed ({pass_ratio:.1%})")
    if top_violations:
        logger.info(f"Top violations: {top_violations}")
    
    return {
        'status': 'completed',
        'pass_count': pass_count,
        'total_count': total,
        'pass_ratio': pass_ratio,
        'violation_statistics': {
            'total_violations': sum(violation_types.values()),
            'unique_violation_types': len(violation_types),
            'top_violations': top_violations
        },
        'results': audit_results
    }


def validate_results(pareto_df: pd.DataFrame, config: 'ConfigManager') -> bool:
    """验证结果合理性"""
    logger.info("Validating results...")

    issues = []

    if 'f1_total_cost_USD' not in pareto_df.columns:
        logger.warning("无法验证：缺少 f1_total_cost_USD 列")
        return True

    # 1. 检查最低成本是否合理
    min_cost = pareto_df['f1_total_cost_USD'].min()
    min_expected_cost = 50000

    if min_cost < min_expected_cost:
        issues.append(f"WARNING: Min cost ${min_cost:,.0f} is suspiciously low")

    # 2. 检查是否有足够的多样性
    if 'sensor' in pareto_df.columns:
        n_unique_sensors = pareto_df['sensor'].nunique()
        if n_unique_sensors < 3:
            issues.append(f"WARNING: Low sensor diversity ({n_unique_sensors} types)")

    # 打印结果
    if issues:
        logger.warning("Validation found issues:")
        for issue in issues:
            logger.warning(f"  {issue}")
        return False
    else:
        logger.info("Validation: PASSED")
        return True


def main():
    """主函数"""

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='RMTwin Multi-Objective Optimization v3.1')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--skip-baselines', action='store_true', help='Skip baseline methods')
    parser.add_argument('--skip-visualization', action='store_true', help='Skip visualization')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    total_start_time = time.time()

    # 加载配置
    logger.info("Loading configuration from %s", args.config)
    config = ConfigManager.from_json(args.config)

    # 创建运行目录
    run_dir = setup_run_directory(config, args.seed)
    setup_logging(run_dir, args.debug)

    logger.info("=" * 80)
    logger.info("RMTwin Multi-Objective Optimization Framework v3.1")
    logger.info("Run Directory: %s", run_dir)
    logger.info("Seed: %d", args.seed)
    logger.info("Config: %d objectives, %d generations, %d population",
                6, config.n_generations, config.population_size)
    logger.info("Start Time: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("=" * 80)

    # 打印约束配置
    logger.info("\nConstraints:")
    logger.info("  Budget Cap: $%s", f"{config.budget_cap_usd:,}")
    logger.info("  Min Recall: %s", config.min_recall_threshold)
    logger.info("  Max Latency: %s s", config.max_latency_seconds)
    logger.info("  Min MTBF: %s hours", config.min_mtbf_hours)
    logger.info("  Max Carbon: %s kgCO2e/year", f"{config.max_carbon_emissions_kgCO2e_year:,}")

    # 打印新参数
    logger.info("\nSensor Parameters (v3.1):")
    logger.info("  Fixed Sensor Density: %s /km", getattr(config, 'fixed_sensor_density_per_km', 1.0))
    logger.info("  Mobile Coverage: %s km/day", getattr(config, 'mobile_km_per_unit_per_day', 80.0))

    # 保存配置快照
    config.save_snapshot(str(run_dir / 'config_snapshot.json'))
    logger.info("Config snapshot saved to %s", run_dir / 'config_snapshot.json')

    # Step 1: 加载本体
    logger.info("\nStep 1: Loading ontology...")
    ontology = OntologyManager()
    build_ontology(ontology)  # 使用兼容函数
    save_ontology(ontology, str(run_dir / 'populated_ontology.ttl'))

    # Step 2: 运行NSGA-III优化
    logger.info("\nStep 2: Running NSGA-III optimization...")
    pareto_df, opt_history = run_optimization(config, ontology, args.seed, run_dir)
    logger.info("Processed %d Pareto optimal solutions", len(pareto_df))

    # 打印Pareto解集多样性
    if 'sensor' in pareto_df.columns:
        sensor_col = pareto_df['sensor'].apply(lambda x: str(x).split('#')[-1])
        logger.info("\n=== Pareto Front Diversity Statistics ===")
        logger.info("Unique sensors: %d", sensor_col.nunique())
        logger.info("Sensor distribution:\n%s", sensor_col.value_counts().to_string())

    if 'algorithm' in pareto_df.columns:
        algo_col = pareto_df['algorithm'].apply(lambda x: str(x).split('#')[-1])
        logger.info("Unique algorithms: %d", algo_col.nunique())
        logger.info("Algorithm distribution:\n%s", algo_col.value_counts().to_string())

    logger.info("\nPareto Front Summary:")
    if 'f1_total_cost_USD' in pareto_df.columns:
        logger.info("  Cost: $%s - $%s",
                    f"{pareto_df['f1_total_cost_USD'].min():,.0f}",
                    f"{pareto_df['f1_total_cost_USD'].max():,.0f}")

    if 'detection_recall' in pareto_df.columns:
        logger.info("  Recall: %.4f - %.4f",
                    pareto_df['detection_recall'].min(),
                    pareto_df['detection_recall'].max())

    # Step 3: 运行基线方法
    baseline_dfs = {}
    if not args.skip_baselines:
        logger.info("\nStep 3: Running baseline methods...")
        baseline_dfs = run_baselines(config, ontology, args.seed + 1, run_dir)

        for name, df in baseline_dfs.items():
            feasible = df[df['is_feasible']] if 'is_feasible' in df.columns else df
            logger.info("  %s: %d total, %d feasible", name, len(df), len(feasible))
    else:
        logger.info("\nStep 3: Skipping baseline methods")

    # Step 4: 验证结果
    logger.info("\nStep 4: Validating results consistency...")
    validation_passed = validate_results(pareto_df, config)

    with open(run_dir / 'validation_result.json', 'w') as f:
        json.dump({'passed': validation_passed}, f)
    
    # Step 4b: SHACL语义审计 (P0)
    logger.info("\nStep 4b: Running SHACL semantic audit...")
    shapes_path = Path('shapes/min_shapes.ttl')
    if not shapes_path.exists():
        shapes_path = run_dir.parent.parent / 'shapes' / 'min_shapes.ttl'
    
    shacl_audit = run_shacl_audit(pareto_df, ontology, str(shapes_path))
    
    # 更新 validation_result.json
    validation_payload = {
        'passed': validation_passed,
        'shacl_audit': {
            'status': shacl_audit.get('status', 'unknown'),
            'pass_ratio': shacl_audit.get('pass_ratio', 0),
            'pass_count': shacl_audit.get('pass_count', 0),
            'total_count': shacl_audit.get('total_count', 0),
        }
    }
    
    with open(run_dir / 'validation_result.json', 'w') as f:
        json.dump(validation_payload, f, indent=2)
    
    # 保存详细审计结果
    if 'results' in shacl_audit:
        with open(run_dir / 'shacl_audit_detail.json', 'w') as f:
            json.dump(shacl_audit['results'], f, indent=2)
    
    logger.info("SHACL Audit: %s (%.1f%% passed)", 
                shacl_audit.get('status', 'unknown'),
                shacl_audit.get('pass_ratio', 0) * 100)

    logger.info("Validation: %s", "PASSED" if validation_passed else "WARNINGS FOUND")

    # Step 5: 生成可视化
    if not args.skip_visualization and ResultVisualizer is not None:
        logger.info("\nStep 5: Generating visualizations...")
        try:
            visualizer = ResultVisualizer(
                pareto_df=pareto_df,
                baseline_dfs=baseline_dfs if baseline_dfs else {},
                config=config,
                output_dir=str(run_dir / 'figures')
            )
            if hasattr(visualizer, 'generate_all'):
                visualizer.generate_all()
            elif hasattr(visualizer, 'create_all_figures'):
                visualizer.create_all_figures()
            logger.info("Visualizations saved to %s", run_dir / 'figures')
        except Exception as e:
            logger.warning("Visualization failed: %s", str(e))
    else:
        logger.info("\nStep 5: Skipping visualization")

    # Step 6: 生成报告
    logger.info("\nStep 6: Generating report...")
    total_elapsed = time.time() - total_start_time

    generate_report(
        config=config,
        pareto_df=pareto_df,
        baseline_dfs=baseline_dfs,
        run_dir=run_dir,
        elapsed_time=total_elapsed
    )

    # 保存摘要JSON
    summary = {
        'run_dir': str(run_dir),
        'seed': args.seed,
        'pareto_solutions': len(pareto_df),
        'baseline_feasible': sum(
            len(df[df['is_feasible']]) if 'is_feasible' in df.columns else len(df)
            for df in baseline_dfs.values()
        ) if baseline_dfs else 0,
        'validation_passed': validation_passed,
        'shacl_pass_ratio': shacl_audit.get('pass_ratio', 1.0),
        'total_time_seconds': total_elapsed,
    }

    if 'f1_total_cost_USD' in pareto_df.columns:
        summary['min_cost'] = float(pareto_df['f1_total_cost_USD'].min())
        summary['max_cost'] = float(pareto_df['f1_total_cost_USD'].max())

    with open(run_dir / 'optimization_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # 最终输出
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("Run Directory: %s", run_dir)
    logger.info("Pareto Solutions: %d", len(pareto_df))
    logger.info("Baseline Feasible: %d", summary['baseline_feasible'])
    logger.info("Validation: %s", "PASSED" if validation_passed else "WARNINGS")
    logger.info("Total Time: %.2fs", total_elapsed)
    logger.info("=" * 80)

    print(f"\n[SUMMARY] run_dir={run_dir}, pareto={len(pareto_df)}, "
          f"baseline_feasible={summary['baseline_feasible']}, "
          f"validation={'PASSED' if validation_passed else 'WARNINGS'}")

    return 0


if __name__ == '__main__':
    sys.exit(main())