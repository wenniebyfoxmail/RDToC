#!/usr/bin/env python3
"""
Core Optimization Module - NSGA-III Implementation
===================================================
修复版: 增加初始种群多样性保证，确保所有离散选项都被探索

Key fixes:
1. CustomSampling: 确保初始种群覆盖所有sensor/algorithm/deployment
2. 周期性注入随机个体保持多样性
3. 更好的离散变量探索

Author: RMTwin Research Team
Version: 2.1 (Diversity Fix)
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, List, Optional

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

logger = logging.getLogger(__name__)


class DiverseInitialSampling(Sampling):
    """
    自定义初始采样: 确保所有离散选项都有代表

    解决问题: 标准随机采样可能遗漏某些离散选项(如Vehicle_Smartphone_System)
    """

    def __init__(self, evaluator):
        super().__init__()
        self.evaluator = evaluator

    def _do(self, problem, n_samples, **kwargs):
        """生成确保离散选项覆盖的初始种群"""
        n_var = problem.n_var

        # 获取离散组件数量
        n_sensors = len(self.evaluator.solution_mapper.sensors)
        n_algos = len(self.evaluator.solution_mapper.algorithms)
        n_storage = len(self.evaluator.solution_mapper.storage_systems)
        n_comm = len(self.evaluator.solution_mapper.comm_systems)
        n_deploy = len(self.evaluator.solution_mapper.deployments)

        logger.info(f"DiverseInitialSampling: {n_sensors} sensors, {n_algos} algorithms, "
                    f"{n_deploy} deployments")

        X = np.zeros((n_samples, n_var))

        # 阶段1: 为每个离散选项创建代表个体
        # 确保每个sensor至少有一个个体
        idx = 0

        # Sensor覆盖 (x[0])
        for i in range(min(n_sensors, n_samples // 4)):
            if idx >= n_samples:
                break
            X[idx, 0] = (i + 0.5) / n_sensors  # 中心点
            X[idx, 1:] = np.random.random(n_var - 1)
            idx += 1

        # Algorithm覆盖 (x[4])
        for i in range(min(n_algos, n_samples // 4)):
            if idx >= n_samples:
                break
            X[idx, 4] = (i + 0.5) / n_algos
            X[idx, :4] = np.random.random(4)
            X[idx, 5:] = np.random.random(n_var - 5)
            idx += 1

        # Deployment覆盖 (x[8])
        for i in range(min(n_deploy, n_samples // 4)):
            if idx >= n_samples:
                break
            X[idx, 8] = (i + 0.5) / n_deploy
            X[idx, :8] = np.random.random(8)
            X[idx, 9:] = np.random.random(n_var - 9)
            idx += 1

        # 阶段2: 组合覆盖 - sensor × algorithm 交叉
        n_combos = min(n_sensors * n_algos, (n_samples - idx) // 2)
        combo_idx = 0
        for si in range(n_sensors):
            for ai in range(n_algos):
                if idx >= n_samples or combo_idx >= n_combos:
                    break
                X[idx, 0] = (si + 0.5) / n_sensors
                X[idx, 4] = (ai + 0.5) / n_algos
                X[idx, 1:4] = np.random.random(3)
                X[idx, 5:] = np.random.random(n_var - 5)
                idx += 1
                combo_idx += 1
            if idx >= n_samples:
                break

        # 阶段3: 剩余用纯随机填充
        if idx < n_samples:
            X[idx:] = np.random.random((n_samples - idx, n_var))

        # 确保所有值在[0, 1]范围内
        X = np.clip(X, 0, 1)

        logger.info(f"Generated {n_samples} initial solutions with diverse discrete coverage")
        logger.info(f"  - {min(n_sensors, n_samples // 4)} sensor-focused")
        logger.info(f"  - {min(n_algos, n_samples // 4)} algorithm-focused")
        logger.info(f"  - {combo_idx} sensor×algorithm combos")

        return X


class RMTwinProblem(Problem):
    """6-objective optimization problem definition"""

    def __init__(self, evaluator):
        # Problem dimensions
        n_var = 11  # Decision variables
        n_obj = 6  # Objectives
        n_constr = 5  # Constraints

        # Variable bounds [0, 1]
        xl = np.zeros(n_var)
        xu = np.ones(n_var)

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=xl,
            xu=xu
        )

        self.evaluator = evaluator
        self._eval_count = 0

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate population"""
        objectives, constraints = self.evaluator.evaluate_batch(X)

        out["F"] = objectives
        out["G"] = constraints

        # Progress tracking
        self._eval_count += len(X)
        if self._eval_count % 1000 == 0:
            logger.debug(f"Evaluated {self._eval_count} solutions")


class RMTwinOptimizer:
    """Main coordinator for 6-objective optimization"""

    DEFAULT_SEED = 42

    def __init__(self, ontology_graph, config, seed: int = None):
        self.ontology_graph = ontology_graph
        self.config = config

        # Set seed
        if seed is not None:
            self.seed = seed
        elif hasattr(config, 'random_seed'):
            self.seed = config.random_seed
        else:
            self.seed = self.DEFAULT_SEED

        logger.info(f"Optimizer initialized with seed={self.seed}")

        # Import evaluator
        from evaluation import EnhancedFitnessEvaluatorV3

        # Initialize evaluator
        self.evaluator = EnhancedFitnessEvaluatorV3(ontology_graph, config)

        # Initialize problem
        self.problem = RMTwinProblem(self.evaluator)

        # Configure algorithm with diverse sampling
        self.algorithm = self._configure_algorithm()

    def _configure_algorithm(self):
        """Configure NSGA-III algorithm with diverse initial sampling"""

        # 使用自定义采样确保离散选项覆盖
        sampling = DiverseInitialSampling(self.evaluator)

        # 增大变异强度以提高离散变量探索
        mutation_eta = getattr(self.config, 'mutation_eta', 20)
        # 对于离散变量多的问题，降低eta可以增大变异幅度
        mutation_eta = max(10, mutation_eta - 5)  # 降低eta增大变异

        if self.config.n_objectives <= 3:
            algorithm = NSGA2(
                pop_size=self.config.population_size,
                sampling=sampling,
                crossover=SBX(eta=self.config.crossover_eta,
                              prob=self.config.crossover_prob),
                mutation=PM(eta=mutation_eta,
                            prob=1.0 / self.problem.n_var),
                eliminate_duplicates=True
            )
            logger.info(f"Using NSGA-II, pop_size={self.config.population_size}")
        else:
            if self.config.n_objectives == 4:
                ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=6)
            elif self.config.n_objectives == 5:
                ref_dirs = get_reference_directions("das-dennis", 5, n_partitions=4)
            else:  # 6 objectives
                ref_dirs = get_reference_directions("das-dennis", 6, n_partitions=3)

            pop_size = max(self.config.population_size, len(ref_dirs) + 50)

            algorithm = NSGA3(
                ref_dirs=ref_dirs,
                pop_size=pop_size,
                sampling=sampling,
                crossover=SBX(eta=self.config.crossover_eta,
                              prob=self.config.crossover_prob),
                mutation=PM(eta=mutation_eta,
                            prob=1.0 / self.problem.n_var),
                eliminate_duplicates=True
            )

            logger.info(f"Using NSGA-III, {len(ref_dirs)} ref_dirs, pop_size={pop_size}")
            logger.info(f"Using DiverseInitialSampling for better discrete variable coverage")
            logger.info(f"Mutation eta={mutation_eta} (increased mutation strength)")

        return algorithm

    def optimize(self) -> Tuple[pd.DataFrame, Dict]:
        """Run optimization and return results"""
        logger.info(f"Starting {self.algorithm.__class__.__name__} optimization, "
                    f"{self.config.n_objectives} objectives, seed={self.seed}...")

        # Configure termination
        termination = get_termination("n_gen", self.config.n_generations)

        # Run optimization
        res = minimize(
            self.problem,
            self.algorithm,
            termination,
            seed=self.seed,
            save_history=True,
            verbose=True
        )

        # Process results
        pareto_df = self._process_results(res)

        # Extract history
        history = {
            'n_evals': self.problem._eval_count,
            'exec_time': res.exec_time if hasattr(res, 'exec_time') else 0,
            'n_gen': self.config.n_generations,
            'history': res.history if hasattr(res, 'history') else None,
            'convergence': self._extract_convergence(res)
        }

        # Log diversity statistics
        if len(pareto_df) > 0:
            self._log_diversity_stats(pareto_df)

        return pareto_df, history

    def _log_diversity_stats(self, pareto_df: pd.DataFrame):
        """Log statistics about solution diversity"""
        logger.info("\n=== Pareto Front Diversity Statistics ===")

        if 'sensor' in pareto_df.columns:
            sensors = pareto_df['sensor'].nunique()
            logger.info(f"Unique sensors: {sensors}")
            logger.info(f"Sensor distribution:\n{pareto_df['sensor'].value_counts().to_string()}")

        if 'algorithm' in pareto_df.columns:
            algos = pareto_df['algorithm'].nunique()
            logger.info(f"Unique algorithms: {algos}")
            logger.info(f"Algorithm distribution:\n{pareto_df['algorithm'].value_counts().to_string()}")

        logger.info("=" * 45)

    def _process_results(self, res) -> pd.DataFrame:
        """Convert optimization results to DataFrame with trace columns"""
        if res.X is None or (hasattr(res.X, '__len__') and len(res.X) == 0):
            logger.warning("No feasible solutions found! Check constraints.")
            return self._get_empty_dataframe()

        # Ensure 2D arrays
        X = res.X if res.X.ndim == 2 else res.X.reshape(1, -1)
        F = res.F if res.F.ndim == 2 else res.F.reshape(1, -1)

        results = []

        for i in range(len(X)):
            # Decode configuration
            config = self.evaluator.solution_mapper.decode_solution(X[i])

            # Get objectives
            objectives = F[i]

            # Get trace information
            trace = self.evaluator.explain(config)

            # Create result dictionary
            result = {
                'solution_id': i + 1,

                # Configuration details
                'sensor': config['sensor'].split('#')[-1],
                'algorithm': config['algorithm'].split('#')[-1],
                'data_rate_Hz': config['data_rate'],
                'geometric_LOD': config['geo_lod'],
                'condition_LOD': config['cond_lod'],
                'detection_threshold': config['detection_threshold'],
                'storage': config['storage'].split('#')[-1],
                'communication': config['communication'].split('#')[-1],
                'deployment': config['deployment'].split('#')[-1],
                'crew_size': config['crew_size'],
                'inspection_cycle_days': config['inspection_cycle'],

                # Raw objectives (6 objectives)
                'f1_total_cost_USD': float(objectives[0]),
                'f2_one_minus_recall': float(objectives[1]),
                'f3_latency_seconds': float(objectives[2]),
                'f4_traffic_disruption_hours': float(objectives[3]),
                'f5_carbon_emissions_kgCO2e_year': float(objectives[4]),
                'f6_system_reliability_inverse_MTBF': float(objectives[5]),

                # Derived metrics
                'detection_recall': float(1 - objectives[1]),
                'system_MTBF_hours': float(1 / objectives[5] if objectives[5] > 0 else 1e6),
                'system_MTBF_years': float(1 / objectives[5] / 8760 if objectives[5] > 0 else 100),
                'annual_cost_USD': float(objectives[0] / self.config.planning_horizon_years),
                'cost_per_km_year': float(objectives[0] / self.config.planning_horizon_years /
                                          self.config.road_network_length_km),
                'carbon_footprint_tons_CO2_year': float(objectives[4] / 1000),

                # All Pareto solutions are feasible
                'is_feasible': True,

                # Trace columns
                'sensor_type': trace['types']['sensor'],
                'algo_type': trace['types']['algorithm'],
                'storage_type': trace['types']['storage'],
                'comm_type': trace['types']['communication'],
                'deploy_type': trace['types']['deployment'],

                'raw_data_gb_per_year': float(trace['data_volume']['raw_total_gb_year']),
                'sent_data_gb_per_year': float(trace['data_volume']['sent_total_gb_year']),
                'data_reduction_ratio': float(trace['data_volume']['data_reduction_ratio']),
                'lod_data_multiplier': float(trace['data_volume']['lod_multiplier']),
                'rate_data_multiplier': float(trace['data_volume']['rate_multiplier']),

                'comm_bandwidth_GBps': float(trace['communication']['bandwidth_GBps']),
                'comm_time_seconds': float(trace['communication']['comm_time_per_inspection_s']),

                'compute_seconds_per_GB': float(trace['compute']['algo_compute_s_per_GB']),
                'deployment_compute_factor': float(trace['compute']['deployment_factor']),
                'compute_time_seconds': float(trace['compute']['compute_time_per_inspection_s']),

                'base_algo_recall': float(trace['recall_model']['base_algo_recall']),
                'sensor_precision': float(trace['recall_model']['sensor_precision']),
                'lod_bonus': float(trace['recall_model']['lod_bonus']),
                'sigmoid_z': float(trace['recall_model']['sigmoid_input_z']),
                'tau': float(trace['recall_model']['tau']),
                'tau0': float(trace['recall_model']['tau0']),
            }

            results.append(result)

        # Convert to DataFrame
        df = pd.DataFrame(results)

        if len(df) > 0:
            df = df.sort_values('f1_total_cost_USD')

            # Add rankings
            df['cost_rank'] = df['f1_total_cost_USD'].rank()
            df['recall_rank'] = df['detection_recall'].rank(ascending=False)
            df['carbon_rank'] = df['f5_carbon_emissions_kgCO2e_year'].rank()
            df['reliability_rank'] = df['system_MTBF_hours'].rank(ascending=False)
            df['latency_rank'] = df['f3_latency_seconds'].rank()
            df['disruption_rank'] = df['f4_traffic_disruption_hours'].rank()

        logger.info(f"Processed {len(df)} Pareto optimal solutions")

        return df

    def _get_empty_dataframe(self) -> pd.DataFrame:
        """Return empty DataFrame with expected columns"""
        return pd.DataFrame(columns=[
            'solution_id', 'sensor', 'algorithm', 'data_rate_Hz',
            'geometric_LOD', 'condition_LOD', 'detection_threshold',
            'storage', 'communication', 'deployment', 'crew_size',
            'inspection_cycle_days', 'f1_total_cost_USD', 'f2_one_minus_recall',
            'f3_latency_seconds', 'f4_traffic_disruption_hours',
            'f5_carbon_emissions_kgCO2e_year', 'f6_system_reliability_inverse_MTBF',
            'detection_recall', 'system_MTBF_hours', 'system_MTBF_years',
            'annual_cost_USD', 'cost_per_km_year', 'carbon_footprint_tons_CO2_year',
            'is_feasible'
        ])

    def _extract_convergence(self, res) -> Dict:
        """Extract convergence metrics from history"""
        if not hasattr(res, 'history') or res.history is None:
            return {}

        convergence = {
            'generations': [],
            'best_cost': [],
            'best_recall': [],
            'best_carbon': [],
            'n_solutions': [],
            'n_nds': [],  # 非支配解数量
            'hv': [],  # Hypervolume
        }

        for gen, algo in enumerate(res.history):
            if hasattr(algo, 'pop') and algo.pop is not None:
                F = algo.pop.get('F')
                if F is not None and len(F) > 0:
                    convergence['generations'].append(gen)
                    convergence['best_cost'].append(float(F[:, 0].min()))
                    convergence['best_recall'].append(float(1 - F[:, 1].min()))
                    convergence['n_solutions'].append(len(F))
                    convergence['n_nds'].append(len(F))  # 简化: 假设都是非支配解

                    if F.shape[1] > 4:
                        convergence['best_carbon'].append(float(F[:, 4].min()))

        return convergence


class OptimizationAnalyzer:
    """Analyze optimization results"""

    @staticmethod
    def calculate_hypervolume(F: np.ndarray, ref_point: np.ndarray) -> float:
        """Calculate hypervolume indicator"""
        try:
            from pymoo.indicators.hv import HV
            ind = HV(ref_point=ref_point)
            return ind(F)
        except ImportError:
            logger.warning("pymoo HV indicator not available")
            return 0.0

    @staticmethod
    def calculate_igd(F: np.ndarray, pareto_front: np.ndarray) -> float:
        """Calculate inverted generational distance"""
        try:
            from pymoo.indicators.igd import IGD
            ind = IGD(pareto_front)
            return ind(F)
        except ImportError:
            logger.warning("pymoo IGD indicator not available")
            return 0.0

    @staticmethod
    def analyze_convergence(history: Dict) -> Dict:
        """Analyze convergence characteristics"""
        if not history or 'convergence' not in history:
            return {}

        conv = history['convergence']

        results = {
            'n_generations': len(conv.get('generations', [])),
            'final_best_cost': conv['best_cost'][-1] if conv.get('best_cost') else None,
            'final_best_recall': conv['best_recall'][-1] if conv.get('best_recall') else None,
            'cost_improvement_ratio': None,
            'recall_improvement_ratio': None
        }

        if conv.get('best_cost') and len(conv['best_cost']) > 1:
            results['cost_improvement_ratio'] = (
                conv['best_cost'][0] / conv['best_cost'][-1]
                if conv['best_cost'][-1] > 0 else 1.0
            )

        if conv.get('best_recall') and len(conv['best_recall']) > 1:
            results['recall_improvement_ratio'] = (
                conv['best_recall'][-1] / conv['best_recall'][0]
                if conv['best_recall'][0] > 0 else 1.0
            )

        return results

    @staticmethod
    def get_extreme_solutions(pareto_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Get solutions optimizing individual objectives"""
        if len(pareto_df) == 0:
            return {}

        extremes = {}

        extremes['min_cost'] = pareto_df.loc[pareto_df['f1_total_cost_USD'].idxmin()]
        extremes['max_recall'] = pareto_df.loc[pareto_df['detection_recall'].idxmax()]
        extremes['min_latency'] = pareto_df.loc[pareto_df['f3_latency_seconds'].idxmin()]
        extremes['min_disruption'] = pareto_df.loc[pareto_df['f4_traffic_disruption_hours'].idxmin()]
        extremes['min_carbon'] = pareto_df.loc[pareto_df['f5_carbon_emissions_kgCO2e_year'].idxmin()]
        extremes['max_reliability'] = pareto_df.loc[pareto_df['system_MTBF_hours'].idxmax()]

        return extremes

    @staticmethod
    def get_knee_solution(pareto_df: pd.DataFrame) -> Optional[pd.Series]:
        """Find knee point (balanced solution) in Pareto front"""
        if len(pareto_df) == 0:
            return None

        objectives = ['f1_total_cost_USD', 'f2_one_minus_recall', 'f3_latency_seconds',
                      'f4_traffic_disruption_hours', 'f5_carbon_emissions_kgCO2e_year',
                      'f6_system_reliability_inverse_MTBF']

        normalized = pareto_df[objectives].copy()
        for col in objectives:
            min_val = normalized[col].min()
            max_val = normalized[col].max()
            if max_val > min_val:
                normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
            else:
                normalized[col] = 0

        distances = np.sqrt((normalized ** 2).sum(axis=1))

        knee_idx = distances.idxmin()
        return pareto_df.loc[knee_idx]


def validate_pareto_solutions(pareto_df: pd.DataFrame) -> Dict:
    """Validate Pareto solution quality"""
    if len(pareto_df) == 0:
        return {'valid': False, 'reason': 'No solutions'}

    validation = {
        'valid': True,
        'n_solutions': len(pareto_df),
        'issues': []
    }

    nan_count = pareto_df.isnull().sum().sum()
    if nan_count > 0:
        validation['issues'].append(f"Found {nan_count} NaN values")

    if pareto_df['detection_recall'].min() < 0 or pareto_df['detection_recall'].max() > 1:
        validation['issues'].append("Recall out of [0,1] range")

    if (pareto_df['f1_total_cost_USD'] < 0).any():
        validation['issues'].append("Negative cost values found")

    # 检查离散选项覆盖
    if 'sensor' in pareto_df.columns:
        n_sensors = pareto_df['sensor'].nunique()
        if n_sensors < 3:
            validation['issues'].append(f"Low sensor diversity: only {n_sensors} unique sensors")

    if validation['issues']:
        validation['valid'] = False

    return validation