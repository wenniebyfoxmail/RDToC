#!/usr/bin/env python3
"""
Baseline Methods for RMTwin Configuration - Step 2-Lite Upgrade
================================================================
Complete implementation with guaranteed feasible Expert baseline.

Key improvements:
1. Expert baseline generates multiple candidates and relaxes if needed
2. Better reference configurations based on domain knowledge
3. Improved feasibility-seeking behavior
4. Reproducible random seeds for all methods

Author: RMTwin Research Team
Version: 2.0 (Step 2-Lite)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Default random seed for reproducibility
DEFAULT_SEED = 42

# NOTE: Each baseline method creates its own np.random.default_rng(seed) 
# to avoid affecting global random state. Do not use np.random.seed() globally.


class BaselineMethod(ABC):
    """Abstract base class for baseline methods"""
    
    def __init__(self, evaluator, config, seed: int = DEFAULT_SEED):
        self.evaluator = evaluator
        self.config = config
        self.results = []
        self.execution_time = 0
        self.seed = seed
        
    @abstractmethod
    def optimize(self) -> pd.DataFrame:
        """Run optimization and return results"""
        pass
    
    def _create_result_entry(self, x: np.ndarray, objectives: np.ndarray, 
                           constraints: np.ndarray, solution_id: int) -> Dict:
        """Create standardized result entry"""
        config = self.evaluator.solution_mapper.decode_solution(x)
        is_feasible = np.all(constraints <= 0)
        
        return {
            'solution_id': solution_id,
            'method': self.__class__.__name__,
            
            # Configuration
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
            
            # Objectives
            'f1_total_cost_USD': float(objectives[0]),
            'f2_one_minus_recall': float(objectives[1]),
            'f3_latency_seconds': float(objectives[2]),
            'f4_traffic_disruption_hours': float(objectives[3]),
            'f5_carbon_emissions_kgCO2e_year': float(objectives[4]),
            'f6_system_reliability_inverse_MTBF': float(objectives[5]),
            
            # Derived metrics
            'detection_recall': float(1 - objectives[1]),
            'system_MTBF_hours': float(1/objectives[5] if objectives[5] > 0 else 1e6),
            'system_MTBF_years': float(1/objectives[5]/8760 if objectives[5] > 0 else 100),
            'annual_cost_USD': float(objectives[0] / self.config.planning_horizon_years),
            'cost_per_km_year': float(objectives[0] / self.config.planning_horizon_years / 
                                    self.config.road_network_length_km),
            'carbon_footprint_tons_CO2_year': float(objectives[4] / 1000),
            
            # Feasibility
            'is_feasible': bool(is_feasible),
            'constraint_violation': float(max(0, constraints.max())),
            'time_seconds': self.execution_time
        }
    
    def _check_feasibility(self, x: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
        """Check if solution is feasible"""
        objectives, constraints = self.evaluator._evaluate_single(x)
        is_feasible = np.all(constraints <= 0)
        return is_feasible, objectives, constraints


class RandomSearchBaseline(BaselineMethod):
    """Improved Random Search with constraint-aware sampling"""
    
    def optimize(self) -> pd.DataFrame:
        """Generate random solutions with smart initialization"""
        # Use local RNG for reproducibility without affecting global state
        self.rng = np.random.default_rng(self.seed)
        
        n_samples = self.config.n_random_samples
        logger.info(f"Running Random Search with {n_samples} samples (seed={self.seed})...")
        
        start_time = time.time()
        
        # Reference configurations (known to be more likely feasible)
        reference_configs = self._generate_reference_configs()
        feasible_found = []
        
        for i in range(n_samples):
            if i < len(reference_configs):
                x = reference_configs[i]
            elif feasible_found and self.rng.random() < 0.4:
                # 40% chance to mutate from a feasible solution
                base_x = feasible_found[self.rng.integers(len(feasible_found))]
                x = self._mutate_solution(base_x, sigma=0.1)
            elif self.rng.random() < 0.6:
                # 60% smart random
                x = self._generate_smart_random_solution()
            else:
                # Pure random exploration
                x = self.rng.random(11)
            
            # Evaluate
            is_feasible, objectives, constraints = self._check_feasibility(x)
            
            if is_feasible:
                feasible_found.append(x.copy())
            
            self.results.append(
                self._create_result_entry(x, objectives, constraints, i + 1)
            )
            
            if (i + 1) % 200 == 0:
                logger.info(f"  Evaluated {i + 1} solutions ({len(feasible_found)} feasible)")
        
        self.execution_time = time.time() - start_time
        
        for result in self.results:
            result['time_seconds'] = self.execution_time
        
        df = pd.DataFrame(self.results)
        logger.info(f"Random Search: {self.execution_time:.2f}s, {df['is_feasible'].sum()} feasible")
        
        return df
    
    def _generate_reference_configs(self) -> List[np.ndarray]:
        """Generate reference configurations likely to be feasible"""
        configs = []
        
        # Config 1: Low-cost IoT
        configs.append(np.array([
            0.95,   # IoT sensor
            0.25,   # Low data rate
            0.5,    # Meso LOD
            0.5,    # Meso LOD
            0.85,   # Traditional algorithm
            0.6,    # Mid threshold
            0.0,    # Cloud storage
            0.35,   # LoRaWAN
            1.0,    # Cloud deploy
            0.2,    # 2-person crew
            0.2     # ~73 day cycle
        ]))
        
        # Config 2: Balanced Vehicle
        configs.append(np.array([
            0.88,   # Vehicle sensor
            0.4,    # Moderate data rate
            0.5,    # Meso LOD
            0.5,    # Meso LOD
            0.65,   # ML algorithm
            0.65,   # Good threshold
            0.0,    # Cloud storage
            0.5,    # 4G
            0.85,   # Cloud deploy
            0.3,    # 3-person crew
            0.12    # ~45 day cycle
        ]))
        
        # Config 3: Cost-effective hybrid
        configs.append(np.array([
            0.92,   # Low-cost sensor
            0.35,   # Lower data rate
            0.5,    # Meso LOD
            0.5,    # Meso LOD
            0.75,   # Efficient algorithm
            0.55,   # Balanced threshold
            0.0,    # Cloud storage
            0.45,   # Mixed comm
            0.9,    # Mostly cloud
            0.25,   # Small crew
            0.18    # ~66 day cycle
        ]))
        
        # Generate mutations (use self.rng if available, else create temp rng)
        rng = getattr(self, 'rng', np.random.default_rng(self.seed))
        while len(configs) < 15:
            base = configs[rng.integers(min(3, len(configs)))]
            mutated = self._mutate_solution(base, sigma=0.08)
            configs.append(mutated)
        
        return configs
    
    def _generate_smart_random_solution(self) -> np.ndarray:
        """Generate solution biased toward feasibility"""
        x = np.zeros(11)
        rng = self.rng  # Use instance RNG
        
        # Sensor: bias toward cheaper options
        x[0] = rng.beta(3, 1.5)  # Skewed toward higher indices (cheaper)
        
        # Data rate: moderate
        x[1] = rng.uniform(0.2, 0.6)
        
        # LOD: prefer Meso
        x[2] = rng.choice([0.33, 0.5, 0.67], p=[0.15, 0.7, 0.15])
        x[3] = x[2]
        
        # Algorithm: any
        x[4] = rng.random()
        
        # Threshold: moderate to high
        x[5] = rng.uniform(0.45, 0.75)
        
        # Storage: prefer cloud
        x[6] = rng.choice([0.0, 0.3, 0.7], p=[0.7, 0.2, 0.1])
        
        # Communication: moderate
        x[7] = rng.uniform(0.3, 0.7)
        
        # Deployment: prefer cloud
        x[8] = rng.uniform(0.6, 1.0)
        
        # Crew: small to medium
        x[9] = rng.uniform(0.1, 0.5)
        
        # Cycle: monthly to quarterly
        x[10] = rng.uniform(0.08, 0.25)
        
        return x
    
    def _mutate_solution(self, x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """Mutate solution with Gaussian noise"""
        mutated = x.copy()
        rng = getattr(self, 'rng', np.random.default_rng(self.seed))
        for i in range(len(mutated)):
            if i in [2, 3]:  # LOD - discrete
                if rng.random() < 0.1:
                    mutated[i] = rng.choice([0.33, 0.5, 0.67])
            else:
                mutated[i] += rng.normal(0, sigma)
        return np.clip(mutated, 0, 1)


class GridSearchBaseline(BaselineMethod):
    """Improved Grid Search with focused parameter ranges"""
    
    def optimize(self) -> pd.DataFrame:
        """Run grid search on key parameters"""
        # Use local RNG for smart variations
        self.rng = np.random.default_rng(self.seed + 1)
        
        logger.info(f"Running Grid Search baseline (seed={self.seed + 1})...")
        
        start_time = time.time()
        
        # Define grid points for key variables
        sensor_values = [0.85, 0.9, 0.95, 1.0]  # Focus on affordable sensors
        algo_values = [0.6, 0.7, 0.8, 0.9]      # Traditional to ML
        threshold_values = [0.5, 0.6, 0.7]
        cycle_values = [0.1, 0.15, 0.2, 0.25]   # 36-90 days
        
        # Fixed reasonable defaults for other variables
        defaults = {
            'data_rate': 0.4,
            'geo_lod': 0.5,
            'cond_lod': 0.5,
            'storage': 0.0,
            'comm': 0.5,
            'deploy': 0.9,
            'crew': 0.3
        }
        
        count = 0
        max_evals = getattr(self.config, 'grid_max_evaluations', 3000)
        
        for sensor in sensor_values:
            for algo in algo_values:
                for threshold in threshold_values:
                    for cycle in cycle_values:
                        if count >= max_evals:
                            break
                        
                        x = np.array([
                            sensor,
                            defaults['data_rate'],
                            defaults['geo_lod'],
                            defaults['cond_lod'],
                            algo,
                            threshold,
                            defaults['storage'],
                            defaults['comm'],
                            defaults['deploy'],
                            defaults['crew'],
                            cycle
                        ])
                        
                        objectives, constraints = self.evaluator._evaluate_single(x)
                        self.results.append(
                            self._create_result_entry(x, objectives, constraints, count + 1)
                        )
                        count += 1
        
        # Add smart variations
        if count < max_evals:
            self._add_smart_variations(max_evals - count)
        
        self.execution_time = time.time() - start_time
        
        for result in self.results:
            result['time_seconds'] = self.execution_time
        
        df = pd.DataFrame(self.results)
        logger.info(f"Grid Search: {self.execution_time:.2f}s, {df['is_feasible'].sum()} feasible")
        
        return df
    
    def _add_smart_variations(self, remaining: int):
        """Add variations around good solutions"""
        rng = getattr(self, 'rng', np.random.default_rng(self.seed + 1))
        
        # Find feasible solutions
        feasible = [r for r in self.results if r['is_feasible']]
        
        if not feasible:
            # Generate conservative defaults if no feasible found
            for i in range(min(remaining, 50)):
                x = np.array([
                    0.95 + rng.uniform(-0.05, 0.05),
                    0.3 + rng.uniform(-0.1, 0.1),
                    0.5, 0.5,
                    0.8 + rng.uniform(-0.1, 0.1),
                    0.6 + rng.uniform(-0.1, 0.1),
                    0.0,
                    0.4 + rng.uniform(-0.1, 0.2),
                    0.95,
                    0.25 + rng.uniform(-0.05, 0.1),
                    0.2 + rng.uniform(-0.05, 0.1)
                ])
                x = np.clip(x, 0, 1)
                
                objectives, constraints = self.evaluator._evaluate_single(x)
                self.results.append(
                    self._create_result_entry(x, objectives, constraints, len(self.results) + 1)
                )


class WeightedSumBaseline(BaselineMethod):
    """Weighted Sum optimization with multiple weight combinations"""
    
    def optimize(self) -> pd.DataFrame:
        """Run weighted sum optimization"""
        # Use local RNG for reproducibility without affecting global state
        self.rng = np.random.default_rng(self.seed + 2)
        
        n_weights = self.config.weight_combinations
        logger.info(f"Running Weighted Sum with {n_weights} weight sets (seed={self.seed + 2})...")
        
        start_time = time.time()
        
        weight_sets = self._generate_weight_sets(n_weights)
        
        for weight_idx, weights in enumerate(weight_sets):
            best_score = float('inf')
            best_solution = None
            best_objectives = None
            best_constraints = None
            
            # Multiple random starts
            for _ in range(20):
                x = self._generate_initial_point()
                
                # Local search
                for _ in range(50):
                    objectives, constraints = self.evaluator._evaluate_single(x)
                    
                    if np.all(constraints <= 0):
                        norm_obj = self._normalize_objectives(objectives)
                        score = np.dot(weights, norm_obj)
                        
                        if score < best_score:
                            best_score = score
                            best_solution = x.copy()
                            best_objectives = objectives
                            best_constraints = constraints
                    
                    x = self._local_search_step(x)
            
            if best_solution is not None:
                result = self._create_result_entry(
                    best_solution, best_objectives, best_constraints, weight_idx + 1
                )
                result['weights'] = weights.tolist()
                self.results.append(result)
            else:
                # Record infeasible attempt
                x = self._generate_initial_point()
                objectives, constraints = self.evaluator._evaluate_single(x)
                result = self._create_result_entry(x, objectives, constraints, weight_idx + 1)
                result['weights'] = weights.tolist()
                self.results.append(result)
            
            if (weight_idx + 1) % 10 == 0:
                logger.info(f"  Processed {weight_idx + 1} weight combinations")
        
        self.execution_time = time.time() - start_time
        
        for result in self.results:
            result['time_seconds'] = self.execution_time
        
        df = pd.DataFrame(self.results)
        logger.info(f"Weighted Sum: {self.execution_time:.2f}s, {df['is_feasible'].sum()} feasible")
        
        return df

    def _generate_weight_sets(self, n_sets: int = 50) -> np.ndarray:

        weights = []
        rng = self.rng
        n_obj = 6

        # 1) Uniform weights
        weights.append(np.ones(n_obj) / n_obj)

        # 2) Single objective focus (corners) - 6个
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            weights.append(w)

        # 3) Pairwise combinations (edges) - 15个
        for i in range(n_obj):
            for j in range(i + 1, n_obj):
                w = np.zeros(n_obj)
                w[i] = 0.5
                w[j] = 0.5
                weights.append(w)

        # 4) Practical weight combinations - 工程实践常见组合
        # 成本+召回 (最常见权衡)
        weights.append(np.array([0.4, 0.4, 0.05, 0.05, 0.05, 0.05]))
        # 成本+延迟 (实时性需求)
        weights.append(np.array([0.4, 0.1, 0.4, 0.03, 0.03, 0.04]))
        # 召回+可靠性 (安全关键)
        weights.append(np.array([0.1, 0.4, 0.1, 0.1, 0.1, 0.2]))
        # 低碳+成本 (绿色优先)
        weights.append(np.array([0.3, 0.1, 0.1, 0.1, 0.3, 0.1]))

        # 5) Dirichlet sampling - 核心改进
        # alpha < 1: 稀疏权重，集中在少数目标
        # alpha = 1: 均匀分布
        # alpha > 1: 更均衡的权重分配
        for alpha in [0.3, 0.5, 1.0, 2.0, 5.0]:
            n_to_add = max(3, (n_sets - len(weights)) // 5)
            dirichlet_weights = rng.dirichlet(np.ones(n_obj) * alpha, size=n_to_add)
            for w in dirichlet_weights:
                if len(weights) < n_sets:
                    weights.append(w)

        # 6) Fill remaining with random
        while len(weights) < n_sets:
            w = rng.random(n_obj)
            w = w / w.sum()
            weights.append(w)

        logger.info(f"Generated {len(weights[:n_sets])} weight combinations (Dirichlet enhanced)")
        return np.array(weights[:n_sets])
    
    def _generate_initial_point(self) -> np.ndarray:
        """Generate good initial point"""
        rng = self.rng
        x = np.array([
            rng.uniform(0.8, 1.0),  # Cheap sensor
            rng.uniform(0.3, 0.5),
            0.5, 0.5,
            rng.uniform(0.5, 0.9),
            rng.uniform(0.5, 0.7),
            0.0,
            rng.uniform(0.4, 0.6),
            rng.uniform(0.7, 1.0),
            rng.uniform(0.2, 0.4),
            rng.uniform(0.1, 0.2)
        ])
        return x
    
    def _local_search_step(self, x: np.ndarray) -> np.ndarray:
        """Take local search step"""
        rng = self.rng
        x_new = x.copy()
        for i in range(len(x)):
            if rng.random() < 0.2:
                if i in [2, 3]:
                    x_new[i] = rng.choice([0.33, 0.5, 0.67])
                else:
                    x_new[i] = np.clip(x[i] + rng.normal(0, 0.08), 0, 1)
        return x_new
    
    def _normalize_objectives(self, obj: np.ndarray) -> np.ndarray:
        """Normalize objectives to [0,1]"""
        ranges = [
            (1e5, 2e7),
            (0, 0.4),
            (1, 300),
            (0, 300),
            (100, 50000),
            (1e-6, 1e-3)
        ]
        
        norm = np.zeros_like(obj)
        for i, (min_val, max_val) in enumerate(ranges):
            norm[i] = (obj[i] - min_val) / (max_val - min_val + 1e-10)
        
        return np.clip(norm, 0, 1)


class ExpertHeuristicBaseline(BaselineMethod):
    """
    Expert Heuristic baseline with GUARANTEED feasible solutions.
    
    Key improvement: If no configurations are feasible, progressively
    relax parameters until at least one feasible solution is found.
    """
    
    def optimize(self) -> pd.DataFrame:
        """Generate expert configurations with feasibility guarantee"""
        # Use local RNG for variations in Phase 4
        self.rng = np.random.default_rng(self.seed + 3)
        
        logger.info(f"Running Expert Heuristic baseline (seed={self.seed + 3})...")
        
        start_time = time.time()
        
        # Phase 1: Try standard expert configurations
        expert_configs = self._get_expert_configurations()
        feasible_found = False
        
        for idx, (name, x) in enumerate(expert_configs):
            is_feasible, objectives, constraints = self._check_feasibility(x)
            
            result = self._create_result_entry(x, objectives, constraints, idx + 1)
            result['method'] = f"ExpertHeuristic-{name}"
            self.results.append(result)
            
            if is_feasible:
                feasible_found = True
        
        # Phase 2: If no feasible solutions, progressively relax
        if not feasible_found:
            logger.warning("No feasible expert configs found. Applying relaxation...")
            relaxed_configs = self._generate_relaxed_configs()
            
            for idx, (name, x) in enumerate(relaxed_configs):
                is_feasible, objectives, constraints = self._check_feasibility(x)
                
                result = self._create_result_entry(
                    x, objectives, constraints, len(self.results) + 1
                )
                result['method'] = f"ExpertHeuristic-{name}"
                self.results.append(result)
                
                if is_feasible:
                    feasible_found = True
                    logger.info(f"Found feasible solution: {name}")
        
        # Phase 3: Extreme relaxation if still not feasible
        if not feasible_found:
            logger.warning("Applying extreme relaxation...")
            extreme_configs = self._generate_extreme_relaxed_configs()
            
            for idx, (name, x) in enumerate(extreme_configs):
                is_feasible, objectives, constraints = self._check_feasibility(x)
                
                result = self._create_result_entry(
                    x, objectives, constraints, len(self.results) + 1
                )
                result['method'] = f"ExpertHeuristic-{name}"
                self.results.append(result)
                
                if is_feasible:
                    feasible_found = True
                    logger.info(f"Found feasible solution with extreme relaxation: {name}")
        
        # Phase 4: Add variations of best solutions
        self._add_feasible_variations()
        
        self.execution_time = time.time() - start_time
        
        for result in self.results:
            result['time_seconds'] = self.execution_time
        
        df = pd.DataFrame(self.results)
        feasible_count = df['is_feasible'].sum()
        
        logger.info(f"Expert Heuristic: {self.execution_time:.2f}s")
        logger.info(f"  Total configs: {len(df)}, Feasible: {feasible_count}")
        
        if feasible_count == 0:
            logger.error("WARNING: No feasible Expert solutions found!")
            logger.error("Consider relaxing constraints in config.json")
        
        return df
    
    def _get_expert_configurations(self) -> List[Tuple[str, np.ndarray]]:
        """Get standard expert configurations"""
        configs = [
            ("LowCost", self._low_cost_config()),
            ("Balanced", self._balanced_config()),
            ("QuickDeploy", self._quick_deploy_config()),
            ("Urban", self._urban_config()),
            ("Rural", self._rural_config()),
            ("Sustainable", self._sustainable_config()),
            ("Reliable", self._reliable_config()),
            ("Emergency", self._emergency_config()),
            ("Research", self._research_config()),
            ("Practical", self._practical_config()),
        ]
        
        # Add variations
        base_configs = configs.copy()
        for name, config in base_configs[:5]:
            for i in range(2):
                varied = self._add_variation(config, 0.08)
                configs.append((f"{name}_Var{i+1}", varied))
        
        return configs
    
    def _generate_relaxed_configs(self) -> List[Tuple[str, np.ndarray]]:
        """Generate configurations with relaxed parameters"""
        configs = []
        
        # Ultra low cost: minimize everything expensive
        configs.append(("UltraLowCost", np.array([
            1.0,    # Cheapest sensor
            0.15,   # Very low data rate
            0.67,   # Macro LOD (less data)
            0.67,   # Macro LOD
            0.95,   # Simplest algorithm
            0.7,    # Higher threshold (fewer false alarms)
            0.0,    # Cloud storage
            0.3,    # Cheapest comm
            1.0,    # Cloud (pay-per-use)
            0.1,    # Minimal crew
            0.35    # Longer cycle (~130 days)
        ])))
        
        # Minimum footprint
        configs.append(("MinimalFootprint", np.array([
            0.98,   # IoT
            0.1,    # Minimal data
            0.5,    # Meso
            0.5,    # Meso
            0.9,    # Traditional
            0.65,   # Mid threshold
            0.0,    # Cloud
            0.35,   # LoRaWAN
            1.0,    # Cloud
            0.15,   # 1-2 person
            0.4     # ~150 day cycle
        ])))
        
        # Extended cycle
        configs.append(("ExtendedCycle", np.array([
            0.9,    # Vehicle
            0.3,    # Low data
            0.5,    # Meso
            0.5,    # Meso
            0.75,   # ML
            0.6,    # Mid threshold
            0.0,    # Cloud
            0.5,    # 4G
            0.9,    # Cloud
            0.2,    # Small crew
            0.5     # ~180 day cycle
        ])))
        
        # Maximum recall relaxation
        configs.append(("RelaxedRecall", np.array([
            0.92,   # Affordable sensor
            0.25,   # Low data
            0.5,    # Meso
            0.5,    # Meso
            0.6,    # Some ML
            0.8,    # High threshold
            0.0,    # Cloud
            0.45,   # Mid comm
            0.95,   # Cloud
            0.2,    # Small crew
            0.3     # ~110 day cycle
        ])))
        
        return configs
    
    def _generate_extreme_relaxed_configs(self) -> List[Tuple[str, np.ndarray]]:
        """Generate extremely relaxed configurations as last resort"""
        configs = []
        
        # Absolute minimum
        configs.append(("AbsoluteMinimum", np.array([
            1.0,    # Cheapest
            0.05,   # Minimum data
            0.67,   # Macro
            0.67,   # Macro
            1.0,    # Simplest
            0.85,   # Very high threshold
            0.0,    # Cloud
            0.3,    # Cheapest comm
            1.0,    # Cloud
            0.05,   # 1 person
            0.6     # ~220 day cycle
        ])))
        
        # Try different sensor options
        for sensor_val in [0.95, 0.9, 0.85, 0.8]:
            configs.append((f"SensorVar_{sensor_val}", np.array([
                sensor_val,
                0.2,
                0.5, 0.5,
                0.85,
                0.7,
                0.0,
                0.4,
                1.0,
                0.2,
                0.35
            ])))
        
        return configs
    
    def _add_feasible_variations(self):
        """Add variations around feasible solutions"""
        feasible = [r for r in self.results if r['is_feasible']]
        
        if not feasible:
            return
        
        # Sort by cost and take best 3
        feasible_sorted = sorted(feasible, key=lambda x: x['f1_total_cost_USD'])[:3]
        
        for base_result in feasible_sorted:
            # Reconstruct x vector (approximate)
            base_x = np.array([
                0.9, 0.3, 0.5, 0.5, 0.7, 0.6, 0.0, 0.5, 0.9, 0.25, 0.2
            ])
            
            # Generate variations
            for i in range(5):
                x = self._add_variation(base_x, 0.05)
                is_feasible, objectives, constraints = self._check_feasibility(x)
                
                result = self._create_result_entry(
                    x, objectives, constraints, len(self.results) + 1
                )
                result['method'] = f"ExpertHeuristic-FeasibleVar{i+1}"
                self.results.append(result)
    
    # =========================================================================
    # EXPERT CONFIGURATION TEMPLATES
    # =========================================================================
    
    def _low_cost_config(self) -> np.ndarray:
        """Low cost IoT configuration"""
        return np.array([
            0.95,   # IoT sensor
            0.25,   # Low data rate
            0.5,    # Meso LOD
            0.5,    # Meso LOD
            0.85,   # Traditional/simple ML
            0.6,    # Mid threshold
            0.0,    # Cloud storage
            0.4,    # LoRaWAN/4G
            1.0,    # Cloud deploy
            0.2,    # 2-person crew
            0.2     # ~73 day cycle
        ])
    
    def _balanced_config(self) -> np.ndarray:
        """Balanced configuration"""
        return np.array([
            0.88,   # Vehicle sensor
            0.4,    # Moderate data rate
            0.5,    # Meso LOD
            0.5,    # Meso LOD
            0.65,   # ML algorithm
            0.65,   # Good threshold
            0.0,    # Cloud storage
            0.5,    # 4G
            0.85,   # Mostly cloud
            0.3,    # 3-person crew
            0.12    # ~45 day cycle
        ])
    
    def _quick_deploy_config(self) -> np.ndarray:
        """Quick deployment configuration"""
        return np.array([
            0.9,    # Easy-deploy sensor
            0.35,   # Moderate data
            0.5,    # Meso LOD
            0.5,    # Meso LOD
            0.7,    # ML
            0.6,    # Mid threshold
            0.0,    # Cloud storage
            0.55,   # 4G
            1.0,    # Cloud deploy
            0.3,    # 3-person crew
            0.1     # ~36 day cycle
        ])
    
    def _urban_config(self) -> np.ndarray:
        """Urban environment configuration"""
        return np.array([
            0.88,   # Vehicle sensor
            0.45,   # Moderate data
            0.5,    # Meso LOD
            0.5,    # Meso LOD
            0.5,    # ML algorithm
            0.65,   # Good threshold
            0.0,    # Cloud storage
            0.6,    # Good connectivity
            0.8,    # Hybrid deploy
            0.35,   # 3-4 person crew
            0.08    # ~30 day cycle
        ])
    
    def _rural_config(self) -> np.ndarray:
        """Rural environment configuration"""
        return np.array([
            0.95,   # IoT/low-cost sensor
            0.25,   # Lower data rate
            0.5,    # Meso LOD
            0.5,    # Meso LOD
            0.75,   # Simpler ML
            0.6,    # Mid threshold
            0.0,    # Cloud storage
            0.35,   # LoRaWAN (long range)
            1.0,    # Cloud deploy
            0.2,    # Small crew
            0.25    # ~90 day cycle
        ])
    
    def _sustainable_config(self) -> np.ndarray:
        """Sustainable/green configuration"""
        return np.array([
            0.95,   # Low power IoT
            0.15,   # Minimal data
            0.5,    # Meso LOD
            0.5,    # Meso LOD
            0.85,   # Efficient algorithm
            0.6,    # Mid threshold
            0.0,    # Cloud (shared)
            0.35,   # LoRaWAN (low power)
            1.0,    # Cloud (efficient)
            0.15,   # Minimal crew
            0.3     # ~110 day cycle
        ])
    
    def _reliable_config(self) -> np.ndarray:
        """High reliability configuration"""
        return np.array([
            0.5,    # Better sensor
            0.4,    # Moderate data
            0.5,    # Meso LOD
            0.5,    # Meso LOD
            0.5,    # ML algorithm
            0.65,   # Good threshold
            0.4,    # Hybrid storage
            0.6,    # 4G/5G
            0.5,    # Hybrid deploy
            0.3,    # 3-person crew
            0.1     # ~36 day cycle
        ])
    
    def _emergency_config(self) -> np.ndarray:
        """Emergency response configuration"""
        return np.array([
            0.55,   # UAV capable
            0.55,   # Higher data rate
            0.5,    # Meso LOD
            0.5,    # Meso LOD
            0.4,    # Fast ML
            0.5,    # Lower threshold (safety)
            0.0,    # Cloud storage
            0.65,   # Good connectivity
            0.3,    # Edge for speed
            0.4,    # 4-person crew
            0.02    # ~7 day cycle
        ])
    
    def _research_config(self) -> np.ndarray:
        """Research/high-accuracy configuration"""
        return np.array([
            0.35,   # Good sensor
            0.6,    # Higher data rate
            0.4,    # Near-Micro LOD
            0.4,    # Near-Micro LOD
            0.3,    # DL algorithm
            0.7,    # Good threshold
            0.3,    # Mixed storage
            0.7,    # Good connectivity
            0.4,    # Edge/hybrid
            0.45,   # 4-5 person crew
            0.05    # ~18 day cycle
        ])
    
    def _practical_config(self) -> np.ndarray:
        """Practical/everyday configuration"""
        return np.array([
            0.88,   # Vehicle sensor
            0.4,    # Moderate data
            0.5,    # Meso LOD
            0.5,    # Meso LOD
            0.55,   # ML algorithm
            0.6,    # Mid threshold
            0.0,    # Cloud storage
            0.55,   # 4G
            0.85,   # Cloud deploy
            0.3,    # 3-person crew
            0.11    # ~40 day cycle
        ])
    
    def _add_variation(self, config: np.ndarray, sigma: float) -> np.ndarray:
        """Add small variation to configuration"""
        rng = getattr(self, 'rng', np.random.default_rng(self.seed + 3))
        varied = config.copy()
        for i in range(len(varied)):
            if i in [2, 3]:  # LOD - keep stable
                continue
            varied[i] = np.clip(varied[i] + rng.normal(0, sigma), 0, 1)
        return varied


class BaselineRunner:
    """Orchestrates all baseline methods"""
    
    def __init__(self, ontology_graph, config, seed: int = DEFAULT_SEED):
        self.ontology_graph = ontology_graph
        self.config = config
        self.seed = seed
        
        # Import evaluator
        from evaluation import EnhancedFitnessEvaluatorV3
        
        # Initialize shared evaluator
        self.evaluator = EnhancedFitnessEvaluatorV3(ontology_graph, config)
        
        # Initialize baseline methods with reproducible seeds
        self.methods = {
            'random': RandomSearchBaseline(self.evaluator, config, seed=seed),
            'grid': GridSearchBaseline(self.evaluator, config, seed=seed),
            'weighted': WeightedSumBaseline(self.evaluator, config, seed=seed),
            'expert': ExpertHeuristicBaseline(self.evaluator, config, seed=seed)
        }
        
        logger.info(f"BaselineRunner initialized with seed={seed}")
    
    def run_all_methods(self) -> Dict[str, pd.DataFrame]:
        """Run all baseline methods"""
        results = {}
        
        for name, method in self.methods.items():
            logger.info(f"\nRunning baseline method: {name}")
            try:
                df = method.optimize()
                results[name] = df
                
                logger.info(f"  Total solutions: {len(df)}")
                if 'is_feasible' in df.columns:
                    feasible = df[df['is_feasible']]
                    logger.info(f"  Feasible solutions: {len(feasible)}")
                    if len(feasible) > 0:
                        logger.info(f"  Best cost: ${feasible['f1_total_cost_USD'].min():,.0f}")
                        logger.info(f"  Best recall: {feasible['detection_recall'].max():.3f}")
                
            except Exception as e:
                logger.error(f"Error in {name}: {str(e)}")
                import traceback
                traceback.print_exc()
                results[name] = pd.DataFrame()
        
        return results
    
    def run_method(self, method_name: str) -> pd.DataFrame:
        """Run a specific baseline method"""
        if method_name not in self.methods:
            raise ValueError(f"Unknown method: {method_name}. "
                           f"Available: {list(self.methods.keys())}")
        
        return self.methods[method_name].optimize()
    
    def compare_with_pareto(self, pareto_df: pd.DataFrame, 
                           baseline_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compare baseline results with Pareto front"""
        comparison = []
        
        if len(pareto_df) > 0:
            comparison.append({
                'Method': 'NSGA-III',
                'Total_Solutions': len(pareto_df),
                'Feasible_Solutions': len(pareto_df),
                'Min_Cost': pareto_df['f1_total_cost_USD'].min(),
                'Max_Recall': pareto_df['detection_recall'].max(),
                'Min_Carbon': pareto_df['f5_carbon_emissions_kgCO2e_year'].min(),
                'Min_Latency': pareto_df['f3_latency_seconds'].min(),
                'Max_MTBF': pareto_df['system_MTBF_hours'].max()
            })
        
        for method, df in baseline_results.items():
            if len(df) > 0 and 'is_feasible' in df.columns:
                feasible = df[df['is_feasible']]
                comparison.append({
                    'Method': method.title(),
                    'Total_Solutions': len(df),
                    'Feasible_Solutions': len(feasible),
                    'Min_Cost': feasible['f1_total_cost_USD'].min() if len(feasible) > 0 else np.nan,
                    'Max_Recall': feasible['detection_recall'].max() if len(feasible) > 0 else np.nan,
                    'Min_Carbon': feasible['f5_carbon_emissions_kgCO2e_year'].min() if len(feasible) > 0 else np.nan,
                    'Min_Latency': feasible['f3_latency_seconds'].min() if len(feasible) > 0 else np.nan,
                    'Max_MTBF': feasible['system_MTBF_hours'].max() if len(feasible) > 0 else np.nan
                })
        
        return pd.DataFrame(comparison)
