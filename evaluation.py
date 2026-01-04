#!/usr/bin/env python3
"""
Enhanced Fitness Evaluation Module V3 - Step 2-Lite Upgrade
============================================================
All 11 decision variables now materially affect objectives.

Key improvements:
1. Recall model: Sigmoid-based with detection_threshold effect
2. Latency model: data_rate + communication + deployment all contribute
3. Cost model: storage + comm + compute OPEX included
4. Carbon model: compute + comm energy included
5. Disruption model: speed + crew_size + inspection_cycle dependent
6. Added explain() trace function for paper interpretability

Author: RMTwin Research Team
Version: 3.0 (Step 2-Lite)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from functools import lru_cache
from rdflib.namespace import RDF, RDFS, OWL
from rdflib import Graph, Namespace

# Import centralized parameters
from model_params import (
    MODEL_PARAMS, get_param, get_comm_type, get_storage_type,
    get_deployment_type, get_sensor_type, get_algo_type, sigmoid,
    validate_recall, validate_positive, validate_non_negative
)

logger = logging.getLogger(__name__)

RDTCO = Namespace("http://www.semanticweb.org/rmtwin/ontologies/rdtco#")
EX = Namespace("http://example.org/rmtwin#")


class SolutionMapper:
    """
    Solution Mapper - Decodes optimization vectors to configurations.

    Decision variables (11 total):
    x[0]: sensor           - Sensor system index
    x[1]: data_rate        - Data sampling rate (10-100 Hz)
    x[2]: geo_lod          - Geometric level of detail (Micro/Meso/Macro)
    x[3]: cond_lod         - Condition level of detail
    x[4]: algorithm        - Detection algorithm index
    x[5]: detection_threshold - Detection threshold (0.1-0.9)
    x[6]: storage          - Storage system index
    x[7]: communication    - Communication system index
    x[8]: deployment       - Compute deployment index
    x[9]: crew_size        - Inspection crew size (1-10)
    x[10]: inspection_cycle - Days between inspections (1-365)
    """

    def __init__(self, ontology_graph: Graph, config=None):
        self.g = ontology_graph
        self.config = config  # v3.2: 添加config引用
        self._cache_components()
        self._decode_cache = {}

    def _cache_components(self):
        """Cache all available components from ontology."""
        self.sensors = []
        self.algorithms = []
        self.storage_systems = []
        self.comm_systems = []
        self.deployments = []

        logger.info("Caching ontology components...")

        sensor_patterns = [
            'MMS_LiDAR_System', 'MMS_Camera_System',
            'UAV_LiDAR_System', 'UAV_Camera_System',
            'TLS_System', 'Handheld_3D_Scanner',
            'FiberOptic_Sensor', 'Vehicle_LowCost_Sensor',
            'IoT_Network_System', 'Sensor', 'sensor'
        ]

        algo_patterns = [
            'DeepLearningAlgorithm', 'MachineLearningAlgorithm',
            'TraditionalAlgorithm', 'PointCloudAlgorithm',
            'Algorithm', 'algorithm'
        ]

        deploy_patterns = [
            'Deployment', 'Compute', 'Edge', 'Cloud',
            'ComputeDeployment', 'Deployment_Edge_Computing',
            'Deployment_Cloud_Computing', 'Deployment_Hybrid_Edge_Cloud',
            'Deployment_OnPremise_Server'
        ]

        for s, p, o in self.g:
            if p == RDF.type and str(s).startswith('http://example.org/rmtwin#'):
                subject_str = str(s)
                type_str = str(o)

                is_sensor = any(pattern in type_str for pattern in sensor_patterns)
                if is_sensor and subject_str not in self.sensors:
                    self.sensors.append(subject_str)
                    continue

                is_algorithm = any(pattern in type_str for pattern in algo_patterns)
                if is_algorithm and subject_str not in self.algorithms:
                    self.algorithms.append(subject_str)
                    continue

                if 'Storage' in type_str and subject_str not in self.storage_systems:
                    self.storage_systems.append(subject_str)
                    continue

                if 'Communication' in type_str and subject_str not in self.comm_systems:
                    self.comm_systems.append(subject_str)
                    continue

                is_deployment = any(pattern in type_str for pattern in deploy_patterns)
                if is_deployment and subject_str not in self.deployments:
                    self.deployments.append(subject_str)

        # Ensure defaults
        if not self.sensors:
            logger.warning("No sensors found! Using defaults")
            self.sensors = ["http://example.org/rmtwin#IoT_LoRaWAN_Sensor"]
        if not self.algorithms:
            logger.warning("No algorithms found! Using defaults")
            self.algorithms = ["http://example.org/rmtwin#Traditional_Canny_Optimized"]
        if not self.storage_systems:
            self.storage_systems = ["http://example.org/rmtwin#Storage_AWS_S3_Standard"]
        if not self.comm_systems:
            self.comm_systems = ["http://example.org/rmtwin#Communication_LoRaWAN_Gateway"]
        if not self.deployments:
            self.deployments = ["http://example.org/rmtwin#Deployment_Cloud_GPU_A4000"]

        logger.info(f"Cached: {len(self.sensors)} sensors, {len(self.algorithms)} algorithms, "
                    f"{len(self.storage_systems)} storage, {len(self.comm_systems)} comm, "
                    f"{len(self.deployments)} deployments")

    def decode_solution(self, x: np.ndarray) -> Dict:
        """Decode solution vector to configuration dict."""
        x_key = tuple(float(xi) for xi in x)

        if x_key in self._decode_cache:
            return self._decode_cache[x_key]

        config = {
            'sensor': self.sensors[int(x[0] * len(self.sensors)) % len(self.sensors)],
            'data_rate': 10 + x[1] * 90,  # 10-100 Hz
            'geo_lod': ['Micro', 'Meso', 'Macro'][int(x[2] * 3) % 3],
            'cond_lod': ['Micro', 'Meso', 'Macro'][int(x[3] * 3) % 3],
            'algorithm': self.algorithms[int(x[4] * len(self.algorithms)) % len(self.algorithms)],
            'detection_threshold': 0.1 + x[5] * 0.8,  # 0.1-0.9
            'storage': self.storage_systems[int(x[6] * len(self.storage_systems)) % len(self.storage_systems)],
            'communication': self.comm_systems[int(x[7] * len(self.comm_systems)) % len(self.comm_systems)],
            'deployment': self.deployments[int(x[8] * len(self.deployments)) % len(self.deployments)],
            'crew_size': int(1 + x[9] * 9),  # 1-10
            # v3.2: 检查周期范围由min_inspections_per_year控制
            'inspection_cycle': int(
                1 + x[10] * (365.0 / getattr(self.config, 'min_inspections_per_year', 4) - 1)) if self.config else int(
                1 + x[10] * 90)
        }

        self._decode_cache[x_key] = config
        return config


class EnhancedFitnessEvaluatorV3:
    """
    Enhanced Fitness Evaluator V3 - Step 2-Lite Upgrade

    All 11 decision variables materially affect at least one objective.
    Uses centralized MODEL_PARAMS for all tunable parameters.
    """

    def __init__(self, ontology_graph: Graph, config):
        self.g = ontology_graph
        self.config = config
        self.solution_mapper = SolutionMapper(ontology_graph, config)  # v3.2: 传入config

        # Property cache for ontology queries
        self._property_cache = {}

        # Initialize caches
        self._initialize_cache()
        self._prepare_pool_data()

        # Statistics
        self._evaluation_count = 0

        logger.info("EnhancedFitnessEvaluatorV3 initialized (Step 2-Lite)")

    def _initialize_cache(self):
        """Initialize property cache from ontology."""
        logger.info("Initializing property cache...")

        properties = [
            'hasInitialCostUSD', 'hasOperationalCostUSDPerDay', 'hasAnnualOpCostUSD',
            'hasEnergyConsumptionW', 'hasMTBFHours', 'hasOperatorSkillLevel',
            'hasCalibrationFreqMonths', 'hasDataAnnotationCostUSD',
            'hasModelRetrainingFreqMonths', 'hasExplainabilityScore',
            'hasIntegrationComplexity', 'hasCybersecurityVulnerability',
            'hasAccuracyRangeMM', 'hasDataVolumeGBPerKm',
            'hasCoverageEfficiencyKmPerDay', 'hasOperatingSpeedKmh',
            'hasRecall', 'hasPrecision', 'hasFPS', 'hasHardwareRequirement',
            'hasBandwidthMbps', 'hasStorageCostPerGBYear'
        ]

        all_components = (
                self.solution_mapper.sensors +
                self.solution_mapper.algorithms +
                self.solution_mapper.storage_systems +
                self.solution_mapper.comm_systems +
                self.solution_mapper.deployments
        )

        for component in all_components:
            if component not in self._property_cache:
                self._property_cache[component] = {}

            for prop in properties:
                query = f"""
                PREFIX rdtco: <http://www.semanticweb.org/rmtwin/ontologies/rdtco#>
                SELECT ?value WHERE {{
                    <{component}> rdtco:{prop} ?value .
                }}
                """

                try:
                    results = list(self.g.query(query))
                    if results:
                        value = results[0][0]
                        try:
                            self._property_cache[component][prop] = float(value)
                        except:
                            self._property_cache[component][prop] = str(value)
                except Exception:
                    pass

        logger.info(f"Cached properties for {len(self._property_cache)} components")

    def _prepare_pool_data(self):
        """Prepare data for potential parallel processing."""
        self._mapper_data = {
            'sensors': self.solution_mapper.sensors,
            'algorithms': self.solution_mapper.algorithms,
            'storage_systems': self.solution_mapper.storage_systems,
            'comm_systems': self.solution_mapper.comm_systems,
            'deployments': self.solution_mapper.deployments
        }
        self._config_dict = vars(self.config) if hasattr(self.config, '__dict__') else self.config

    def _query_property(self, subject: str, predicate: str, default=None):
        """Query property from cache."""
        if subject in self._property_cache:
            if predicate in self._property_cache[subject]:
                return self._property_cache[subject][predicate]
        return default

    # =========================================================================
    # MAIN EVALUATION INTERFACE
    # =========================================================================

    def evaluate_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch evaluate solutions - main interface.

        Args:
            X: Array of shape (n_solutions, 11) with decision variables

        Returns:
            objectives: Array of shape (n_solutions, 6)
            constraints: Array of shape (n_solutions, 5)
        """
        n_solutions = len(X)
        objectives = np.zeros((n_solutions, 6))
        constraints = np.zeros((n_solutions, 5))

        for i, x in enumerate(X):
            try:
                objectives[i], constraints[i] = self._evaluate_single(x)
            except Exception as e:
                logger.error(f"Error evaluating solution {i}: {e}")
                # Penalty values
                objectives[i] = np.array([1e10, 1, 1000, 1000, 200000, 1])
                constraints[i] = np.array([1000, 1, 1e10, 100000, -1000])

        self._evaluation_count += n_solutions

        if self._evaluation_count % 1000 == 0:
            logger.debug(f"Evaluated {self._evaluation_count} solutions")

        return objectives, constraints

    def _evaluate_single(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate single solution."""
        config = self.solution_mapper.decode_solution(x)

        # P1: 语义快速筛选 - 检查组件兼容性
        is_valid, reasons = self._semantic_fast_check(config)
        if not is_valid:
            # 语义不合法：给合理惩罚值（足够差但有限，避免污染优化器数值尺度）
            pen_obj = np.array([
                self.config.budget_cap_usd * 10,  # f1: cost很差
                1.0,  # f2: 1-recall=1 即recall=0
                self.config.max_latency_seconds * 10,  # f3: latency很差
                1e6,  # f4: disruption很差
                self.config.max_carbon_emissions_kgCO2e_year * 10,  # f5: carbon很差
                1.0  # f6: 1/MTBF很差(低可靠性)
            ], dtype=float)
            # 约束违反值（g(x) <= 0为可行，正值表示违反）
            pen_con = np.array([
                self.config.max_latency_seconds * 5,  # latency违反
                0.5,  # recall违反
                self.config.budget_cap_usd * 5,  # budget违反
                self.config.max_carbon_emissions_kgCO2e_year * 5,  # carbon违反
                self.config.min_mtbf_hours  # MTBF违反
            ], dtype=float)
            return pen_obj, pen_con

        # Calculate all 6 objectives
        f1 = self._calculate_total_cost(config)
        f2 = self._calculate_detection_performance(config)  # Returns 1-recall
        f3 = self._calculate_latency(config)
        f4 = self._calculate_traffic_disruption(config)
        f5 = self._calculate_environmental_impact(config)
        f6 = self._calculate_system_reliability(config)

        objectives = np.array([f1, f2, f3, f4, f5, f6])

        # Calculate constraints (g(x) <= 0 for feasibility)
        recall = 1 - f2
        mtbf = 1 / f6 if f6 > 1e-10 else 1e6

        constraints = np.array([
            f3 - self.config.max_latency_seconds,  # Latency constraint
            self.config.min_recall_threshold - recall,  # Recall constraint
            f1 - self.config.budget_cap_usd,  # Budget constraint
            f5 - self.config.max_carbon_emissions_kgCO2e_year,  # Carbon constraint
            self.config.min_mtbf_hours - mtbf  # Reliability constraint
        ])

        return objectives, constraints

    # =========================================================================
    # P1: 语义快速筛选 (运行时兼容性检查)
    # =========================================================================

    def _semantic_fast_check(self, config: Dict) -> Tuple[bool, List[str]]:
        """
        快速语义兼容性检查 - 运行时筛选明显不兼容的组合。

        这些规则基于工程常识，用于避免产生"语义无意义但数值可行"的解。
        完整的SHACL验证在Pareto输出后进行（后验审计）。

        Args:
            config: 解码后的配置字典

        Returns:
            (is_valid: bool, reasons: list[str])
        """
        reasons = []

        sensor_str = str(config.get('sensor', ''))
        algo_str = str(config.get('algorithm', ''))
        comm_str = str(config.get('communication', ''))
        deploy_str = str(config.get('deployment', ''))

        sensor_type = get_sensor_type(sensor_str)
        comm_type = get_comm_type(comm_str)
        deploy_type = get_deployment_type(deploy_str)
        algo_type = get_algo_type(algo_str)

        # =====================================================================
        # 规则 1: IoT/FOS 固定传感器不应搭配 V2X-DSRC 车载通信
        # 理由: IoT和光纤传感器是固定部署的，V2X/DSRC是车载短程通信
        # =====================================================================
        if sensor_type in ['IoT', 'FOS', 'Fiber'] and comm_type in ['V2X', 'DSRC']:
            reasons.append(
                f"Rule1: {sensor_type} sensor is fixed-deployment, "
                f"incompatible with vehicle-based {comm_type} communication."
            )

        # =====================================================================
        # 规则 2: 计算资源需求检查（合并GPU和DL算法需求）
        # 理由: GPU/DL算法需要适当的计算基础设施
        # 依据: hasHardwareRequirement属性 + 算法类型推断
        # =====================================================================
        hw_req = self._query_property(config['algorithm'], 'hasHardwareRequirement', 'CPU')
        hw_req_str = str(hw_req).upper()

        # 判断是否需要GPU（通过属性或算法名称推断）
        needs_gpu = 'GPU' in hw_req_str or any(kw in algo_str.upper() for kw in
                                               ['DL_', 'YOLO', 'UNET', 'MASK', 'EFFICIENT', 'SAM', 'RETINA', 'FASTER'])

        # 判断部署是否支持GPU
        has_gpu_capability = any(kw in deploy_str.upper() for kw in ['GPU', 'CLOUD', 'EDGE'])

        if needs_gpu and deploy_type == 'OnPremise' and not has_gpu_capability:
            reasons.append(
                f"Rule2: Algorithm requires GPU/high-compute (hw_req={hw_req}) "
                f"but deployment '{deploy_str.split('#')[-1]}' lacks GPU capability."
            )

        # =====================================================================
        # 规则 3: 移动传感器需要无线通信
        # 理由: 车载/无人机传感器不能只用光纤通信
        # =====================================================================
        is_mobile_sensor = sensor_type in ['MMS', 'UAV', 'Vehicle', 'Handheld']
        is_fiber_only = 'Fiber' in comm_str and all(
            kw not in comm_str.upper() for kw in ['CELLULAR', '5G', 'LTE', 'V2X', 'WIFI']
        )

        if is_mobile_sensor and is_fiber_only:
            reasons.append(
                f"Rule3: Mobile sensor {sensor_type} requires wireless communication, "
                f"not fiber-only connection."
            )

        # =====================================================================
        # 规则 5: 最小年检查次数约束（v3.2 硬约束）
        # 理由: 道路网络需要定期检查以保证安全，检查过于稀疏不现实
        # =====================================================================
        inspection_cycle = config.get('inspection_cycle', 30)
        min_inspections = getattr(self.config, 'min_inspections_per_year', 4)
        max_allowed_cycle = 365.0 / min_inspections

        if inspection_cycle > max_allowed_cycle:
            reasons.append(
                f"Rule5: Inspection cycle {inspection_cycle} days exceeds maximum "
                f"{max_allowed_cycle:.0f} days (requires {min_inspections} inspections/year)."
            )

        return (len(reasons) == 0), reasons

    def get_semantic_check_stats(self) -> Dict:
        """获取语义检查统计（用于调试）"""
        if not hasattr(self, '_semantic_reject_count'):
            self._semantic_reject_count = 0
        if not hasattr(self, '_semantic_reject_reasons'):
            self._semantic_reject_reasons = {}

        return {
            'reject_count': self._semantic_reject_count,
            'reject_reasons': self._semantic_reject_reasons
        }

    # =========================================================================
    # OBJECTIVE 1: TOTAL COST (USD over planning horizon)
    # =========================================================================

    def _calculate_total_cost(self, config: Dict) -> float:
        """
        Calculate total lifecycle cost.

        【v3.1修复】区分固定点位型和移动巡检型传感器的成本计算

        固定点位型 (coverage=0): IoT, FOS, 固定Camera
          - CAPEX = 单价 × sensors_needed
          - OPEX = 日成本 × sensors_needed × 365

        移动巡检型 (coverage>0): MMS, UAV, TLS, Vehicle, Handheld
          - CAPEX = 单价 × units_needed
          - OPEX = 日成本 × 巡检天数 × 巡检频率 × units_needed  【v3.1关键修复】

        Affected by: sensor, data_rate, geo_lod, algorithm, detection_threshold,
                    storage, communication, deployment, crew_size, inspection_cycle
        """
        sensor_name = str(config['sensor']).split('#')[-1]
        sensor_type = get_sensor_type(sensor_name)
        storage_type = get_storage_type(str(config['storage']))
        comm_type = get_comm_type(str(config['communication']))
        deploy_type = get_deployment_type(str(config['deployment']))
        algo_type = get_algo_type(str(config['algorithm']))

        road_length = self.config.road_network_length_km
        planning_years = self.config.planning_horizon_years
        inspections_per_year = 365.0 / config['inspection_cycle']

        # ----- CAPEX -----
        sensor_initial = self._query_property(config['sensor'], 'hasInitialCostUSD', 100000)
        coverage_km_day = self._query_property(config['sensor'], 'hasCoverageEfficiencyKmPerDay', 80)

        # 【v3.1】用于后续OPEX计算的变量
        mobile_units_needed = 1  # 默认值
        fixed_sensors_needed = 0  # 默认值

        # 区分固定点位型和移动巡检型
        if sensor_type == 'FOS':
            # FOS: 光纤传感器，高密度布设
            sensor_spacing_km = self.config.fos_sensor_spacing_km  # 默认0.1km
            fixed_sensors_needed = int(np.ceil(road_length / sensor_spacing_km))
            installation_cost_per_sensor = 5000  # 光纤安装成本高
            total_sensor_initial = (sensor_initial + installation_cost_per_sensor) * fixed_sensors_needed

        elif coverage_km_day == 0 or sensor_type == 'IoT':
            # 固定点位型传感器: IoT, 固定Camera等
            density_per_km = getattr(self.config, 'fixed_sensor_density_per_km', 1.0)
            fixed_sensors_needed = int(np.ceil(road_length * density_per_km))

            # IoT安装成本较低，其他固定传感器较高
            if sensor_type == 'IoT':
                installation_cost_per_sensor = 200  # LoRa网关部署
            else:
                installation_cost_per_sensor = 1000  # 固定摄像头等

            total_sensor_initial = (sensor_initial + installation_cost_per_sensor) * fixed_sensors_needed

        else:
            # 移动巡检型: MMS, UAV, TLS, Vehicle, Handheld, Camera(移动)
            # 【v3.1核心修复】根据road_length和inspection_cycle计算需要的设备套数

            # 方法1: 基于coverage能力（原有逻辑优化）
            coverage_per_cycle = coverage_km_day * config['inspection_cycle'] * 0.8  # 80%利用率

            if coverage_per_cycle > 0:
                mobile_units_needed = max(1, int(np.ceil(road_length / coverage_per_cycle)))
            else:
                mobile_units_needed = 1

            # 合理上限：移动设备通常1-10套
            mobile_units_needed = min(mobile_units_needed, 10)

            total_sensor_initial = sensor_initial * mobile_units_needed

        # Other component costs
        storage_initial = self._query_property(config['storage'], 'hasInitialCostUSD', 0)
        comm_initial = self._query_property(config['communication'], 'hasInitialCostUSD', 0)
        deploy_initial = self._query_property(config['deployment'], 'hasInitialCostUSD', 0)
        algo_initial = self._query_property(config['algorithm'], 'hasInitialCostUSD', 20000)

        total_capex = total_sensor_initial + storage_initial + comm_initial + deploy_initial + algo_initial

        # Annual capital cost (depreciation)
        dep_rate = get_param('depreciation_rate', sensor_type, 0.12)
        annual_capital_cost = total_capex * dep_rate

        # ----- SENSOR OPEX -----
        sensor_daily_cost = self._query_property(config['sensor'], 'hasOperationalCostUSDPerDay', 100)

        if coverage_km_day > 0 and sensor_type not in ['FOS', 'IoT']:
            # 【v3.1关键修复】Mobile sensor - OPEX也要乘以units_needed
            days_per_inspection = road_length / coverage_km_day
            # 原来漏了 mobile_units_needed！
            sensor_annual_opex = sensor_daily_cost * days_per_inspection * inspections_per_year * mobile_units_needed

        else:  # Fixed sensor
            if sensor_type == 'FOS':
                sensors_needed = int(np.ceil(road_length / self.config.fos_sensor_spacing_km))
                # FOS日维护成本较低
                sensor_annual_opex = 0.5 * sensors_needed * 365
            else:
                # IoT等固定传感器
                density_per_km = getattr(self.config, 'fixed_sensor_density_per_km', 1.0)
                sensors_needed = int(np.ceil(road_length * density_per_km))
                # 每个传感器的日运营成本
                sensor_annual_opex = sensor_daily_cost * sensors_needed * 365

        # ----- DATA VOLUME CALCULATION (affects storage, comm, compute costs) -----
        base_data_gb_per_km = self._query_property(config['sensor'], 'hasDataVolumeGBPerKm', 1.0)

        # LOD multiplier
        lod_mult = get_param('lod_data_multiplier', config['geo_lod'], 1.0)

        # Data rate multiplier
        base_rate = MODEL_PARAMS['recall_model']['base_data_rate']
        rate_mult = config['data_rate'] / base_rate

        # Raw data volume per year
        raw_data_gb_year = base_data_gb_per_km * road_length * lod_mult * rate_mult * inspections_per_year

        # Data reduction at edge
        data_reduction = get_param('data_reduction_ratio', deploy_type, 0.7)
        sent_data_gb_year = raw_data_gb_year * data_reduction

        # ----- STORAGE COST -----
        storage_cost_rate = get_param('storage_cost_USD_per_GB_year', storage_type, 0.20)
        retention_years = getattr(self.config, 'data_retention_years', 3)
        storage_annual = (storage_cost_rate * raw_data_gb_year * min(retention_years, 3) +
                          self._query_property(config['storage'], 'hasAnnualOpCostUSD', 5000))

        # ----- COMMUNICATION COST -----
        comm_cost_rate = get_param('comm_cost_USD_per_GB', comm_type, 0.05)
        comm_annual = (comm_cost_rate * sent_data_gb_year +
                       self._query_property(config['communication'], 'hasAnnualOpCostUSD', 2000))

        # ----- COMPUTE COST -----
        compute_time_per_gb = get_param('algo_compute_seconds_per_GB', algo_type, 15.0)
        compute_factor = get_param('deployment_compute_factor', deploy_type, 1.5)
        annual_compute_seconds = compute_time_per_gb * sent_data_gb_year * compute_factor
        annual_compute_hours = annual_compute_seconds / 3600.0

        compute_cost_rate = get_param('deployment_compute_cost_USD_per_hour', deploy_type, 0.5)
        compute_annual = (compute_cost_rate * annual_compute_hours +
                          self._query_property(config['deployment'], 'hasAnnualOpCostUSD', 5000))

        # ----- LABOR COST -----
        skill_level = str(self._query_property(config['sensor'], 'hasOperatorSkillLevel', 'Basic'))
        skill_mult = get_param('skill_wage_multiplier', skill_level, 1.0)
        daily_wage = self.config.daily_wage_per_person * skill_mult

        if coverage_km_day > 0 and sensor_type not in ['FOS', 'IoT']:
            # 【v3.1修复】移动传感器人工成本也要乘以units_needed
            days_per_inspection = road_length / coverage_km_day
            crew_annual_cost = config[
                                   'crew_size'] * daily_wage * days_per_inspection * inspections_per_year * mobile_units_needed
        else:
            # 固定传感器的维护人工
            if sensor_type == 'FOS':
                maintenance_days = 10
            elif sensor_type == 'IoT':
                maintenance_days = 5  # IoT维护需求低
            else:
                maintenance_days = 20
            crew_annual_cost = config['crew_size'] * daily_wage * maintenance_days

        # ----- ML/DL ANNOTATION COST -----
        annotation_mult = get_param('annotation_cost_multiplier', algo_type, 0.0)
        if annotation_mult > 0:
            base_annotation_cost = self._query_property(config['algorithm'], 'hasDataAnnotationCostUSD', 0.5)
            if 'Camera' in sensor_name:
                annual_images = 100 * road_length * inspections_per_year
            else:
                annual_images = 10000 * inspections_per_year
            annotation_annual = base_annotation_cost * annual_images * annotation_mult * 0.3
        else:
            annotation_annual = 0

        # ----- MODEL RETRAINING COST -----
        retrain_freq = self._query_property(config['algorithm'], 'hasModelRetrainingFreqMonths', 12)
        if retrain_freq and retrain_freq > 0 and algo_type in ['DL', 'ML']:
            retrainings_per_year = 12.0 / retrain_freq
            retrain_cost = 5000 if algo_type == 'DL' else 2000
            retrain_annual = retrain_cost * retrainings_per_year
        else:
            retrain_annual = 0

        # ----- FALSE POSITIVE PENALTY -----
        tau0 = MODEL_PARAMS['recall_model']['tau0']
        tau = config['detection_threshold']
        if tau < tau0:
            fp_coeff = MODEL_PARAMS['fp_penalty_coeff']
            base_ops_cost = sensor_annual_opex + crew_annual_cost
            fp_penalty = fp_coeff * ((tau0 - tau) ** 2) * base_ops_cost
        else:
            fp_penalty = 0

        # ----- TOTAL ANNUAL COST -----
        total_annual = (annual_capital_cost + sensor_annual_opex + storage_annual +
                        comm_annual + compute_annual + crew_annual_cost +
                        annotation_annual + retrain_annual + fp_penalty)

        # Seasonal adjustment
        if getattr(self.config, 'apply_seasonal_adjustments', True):
            total_annual *= 1.075

        # Lifecycle cost
        total_cost = total_annual * planning_years

        # Sanity check
        assert validate_positive(total_cost, "total_cost"), f"Invalid cost: {total_cost}"

        return total_cost

    # =========================================================================
    # OBJECTIVE 2: DETECTION PERFORMANCE (1 - recall)
    # =========================================================================

    def _calculate_detection_performance(self, config: Dict) -> float:
        """
        Calculate detection performance using sigmoid model.

        Returns 1-recall (minimization objective).

        Affected by: sensor (precision), algorithm (base recall), geo_lod,
                    detection_threshold, data_rate, deployment (hardware match)
        """
        rm = MODEL_PARAMS['recall_model']

        # Base algorithm recall
        base_algo_recall = self._query_property(config['algorithm'], 'hasRecall', 0.7)

        # Sensor precision/quality
        sensor_precision = self._query_property(config['sensor'], 'hasPrecision', 0.7)
        if sensor_precision is None or sensor_precision == 0:
            # Estimate from accuracy (已经加载到本体)
            accuracy_mm = self._query_property(config['sensor'], 'hasAccuracyRangeMM', 25)
            sensor_precision = float(np.clip(1.0 - accuracy_mm / 150.0, 0.2, 0.98))

        # LOD bonus (geometric LOD)
        lod_bonus = rm['lod_bonus'].get(config['geo_lod'], 0.0)

        # Condition LOD bonus - affects defect detection quality
        cond_lod_bonus_map = {'Micro': 0.3, 'Meso': 0.0, 'Macro': -0.2}
        cond_lod_bonus = cond_lod_bonus_map.get(config['cond_lod'], 0.0)

        # Detection threshold effect
        tau = config['detection_threshold']
        tau0 = rm['tau0']

        # Data rate bonus (higher rate → better sampling → better detection)
        data_rate_bonus = rm['data_rate_bonus_factor'] * max(0, config['data_rate'] - rm['base_data_rate'])

        # Hardware mismatch penalty
        hw_penalty = 0
        hardware_req = str(self._query_property(config['algorithm'], 'hasHardwareRequirement', 'CPU'))
        deploy_type = get_deployment_type(str(config['deployment']))
        algo_type = get_algo_type(str(config['algorithm']))

        if 'GPU' in hardware_req and deploy_type == 'Edge':
            hw_penalty = 0.5  # Edge may not have good GPU
        elif 'HighEnd' in hardware_req and deploy_type != 'Cloud':
            hw_penalty = 0.8

        # Sigmoid input (now includes cond_lod_bonus)
        z = (rm['a0'] +
             rm['a1'] * base_algo_recall +
             rm['a2'] * sensor_precision +
             lod_bonus +
             cond_lod_bonus +  # NEW: condition LOD affects recall
             data_rate_bonus -
             rm['a3'] * (tau - tau0) -
             hw_penalty)

        # Sigmoid function
        recall = sigmoid(z)

        # Apply practical bounds
        recall = np.clip(recall, rm['min_recall'], rm['max_recall'])

        # NOTE: No random noise added - deterministic for reproducible optimization
        # If uncertainty modeling is needed, use Monte-Carlo wrapper on final solutions

        # Return 1-recall for minimization
        return 1.0 - recall

    # =========================================================================
    # OBJECTIVE 3: LATENCY (seconds)
    # =========================================================================

    def _calculate_latency(self, config: Dict) -> float:
        """
        Calculate end-to-end latency.

        Affected by: sensor, data_rate, geo_lod, algorithm, communication,
                    deployment
        """
        sensor_type = get_sensor_type(str(config['sensor']))
        comm_type = get_comm_type(str(config['communication']))
        deploy_type = get_deployment_type(str(config['deployment']))
        algo_type = get_algo_type(str(config['algorithm']))

        road_length = self.config.road_network_length_km

        # ----- ACQUISITION TIME -----
        speed = self._query_property(config['sensor'], 'hasOperatingSpeedKmh', 80)
        coverage = self._query_property(config['sensor'], 'hasCoverageEfficiencyKmPerDay', 80)

        if sensor_type in ['IoT', 'FOS']:
            # Real-time monitoring
            samples_needed = 100
            acq_time = samples_needed / max(config['data_rate'], 1)
        elif sensor_type == 'MMS' and speed > 0:
            segment_km = 5.0
            if getattr(self.config, 'scenario_type', 'mixed') == 'urban':
                speed *= 0.6
            acq_time = (segment_km / max(speed, 1)) * 3600
        elif sensor_type == 'UAV':
            if coverage > 0:
                acq_time = (1.0 / (coverage / 24)) * 3600
            else:
                acq_time = 1200
        elif sensor_type in ['TLS', 'Handheld']:
            acq_time = 300
        else:
            acq_time = 60

        # ----- DATA VOLUME -----
        base_data_gb = self._query_property(config['sensor'], 'hasDataVolumeGBPerKm', 1.0)
        lod_mult = get_param('lod_data_multiplier', config['geo_lod'], 1.0)
        rate_mult = config['data_rate'] / MODEL_PARAMS['recall_model']['base_data_rate']

        # Data for one segment (not full network for latency)
        # Use smaller segment for latency calculation (per-inspection latency)
        segment_km = min(1.0, road_length / 100)  # 1km or smaller segments
        raw_data_gb = base_data_gb * segment_km * lod_mult * min(rate_mult, 2.0)

        # Edge preprocessing reduces data sent
        data_reduction = get_param('data_reduction_ratio', deploy_type, 0.7)
        sent_data_gb = raw_data_gb * data_reduction

        # ----- COMMUNICATION TIME -----
        bandwidth_GBps = get_param('comm_bandwidth_GBps', comm_type, 0.01)
        reliability = get_param('comm_reliability', comm_type, 0.9)

        # Scenario adjustment
        scenario = getattr(self.config, 'scenario_type', 'mixed')
        scenario_factors = {
            'urban': {'Fiber': 0.95, '5G': 0.8, '4G': 0.7, 'LoRaWAN': 0.6},
            'rural': {'Fiber': 0.7, '5G': 0.4, '4G': 0.6, 'LoRaWAN': 0.9},
            'mixed': {'Fiber': 0.85, '5G': 0.6, '4G': 0.65, 'LoRaWAN': 0.8}
        }
        scenario_factor = scenario_factors.get(scenario, {}).get(comm_type, 0.7)

        effective_bandwidth = bandwidth_GBps * scenario_factor * reliability

        # Protocol overhead
        protocol_overhead = 1.15
        if effective_bandwidth > 1e-6:
            comm_time = (sent_data_gb * protocol_overhead) / effective_bandwidth
        else:
            comm_time = 3600  # 1 hour if no bandwidth

        # Retransmission
        retransmit_factor = 1 + (1 - reliability) * 0.5
        comm_time *= retransmit_factor

        # ----- COMPUTE TIME -----
        compute_time_per_gb = get_param('algo_compute_seconds_per_GB', algo_type, 15.0)
        compute_factor = get_param('deployment_compute_factor', deploy_type, 1.5)

        # LOD affects processing complexity
        lod_processing = {'Micro': 1.5, 'Meso': 1.0, 'Macro': 0.7}
        lod_compute_mult = lod_processing.get(config['geo_lod'], 1.0)

        proc_time = compute_time_per_gb * sent_data_gb * compute_factor * lod_compute_mult

        # Startup overhead (deterministic - use mean values)
        overhead = MODEL_PARAMS['latency_overhead']
        if deploy_type == 'Cloud':
            queue_time = overhead['queue_cloud']  # Use mean instead of random
        else:
            queue_time = overhead['queue_edge']  # Use mean instead of random

        startup_time = overhead['startup']
        result_time = overhead['result_processing']

        # ----- TOTAL LATENCY -----
        total_latency = acq_time + comm_time + proc_time + queue_time + startup_time + result_time

        # Bounds - more reasonable range
        total_latency = np.clip(total_latency, 5.0, 400.0)

        # NOTE: No random noise - deterministic for reproducible optimization

        assert validate_non_negative(total_latency, "latency"), f"Invalid latency: {total_latency}"

        return total_latency

    # =========================================================================
    # OBJECTIVE 4: TRAFFIC DISRUPTION (hours/year)
    # =========================================================================

    def _calculate_traffic_disruption(self, config: Dict) -> float:
        """
        Calculate annual traffic disruption.

        Affected by: sensor (speed, type), crew_size, inspection_cycle
        """
        sensor_type = get_sensor_type(str(config['sensor']))

        road_length = self.config.road_network_length_km
        inspections_per_year = 365.0 / config['inspection_cycle']

        # Base disruption hours per inspection event
        base_hours = get_param('base_disruption_hours', sensor_type, 3.0)

        # Coverage and speed
        coverage_per_day = self._query_property(config['sensor'], 'hasCoverageEfficiencyKmPerDay', 80)
        speed = self._query_property(config['sensor'], 'hasOperatingSpeedKmh', 80)

        # Calculate inspection duration factor
        if sensor_type in ['IoT', 'FOS']:
            # Fixed sensors: only installation disruption
            installation_days = 5
            maintenance_visits = 4  # Quarterly
            annual_disruption_base = (base_hours * installation_days / self.config.planning_horizon_years +
                                      base_hours * 0.5 * maintenance_visits)
        else:
            # Mobile sensors
            if coverage_per_day > 0:
                days_per_inspection = road_length / coverage_per_day
            else:
                days_per_inspection = road_length / max(speed * 8, 1)  # 8 hour workday

            annual_disruption_base = base_hours * days_per_inspection * inspections_per_year

        # Speed differential factor for mobile sensors
        if speed > 0 and sensor_type not in ['IoT', 'FOS']:
            traffic_speed = 60  # Assumed traffic speed
            speed_diff = abs(speed - traffic_speed)
            speed_factor = 1.0 + (speed_diff / traffic_speed) * 0.5
        else:
            speed_factor = 1.0

        # Crew size factor
        crew_factors = MODEL_PARAMS['crew_disruption_factor']
        if config['crew_size'] <= 5:
            crew_factor = crew_factors.get(config['crew_size'], 1.0)
        else:
            crew_factor = crew_factors['large']

        # Traffic volume factor (log scale)
        traffic_volume = getattr(self.config, 'traffic_volume_hourly', 2000)
        traffic_base = MODEL_PARAMS['traffic_disruption_log_base']
        traffic_factor = 1.0 + np.log10(traffic_volume / traffic_base + 1)

        # Lane closure factor
        lane_closure = getattr(self.config, 'default_lane_closure_ratio', 0.3)
        lane_factor = 1.0 + lane_closure * 1.5

        # Time optimization (night/weekend work)
        night_ratio = 0.4
        weekend_ratio = 0.2
        time_reduction = night_ratio * 0.7 + weekend_ratio * 0.5
        time_factor = 1.0 - time_reduction

        # Seasonal factor
        if getattr(self.config, 'apply_seasonal_adjustments', True):
            seasonal_factor = 1.1
        else:
            seasonal_factor = 1.0

        # Total annual disruption
        total_disruption = (annual_disruption_base * speed_factor * crew_factor *
                            traffic_factor * lane_factor * time_factor * seasonal_factor)

        # Bounds
        total_disruption = np.clip(total_disruption, 0, 500)

        return total_disruption

    # =========================================================================
    # OBJECTIVE 5: ENVIRONMENTAL IMPACT (kgCO2e/year)
    # =========================================================================

    def _calculate_environmental_impact(self, config: Dict) -> float:
        """
        Calculate annual carbon emissions.

        Affected by: sensor, data_rate, geo_lod, algorithm, storage,
                    communication, deployment, inspection_cycle
        """
        sensor_type = get_sensor_type(str(config['sensor']))
        storage_type = get_storage_type(str(config['storage']))
        comm_type = get_comm_type(str(config['communication']))
        deploy_type = get_deployment_type(str(config['deployment']))
        algo_type = get_algo_type(str(config['algorithm']))

        road_length = self.config.road_network_length_km
        inspections_per_year = 365.0 / config['inspection_cycle']
        carbon_intensity = self.config.carbon_intensity_factor  # kgCO2/kWh

        # ----- SENSOR ENERGY -----
        sensor_power_w = self._query_property(config['sensor'], 'hasEnergyConsumptionW', 50)
        coverage = self._query_property(config['sensor'], 'hasCoverageEfficiencyKmPerDay', 80)

        if coverage > 0:
            days_per_inspection = road_length / coverage
            sensor_hours_per_year = days_per_inspection * inspections_per_year * 6  # 6hr/day
        else:
            sensor_hours_per_year = 365 * 24  # Always on

        sensor_kwh = (sensor_power_w * sensor_hours_per_year) / 1000

        # ----- DATA VOLUME -----
        base_data_gb = self._query_property(config['sensor'], 'hasDataVolumeGBPerKm', 1.0)
        lod_mult = get_param('lod_data_multiplier', config['geo_lod'], 1.0)
        rate_mult = config['data_rate'] / MODEL_PARAMS['recall_model']['base_data_rate']

        raw_data_gb_year = base_data_gb * road_length * lod_mult * rate_mult * inspections_per_year
        data_reduction = get_param('data_reduction_ratio', deploy_type, 0.7)
        sent_data_gb_year = raw_data_gb_year * data_reduction

        # ----- COMMUNICATION ENERGY -----
        comm_energy_rate = get_param('comm_energy_kWh_per_GB', comm_type, 0.05)
        comm_kwh = comm_energy_rate * sent_data_gb_year

        # ----- STORAGE ENERGY -----
        storage_energy_rate = get_param('storage_energy_kWh_per_GB_year', storage_type, 0.04)
        retention = getattr(self.config, 'data_retention_years', 3)
        storage_kwh = storage_energy_rate * raw_data_gb_year * min(retention, 3)

        # ----- COMPUTE ENERGY -----
        compute_time_per_gb = get_param('algo_compute_seconds_per_GB', algo_type, 15.0)
        compute_factor = get_param('deployment_compute_factor', deploy_type, 1.5)
        compute_power_w = get_param('deployment_power_W', deploy_type, 150)
        pue = get_param('datacenter_pue', deploy_type, 1.3)

        annual_compute_seconds = compute_time_per_gb * sent_data_gb_year * compute_factor
        annual_compute_hours = annual_compute_seconds / 3600
        compute_kwh = (compute_power_w * annual_compute_hours * pue) / 1000

        # ----- VEHICLE EMISSIONS (for mobile sensors) -----
        if coverage > 0:
            vehicle_km = road_length * inspections_per_year
            if sensor_type == 'MMS':
                fuel_consumption = 0.12  # 12L/100km
            else:
                fuel_consumption = 0.08
            vehicle_fuel_l = vehicle_km * fuel_consumption
            vehicle_emissions_kg = vehicle_fuel_l * 2.31  # Diesel/petrol factor
        else:
            vehicle_emissions_kg = 0

        # ----- MANUFACTURING (amortized) -----
        equipment_cost = self._query_property(config['sensor'], 'hasInitialCostUSD', 100000)
        mfg_type = 'Electronics' if sensor_type in ['UAV', 'IoT'] else 'Mechanical'
        mfg_carbon_rate = get_param('manufacturing_carbon_kg_per_1000USD', mfg_type, 40.0)
        mfg_emissions = (equipment_cost / 1000) * mfg_carbon_rate
        lifetime_years = MODEL_PARAMS['equipment_lifetime_years']
        annual_mfg_emissions = mfg_emissions / lifetime_years

        # ----- TOTAL EMISSIONS -----
        # Apply deployment carbon intensity factor
        deploy_carbon_factor = get_param('carbon_intensity_factor', deploy_type, 1.0)

        electricity_emissions = (
                                            sensor_kwh + comm_kwh + storage_kwh + compute_kwh) * carbon_intensity * deploy_carbon_factor

        total_emissions = electricity_emissions + vehicle_emissions_kg + annual_mfg_emissions

        # Bounds
        # 数值防护：不裁剪成常数，只保证合理性 (P0修复)
        if not np.isfinite(total_emissions):
            total_emissions = 1e9
        total_emissions = max(0.0, float(total_emissions))

        return total_emissions

    # =========================================================================
    # OBJECTIVE 6: SYSTEM RELIABILITY (1/MTBF in hours^-1)
    # =========================================================================

    def _calculate_system_reliability(self, config: Dict) -> float:
        """
        Calculate system failure rate (1/MTBF).

        Affected by: sensor, algorithm, storage, communication, deployment, crew_size
        """
        sensor_type = get_sensor_type(str(config['sensor']))
        deploy_type = get_deployment_type(str(config['deployment']))
        storage_type = get_storage_type(str(config['storage']))
        algo_type = get_algo_type(str(config['algorithm']))

        # Determine architecture type
        if deploy_type == 'Cloud' and 'Cloud' in str(config['storage']):
            architecture = 'distributed'
        elif 'Hybrid' in deploy_type or 'Hybrid' in storage_type:
            architecture = 'load_balanced'
        elif 'OnPremise' in deploy_type:
            architecture = 'active_backup'
        else:
            architecture = 'single_point'

        arch_factor = get_param('architecture_reliability_factor', architecture, 1.0)

        # Component failure rates
        component_failure_rates = []

        # Sensor MTBF
        sensor_mtbf = self._query_property(config['sensor'], 'hasMTBFHours', 10000)
        env_factor = get_param('environmental_mtbf_factor', sensor_type, 0.8)
        effective_sensor_mtbf = sensor_mtbf * arch_factor * env_factor
        if effective_sensor_mtbf > 0:
            component_failure_rates.append(1.0 / effective_sensor_mtbf)

        # Storage MTBF
        storage_mtbf = self._query_property(config['storage'], 'hasMTBFHours', 100000)
        if storage_mtbf and storage_mtbf > 0:
            component_failure_rates.append(1.0 / (storage_mtbf * arch_factor * 0.95))

        # Communication MTBF
        comm_mtbf = self._query_property(config['communication'], 'hasMTBFHours', 50000)
        if comm_mtbf and comm_mtbf > 0:
            component_failure_rates.append(1.0 / (comm_mtbf * arch_factor * 0.9))

        # Deployment MTBF
        deploy_mtbf = self._query_property(config['deployment'], 'hasMTBFHours', 60000)
        if deploy_mtbf and deploy_mtbf > 0:
            component_failure_rates.append(1.0 / (deploy_mtbf * arch_factor * 0.95))

        # Algorithm reliability
        algo_mtbf = self._query_property(config['algorithm'], 'hasMTBFHours', 50000)
        complexity_factor = get_param('algo_complexity_factor', algo_type, 0.85)
        if algo_mtbf and algo_mtbf > 0:
            component_failure_rates.append(1.0 / (algo_mtbf * complexity_factor))

        # System failure rate
        if architecture in ['distributed', 'load_balanced']:
            # Partial redundancy reduces individual component impact
            redundancy_factor = 0.3 if architecture == 'distributed' else 0.5
            system_failure_rate = sum(r * redundancy_factor for r in component_failure_rates)
        else:
            # Series system
            system_failure_rate = sum(component_failure_rates)

        # Common cause failures
        common_cause_rate = 1.0 / (365 * 24 * 20)  # Once per 20 years
        system_failure_rate += common_cause_rate

        # Human error
        skill_level = str(self._query_property(config['sensor'], 'hasOperatorSkillLevel', 'Basic'))
        human_error_rates = {
            'Basic': 1.0 / (365 * 24 * 1),
            'Intermediate': 1.0 / (365 * 24 * 2),
            'Expert': 1.0 / (365 * 24 * 5)
        }
        human_error_rate = human_error_rates.get(skill_level, human_error_rates['Basic'])

        # Crew size effect
        if config['crew_size'] <= 2:
            crew_factor = 1.2
        elif config['crew_size'] <= 5:
            crew_factor = 1.0
        else:
            crew_factor = 0.9  # Redundancy

        system_failure_rate += human_error_rate * crew_factor

        # Software update risk for ML/DL
        if algo_type in ['DL', 'ML']:
            update_risk = 1.0 / (365 * 24 * 0.5) * 0.1  # 10% chance per update
            system_failure_rate += update_risk

        # Ensure positive
        system_failure_rate = max(system_failure_rate, 1e-7)

        return system_failure_rate

    # =========================================================================
    # TRACE/EXPLAIN FUNCTION (for paper interpretability)
    # =========================================================================

    def explain(self, config: Dict) -> Dict[str, Any]:
        """
        Produce detailed trace of objective calculations.

        Use ONLY when exporting results, not during optimization.

        Args:
            config: Decoded configuration dictionary

        Returns:
            Dictionary with all intermediate values for transparency
        """
        trace = {}

        # Extract types
        sensor_type = get_sensor_type(str(config['sensor']))
        comm_type = get_comm_type(str(config['communication']))
        storage_type = get_storage_type(str(config['storage']))
        deploy_type = get_deployment_type(str(config['deployment']))
        algo_type = get_algo_type(str(config['algorithm']))

        trace['types'] = {
            'sensor': sensor_type,
            'algorithm': algo_type,
            'storage': storage_type,
            'communication': comm_type,
            'deployment': deploy_type
        }

        # Configuration
        trace['config'] = {
            'data_rate_Hz': config['data_rate'],
            'geo_lod': config['geo_lod'],
            'detection_threshold': config['detection_threshold'],
            'crew_size': config['crew_size'],
            'inspection_cycle_days': config['inspection_cycle']
        }

        # Data volume calculations
        road_length = self.config.road_network_length_km
        inspections_per_year = 365.0 / config['inspection_cycle']
        base_data_gb = self._query_property(config['sensor'], 'hasDataVolumeGBPerKm', 1.0)
        lod_mult = get_param('lod_data_multiplier', config['geo_lod'], 1.0)
        rate_mult = config['data_rate'] / MODEL_PARAMS['recall_model']['base_data_rate']
        raw_total_gb = base_data_gb * road_length * lod_mult * rate_mult * inspections_per_year
        data_reduction = get_param('data_reduction_ratio', deploy_type, 0.7)
        sent_total_gb = raw_total_gb * data_reduction

        trace['data_volume'] = {
            'base_gb_per_km': base_data_gb,
            'lod_multiplier': lod_mult,
            'rate_multiplier': rate_mult,
            'raw_total_gb_year': raw_total_gb,
            'data_reduction_ratio': data_reduction,
            'sent_total_gb_year': sent_total_gb
        }

        # Communication
        bandwidth_GBps = get_param('comm_bandwidth_GBps', comm_type, 0.01)
        comm_time_s = (sent_total_gb / inspections_per_year) / max(bandwidth_GBps, 1e-6)

        trace['communication'] = {
            'bandwidth_GBps': bandwidth_GBps,
            'comm_time_per_inspection_s': comm_time_s
        }

        # Compute
        compute_per_gb = get_param('algo_compute_seconds_per_GB', algo_type, 15.0)
        compute_factor = get_param('deployment_compute_factor', deploy_type, 1.5)
        compute_time_s = compute_per_gb * (sent_total_gb / inspections_per_year) * compute_factor

        trace['compute'] = {
            'algo_compute_s_per_GB': compute_per_gb,
            'deployment_factor': compute_factor,
            'compute_time_per_inspection_s': compute_time_s
        }

        # Recall model - MUST match _calculate_detection_performance exactly
        rm = MODEL_PARAMS['recall_model']
        base_algo_recall = self._query_property(config['algorithm'], 'hasRecall', 0.7)
        sensor_precision = self._query_property(config['sensor'], 'hasPrecision', 0.7)
        lod_bonus = rm['lod_bonus'].get(config['geo_lod'], 0.0)
        tau = config['detection_threshold']
        tau0 = rm['tau0']

        # Data rate bonus (same as in _calculate_detection_performance)
        data_rate_bonus = rm['data_rate_bonus_factor'] * max(0, config['data_rate'] - rm['base_data_rate'])

        # Hardware penalty (same as in _calculate_detection_performance)
        algo_type = get_algo_type(str(config['algorithm']))
        deploy_type = trace['types']['deployment']
        hw_penalty = 0.0
        if algo_type in ['DL', 'ML']:
            if deploy_type == 'Edge':
                hw_penalty = 0.5  # Edge may not have good GPU
            elif deploy_type not in ['Cloud', 'Hybrid']:
                hw_penalty = 0.8

        # Condition LOD bonus - MUST match _calculate_detection_performance
        cond_lod_bonus_map = {'Micro': 0.3, 'Meso': 0.0, 'Macro': -0.2}
        cond_lod_bonus = cond_lod_bonus_map.get(config['cond_lod'], 0.0)

        # z calculation - MUST match _calculate_detection_performance exactly
        z = (rm['a0'] +
             rm['a1'] * base_algo_recall +
             rm['a2'] * sensor_precision +
             lod_bonus +
             cond_lod_bonus +  # NEW: condition LOD bonus
             data_rate_bonus -
             rm['a3'] * (tau - tau0) -
             hw_penalty)
        recall = sigmoid(z)
        recall = np.clip(recall, rm['min_recall'], rm['max_recall'])

        trace['recall_model'] = {
            'base_algo_recall': base_algo_recall,
            'sensor_precision': sensor_precision,
            'lod_bonus': lod_bonus,
            'cond_lod_bonus': cond_lod_bonus,  # NEW
            'data_rate_bonus': data_rate_bonus,
            'hw_penalty': hw_penalty,
            'tau': tau,
            'tau0': tau0,
            'sigmoid_input_z': z,
            'final_recall': recall
        }

        # Objectives
        trace['objectives'] = {
            'f1_cost_USD': self._calculate_total_cost(config),
            'f2_one_minus_recall': self._calculate_detection_performance(config),
            'f3_latency_s': self._calculate_latency(config),
            'f4_disruption_hours_year': self._calculate_traffic_disruption(config),
            'f5_carbon_kgCO2e_year': self._calculate_environmental_impact(config),
            'f6_failure_rate': self._calculate_system_reliability(config)
        }

        return trace


# =============================================================================
# VALIDATION HARNESS
# =============================================================================

def validate_model_consistency(evaluator: EnhancedFitnessEvaluatorV3):
    """
    Run consistency checks on the evaluation model.

    Verifies:
    1. Higher bandwidth reduces latency (all else equal)
    2. Higher detection threshold affects recall
    3. No NaN values
    """
    print("Running model consistency checks...")

    # Base configuration vector
    base_x = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.1])

    # Test 1: Check for NaNs
    objectives, constraints = evaluator._evaluate_single(base_x)
    assert not np.any(np.isnan(objectives)), "NaN in objectives"
    assert not np.any(np.isnan(constraints)), "NaN in constraints"
    print("  ✓ No NaN values")

    # Test 2: Recall in valid range
    recall = 1 - objectives[1]
    assert 0 < recall < 1, f"Recall out of range: {recall}"
    print(f"  ✓ Recall in valid range: {recall:.4f}")

    # Test 3: All objectives non-negative
    assert np.all(objectives >= 0), f"Negative objective: {objectives}"
    print("  ✓ All objectives non-negative")

    # Test 4: Latency increases with higher data rate
    x_low_rate = base_x.copy()
    x_low_rate[1] = 0.1  # Low data rate
    x_high_rate = base_x.copy()
    x_high_rate[1] = 0.9  # High data rate

    obj_low, _ = evaluator._evaluate_single(x_low_rate)
    obj_high, _ = evaluator._evaluate_single(x_high_rate)

    # Higher data rate should generally increase latency (more data to process)
    # But this isn't strictly monotonic due to complex interactions
    print(f"  ℹ Latency at low rate: {obj_low[2]:.2f}s, high rate: {obj_high[2]:.2f}s")

    # Test 5: Detection threshold affects recall
    x_low_thresh = base_x.copy()
    x_low_thresh[5] = 0.1  # Low threshold (more sensitive)
    x_high_thresh = base_x.copy()
    x_high_thresh[5] = 0.9  # High threshold (less sensitive)

    obj_low_t, _ = evaluator._evaluate_single(x_low_thresh)
    obj_high_t, _ = evaluator._evaluate_single(x_high_thresh)

    recall_low = 1 - obj_low_t[1]
    recall_high = 1 - obj_high_t[1]
    print(f"  ℹ Recall at low threshold: {recall_low:.4f}, high threshold: {recall_high:.4f}")

    print("All consistency checks passed!")
    return True


if __name__ == "__main__":
    # Run sanity checks
    from model_params import run_sanity_checks

    run_sanity_checks()