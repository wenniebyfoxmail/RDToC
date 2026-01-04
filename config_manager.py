#!/usr/bin/env python3
"""
配置管理器 - 修复版
调整关键约束参数以确保找到可行解
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict, fields
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConfigManager:
    """中央配置管理 - 调整了关键约束"""

    # 文件路径
    config_file: str = 'config.json'
    sensor_csv: str = 'sensors_data.txt'
    algorithm_csv: str = 'algorithms_data.txt'
    infrastructure_csv: str = 'infrastructure_data.txt'
    cost_benefit_csv: str = 'cost_benefit_data.txt'

    # 网络参数
    road_network_length_km: float = 500.0
    planning_horizon_years: int = 10
    budget_cap_usd: float = 20_000_000  # 保持原始预算

    # 运营参数
    daily_wage_per_person: float = 500
    fos_sensor_spacing_km: float = 0.1
    fixed_sensor_density_per_km: float = 1.0  # 【新增】固定传感器密度，每km安装数量
    mobile_km_per_unit_per_day: float = 80.0  # 移动设备每天可覆盖里程(km)
    depreciation_rate: float = 0.1
    scenario_type: str = 'urban'
    carbon_intensity_factor: float = 0.417

    # 约束 - 关键调整以找到可行解
    min_recall_threshold: float = 0.9  # 从0.70进一步降低
    max_latency_seconds: float = 500.0  # 从180增加到300秒
    max_disruption_hours: float = 300.0  # 从100增加到300
    max_energy_kwh_year: float = 200_000  # 从50k增加到150k
    min_mtbf_hours: float = 1_000  # 从5000降低到2000（约3个月）
    max_carbon_emissions_kgCO2e_year: float = 300_000  # 增加到200k
    # v3.2: 检查周期约束 - 使用min_inspections_per_year
    min_inspections_per_year: int = 4  # 每年至少检查4次 → 最大周期91天

    # 额外的运营参数
    apply_seasonal_adjustments: bool = True
    traffic_volume_hourly: int = 2000
    default_lane_closure_ratio: float = 0.3

    # 高级参数
    class_imbalance_penalties: Dict[str, float] = field(default_factory=lambda: {
        'Traditional': 0.05,
        'ML': 0.02,
        'DL': 0.01,
        'PC': 0.03
    })

    network_quality_factors: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'rural': {'Fiber': 0.8, '5G': 0.7, '4G': 0.9, 'LoRaWAN': 1.0},
        'urban': {'Fiber': 1.0, '5G': 1.0, '4G': 1.0, 'LoRaWAN': 0.9},
        'mixed': {'Fiber': 0.9, '5G': 0.85, '4G': 0.95, 'LoRaWAN': 0.95}
    })

    redundancy_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'Cloud': 10.0,
        'OnPremise': 1.5,
        'Edge': 2.0,
        'Hybrid': 5.0
    })

    # 优化参数
    n_objectives: int = 6
    n_partitions: int = 4  # 添加这一行
    population_size: int = 2000
    n_generations: int = 100
    crossover_prob: float = 0.9
    crossover_eta: float = 20
    mutation_eta: float = 20
    random_seed: int = 42  # Seed for reproducibility across all methods

    # 基线参数
    n_random_samples: int = 500
    grid_resolution: int = 5
    weight_combinations: int = 50

    # 并行计算
    use_parallel: bool = True
    n_processes: int = -1  # -1表示使用所有可用核心-1

    # 输出设置
    output_dir: Path = field(default_factory=lambda: Path('./results'))
    log_dir: Path = field(default_factory=lambda: Path('./results/logs'))
    figure_format: List[str] = field(default_factory=lambda: ['png', 'pdf'])

    # 调试选项
    data_retention_years: int = 3
    enable_debug_output: bool = True

    @classmethod
    def from_json(cls, json_path: str) -> 'ConfigManager':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)

        # Filter only known fields
        known_fields = {f.name for f in fields(cls)}
        filtered_dict = {}

        for key, value in config_dict.items():
            if key in known_fields:
                filtered_dict[key] = value
            else:
                logger.warning(f"未知配置键: {key}")

        return cls(**filtered_dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary with JSON-serializable values."""
        result = {}
        for f in self.__dataclass_fields__:
            value = getattr(self, f)
            # 将 Path 对象转换为字符串
            if hasattr(value, '__fspath__'):  # Path-like object
                result[f] = str(value)
            else:
                result[f] = value
        return result

    def save_snapshot(self, path: str):
        """Save configuration snapshot."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def __post_init__(self):
        """初始化后处理"""
        # 加载配置文件（如果存在）
        if Path(self.config_file).exists():
            self.load_from_file(self.config_file)

        # 处理计算值
        self._process_config()

        # 创建目录
        self._create_directories()

    def load_from_file(self, filepath: str):
        """从JSON文件加载配置"""
        logger.info(f"从 {filepath} 加载配置")

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # 获取数据类的所有字段名
            field_names = {f.name for f in fields(self)}

            # 更新属性
            for key, value in data.items():
                if key in field_names:
                    # 处理Path对象
                    if key in ['output_dir', 'log_dir']:
                        value = Path(value)
                    setattr(self, key, value)
                    logger.debug(f"设置 {key} = {value}")
                else:
                    logger.warning(f"未知配置键: {key}")

        except Exception as e:
            logger.error(f"加载配置时出错: {e}")
            logger.info("使用默认配置")

    def _process_config(self):
        """处理和验证配置"""
        # 设置计算默认值
        import multiprocessing as mp
        if self.n_processes == -1:
            self.n_processes = max(1, mp.cpu_count() - 1)

        # 确保路径是Path对象
        self.output_dir = Path(self.output_dir)
        self.log_dir = Path(self.log_dir)

        # 验证数值范围
        if self.population_size < 10:
            logger.warning(f"种群大小 {self.population_size} 太小")

        if self.n_generations < 10:
            logger.warning(f"代数 {self.n_generations} 太少")

        if self.budget_cap_usd < 100_000:
            logger.warning(f"预算上限 ${self.budget_cap_usd} 可能太严格")

        # 验证文件存在性
        for csv_attr in ['sensor_csv', 'algorithm_csv', 'infrastructure_csv', 'cost_benefit_csv']:
            filepath = getattr(self, csv_attr)
            if not Path(filepath).exists():
                logger.error(f"找不到所需的数据文件: {filepath}")

    def _create_directories(self):
        """创建输出目录"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'baseline').mkdir(exist_ok=True)

    def save_to_file(self, filepath: Optional[str] = None):
        """保存配置到JSON文件"""
        filepath = filepath or self.config_file

        # 转换为字典
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                # 转换Path对象为字符串
                if isinstance(value, Path):
                    value = str(value)
                config_dict[key] = value

        # 保存
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"配置保存到 {filepath}")

    def get_summary(self) -> str:
        """获取配置摘要"""
        summary = [
            "配置摘要:",
            f"  网络: {self.road_network_length_km} km",
            f"  预算: ${self.budget_cap_usd:,.0f}",
            f"  规划期: {self.planning_horizon_years} 年",
            f"  目标数: {self.n_objectives}",
            f"  算法: NSGA-II/III",
            f"  种群: {self.population_size}",
            f"  代数: {self.n_generations}",
            f"  并行处理: {self.use_parallel} ({self.n_processes} 核心)",
            f"  输出目录: {self.output_dir}"
        ]
        return '\n'.join(summary)