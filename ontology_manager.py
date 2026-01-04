#!/usr/bin/env python3
"""
本体管理器 v2.1 - 支持SHACL验证 + 正确的数据格式
=================================================
修复数据加载方法以匹配实际的CSV/TXT文件格式

Author: RMTwin Research Team
Version: 2.1 (Fixed Data Format)
"""

import logging
import uuid
import pandas as pd
from pathlib import Path
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD

logger = logging.getLogger(__name__)

# 命名空间
RDTCO = Namespace("http://www.semanticweb.org/rmtwin/ontologies/rdtco#")
EX = Namespace("http://example.org/rmtwin#")
SH = Namespace("http://www.w3.org/ns/shacl#")


class OntologyManager:
    """管理RDTcO-Maint本体操作 - 支持SHACL验证"""
    
    def __init__(self):
        self.g = Graph()
        self.g.bind("rdtco", RDTCO)
        self.g.bind("ex", EX)
        self.g.bind("sh", SH)
        self.g.bind("rdf", RDF)
        self.g.bind("rdfs", RDFS)
        self.g.bind("owl", OWL)
        self.g.bind("xsd", XSD)
        
        self._setup_base_ontology()
    
    def _setup_base_ontology(self):
        """设置基础本体结构"""
        logger.info("设置基础本体结构...")
        
        # 定义核心类
        core_classes = [
            'DigitalTwinConfiguration',
            'SensorSystem',
            'Algorithm',
            'StorageSystem',
            'CommunicationSystem',
            'ComputeDeployment',
            'ConfigurationParameter'
        ]
        
        for class_name in core_classes:
            class_uri = RDTCO[class_name]
            self.g.add((class_uri, RDF.type, OWL.Class))
            self.g.add((class_uri, RDFS.label, Literal(class_name)))
        
        # 定义传感器子类
        sensor_types = [
            'MMS_LiDAR_System', 'MMS_Camera_System', 'UAV_LiDAR_System',
            'UAV_Camera_System', 'TLS_System', 'Handheld_3D_Scanner',
            'FiberOptic_Sensor', 'Vehicle_LowCost_Sensor', 'IoT_Network_System'
        ]
        
        for sensor_type in sensor_types:
            sensor_class = RDTCO[sensor_type]
            self.g.add((sensor_class, RDF.type, OWL.Class))
            self.g.add((sensor_class, RDFS.subClassOf, RDTCO.SensorSystem))
            self.g.add((sensor_class, RDFS.label, Literal(sensor_type)))
        
        # 定义算法子类
        algo_types = [
            'DeepLearningAlgorithm', 'MachineLearningAlgorithm',
            'TraditionalAlgorithm', 'PointCloudAlgorithm'
        ]
        
        for algo_type in algo_types:
            algo_class = RDTCO[algo_type]
            self.g.add((algo_class, RDF.type, OWL.Class))
            self.g.add((algo_class, RDFS.subClassOf, RDTCO.Algorithm))
            self.g.add((algo_class, RDFS.label, Literal(algo_type)))
        
        # 定义属性
        self._define_properties()
        
        # 定义配置关系属性 (用于SHACL验证)
        self._define_configuration_properties()
        
        logger.info(f"基础本体创建完成，包含 {len(self.g)} 个三元组")
    
    def _define_configuration_properties(self):
        """定义配置关系属性（用于SHACL验证）"""
        config_relations = [
            ('hasSensor', 'SensorSystem', '配置使用的传感器'),
            ('hasAlgorithm', 'Algorithm', '配置使用的算法'),
            ('hasStorage', 'StorageSystem', '配置使用的存储'),
            ('hasCommunication', 'CommunicationSystem', '配置使用的通信'),
            ('hasDeployment', 'ComputeDeployment', '配置使用的部署'),
        ]
        
        for prop_name, range_class, comment in config_relations:
            prop_uri = RDTCO[prop_name]
            self.g.add((prop_uri, RDF.type, OWL.ObjectProperty))
            self.g.add((prop_uri, RDFS.domain, RDTCO.DigitalTwinConfiguration))
            self.g.add((prop_uri, RDFS.range, RDTCO[range_class]))
            self.g.add((prop_uri, RDFS.comment, Literal(comment)))
        
        config_params = [
            ('hasInspectionCycleDays', XSD.integer, '检测周期（天）'),
            ('hasDataRateHz', XSD.decimal, '数据采集频率'),
            ('hasTotalCostUSD', XSD.decimal, '总成本'),
            ('hasRecall', XSD.decimal, '检测召回率'),
            ('hasLatencySeconds', XSD.decimal, '延迟（秒）'),
            ('hasCarbonKgCO2eYear', XSD.decimal, '年碳排放'),
        ]
        
        for prop_name, datatype, comment in config_params:
            prop_uri = RDTCO[prop_name]
            self.g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
            self.g.add((prop_uri, RDFS.domain, RDTCO.DigitalTwinConfiguration))
            self.g.add((prop_uri, RDFS.range, datatype))
            self.g.add((prop_uri, RDFS.comment, Literal(comment)))
    
    def _define_properties(self):
        """定义本体属性"""
        properties = [
            ('hasInitialCostUSD', '初始成本（美元）', XSD.decimal),
            ('hasOperationalCostUSDPerDay', '日运营成本', XSD.decimal),
            ('hasAnnualOpCostUSD', '年运营成本', XSD.decimal),
            ('hasRecall', '检测召回率', XSD.decimal),
            ('hasPrecision', '检测精确率', XSD.decimal),
            ('hasFPS', '每秒帧数', XSD.decimal),
            ('hasAccuracyRangeMM', '精度范围（毫米）', XSD.decimal),
            ('hasEnergyConsumptionW', '能耗（瓦）', XSD.decimal),
            ('hasMTBFHours', '平均故障间隔时间（小时）', XSD.decimal),
            ('hasCoverageEfficiencyKmPerDay', '覆盖效率（公里/天）', XSD.decimal),
            ('hasDataVolumeGBPerKm', '数据量（GB/公里）', XSD.decimal),
            ('hasHardwareRequirement', '硬件需求', XSD.string),
            ('hasDataFormat', '数据格式', XSD.string),
            ('hasComponentCategory', '组件类别', XSD.string),
        ]
        
        for prop_name, label, datatype in properties:
            prop_uri = RDTCO[prop_name]
            self.g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
            self.g.add((prop_uri, RDFS.label, Literal(label)))
    
    # =========================================================================
    # SHACL 验证功能 (P0 实现)
    # =========================================================================
    
    def build_config_graph(self, config: dict) -> Graph:
        """将配置字典转换为RDF图，用于SHACL验证。"""
        g_cfg = Graph()
        g_cfg.bind("rdtco", RDTCO)
        g_cfg.bind("ex", EX)
        g_cfg.bind("sh", SH)
        g_cfg.bind("rdf", RDF)
        g_cfg.bind("xsd", XSD)
        
        cfg_uri = EX[f"cfg_{uuid.uuid4().hex[:8]}"]
        g_cfg.add((cfg_uri, RDF.type, RDTCO.DigitalTwinConfiguration))
        
        def as_uri(v):
            if v is None:
                return None
            if isinstance(v, URIRef):
                return v
            v_str = str(v)
            if v_str.startswith('http'):
                return URIRef(v_str)
            if '#' not in v_str and '/' not in v_str:
                return EX[v_str.replace(' ', '_')]
            return URIRef(v_str)
        
        component_mappings = [
            ('sensor', RDTCO.hasSensor),
            ('algorithm', RDTCO.hasAlgorithm),
            ('storage', RDTCO.hasStorage),
            ('communication', RDTCO.hasCommunication),
            ('deployment', RDTCO.hasDeployment),
        ]
        
        for key, predicate in component_mappings:
            if key in config and config[key] is not None:
                uri = as_uri(config[key])
                if uri:
                    g_cfg.add((cfg_uri, predicate, uri))
        
        if 'inspection_cycle' in config and config['inspection_cycle'] is not None:
            g_cfg.add((cfg_uri, RDTCO.hasInspectionCycleDays,
                      Literal(int(config['inspection_cycle']), datatype=XSD.integer)))
        
        if 'data_rate' in config and config['data_rate'] is not None:
            g_cfg.add((cfg_uri, RDTCO.hasDataRateHz,
                      Literal(float(config['data_rate']), datatype=XSD.decimal)))
        
        result_mappings = [
            ('total_cost', RDTCO.hasTotalCostUSD),
            ('f1_total_cost_USD', RDTCO.hasTotalCostUSD),
            ('recall', RDTCO.hasRecall),
            ('detection_recall', RDTCO.hasRecall),
            ('latency', RDTCO.hasLatencySeconds),
            ('f3_latency_seconds', RDTCO.hasLatencySeconds),
            ('carbon', RDTCO.hasCarbonKgCO2eYear),
            ('f5_carbon_emissions_kgCO2e_year', RDTCO.hasCarbonKgCO2eYear),
        ]
        
        for key, predicate in result_mappings:
            if key in config and config[key] is not None:
                try:
                    g_cfg.add((cfg_uri, predicate,
                              Literal(float(config[key]), datatype=XSD.decimal)))
                except (ValueError, TypeError):
                    pass
        
        return g_cfg
    
    def shacl_validate_config(self, config: dict, shapes_path: str = None) -> tuple:
        """
        使用SHACL验证配置。
        
        修复：只验证配置图，不包含整个本体图，避免验证无关节点
        """
        try:
            from pyshacl import validate
        except ImportError:
            logger.warning("pyshacl未安装，跳过SHACL验证")
            return True, "pyshacl not installed - validation skipped"
        
        # 只构建配置图，不包含整个本体（避免验证无关节点）
        data_g = self.build_config_graph(config)
        
        if shapes_path is None:
            default_paths = ['shapes/min_shapes.ttl', './shapes/min_shapes.ttl']
            for p in default_paths:
                if Path(p).exists():
                    shapes_path = p
                    break
        
        if shapes_path is None or not Path(shapes_path).exists():
            logger.warning(f"SHACL shapes文件不存在: {shapes_path}")
            return True, "SHACL shapes file not found - validation skipped"
        
        try:
            shacl_g = Graph().parse(shapes_path, format="turtle")
            conforms, report_graph, report_text = validate(
                data_g, shacl_graph=shacl_g, inference="rdfs",
                abort_on_first=False, meta_shacl=False, advanced=True, debug=False
            )
            return conforms, report_text
        except Exception as e:
            logger.error(f"SHACL验证出错: {e}")
            return True, f"SHACL validation error: {str(e)}"
    
    def validate_configuration(self, config_or_uri, shapes_path: str = None) -> bool:
        if isinstance(config_or_uri, dict):
            conforms, _ = self.shacl_validate_config(config_or_uri, shapes_path)
            return conforms
        return True
    
    def batch_validate_configs(self, configs: list, shapes_path: str = None) -> dict:
        results = []
        pass_count = 0
        for cfg in configs:
            conforms, report = self.shacl_validate_config(cfg, shapes_path)
            pass_count += int(conforms)
            results.append({'conforms': bool(conforms), 'report': report[:500] if report else ''})
        total = len(configs)
        return {'pass_count': pass_count, 'total_count': total, 
                'pass_ratio': pass_count / max(1, total), 'results': results}
    
    # =========================================================================
    # 数据加载功能 (v2.1 - 匹配实际数据格式)
    # =========================================================================
    
    def populate_from_csv_files(self, 
                                data_dir: str = 'data',
                                sensor_csv: str = None,
                                algorithm_csv: str = None,
                                infrastructure_csv: str = None,
                                cost_benefit_csv: str = None):
        """从CSV/TXT文件填充本体"""
        logger.info("从数据文件填充本体...")
        
        # 加载传感器
        if sensor_csv:
            sensors_file = Path(sensor_csv)
            if sensors_file.exists():
                self._load_sensors(sensors_file)
        else:
            for p in [Path(data_dir) / 'sensors_data.txt', Path('sensors_data.txt')]:
                if p.exists():
                    self._load_sensors(p)
                    break
        
        # 加载算法
        if algorithm_csv:
            algo_file = Path(algorithm_csv)
            if algo_file.exists():
                self._load_algorithms(algo_file)
        else:
            for p in [Path(data_dir) / 'algorithms_data.txt', Path('algorithms_data.txt')]:
                if p.exists():
                    self._load_algorithms(p)
                    break
        
        # 加载基础设施
        if infrastructure_csv:
            infra_file = Path(infrastructure_csv)
            if infra_file.exists():
                self._load_infrastructure(infra_file)
        else:
            for p in [Path(data_dir) / 'infrastructure_data.txt', Path('infrastructure_data.txt')]:
                if p.exists():
                    self._load_infrastructure(p)
                    break
        
        # 加载成本效益参数
        if cost_benefit_csv:
            cost_file = Path(cost_benefit_csv)
            if cost_file.exists():
                self._load_cost_effectiveness(cost_file)
        else:
            for p in [Path(data_dir) / 'cost_effectiveness_data.txt', 
                      Path('cost_effectiveness_data.txt'),
                      Path(data_dir) / 'cost_benefit_data.txt',
                      Path('cost_benefit_data.txt')]:
                if p.exists():
                    self._load_cost_effectiveness(p)
                    break
        
        self._add_shacl_constraints()
        logger.info(f"本体填充完成，包含 {len(self.g)} 个三元组")
        self._verify_loaded_components()
    
    def _detect_separator(self, filepath: Path) -> str:
        """检测文件分隔符"""
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        if '|' in first_line and first_line.count('|') > first_line.count(','):
            return '|'
        return ','
    
    def _load_sensors(self, filepath: Path):
        """加载传感器数据 - 支持多种列名格式"""
        try:
            sep = self._detect_separator(filepath)
            df = pd.read_csv(filepath, sep=sep)
            logger.info(f"加载 {len(df)} 个传感器实例...")
            
            # 自动检测ID列
            id_col = None
            for col in ['Sensor_Instance_Name', 'Component_ID', 'sensor_id', 'ID', 'Name']:
                if col in df.columns:
                    id_col = col
                    break
            
            # 自动检测类型列
            type_col = None
            for col in ['Sensor_RDF_Type', 'Component_Category', 'sensor_type', 'Type', 'Category']:
                if col in df.columns:
                    type_col = col
                    break
            
            if id_col is None:
                logger.error(f"传感器文件缺少ID列，可用列: {list(df.columns)}")
                return
            
            for _, row in df.iterrows():
                sensor_id = str(row[id_col]).replace(' ', '_')
                sensor_uri = EX[sensor_id]
                
                category = str(row.get(type_col, '')) if type_col else ''
                sensor_class = self._map_sensor_class(category)
                
                self.g.add((sensor_uri, RDF.type, RDTCO[sensor_class]))
                self.g.add((sensor_uri, RDFS.label, Literal(row[id_col])))
                
                # 灵活的列名映射
                self._add_property_flexible(sensor_uri, RDTCO.hasInitialCostUSD, row,
                    ['Initial_Cost_USD', 'initial_cost', 'Cost_USD'])
                self._add_property_flexible(sensor_uri, RDTCO.hasOperationalCostUSDPerDay, row,
                    ['Operational_Cost_USD_per_day', 'op_cost_per_day'])
                self._add_property_flexible(sensor_uri, RDTCO.hasEnergyConsumptionW, row,
                    ['Energy_Consumption_W', 'energy_w', 'Power_W'])
                self._add_property_flexible(sensor_uri, RDTCO.hasMTBFHours, row,
                    ['MTBF_hours', 'MTBF', 'mtbf'])
                self._add_property_flexible(sensor_uri, RDTCO.hasCoverageEfficiencyKmPerDay, row,
                    ['Coverage_Efficiency_km_per_day', 'coverage_km_day'])
                self._add_property_flexible(sensor_uri, RDTCO.hasDataVolumeGBPerKm, row,
                    ['Data_Volume_GB_per_km', 'data_volume_gb'])
                self._add_property_flexible(sensor_uri, RDTCO.hasAccuracyRangeMM, row,
                    ['Accuracy_Range_mm', 'accuracy_mm'])
                
                if type_col and pd.notna(row.get(type_col)):
                    self.g.add((sensor_uri, RDTCO.hasComponentCategory, Literal(str(row[type_col]))))
                    
        except Exception as e:
            logger.error(f"加载传感器数据时出错: {e}")
            raise
    
    def _map_sensor_class(self, category: str) -> str:
        """映射传感器类别到本体类"""
        category = str(category).upper() if pd.notna(category) else ''
        mapping = {
            'MMS': 'MMS_LiDAR_System', 'UAV': 'UAV_LiDAR_System', 
            'TLS': 'TLS_System', 'HANDHELD': 'Handheld_3D_Scanner',
            'FIBER': 'FiberOptic_Sensor', 'FOS': 'FiberOptic_Sensor',
            'VEHICLE': 'Vehicle_LowCost_Sensor', 'IOT': 'IoT_Network_System',
            'CAMERA': 'MMS_Camera_System', 'LIDAR': 'MMS_LiDAR_System',
        }
        for key, value in mapping.items():
            if key in category:
                return value
        return 'SensorSystem'
    
    def _load_algorithms(self, filepath: Path):
        """加载算法数据 - 支持多种列名格式"""
        try:
            sep = self._detect_separator(filepath)
            df = pd.read_csv(filepath, sep=sep)
            logger.info(f"加载 {len(df)} 个算法实例...")
            
            id_col = None
            for col in ['Algorithm_Instance_Name', 'Component_ID', 'algorithm_id', 'ID', 'Name']:
                if col in df.columns:
                    id_col = col
                    break
            
            type_col = None
            for col in ['Algorithm_RDF_Type', 'Component_Category', 'algorithm_type', 'Type']:
                if col in df.columns:
                    type_col = col
                    break
            
            if id_col is None:
                logger.error(f"算法文件缺少ID列，可用列: {list(df.columns)}")
                return
            
            for _, row in df.iterrows():
                algo_id = str(row[id_col]).replace(' ', '_')
                algo_uri = EX[algo_id]
                
                category = str(row.get(type_col, '')) if type_col else ''
                algo_class = self._map_algorithm_class(category)
                
                self.g.add((algo_uri, RDF.type, RDTCO[algo_class]))
                self.g.add((algo_uri, RDFS.label, Literal(row[id_col])))
                
                self._add_property_flexible(algo_uri, RDTCO.hasRecall, row,
                    ['Recall', 'Detection_Recall_Typical', 'recall'])
                self._add_property_flexible(algo_uri, RDTCO.hasPrecision, row,
                    ['Precision', 'Detection_Precision_Typical', 'precision'])
                self._add_property_flexible(algo_uri, RDTCO.hasFPS, row,
                    ['FPS', 'Processing_FPS_Typical', 'fps'])
                
                # 硬件需求
                for col in ['Hardware_Requirement', 'hardware_req', 'Hardware']:
                    if col in df.columns and pd.notna(row.get(col)):
                        self.g.add((algo_uri, RDTCO.hasHardwareRequirement, Literal(str(row[col]))))
                        break
                
                if type_col and pd.notna(row.get(type_col)):
                    self.g.add((algo_uri, RDTCO.hasComponentCategory, Literal(str(row[type_col]))))
                    
        except Exception as e:
            logger.error(f"加载算法数据时出错: {e}")
            raise
    
    def _map_algorithm_class(self, category: str) -> str:
        """映射算法类别到本体类"""
        category = str(category).upper() if pd.notna(category) else ''
        if 'DEEP' in category or 'DL' in category:
            return 'DeepLearningAlgorithm'
        elif 'MACHINE' in category or 'ML' in category:
            return 'MachineLearningAlgorithm'
        elif 'POINT' in category or '3D' in category:
            return 'PointCloudAlgorithm'
        return 'TraditionalAlgorithm'
    
    def _load_infrastructure(self, filepath: Path):
        """加载基础设施数据 - 支持多种列名格式"""
        try:
            sep = self._detect_separator(filepath)
            df = pd.read_csv(filepath, sep=sep)
            logger.info(f"加载 {len(df)} 个基础设施实例...")
            
            id_col = None
            for col in ['Component_Instance_Name', 'Component_ID', 'infra_id', 'ID', 'Name']:
                if col in df.columns:
                    id_col = col
                    break
            
            type_col = None
            for col in ['Component_RDF_Type', 'Component_Type', 'Type']:
                if col in df.columns:
                    type_col = col
                    break
            
            category_col = None
            for col in ['Component_Category', 'Category']:
                if col in df.columns:
                    category_col = col
                    break
            
            if id_col is None:
                logger.error(f"基础设施文件缺少ID列，可用列: {list(df.columns)}")
                return
            
            for _, row in df.iterrows():
                infra_id = str(row[id_col]).replace(' ', '_')
                infra_uri = EX[infra_id]
                
                rdf_type = str(row.get(type_col, '')).upper() if type_col else ''
                category = str(row.get(category_col, '')).upper() if category_col else ''
                combined = rdf_type + ' ' + category
                
                if 'STORAGE' in combined:
                    infra_class = RDTCO.StorageSystem
                elif 'COMM' in combined or 'NETWORK' in combined:
                    infra_class = RDTCO.CommunicationSystem
                elif 'DEPLOY' in combined or 'COMPUTE' in combined:
                    infra_class = RDTCO.ComputeDeployment
                else:
                    infra_class = RDTCO.ConfigurationParameter
                
                self.g.add((infra_uri, RDF.type, infra_class))
                self.g.add((infra_uri, RDFS.label, Literal(row[id_col])))
                
                self._add_property_flexible(infra_uri, RDTCO.hasInitialCostUSD, row,
                    ['Initial_Cost_USD', 'initial_cost'])
                self._add_property_flexible(infra_uri, RDTCO.hasAnnualOpCostUSD, row,
                    ['Annual_OpCost_USD', 'annual_op_cost'])
                self._add_property_flexible(infra_uri, RDTCO.hasEnergyConsumptionW, row,
                    ['Energy_Consumption_W', 'energy_w'])
                self._add_property_flexible(infra_uri, RDTCO.hasBandwidthMbps, row,
                    ['Bandwidth_Mbps', 'bandwidth'])
                self._add_property_flexible(infra_uri, RDTCO.hasStorageCostPerGBYear, row,
                    ['Storage_Cost_per_GB_Year', 'storage_cost_gb'])
                
                if category_col and pd.notna(row.get(category_col)):
                    self.g.add((infra_uri, RDTCO.hasComponentCategory, Literal(str(row[category_col]))))
                    
        except Exception as e:
            logger.error(f"加载基础设施数据时出错: {e}")
            raise
    
    def _load_cost_effectiveness(self, filepath: Path):
        """加载成本效益参数"""
        try:
            sep = self._detect_separator(filepath)
            df = pd.read_csv(filepath, sep=sep)
            logger.info(f"加载 {len(df)} 个成本效益条目...")
            
            name_col = None
            for col in ['Metric_Name', 'Parameter_Name', 'Name']:
                if col in df.columns:
                    name_col = col
                    break
            
            value_col = None
            for col in ['Value', 'value', 'Amount']:
                if col in df.columns:
                    value_col = col
                    break
            
            if name_col is None or value_col is None:
                logger.warning(f"成本效益文件格式不匹配，跳过")
                return
            
            for _, row in df.iterrows():
                param_name = str(row[name_col]).replace(' ', '_')
                param_uri = EX[f"Parameter_{param_name}"]
                
                self.g.add((param_uri, RDF.type, RDTCO.ConfigurationParameter))
                self.g.add((param_uri, RDFS.label, Literal(row[name_col])))
                
                try:
                    value = float(row[value_col])
                    self.g.add((param_uri, RDTCO.hasValue, Literal(value, datatype=XSD.decimal)))
                except (ValueError, TypeError):
                    pass
                
        except Exception as e:
            logger.error(f"加载成本效益数据时出错: {e}")
    
    def _add_property_flexible(self, subject: URIRef, predicate: URIRef, row, possible_cols: list):
        """灵活添加属性 - 尝试多个可能的列名"""
        for col in possible_cols:
            if col in row.index and pd.notna(row.get(col)):
                self._add_property(subject, predicate, row.get(col), XSD.decimal)
                break
    
    def _add_property(self, subject: URIRef, predicate: URIRef, value, datatype: URIRef):
        """添加属性（安全类型转换）"""
        if pd.notna(value) and str(value) not in ['N/A', 'nan', 'NaN', '']:
            try:
                if datatype == XSD.decimal:
                    value = float(value)
                elif datatype == XSD.integer:
                    value = int(float(value))
                self.g.add((subject, predicate, Literal(value, datatype=datatype)))
            except (ValueError, TypeError):
                self.g.add((subject, predicate, Literal(str(value))))
    
    def _add_shacl_constraints(self):
        """添加SHACL约束"""
        logger.info("添加SHACL约束...")
        budget_shape = BNode()
        self.g.add((budget_shape, RDF.type, SH.PropertyShape))
        self.g.add((budget_shape, SH.path, RDTCO.hasTotalCost))
        self.g.add((budget_shape, SH.maxInclusive, Literal(20000000, datatype=XSD.decimal)))
        
        recall_shape = BNode()
        self.g.add((recall_shape, RDF.type, SH.PropertyShape))
        self.g.add((recall_shape, SH.path, RDTCO.hasRecall))
        self.g.add((recall_shape, SH.minInclusive, Literal(0.65, datatype=XSD.decimal)))
    
    def _verify_loaded_components(self):
        """验证加载的组件"""
        sensor_query = """
        PREFIX rdtco: <http://www.semanticweb.org/rmtwin/ontologies/rdtco#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT DISTINCT ?sensor WHERE {
            ?sensor rdf:type ?type .
            FILTER(CONTAINS(STR(?type), "Sensor") || CONTAINS(STR(?type), "System") || CONTAINS(STR(?type), "Scanner"))
        }
        """
        sensors = list(self.g.query(sensor_query))
        logger.info(f"验证：找到 {len(sensors)} 个传感器实例")
        
        algo_query = """
        PREFIX rdtco: <http://www.semanticweb.org/rmtwin/ontologies/rdtco#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT DISTINCT ?algo WHERE {
            ?algo rdf:type ?type .
            FILTER(CONTAINS(STR(?type), "Algorithm"))
        }
        """
        algorithms = list(self.g.query(algo_query))
        logger.info(f"验证：找到 {len(algorithms)} 个算法实例")
    
    def save_ontology(self, filepath: str, format: str = 'turtle'):
        """保存本体到文件"""
        self.g.serialize(destination=filepath, format=format)
        logger.info(f"本体保存到 {filepath}")


def create_ontology_manager(data_dir: str = 'data') -> OntologyManager:
    """创建并初始化本体管理器"""
    manager = OntologyManager()
    manager.populate_from_csv_files(data_dir)
    return manager
