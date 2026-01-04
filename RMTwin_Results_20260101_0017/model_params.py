#!/usr/bin/env python3
"""
Model Parameters for RMTwin Fitness Evaluation
==============================================
Centralized parameter configuration with units documented.

This module provides MODEL_PARAMS dict containing all tunable parameters
for the enhanced fitness evaluation model. All decision variables (11 total)
now materially affect objectives.

Author: RMTwin Research Team
Version: 2.0 (Step 2-Lite Upgrade)
"""

import numpy as np
from typing import Dict, Any

# =============================================================================
# CENTRALIZED MODEL PARAMETERS
# =============================================================================

MODEL_PARAMS: Dict[str, Any] = {
    
    # =========================================================================
    # 1. COMMUNICATION PARAMETERS
    # =========================================================================
    # Bandwidth in GB/s for different communication systems
    # Used in latency calculation: comm_time = data_volume / bandwidth
    'comm_bandwidth_GBps': {
        'Fiber': 1.25,           # 10 Gbps fiber → 1.25 GB/s
        '5G': 0.125,             # 1 Gbps 5G → 0.125 GB/s (realistic average)
        '4G': 0.0125,            # 100 Mbps 4G → 0.0125 GB/s
        'LoRaWAN': 0.00000625,   # 50 kbps LoRaWAN → 0.00000625 GB/s
        'V2X': 0.00125,          # 10 Mbps V2X → 0.00125 GB/s
        'default': 0.01          # 80 Mbps default
    },
    
    # Communication energy consumption in kWh per GB transferred
    'comm_energy_kWh_per_GB': {
        'Fiber': 0.006,          # Very efficient
        '5G': 0.05,              # Moderate
        '4G': 0.08,              # Higher due to older tech
        'LoRaWAN': 0.001,        # Very low power but low throughput
        'V2X': 0.02,             # Short range, moderate
        'default': 0.05
    },
    
    # Communication cost in USD per GB transferred
    'comm_cost_USD_per_GB': {
        'Fiber': 0.01,           # Cheap after installation
        '5G': 0.05,              # Moderate data cost
        '4G': 0.08,              # Higher cellular cost
        'LoRaWAN': 0.001,        # Very cheap (low volume)
        'V2X': 0.02,             # Moderate
        'default': 0.05
    },
    
    # Network reliability factor (0-1)
    'comm_reliability': {
        'Fiber': 0.999,
        '5G': 0.95,
        '4G': 0.92,
        'LoRaWAN': 0.88,
        'V2X': 0.90,
        'default': 0.90
    },
    
    # =========================================================================
    # 2. STORAGE PARAMETERS
    # =========================================================================
    # Storage cost in USD per GB per year
    'storage_cost_USD_per_GB_year': {
        'Hot': 0.276,            # AWS S3 Standard (~$0.023/GB/month)
        'Cold': 0.048,           # AWS S3 Glacier (~$0.004/GB/month)
        'Archive': 0.012,        # AWS S3 Deep Archive
        'OnPremise': 0.10,       # Amortized local storage
        'Hybrid': 0.15,          # Mix of hot and cold
        'default': 0.20
    },
    
    # Storage energy in kWh per GB per year
    'storage_energy_kWh_per_GB_year': {
        'Hot': 0.05,             # Active storage
        'Cold': 0.01,            # Reduced access
        'Archive': 0.005,        # Minimal energy
        'OnPremise': 0.08,       # Less efficient
        'Hybrid': 0.03,
        'default': 0.04
    },
    
    # =========================================================================
    # 3. DEPLOYMENT/COMPUTE PARAMETERS
    # =========================================================================
    # Compute factor: multiplier for base processing time
    # Higher = slower processing
    'deployment_compute_factor': {
        'Cloud': 1.0,            # Baseline (powerful GPUs)
        'Edge': 2.5,             # Limited compute on edge
        'OnPremise': 1.5,        # Good but not cloud-scale
        'Hybrid': 1.3,           # Mix benefits
        'default': 1.5
    },
    
    # Power consumption in Watts during compute
    'deployment_power_W': {
        'Cloud': 300,            # GPU server (allocated share)
        'Edge': 60,              # Jetson-class device
        'OnPremise': 200,        # Local server
        'Hybrid': 150,           # Average
        'default': 150
    },
    
    # Compute cost in USD per hour
    'deployment_compute_cost_USD_per_hour': {
        'Cloud': 1.0,            # GPU instance hourly rate
        'Edge': 0.05,            # Amortized edge device
        'OnPremise': 0.2,        # Amortized local compute
        'Hybrid': 0.5,
        'default': 0.5
    },
    
    # Data reduction ratio: how much data is sent after edge preprocessing
    # 1.0 = all data sent, 0.1 = 90% filtered at edge
    'data_reduction_ratio': {
        'Cloud': 1.0,            # All data sent to cloud
        'Edge': 0.15,            # 85% filtered at edge (only anomalies sent)
        'OnPremise': 0.8,        # Some local filtering
        'Hybrid': 0.4,           # Significant edge filtering
        'default': 0.7
    },
    
    # PUE (Power Usage Effectiveness) for data centers
    'datacenter_pue': {
        'Cloud': 1.1,            # Hyperscale efficiency
        'Edge': 1.0,             # No datacenter overhead
        'OnPremise': 1.4,        # Typical enterprise
        'Hybrid': 1.2,
        'default': 1.3
    },
    
    # =========================================================================
    # 4. RECALL MODEL PARAMETERS
    # =========================================================================
    # Sigmoid model: recall = sigmoid(z) where
    # z = a0 + a1*base_algo + a2*sensor_q + lod_bonus - a3*(tau - tau0)
    'recall_model': {
        'a0': -1.5,              # Intercept (shifts sigmoid center)
        'a1': 3,               # Algorithm recall weight
        'a2': 2.5,               # Sensor precision weight
        'a3': 1.0,               # Detection threshold penalty weight (increased)
        'tau0': 0.5,             # Optimal detection threshold
        'lod_bonus': {           # LOD impact on recall
            'Micro': 0.6,        # High detail → better recall
            'Meso': 0.0,         # Baseline
            'Macro': -0.4        # Low detail → worse recall
        },
        'data_rate_bonus_factor': 0.005,   # Per Hz above base_rate
        'base_data_rate': 50.0,            # Hz, baseline data rate
        'max_recall': 0.99,      # Practical ceiling
        'min_recall': 0.6       # Practical floor
    },
    
    # False positive penalty coefficient
    # When tau < tau0, lower threshold means more false positives
    # fp_penalty = fp_coeff * max(0, tau0 - tau)^2 * base_cost
    'fp_penalty_coeff': 0.15,    # 15% cost increase per unit threshold below optimal
    
    # =========================================================================
    # 5. LATENCY MODEL PARAMETERS
    # =========================================================================
    # LOD multiplier for data volume
    'lod_data_multiplier': {
        'Micro': 1.8,            # High resolution = more data
        'Meso': 1.0,             # Baseline
        'Macro': 0.5             # Low resolution = less data
    },
    
    # Base data rate for normalization (Hz)
    'base_data_rate_Hz': 50.0,   # Higher base = less rate_mult impact
    
    # Algorithm compute time in seconds per GB of data
    'algo_compute_seconds_per_GB': {
        'DL': 8.0,               # Deep learning: GPU-intensive but optimized
        'ML': 3.0,               # Machine learning: moderate
        'Traditional': 0.5,      # Traditional: fast CPU
        'PC': 5.0,               # Point cloud: memory-intensive
        'Hybrid': 4.0,           # Mix
        'default': 3.0
    },
    
    # Fixed overhead times in seconds
    'latency_overhead': {
        'queue_cloud': 5.0,      # Cloud queue average
        'queue_edge': 1.0,       # Edge queue
        'startup': 2.0,          # System startup
        'result_processing': 3.0 # Result aggregation
    },
    
    # =========================================================================
    # 6. COST MODEL PARAMETERS
    # =========================================================================
    # Depreciation rates by sensor type (annual)
    'depreciation_rate': {
        'MMS': 0.15,
        'UAV': 0.20,
        'TLS': 0.12,
        'Handheld': 0.12,
        'Vehicle': 0.15,
        'IoT': 0.10,
        'FOS': 0.08,
        'Camera': 0.12,
        'default': 0.12
    },
    
    # Skill level wage multipliers
    'skill_wage_multiplier': {
        'Basic': 1.0,
        'Intermediate': 1.5,
        'Expert': 2.0
    },
    
    # ML/DL annotation cost multipliers
    'annotation_cost_multiplier': {
        'DL': 1.0,               # Full annotation needed
        'ML': 0.6,               # Some feature engineering
        'Traditional': 0.0,      # No annotation
        'default': 0.3
    },
    
    # =========================================================================
    # 7. CARBON MODEL PARAMETERS
    # =========================================================================
    # Carbon intensity by deployment type (relative factor)
    'carbon_intensity_factor': {
        'Cloud': 0.7,            # Hyperscale uses more renewables
        'Edge': 1.0,             # Grid average
        'OnPremise': 1.1,        # Less efficient
        'Hybrid': 0.85,
        'default': 1.0
    },
    
    # Manufacturing carbon in kgCO2 per $1000 equipment cost
    'manufacturing_carbon_kg_per_1000USD': {
        'Electronics': 50.0,
        'Mechanical': 30.0,
        'Vehicle': 80.0,
        'Software': 10.0,
        'default': 40.0
    },
    
    # Equipment lifetime in years (for amortizing manufacturing carbon)
    'equipment_lifetime_years': 8,
    
    # =========================================================================
    # 8. DISRUPTION MODEL PARAMETERS
    # =========================================================================
    # Base disruption hours per inspection event by sensor type
    'base_disruption_hours': {
        'MMS': 4.0,              # Mobile mapping: lane closure
        'UAV': 0.5,              # Minimal ground disruption
        'TLS': 6.0,              # Stationary: significant
        'Handheld': 3.0,         # Manual inspection
        'Vehicle': 2.0,          # Flows with traffic
        'IoT': 0.1,              # Fixed sensor: minimal
        'FOS': 0.1,              # Fiber optic: minimal
        'Camera': 1.0,           # Fixed camera: low disruption
        'default': 3.0
    },
    
    # Crew size factor for disruption
    # Larger crews may require more lane closures but work faster
    'crew_disruption_factor': {
        1: 0.5,                  # Single person: minimal footprint
        2: 0.7,
        3: 0.9,
        4: 1.0,
        5: 1.1,
        'large': 1.3             # 6+ people
    },
    
    # Traffic volume impact (log scale factor)
    'traffic_disruption_log_base': 1000,  # vehicles/hour baseline
    
    # =========================================================================
    # 9. RELIABILITY MODEL PARAMETERS
    # =========================================================================
    # Architecture reliability multipliers
    'architecture_reliability_factor': {
        'single_point': 1.0,
        'active_backup': 1.8,
        'load_balanced': 2.5,
        'distributed': 3.0
    },
    
    # Environmental degradation factors
    'environmental_mtbf_factor': {
        'UAV': 0.6,
        'Vehicle': 0.7,
        'FOS': 0.9,
        'IoT': 0.9,
        'Indoor': 0.95,
        'default': 0.8
    },
    
    # Algorithm complexity factor (affects reliability)
    'algo_complexity_factor': {
        'DL': 0.7,               # Complex, more failure modes
        'ML': 0.85,
        'Traditional': 0.95,
        'default': 0.85
    },
    
    # =========================================================================
    # 10. NORMALIZATION BOUNDS (for objective scaling)
    # =========================================================================
    'normalization_bounds': {
        'cost': {'min': 100_000, 'max': 10_000_000},      # USD
        'recall': {'min': 0.0, 'max': 0.6},               # 1-recall
        'latency': {'min': 1.0, 'max': 600.0},            # seconds
        'disruption': {'min': 0.0, 'max': 500.0},         # hours/year
        'carbon': {'min': 100, 'max': 100_000},           # kgCO2e/year
        'reliability': {'min': 1e-7, 'max': 1e-3}         # 1/MTBF
    }
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_param(category: str, key: str, default: Any = None) -> Any:
    """
    Safely retrieve a parameter from MODEL_PARAMS.
    
    Args:
        category: Top-level category in MODEL_PARAMS
        key: Key within the category
        default: Default value if not found
        
    Returns:
        Parameter value or default
    """
    if category in MODEL_PARAMS:
        params = MODEL_PARAMS[category]
        if isinstance(params, dict):
            return params.get(key, params.get('default', default))
        return params
    return default


def get_comm_type(comm_str: str) -> str:
    """
    Extract communication type from URI/string.
    
    Args:
        comm_str: Communication system string (e.g., 'Communication_5G_SmallCell')
        
    Returns:
        Communication type key matching MODEL_PARAMS keys (e.g., 'Fiber', '5G')
    """
    comm_str_upper = str(comm_str).upper()
    # Mapping from detection pattern to correct parameter key
    key_mapping = {
        'FIBER': 'Fiber',
        '5G': '5G',
        '4G': '4G',
        'LORAWAN': 'LoRaWAN',
        'LORA': 'LoRaWAN',
        'V2X': 'V2X'
    }
    for pattern, key in key_mapping.items():
        if pattern in comm_str_upper:
            return key
    return 'default'


def get_storage_type(storage_str: str) -> str:
    """
    Extract storage type from URI/string.
    
    Args:
        storage_str: Storage system string
        
    Returns:
        Storage type key (e.g., 'Hot', 'Cold')
    """
    storage_str = str(storage_str).upper()
    if 'GLACIER' in storage_str or 'ARCHIVE' in storage_str:
        return 'Archive'
    elif 'COLD' in storage_str:
        return 'Cold'
    elif 'HYBRID' in storage_str:
        return 'Hybrid'
    elif 'ONPREM' in storage_str or 'LOCAL' in storage_str:
        return 'OnPremise'
    return 'Hot'


def get_deployment_type(deploy_str: str) -> str:
    """
    Extract deployment type from URI/string.
    
    Args:
        deploy_str: Deployment system string
        
    Returns:
        Deployment type key (e.g., 'Cloud', 'Edge')
    """
    deploy_str = str(deploy_str).upper()
    if 'EDGE' in deploy_str:
        return 'Edge'
    elif 'HYBRID' in deploy_str:
        return 'Hybrid'
    elif 'ONPREM' in deploy_str or 'LOCAL' in deploy_str:
        return 'OnPremise'
    elif 'CLOUD' in deploy_str or 'GPU' in deploy_str:
        return 'Cloud'
    return 'default'


def get_sensor_type(sensor_str: str) -> str:
    """
    Extract sensor type from URI/string.
    
    Args:
        sensor_str: Sensor system string
        
    Returns:
        Sensor type key matching MODEL_PARAMS keys (e.g., 'MMS', 'Vehicle', 'IoT')
    """
    sensor_str_upper = str(sensor_str).upper()
    # Mapping from detection pattern to correct parameter key
    # Keys must match MODEL_PARAMS['depreciation_rate'], ['base_disruption_hours'], etc.
    key_mapping = {
        'MMS': 'MMS',
        'UAV': 'UAV',
        'TLS': 'TLS',
        'HANDHELD': 'Handheld',
        'VEHICLE': 'Vehicle',
        'IOT': 'IoT',
        'FOS': 'FOS',
        'FIBER': 'FOS',       # Fiber optic sensing maps to FOS
        'CAMERA': 'Camera'
    }
    for pattern, key in key_mapping.items():
        if pattern in sensor_str_upper:
            return key
    return 'default'


def get_algo_type(algo_str: str) -> str:
    """
    Extract algorithm type from URI/string.
    
    Args:
        algo_str: Algorithm string
        
    Returns:
        Algorithm type key (e.g., 'DL', 'ML', 'Traditional')
    """
    algo_str = str(algo_str).upper()
    if 'DL_' in algo_str or 'DEEP' in algo_str or 'YOLO' in algo_str or 'UNET' in algo_str:
        return 'DL'
    elif 'ML_' in algo_str or 'SVM' in algo_str or 'RANDOM' in algo_str or 'XGBOOST' in algo_str:
        return 'ML'
    elif 'TRADITIONAL' in algo_str or 'CANNY' in algo_str or 'OTSU' in algo_str:
        return 'Traditional'
    elif 'HYBRID' in algo_str:
        return 'Hybrid'
    elif 'PC_' in algo_str or 'POINT' in algo_str:
        return 'PC'
    return 'default'


def sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid function.
    
    Args:
        x: Input value
        
    Returns:
        Sigmoid output in (0, 1)
    """
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_recall(recall: float) -> bool:
    """Check recall is in valid range (0, 1)."""
    return 0 < recall < 1


def validate_positive(value: float, name: str = "value") -> bool:
    """Check value is positive and finite."""
    return np.isfinite(value) and value > 0


def validate_non_negative(value: float, name: str = "value") -> bool:
    """Check value is non-negative and finite."""
    return np.isfinite(value) and value >= 0


def run_sanity_checks():
    """
    Run basic sanity checks on model parameters.
    
    Raises:
        AssertionError: If any check fails
    """
    # Check bandwidth ordering (higher tech = higher bandwidth)
    bw = MODEL_PARAMS['comm_bandwidth_GBps']
    assert bw['Fiber'] > bw['5G'] > bw['4G'] > bw['LoRaWAN'], \
        "Bandwidth ordering violated"
    
    # Check compute factor ordering (cloud fastest)
    cf = MODEL_PARAMS['deployment_compute_factor']
    assert cf['Cloud'] <= cf['Hybrid'] <= cf['OnPremise'] < cf['Edge'], \
        "Compute factor ordering violated"
    
    # Check data reduction makes sense
    dr = MODEL_PARAMS['data_reduction_ratio']
    assert dr['Cloud'] == 1.0, "Cloud should send all data"
    assert dr['Edge'] < dr['Cloud'], "Edge should reduce data"
    
    # Check recall model parameters
    rm = MODEL_PARAMS['recall_model']
    assert rm['max_recall'] < 1.0, "Max recall should be < 1"
    assert rm['min_recall'] > 0.0, "Min recall should be > 0"
    assert rm['min_recall'] < rm['max_recall'], "Min < Max recall"
    
    print("✓ All sanity checks passed")


if __name__ == "__main__":
    run_sanity_checks()
    print("\nModel parameters loaded successfully.")
    print(f"Total parameter categories: {len(MODEL_PARAMS)}")
