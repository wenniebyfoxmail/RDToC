# RMTwin 语义验证更新包 v2.1
## Semantic Validation Update (P0 + P1 Implementation)

### 📋 更新内容

基于导师建议，本更新包实现了**分层验证策略**：

| 层 | 方法 | 职责 | 执行时机 |
|----|------|------|----------|
| **SHACL** | 形式化验证 | 配置完整性（结构约束） | Pareto输出后 |
| **Fast-Check** | 运行时规则 | 语义兼容性（工程规则） | 每次评估时 |

#### P0: 后验SHACL审计
- ✅ `shapes/min_shapes.ttl` - 5条完整性约束
- ✅ `ontology_manager.py` - `build_config_graph()` + `shacl_validate_config()`
- ✅ `main.py` - Step 4b SHACL审计 + violation统计
- ✅ 移除 carbon clip 下限（改为非负+finite防护）

#### P1: 运行时语义筛选
- ✅ `evaluation.py` - `_semantic_fast_check()` 3条规则：
  1. IoT/FOS传感器不兼容V2X/DSRC通信
  2. GPU/DL算法需要GPU部署环境
  3. 移动传感器需要无线通信
- ✅ 惩罚值使用合理尺度，避免污染优化器

#### 负对照测试
- ✅ `test_shacl_negative_control.py` - 验证SHACL工作正常

### 📁 文件清单

```
rmtwin_semantic_update/
├── shapes/
│   └── min_shapes.ttl              # SHACL完整性约束
├── ontology_manager.py             # 本体管理器 v2.1
├── evaluation.py                   # 评估模块 v2.1
├── main.py                         # 主程序 v2.1
├── test_shacl_negative_control.py  # SHACL测试脚本
└── README.md
```

### 🚀 使用方法

```bash
# 1. 替换文件
mkdir -p shapes
cp rmtwin_semantic_update/shapes/min_shapes.ttl ./shapes/
cp rmtwin_semantic_update/ontology_manager.py ./
cp rmtwin_semantic_update/evaluation.py ./
cp rmtwin_semantic_update/main.py ./
cp rmtwin_semantic_update/test_shacl_negative_control.py ./

# 2. 安装依赖
pip install pyshacl

# 3. 验证SHACL工作正常
python test_shacl_negative_control.py

# 4. 运行优化
python main.py --config config.json --seed 42
```

### 📊 预期测试输出

```
SHACL 完整性验证测试 (v2.0)
============================================================

[1/2] 测试完整配置 (应该通过 SHACL)...
  ✅ PASS Complete_MMS_Cloud
  ✅ PASS Complete_IoT_Edge
  ✅ PASS Complete_UAV_Cloud
  ✅ PASS Complete_DL_GPU

[2/2] 测试不完整配置 (应该被 SHACL 拒绝)...
  ✅ CORRECTLY REJECTED Missing_Sensor
  ✅ CORRECTLY REJECTED Missing_Algorithm
  ...

✅ SHACL 完整性验证器正常工作！
```

### 📝 分层验证策略说明

**为什么采用分层？**

| 方案 | 优点 | 缺点 |
|------|------|------|
| 纯SHACL | 形式化、可追溯 | 需要完整本体图、慢 |
| 纯规则 | 快、简单 | 无形式化保证 |
| **分层** | 兼顾两者优点 | ✅ 推荐 |

**SHACL 负责**：
- 配置必须有5个必要组件（sensor, algorithm, deployment, storage, communication）
- 参数范围（inspection_cycle: 1-365, data_rate > 0）

**Fast-Check 负责**：
- GPU算法↔GPU部署
- IoT传感器↔非V2X通信
- 移动传感器↔无线通信

---

**Author**: RMTwin Research Team  
**Version**: 2.1 (Layered Validation)  
**Date**: 2024-12
