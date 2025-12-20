# Noah-MP 输出数据处理流程

## 概述

本文档记录了从 Noah-MP 模型输出文件 (NetCDF) 到用于验证分析的 CSV 文件的处理流程。

## 数据源

### Noah-MP 输出文件
- **位置**: `validation_results/<timestamp>/<calibration_id>/run_<type>/output/`
- **格式**: NetCDF (`.LDASOUT_DOMAIN1`)
- **时间分辨率**: 30 分钟
- **时区**: UTC+0 (协调世界时)
- **示例文件**: `201507301730.LDASOUT_DOMAIN1`

### 观测数据文件
- **位置**: `data/obs/Panama_BCI_v5.1_fluxtowerdata.csv`
- **格式**: CSV
- **时间分辨率**: 30 分钟
- **时区**: UTC-5 (巴拿马当地时间)

## 处理步骤

### 步骤 1: 打开 NetCDF 文件
```python
ds = xr.open_dataset(nc_file, engine='netcdf4')
```

### 步骤 2: 提取时间戳并进行时区转换

**问题**: Noah-MP 使用 UTC 时间，而观测数据使用当地时间 (UTC-5)

**解决方案**: 在提取时将 UTC 时间转换为当地时间

```python
# Noah-MP 时间格式: 'YYYY-MM-DD_HH:MM:SS'
times_utc = pd.to_datetime(times_str, format='%Y-%m-%d_%H:%M:%S')

# UTC -> UTC-5 转换 (减 5 小时)
times_local = times_utc + timedelta(hours=-5)
```

**示例**:
- Noah-MP UTC: `2015-07-30 17:00:00`
- 转换后当地时间: `2015-07-30 12:00:00`

### 步骤 3: 提取目标变量

从 NetCDF 文件中提取以下变量：

| Noah-MP 变量 | 标准名称 | 描述 | 单位 |
|-------------|---------|------|------|
| HFX | HFX | 感热通量 | W/m² |
| LH | LH | 潜热通量 | W/m² |
| SOIL_M | SOIL_M | 土壤湿度（第一层） | m³/m³ |
| ECAN | ECAN | 冠层蒸发 | mm/s |
| ETRAN | ETRAN | 植物蒸腾 | mm/s |
| EDIR | EDIR | 土壤直接蒸发 | mm/s |

**多层变量处理**:
- `SOIL_M` 有 4 个土壤层，仅提取第一层（表层土壤）
- 其他多维变量取平均

### 步骤 4: 计算衍生变量

总蒸散发 (ET):
```python
# ET = ECAN + ETRAN + EDIR
# 单位转换: mm/s -> mm/30min (乘以 1800 秒)
ET = (ECAN + ETRAN + EDIR) * 1800
```

### 步骤 5: 保存为 CSV

**默认行为**: 保持 30 分钟原分辨率

**可选**: 使用 `--daily` 参数聚合为日均值

## 脚本使用

```bash
# 默认: 30分钟分辨率 + 时区转换
python parse_noahmp_outputs.py --input output.nc --output output.csv

# 聚合为日均值
python parse_noahmp_outputs.py --input output.nc --output output.csv --daily

# 保持 UTC 时间（不转换时区）
python parse_noahmp_outputs.py --input output.nc --output output.csv --no-timezone
```

## 与观测数据对齐

处理后的模拟数据与观测数据：
- **时间分辨率**: 均为 30 分钟
- **时区**: 均为 UTC-5（巴拿马当地时间）
- **变量名映射**:

| 观测数据列名 | 模拟数据列名 | 描述 |
|------------|------------|------|
| LE | LH | 潜热通量 |
| H | HFX | 感热通量 |
| SWC | SOIL_M | 土壤水分含量 |

## 验证分析脚本

`07_calibration_validation_enhanced.py` 会自动应用列名映射：
```python
OBS_TO_SIM_MAPPING = {
    'LE': 'LH',
    'H': 'HFX',
    'SWC': 'SOIL_M',
}
```

## 输出文件

- `emulator_output.csv`: 模拟器校准参数的模型输出
- `expert_output.csv`: 专家校准参数的模型输出
- `default_output.csv`: 默认参数的模型输出
- `validation_metrics_by_period.csv`: 分时段验证指标
- `metrics_summary.txt`: 指标摘要文本文件

## 更新日志

- **2024-12-19**: 
  - 修改为默认保持 30 分钟原分辨率
  - 添加 UTC -> UTC-5 时区转换
  - 更新文档
