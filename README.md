# LSTM Emulator for Noah-MP Land Surface Model

Deep learning emulator for the Noah-MP land surface model using LSTM networks to predict energy and water cycle variables from model parameters and meteorological forcing.

## Overview

This project provides a comprehensive LSTM-based emulator that:
- Predicts **29 energy and water cycle variables** from Noah-MP simulations
- Enables **fast parameter sensitivity analysis** and uncertainty quantification
- Maintains **physical conservation** of energy and water budgets
- Supports multiple LSTM architectures (LSTM, BiLSTM, Attention-LSTM)

### Key Features

- **Comprehensive Output**: 29 variables covering energy balance, water fluxes, storage, and temperature
- **Physical Consistency**: Full energy and water conservation validation
- **Multiple Architectures**: Choose from LSTM, BiLSTM, or Attention-LSTM models
- **Flexible Training**: Configurable loss weights to prioritize critical variables
- **Automated Validation**: Built-in metrics computation and visualization
- **Production Ready**: Standardized inference interface for real-world applications

## Project Structure

```
LSTM_Emulator_4_Noahmp/
├── config_forward_comprehensive.py          # Model and training configuration
├── lstm_model_forward.py                    # LSTM model architectures
│
├── Data Processing:
│   └── 01_data_preprocessing_forward_comprehensive.py
│
├── Training & Validation:
│   ├── 02_train_forward_comprehensive.py    # Training with integrated validation
│   └── 03_validation.py                     # Time series validation for test samples
│
├── Inference & Analysis:
│   ├── 04_inference.py                      # Production inference script
│   └── 05_comprehensive_conservation_validation.py
│
├── Utilities:
│   ├── conservation_check_comprehensive.py  # Conservation validation
│   └── test_comprehensive_model.py          # Model testing
│
└── Data:
    ├── raw/param/noahmp_param_sets.txt     # 9 Noah-MP parameters
    ├── raw/sim_results/                     # Simulation outputs
    └── processed_data_forward_comprehensive.pkl
```

## Target Variables (29 Total)

### Energy Balance (5 variables)
Core energy conservation: `FSA - FIRA = HFX + LH + GRDFLX`

```
FSA       - Total absorbed SW radiation (W/m²)
FIRA      - Total net LW radiation to atmosphere (W/m²)
HFX       - Total sensible heat to atmosphere (W/m²) ⭐
LH        - Total latent heat to atmosphere (W/m²) ⭐
GRDFLX    - Heat flux into the soil (W/m²)
```

### Energy Components (9 variables)
Detailed energy budget breakdown

```
SAV, SAG   - Solar radiation (canopy/ground)
IRC, IRG   - Net LW radiation (canopy/ground)
SHC, SHG   - Sensible heat (canopy/ground)
EVC, EVG   - Evaporation heat (canopy/ground)
GHV        - Ground heat to soil
```

### Water Fluxes (5 variables)
Water cycle fluxes

```
ECAN             - Canopy water evaporation rate (mm/s)
ETRAN            - Transpiration rate (mm/s)
EDIR             - Direct soil evaporation rate (mm/s)
UGDRNOFF_RATE    - Underground runoff rate (mm/day)
SFCRNOFF_RATE    - Surface runoff rate (mm/day)
```

### Water Storage (5 variables)
Water storage state variables

```
SOIL_M    - Volumetric soil moisture Layer 1 (m³/m³) ⭐
SOIL_M_L2 - Volumetric soil moisture Layer 2 (m³/m³)
SOIL_M_L3 - Volumetric soil moisture Layer 3 (m³/m³)
SOIL_M_L4 - Volumetric soil moisture Layer 4 (m³/m³)
CANLIQ    - Canopy liquid water content (mm)
```

### Temperature (5 variables)
Temperature state variables

```
SOIL_T    - Soil temperature Layer 1 (K)
SOIL_T_L2 - Soil temperature Layer 2 (K)
TG        - Ground temperature (K)
TV        - Vegetation temperature (K)
TRAD      - Surface radiative temperature (K)
```

⭐ = Main target variables (SOIL_M, LH, HFX)

## Quick Start

### 1. Environment Setup

```bash
conda activate dfm  # Or your Python environment
pip install torch numpy pandas xarray matplotlib scikit-learn
```

### 2. Data Preprocessing

Extract and normalize all 31 variables from Noah-MP simulation outputs:

```bash
python 01_data_preprocessing_forward_comprehensive.py
```

**Output:** `data/processed_data_forward_comprehensive.pkl`

### 3. Model Training

Train the comprehensive LSTM emulator with integrated validation:

```bash
python 02_train_forward_comprehensive.py
```

**Outputs** (saved to timestamped directory, e.g., `results_forward_comprehensive/AttentionLSTM_20251116_143022_dim-512_layer-2/`):
- `best_model.pth` - Trained model weights
- `config.json` - Complete model configuration and metrics
- `validation_metrics.json` - Detailed metrics for all 31 variables
- `training_history.png` - Training and validation loss curves
- `validation_scatter_main.png` - Scatter plots for SOIL_M, LH, HFX
- `validation_scatter_all.png` - Scatter plots for all variables
- `loss_history.npz` - Training metrics

### 4. Detailed Validation (Time Series)

Generate detailed time series comparison plots for specific test samples:

```bash
# Validate samples 0-4 from validation set
python 03_validation.py \
  --model_dir results_forward_comprehensive/AttentionLSTM_20251116_143022_dim-512_layer-2 \
  --sample 0 5

# Validate single sample
python 03_validation.py \
  --model_dir results_forward_comprehensive/AttentionLSTM_20251116_143022_dim-512_layer-2 \
  --sample 0 1

# Validate all test samples (use -1 for end)
python 03_validation.py \
  --model_dir results_forward_comprehensive/AttentionLSTM_20251116_143022_dim-512_layer-2 \
  --sample 0 -1
```

**Outputs** (saved to `model_dir/validation_timeseries/`):
- `timeseries_sample_*_main.png` - Time series plots for SOIL_M, LH, HFX
- `timeseries_sample_*_all.png` - Time series plots for all variables
- `validation_metrics_summary.json` - Metrics for each validated sample

### 5. Inference

Run predictions with new forcing data and parameters:

```bash
# Single parameter set (no observations)
python 04_inference.py \
  --model_dir results_forward_comprehensive/AttentionLSTM_20251116_143022_dim-512_layer-2 \
  --forcing data/raw/forcing/forcing_sample_1.nc \
  --params data/raw/param/test_params.txt \
  --output predictions.csv \
  --plot

# With observations for validation
python 04_inference.py \
  --model_dir results_forward_comprehensive/AttentionLSTM_20251116_143022_dim-512_layer-2 \
  --forcing data/raw/forcing/forcing_sample_1.nc \
  --params data/raw/param/test_params.txt \
  --obs data/obs/observations.csv \
  --output predictions.csv \
  --plot
```

### 6. Conservation Validation

Validate physical conservation of energy and water:

```bash
python 05_comprehensive_conservation_validation.py \
  --model_dir results_forward_comprehensive/AttentionLSTM_20251116_143022_dim-512_layer-2 \
  --n_samples 20 \
  --sample_detail 0
```

## Configuration

Edit `config_forward_comprehensive.py` to customize:

### Model Architecture

```python
MODEL_CONFIG = {
    'model_type': 'AttentionLSTM',  # Options: 'LSTM', 'BiLSTM', 'AttentionLSTM'
    'hidden_dim': 512,              # Hidden layer size
    'num_layers': 2,                # Number of LSTM layers
    'dropout': 0.3,                 # Dropout rate
    'param_embedding_dim': 128,     # Parameter embedding dimension
}
```

### Training Parameters

```python
TRAINING_CONFIG = {
    'learning_rate': 0.0005,
    'batch_size': 8,
    'num_epochs': 1000,
    'patience': 100,               # Early stopping patience
    'train_ratio': 0.8,
}
```

### Loss Weights

Prioritize critical variables:

```python
OUTPUT_WEIGHTS = {
    # Energy balance (highest priority)
    'FSA': 2.0, 'FIRA': 2.0, 'HFX': 2.0, 'LH': 2.0, 'GRDFLX': 2.0,

    # Energy components
    'SAV': 1.5, 'SAG': 1.5, 'IRC': 1.5, ...

    # Water fluxes
    'ECAN': 2.0, 'ETRAN': 2.0, 'EDIR': 2.0,

    # Water storage
    'SOIL_M': 2.0, 'SOIL_M_L2': 1.5, ...
}
```

## Input Data Requirements

### Parameters
9 Noah-MP parameters in text file (space-separated):
```
CWPVT VCMX25 MP DLEAF Z0MVT HVT HVB BEXP SMCMAX
2.5   50.0   9  0.04  0.8   20.0 0.5 5.3  0.45
```

### Forcing Data
NetCDF file with hourly/sub-daily data:
- `LWFORC` - Longwave radiation (W/m²)
- `SWFORC` - Shortwave radiation (W/m²)
- `RAINRATE` - Precipitation rate (mm/s)
- `T2MV` - 2m air temperature (K)
- `time` - Time coordinate

Data is automatically aggregated to daily timesteps.

### Observations (Optional)
CSV file with date column and target variables:
```
date,SOIL_M,LH,HFX,...
2015-07-30,0.25,120.5,45.2,...
2015-07-31,0.24,118.3,43.1,...
```

## Model Performance

### Expected Metrics

**Main Target Variables:**
- SOIL_M: R² > 0.90
- LH: R² > 0.85
- HFX: R² > 0.85

**Conservation:**
- Energy balance RMSE: 10-30 W/m² (Good-Excellent)
- Water balance RMSE: 0.1-0.5 mm/day (Good-Excellent)

### Validation Metrics

For each variable, the model computes:
- **R²** - Coefficient of determination
- **RMSE** - Root mean squared error
- **PBIAS** - Percent bias

## Conservation Validation

### Energy Conservation

**Primary Balance:**
```
FSA - FIRA = HFX + LH + GRDFLX
```

**Component Checks:**
- Canopy: `SAV = IRC + SHC + EVC`
- Ground: `SAG = IRG + SHG + EVG + GHV`

### Water Conservation

**Primary Balance:**
```
ΔStorage = Precipitation - ET - Runoff
```

**Components:**
- Total ET = `ECAN + ETRAN + EDIR` (converted to mm/day)
- Total Runoff = `UGDRNOFF_RATE + SFCRNOFF_RATE` (mm/day)
- Storage = Sum of all soil layers + canopy + snow

## Troubleshooting

### Memory Issues

Reduce batch size or model size:
```python
TRAINING_CONFIG = {'batch_size': 4}  # Reduce from 8
MODEL_CONFIG = {'hidden_dim': 256}   # Reduce from 512
```

### Poor Performance

1. Check data quality (no NaN/Inf values)
2. Increase loss weights for underperforming variables
3. Increase model capacity (hidden_dim, num_layers)
4. Train longer (increase patience)

### Training Not Converging

1. Reduce learning rate (try 0.0001)
2. Check data normalization
3. Reduce model complexity
4. Adjust loss weights

## Advanced Usage

### Custom Variable Selection

Modify `TARGET_VARIABLES` in config to predict specific variables:

```python
# Example: Energy balance only
TARGET_VARIABLES = ENERGY_BALANCE_TARGETS
```

### Multi-Parameter Set Inference

Process multiple parameter sets simultaneously:

```bash
python 04_inference.py \
  --model_dir results_forward_comprehensive/AttentionLSTM_20251116_143022_dim-512_layer-2 \
  --forcing data/raw/forcing/forcing_sample_1.nc \
  --params data/raw/param/noahmp_param_sets.txt \
  --output predictions.csv \
  --max_samples 10 \
  --obs observations.csv \
  --plot
```

## File Outputs

### Training Output Structure

Each training run creates a timestamped directory:
```
results_forward_comprehensive/
└── AttentionLSTM_20251116_143022_dim-512_layer-2/
    ├── best_model.pth                    # Model weights
    ├── config.json                       # Configuration + metrics
    ├── validation_metrics.json           # Detailed metrics
    ├── training_history.png              # Loss curves
    ├── validation_scatter_main.png       # Main variables scatter
    ├── validation_scatter_all.png        # All variables scatter
    ├── loss_history.npz                  # Training history
    └── train_val_indices.npz             # Train/val split indices
```

### Validation Output Structure

Time series validation creates detailed plots:
```
results_forward_comprehensive/
└── AttentionLSTM_20251116_143022_dim-512_layer-2/
    └── validation_timeseries/
        ├── timeseries_sample_0_main.png   # Main vars for sample 0
        ├── timeseries_sample_0_all.png    # All vars for sample 0
        ├── timeseries_sample_1_main.png   # Main vars for sample 1
        ├── timeseries_sample_1_all.png    # All vars for sample 1
        └── validation_metrics_summary.json # Metrics for all samples
```

### Inference Outputs

```
predictions.csv                    # Predicted time series
predictions_metrics.json           # Performance metrics (if obs provided)
predictions.png                    # Time series plots
predictions_scatter.png            # Scatter plots (if obs provided)
```

## Model Architectures

### LSTM
Standard LSTM with parameter embedding.

### BiLSTM
Bidirectional LSTM for capturing future context (useful for gap-filling).

### Attention-LSTM (Recommended)
LSTM with attention mechanism for better long-term dependencies and parameter sensitivity.

## Citation

If you use this emulator in your research, please cite:
- Noah-MP land surface model
- This emulator framework (paper in preparation)

## Version History

- **v2.0** (2025-11-16): Comprehensive model with 31 variables, integrated validation
- **v1.0** (2025-11-13): Initial 3-variable model

## Support

For issues or questions:
1. Check this README
2. Review configuration in `config_forward_comprehensive.py`
3. Run `python test_comprehensive_model.py` for diagnostics
4. Open an issue on the repository

---

**Last Updated:** 2025-11-16
**Status:** Production Ready
**Model Variants:** LSTM, BiLSTM, Attention-LSTM
**Target Variables:** 31
**Conservation:** Full Energy + Water
