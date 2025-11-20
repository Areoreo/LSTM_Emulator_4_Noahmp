# Calibration Script Usage Guide

## Overview

The `06_calibration_applying_emulator.py` script calibrates NoahMP parameters using the trained LSTM emulator by minimizing the error between emulator predictions and observations.

**Key Feature**: Unlike traditional calibration that returns only the best result, this script saves the **top N calibration results** (specified by `--num_calibration`) for ensemble prediction and sensitivity analysis.

## Requirements

1. **Trained emulator model**: A trained model directory (e.g., `results_forward_comprehensive/AttentionLSTM_20251116_171820_dim-512_layer-2/`)
2. **Forcing data**: NetCDF file with meteorological forcing (e.g., `data/raw/forcing/forcing_sample_1.nc`)
3. **Observation data**: CSV file with daily observations at `data/obs/Panama_BCI_obs_2015-07-30_2016-07-29.csv`
4. **Parameter bounds**: CSV file with parameter bounds (`value_bounds.csv`)

## Basic Usage

### Example 1: Get top 5 calibration results (default)

```bash
python 06_calibration_applying_emulator.py \
  --model_dir results_forward_comprehensive/AttentionLSTM_20251116_171820_dim-512_layer-2 \
  --forcing data/raw/forcing/forcing_sample_1.nc \
  --obs data/obs/Panama_BCI_obs_2015-07-30_2016-07-29.csv \
  --bounds value_bounds.csv \
  --output calibration_results
```

### Example 2: Get top 10 results for more comprehensive ensemble

```bash
python 06_calibration_applying_emulator.py \
  --model_dir results_forward_comprehensive/AttentionLSTM_20251116_171820_dim-512_layer-2 \
  --forcing data/raw/forcing/forcing_sample_1.nc \
  --obs data/obs/Panama_BCI_obs_2015-07-30_2016-07-29.csv \
  --bounds value_bounds.csv \
  --num_calibration 10 \
  --output calibration_results
```

### Example 3: Calibrate specific parameters only

```bash
python 06_calibration_applying_emulator.py \
  --model_dir results_forward_comprehensive/AttentionLSTM_20251116_171820_dim-512_layer-2 \
  --forcing data/raw/forcing/forcing_sample_1.nc \
  --obs data/obs/Panama_BCI_obs_2015-07-30_2016-07-29.csv \
  --bounds value_bounds.csv \
  --calibrate_params VCMX25 HVT SATDK \
  --num_calibration 5 \
  --max_iter 50
```

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model_dir` | Yes | - | Directory containing trained emulator model |
| `--forcing` | Yes | - | NetCDF file with forcing data |
| `--obs` | Yes | - | CSV file with observation data |
| `--bounds` | Yes | - | CSV file with parameter bounds |
| `--calibrate_params` | No | All params | List of parameter names to calibrate |
| `--num_calibration` | No | 5 | Number of best results to save |
| `--output` | No | `calibration_results` | Output directory |
| `--max_iter` | No | 100 | Maximum optimization iterations |
| `--popsize` | No | 15 | Population size for differential evolution |

## Output Structure

The script creates the following output structure:

```
calibration_results/
├── ensemble_summary.json          # Summary of all calibration results
├── calibration_1/                 # Best calibration result
│   ├── calibrated_parameters.csv
│   ├── calibration_result.json
│   ├── predictions_comparison.csv
│   └── calibration_results.png
├── calibration_2/                 # 2nd best result
│   ├── ...
├── calibration_3/                 # 3rd best result
│   ├── ...
└── ...
```

### File Descriptions

1. **`ensemble_summary.json`**: Overview of all calibration results including:
   - Number of calibrations performed
   - Optimization method used
   - NRMSE for each result
   - Variable-specific errors

2. **`calibration_X/calibrated_parameters.csv`**: Parameter values for each calibration
   - Baseline (original) values
   - Calibrated values
   - Flag indicating which parameters were calibrated

3. **`calibration_X/calibration_result.json`**: Detailed metrics for each calibration
   - Rank
   - Overall NRMSE
   - Variable-specific RMSE and NRMSE
   - Calibrated parameter values

4. **`calibration_X/predictions_comparison.csv`**: Time series comparison
   - Date
   - Predicted values for each variable
   - Observed values (where available)

5. **`calibration_X/calibration_results.png`**: Visualization plots
   - Time series comparison for each observed variable
   - R² and RMSE statistics

## Why Multiple Calibration Results?

### 1. **Ensemble Prediction**
Instead of relying on a single "best" calibration, you can:
- Average predictions from top N results
- Quantify prediction uncertainty using ensemble spread
- Improve robustness against overfitting

### 2. **Sensitivity Analysis**
Compare parameter variations across top results to:
- Identify which parameters are consistently similar (well-constrained)
- Find parameters with high variability (poorly constrained)
- Understand parameter equifinality

### 3. **Uncertainty Quantification**
Use the range of calibration results to:
- Estimate parameter uncertainty
- Create confidence intervals for predictions
- Identify observation-constrained vs. unconstrained parameters

## Example Workflow

### Step 1: Run Calibration
```bash
python 06_calibration_applying_emulator.py \
  --model_dir results_forward_comprehensive/AttentionLSTM_20251116_171820_dim-512_layer-2 \
  --forcing data/raw/forcing/forcing_sample_1.nc \
  --obs data/obs/Panama_BCI_obs_2015-07-30_2016-07-29.csv \
  --bounds value_bounds.csv \
  --num_calibration 10 \
  --output calibration_results
```

### Step 2: Analyze Ensemble Summary
```python
import json
import pandas as pd

# Load ensemble summary
with open('calibration_results/ensemble_summary.json') as f:
    summary = json.load(f)

# Check NRMSE range
errors = [r['nrmse'] for r in summary['results']]
print(f"Best NRMSE: {min(errors):.4f}")
print(f"Worst NRMSE (in top 10): {max(errors):.4f}")
print(f"Mean NRMSE: {sum(errors)/len(errors):.4f}")
```

### Step 3: Compare Parameter Variability
```python
import numpy as np

# Load all calibrated parameters
params_list = []
for i in range(1, 11):
    df = pd.read_csv(f'calibration_results/calibration_{i}/calibrated_parameters.csv')
    params_list.append(df[df['is_calibrated'] == 'Yes']['calibrated'].values)

params_array = np.array(params_list)

# Calculate coefficient of variation for each parameter
param_names = df[df['is_calibrated'] == 'Yes']['parameter'].values
cv = np.std(params_array, axis=0) / np.mean(params_array, axis=0)

# Identify well-constrained parameters (low CV)
well_constrained = param_names[cv < 0.1]
print("Well-constrained parameters:", well_constrained)

# Identify poorly-constrained parameters (high CV)
poorly_constrained = param_names[cv > 0.3]
print("Poorly-constrained parameters:", poorly_constrained)
```

### Step 4: Create Ensemble Predictions
```python
# Load predictions from all calibrations
predictions = []
for i in range(1, 11):
    df = pd.read_csv(f'calibration_results/calibration_{i}/predictions_comparison.csv')
    predictions.append(df[['SOIL_M_pred', 'LH_pred', 'HFX_pred']].values)

predictions_array = np.array(predictions)

# Ensemble mean
ensemble_mean = np.mean(predictions_array, axis=0)

# Ensemble spread (uncertainty)
ensemble_std = np.std(predictions_array, axis=0)

# Create ensemble prediction DataFrame
ensemble_df = pd.DataFrame({
    'date': df['date'],
    'SOIL_M_mean': ensemble_mean[:, 0],
    'SOIL_M_std': ensemble_std[:, 0],
    'LH_mean': ensemble_mean[:, 1],
    'LH_std': ensemble_std[:, 1],
    'HFX_mean': ensemble_mean[:, 2],
    'HFX_std': ensemble_std[:, 2],
})
```

## Observation Data Format

The observation CSV file should have the following format:

```csv
date,SOIL_M,LH,HFX
2015-07-30,0.352,64.16,7.65
2015-07-31,0.349,103.75,66.61
...
```

Where:
- `date`: Date in YYYY-MM-DD format
- `SOIL_M`: Volumetric soil moisture (m³/m³)
- `LH`: Latent heat flux (W/m²)
- `HFX`: Sensible heat flux (W/m²)

## Tips for Better Calibration

1. **Increase population size** for better exploration:
   ```bash
   --popsize 30 --max_iter 200
   ```

2. **Focus on specific parameters** if you know which ones matter:
   ```bash
   --calibrate_params VCMX25 HVT SATDK
   ```

3. **Get more ensemble members** for better uncertainty quantification:
   ```bash
   --num_calibration 20
   ```

4. **Use GPU** if available (automatically detected):
   - The script will use CUDA if available
   - Speeds up emulator predictions significantly

## Troubleshooting

### Issue: "No target variables found in observation data"
**Solution**: Check that your observation CSV has columns named `SOIL_M`, `LH`, and/or `HFX`.

### Issue: Calibration results are very similar
**Solution**:
- Increase `--popsize` for more diversity
- Increase `--max_iter` for better convergence
- Try calibrating fewer parameters if many are unconstrained

### Issue: High NRMSE values
**Solution**:
- Check if observation data matches the forcing data period
- Verify the emulator model was trained properly
- Consider if the parameters being calibrated are relevant to the observed variables

## Integration with Existing Scripts

This calibration script integrates seamlessly with your existing workflow:

1. **After training** (`02_train_forward_comprehensive.py`):
   - Use the trained model directory with `--model_dir`

2. **Before inference** (`04_inference.py`):
   - Use calibrated parameters instead of default values
   - Can run inference with ensemble mean parameters

3. **For validation** (`03_emulator_validation.py`):
   - Compare calibrated vs. uncalibrated predictions
   - Validate that calibration improves fit to observations
