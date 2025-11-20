# Parameter Uncertainty Quantification Guide

## Problem Statement

The initial calibration implementation had a critical flaw: it was selecting multiple results from a **single optimization run**, which meant all selected parameter sets were nearly identical. This doesn't provide meaningful uncertainty quantification.

## Solution: Multiple Independent Calibration Runs

The updated `06_calibration_applying_emulator.py` now performs **multiple independent calibration runs**, each with a different random seed. This provides genuine parameter diversity and proper uncertainty quantification.

### What Changed

**Before:**
```python
# Single optimization run, selecting top N from all evaluations
result = differential_evolution(..., seed=42)
top_results = sorted(all_evaluations)[:N]  # ❌ All very similar!
```

**After:**
```python
# Multiple independent optimization runs
for i in range(num_calibration):
    result = differential_evolution(..., seed=42 + i)  # Different seed each time
    independent_results.append(result)  # ✅ Genuinely diverse results!
```

## How It Works

### 1. Run Independent Calibrations

Each calibration run:
- Uses a **different random seed** (42, 43, 44, ...)
- Starts from **different initial populations**
- Explores **different regions** of parameter space
- May converge to **different local optima**

```bash
python 06_calibration_applying_emulator.py \
  --model_dir results_forward_comprehensive/AttentionLSTM_20251116_171820_dim-512_layer-2 \
  --forcing data/raw/forcing/forcing_sample_1.nc \
  --obs data/obs/Panama_BCI_obs_2015-07-30_2016-07-29.csv \
  --bounds value_bounds.csv \
  --num_calibration 20 \
  --max_iter 100
```

This runs **20 independent calibrations**, each potentially finding different parameter combinations.

### 2. Analyze Parameter Distributions

The script automatically creates:

#### A. Parameter Distribution Plot (`parameter_distributions.png`)
- Histograms for each parameter
- Shows mean, median, and standard deviation
- Displays Coefficient of Variation (CV) for uncertainty quantification

#### B. Parameter Correlation Matrix (`parameter_correlations.png`)
- Shows which parameters co-vary across calibration runs
- High correlation → parameters compensate for each other (equifinality)
- Low correlation → parameters are independently constrained

#### C. Parameter Statistics (`parameter_statistics.json`)
```json
{
  "VCMX25": {
    "baseline": 50.0,
    "mean": 48.3,
    "std": 2.1,
    "cv": 0.043,    // Coefficient of Variation (std/mean)
    "min": 45.2,
    "max": 51.8,
    "median": 48.5,
    "all_values": [48.3, 47.9, 49.1, ...]
  }
}
```

### 3. Interpret Results

Use the additional analysis script:

```bash
python analyze_calibration_results.py \
  --calibration_dir calibration_results
```

This creates comprehensive visualizations showing:

1. **Parameter Uncertainty**: Error bars on each parameter
2. **Constraint Quality**: Which parameters are well-determined vs. uncertain
3. **Calibration Impact**: How much parameters changed from baseline
4. **NRMSE Distribution**: Performance variation across runs

## Understanding Parameter Uncertainty

### Coefficient of Variation (CV)

CV = Standard Deviation / Mean

**Interpretation:**
- **CV < 5%**: Well-constrained parameter
  - Observation data strongly constrains this parameter
  - Can use with confidence

- **CV 5-15%**: Moderately-constrained parameter
  - Some uncertainty remains
  - Consider using ensemble mean

- **CV > 15%**: Poorly-constrained parameter
  - High uncertainty
  - Observations don't strongly constrain this parameter
  - Multiple parameter values fit the data equally well (equifinality)

### Example Output

```
Parameter Statistics:
--------------------------------------------------------------------------------
Parameter       Mean         Std          CV (%)     Min          Max
--------------------------------------------------------------------------------
VCMX25          48.3245      2.0812       4.31       45.2100      51.7800  ✓ Well-constrained
HVT             19.8234      5.2341       26.40      12.3400      28.9100  ⚠ Poorly-constrained
SATDK           0.0234       0.0012       5.13       0.0215       0.0253   ✓ Well-constrained
```

## Using Results for Prediction

### Option 1: Use Best Calibration
```python
# Use the calibration with lowest NRMSE
best_params = calibration_results['calibration_1/calibrated_parameters.csv']
```

### Option 2: Ensemble Mean (Recommended)
```python
import pandas as pd
import numpy as np

# Load all calibrated parameters
all_params = []
for i in range(1, 21):  # 20 calibrations
    df = pd.read_csv(f'calibration_results/calibration_{i}/calibrated_parameters.csv')
    all_params.append(df['calibrated'].values)

# Calculate ensemble mean
ensemble_mean_params = np.mean(all_params, axis=0)
```

### Option 3: Weighted Ensemble
```python
# Weight by inverse NRMSE
weights = [1/r['nrmse'] for r in results]
weights = np.array(weights) / np.sum(weights)

weighted_mean_params = np.average(all_params, axis=0, weights=weights)
```

## Generating Prediction Uncertainty

Create ensemble predictions to quantify prediction uncertainty:

```python
import numpy as np
import pandas as pd

# Load predictions from all calibrations
predictions = []
for i in range(1, 21):
    df = pd.read_csv(f'calibration_results/calibration_{i}/predictions_comparison.csv')
    predictions.append(df[['SOIL_M_pred', 'LH_pred', 'HFX_pred']].values)

predictions_array = np.array(predictions)  # Shape: (20, n_timesteps, 3)

# Calculate ensemble statistics
ensemble_mean = np.mean(predictions_array, axis=0)
ensemble_std = np.std(predictions_array, axis=0)
ensemble_min = np.min(predictions_array, axis=0)
ensemble_max = np.max(predictions_array, axis=0)

# Create prediction intervals
lower_95 = ensemble_mean - 1.96 * ensemble_std
upper_95 = ensemble_mean + 1.96 * ensemble_std

# Plot with uncertainty bands
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

variables = ['SOIL_M', 'LH', 'HFX']
for i, (ax, var) in enumerate(zip(axes, variables)):
    # Plot ensemble mean
    ax.plot(dates, ensemble_mean[:, i], 'b-', linewidth=2, label='Ensemble Mean')

    # Plot uncertainty band
    ax.fill_between(dates, lower_95[:, i], upper_95[:, i],
                     alpha=0.3, color='blue', label='95% Prediction Interval')

    # Plot individual calibrations (optional, for visualization)
    for j in range(min(5, len(predictions))):  # Plot first 5
        ax.plot(dates, predictions[j][:, i], 'gray', alpha=0.3, linewidth=0.5)

    # Plot observations
    if f'{var}_obs' in df.columns:
        ax.scatter(dates, df[f'{var}_obs'], color='red', s=20,
                  alpha=0.6, label='Observations', zorder=10)

    ax.set_ylabel(var)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.savefig('ensemble_predictions_with_uncertainty.png', dpi=300, bbox_inches='tight')
```

## Recommended Workflow

### 1. Quick Test (5 runs)
```bash
# Fast check with 5 independent runs
python 06_calibration_applying_emulator.py \
  --num_calibration 5 \
  --max_iter 50 \
  ...
```

### 2. Production Run (20+ runs)
```bash
# Robust uncertainty quantification with 20+ runs
python 06_calibration_applying_emulator.py \
  --num_calibration 20 \
  --max_iter 100 \
  ...
```

### 3. Analyze Results
```bash
# Create comprehensive analysis
python analyze_calibration_results.py \
  --calibration_dir calibration_results
```

### 4. Check Key Metrics

Look at the analysis output:

1. **NRMSE Statistics**:
   - Low std → Consistent performance across runs ✓
   - High std → Results vary significantly ⚠

2. **Parameter CV**:
   - Most CV < 10% → Well-constrained problem ✓
   - Many CV > 20% → Equifinality issues ⚠

3. **Parameter Correlations**:
   - High correlations → Parameters compensate for each other
   - Consider calibrating only well-constrained subset

## Troubleshooting

### Issue: All calibration results are still very similar

**Causes:**
- Too few iterations (increase `--max_iter`)
- Too small parameter bounds
- Observations don't constrain parameters well

**Solutions:**
1. Increase iterations: `--max_iter 200`
2. Check parameter bounds in `value_bounds.csv`
3. Add more/better quality observations
4. Calibrate fewer parameters (only the most sensitive ones)

### Issue: High parameter uncertainty (CV > 30%)

**Interpretation:** Multiple parameter combinations fit the data equally well (equifinality)

**Solutions:**
1. **Use ensemble mean** instead of single calibration
2. **Add more observations** (different times, different variables)
3. **Calibrate fewer parameters** (fix uncertain ones at default values)
4. **Add regularization** (penalize large deviations from baseline)

### Issue: Parameters hitting bounds

**Symptoms:**
- Many calibrations have parameters at min/max bounds
- High CV for bounded parameters

**Solutions:**
1. **Widen bounds** if physically reasonable
2. **Check if bounds are too tight**
3. **Examine if model/data mismatch** causing extreme parameter values

## Scientific Interpretation

### Well-Constrained Parameters
- **CV < 5%**: Observations provide strong information
- Can use calibrated value with confidence
- Important for predictions

### Poorly-Constrained Parameters
- **CV > 15%**: Multiple values fit observations equally well
- Indicates:
  - Parameter insensitivity (doesn't affect observed variables much)
  - Parameter compensation (other parameters can adjust to compensate)
  - Insufficient observational constraints
- **Don't worry!** Use ensemble mean for robust predictions

### Parameter Correlations
- **High correlation** (|r| > 0.7): Parameters compensate for each other
  - Example: `VCMX25` and `HVT` might trade off in affecting `LH`
  - This is equifinality
  - Use ensemble approach

- **Low correlation** (|r| < 0.3): Parameters independently constrained
  - Good! Each parameter uniquely determined
  - High confidence in calibrated values

## Summary

The updated calibration approach provides:

✅ **True uncertainty quantification** through independent runs
✅ **Parameter distributions** showing probability ranges
✅ **Correlation analysis** revealing parameter interactions
✅ **Ensemble predictions** with uncertainty bands
✅ **Statistical diagnostics** (CV, ranges, confidence intervals)

This enables robust parameter estimation and prediction uncertainty quantification for your Noah-MP emulator!
