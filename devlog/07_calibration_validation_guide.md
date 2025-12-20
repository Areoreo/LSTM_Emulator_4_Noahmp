# Calibration Validation Guide

## Overview

The calibration validation workflow (Step 7) validates emulator-based calibration results by comparing Noah-MP model outputs with three different parameter sets against real observations:

1. **Emulator-calibrated parameters** - Parameters obtained from the emulator-based calibration process
2. **Expert-calibrated parameters** - Manually calibrated parameters by domain experts
3. **Default parameters** - Default Noah-MP parameter values

## Files

### Main Scripts

- **`07_calibration_validation.sh`** - Shell script that orchestrates the entire validation workflow
- **`07_calibration_validation.py`** - Python script for metrics calculation and visualization

### Input Files

- **Calibration results**: `calibration_results/{CALIBRATION_ID}/calibrated_parameters.csv`
- **Expert calibration**: `data/raw/param/expert_calibration.txt`
- **Default parameters**: `data/raw/param/default_param.txt`
- **Observations**: `data/obs/Panama_BCI_obs_2015-07-30_2016-07-29.csv`

### Output Files

All outputs are saved to `validation_results/{CALIBRATION_ID}/`:

- **`validation_metrics.csv`** - Comprehensive metrics table (BIAS, MAE, RMSE, PBIAS)
- **`timeseries_*.png`** - Time series comparison plots for each variable
- **`scatter_*.png`** - Scatter plots comparing simulations vs observations
- **`metrics_comparison_*.png`** - Bar charts comparing metrics across calibrations
- **`emulator_output.csv`** - Noah-MP output with emulator-calibrated parameters
- **`expert_output.csv`** - Noah-MP output with expert-calibrated parameters
- **`default_output.csv`** - Noah-MP output with default parameters

## Usage

### Basic Usage

```bash
cd /home/petrichor/ymwang/snap/LSTM_Emulator_4_Noahmp

# Validate a specific calibration result
bash 07_calibration_validation.sh calibration_1
```

### Advanced Usage

```bash
# Use custom observation file
bash 07_calibration_validation.sh calibration_1 \\
    --obs_file data/obs/Panama_BCI_obs_2016-07-30_2017-07-29.csv

# Validate specific variables only
bash 07_calibration_validation.sh calibration_1 \\
    --variables "ET GPP"

# Skip default parameter run (faster)
bash 07_calibration_validation.sh calibration_1 \\
    --skip_default

# Use existing Noah-MP outputs (for re-running analysis only)
bash 07_calibration_validation.sh calibration_1 \\
    --skip_noahmp

# Show help
bash 07_calibration_validation.sh --help
```

## Workflow Steps

The validation script performs the following steps:

### Step 1: Convert Calibrated Parameters

Converts the emulator-calibrated parameters from CSV format to the space-delimited format required by the TBL generator.

**Input**: `calibration_results/{CALIBRATION_ID}/calibrated_parameters.csv`  
**Output**: `validation_results/{CALIBRATION_ID}/emulator_calibrated_params.txt`

### Step 2: Generate TBL Files

Uses the TBL generator to create Noah-MP parameter table files for each parameter set.

**Outputs**:
- `validation_results/{CALIBRATION_ID}/emulator_tbl/NoahmpTable_4emu_1/NoahmpTable.TBL`
- `validation_results/{CALIBRATION_ID}/expert_tbl/NoahmpTable_4emu_1/NoahmpTable.TBL`
- `validation_results/{CALIBRATION_ID}/default_tbl/NoahmpTable_4emu_1/NoahmpTable.TBL`

### Step 3: Run Noah-MP Simulations

Runs the Noah-MP model with each parameter table file.

**Outputs**:
- `validation_results/{CALIBRATION_ID}/run_emulator/output/`
- `validation_results/{CALIBRATION_ID}/run_expert/output/`
- `validation_results/{CALIBRATION_ID}/run_default/output/`

### Step 4: Parse Noah-MP Outputs

Converts Noah-MP output files (NetCDF or text) to CSV format for analysis.

**Outputs**:
- `validation_results/{CALIBRATION_ID}/emulator_output.csv`
- `validation_results/{CALIBRATION_ID}/expert_output.csv`
- `validation_results/{CALIBRATION_ID}/default_output.csv`

### Step 5: Validation Analysis

Calculates metrics and generates comparison plots.

**Metrics Calculated**:
- **BIAS**: Mean difference between simulated and observed values
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **PBIAS**: Percent Bias

**Visualizations**:
- Time series plots showing all three calibrations vs observations
- Scatter plots for each calibration vs observations (with 1:1 line)
- Bar charts comparing metrics across calibrations

## Validation Metrics

### BIAS (Bias)
- Mean difference between simulated and observed values
- **Formula**: BIAS = mean(sim - obs)
- **Ideal value**: 0
- **Interpretation**: Positive values indicate overestimation, negative indicate underestimation

### MAE (Mean Absolute Error)
- Average magnitude of errors
- **Formula**: MAE = mean(|sim - obs|)
- **Ideal value**: 0
- **Range**: [0, ∞)

### RMSE (Root Mean Square Error)
- Square root of average squared errors
- **Formula**: RMSE = sqrt(mean((sim - obs)²))
- **Ideal value**: 0
- **Range**: [0, ∞)
- **Note**: More sensitive to large errors than MAE

### PBIAS (Percent Bias)
- Tendency of simulated values to be larger or smaller than observed
- **Formula**: PBIAS = 100 × sum(sim - obs) / sum(obs)
- **Ideal value**: 0
- **Interpretation**:
  - PBIAS < -10%: Model overestimates
  - -10% ≤ PBIAS ≤ 10%: Very good
  - 10% < PBIAS ≤ 25%: Good
  - PBIAS > 25%: Model underestimates significantly

## Example Output

After running the validation, you'll see output like:

```
==================================================================
Processing variable: ET
==================================================================

Emulator Calibration - ET:
  BIAS:  0.0234
  MAE:   0.3456
  RMSE:  0.4567
  PBIAS: 5.23%
  N:     8760

Expert Calibration - ET:
  BIAS:  -0.0456
  MAE:   0.3890
  RMSE:  0.5012
  PBIAS: -10.45%
  N:     8760

Default Calibration - ET:
  BIAS:  0.1234
  MAE:   0.5678
  RMSE:  0.7890
  PBIAS: 28.90%
  N:     8760
```

## Interpreting Results

### Good Calibration Indicators

1. **Lower metrics are better**: BIAS ≈ 0, low MAE/RMSE, |PBIAS| < 10%
2. **Emulator calibration should outperform default**: Validates emulator effectiveness
3. **Emulator competitive with expert**: Suggests automated calibration is reliable

### Common Issues

- **High BIAS**: Systematic over/under-estimation → Check parameter ranges
- **Low N values**: Data alignment issues → Check timestamps in observation/simulation files
- **NaN metrics**: Missing variables → Check variable names match between files
- **All runs fail**: Noah-MP configuration issue → Check `noahmp.log` in run directories

## Troubleshooting

### Noah-MP Fails to Run

Check the log files:
```bash
cat validation_results/{CALIBRATION_ID}/run_emulator/noahmp.log
```

Common issues:
- Missing forcing data files
- Incorrect paths in namelist.hrldas
- TBL file format errors

### Variable Not Found

Check variable names in observation file:
```bash
head -1 data/obs/Panama_BCI_obs_2015-07-30_2016-07-29.csv
```

Adjust variables in command:
```bash
bash 07_calibration_validation.sh calibration_1 --variables "H LE GPP"
```

### Re-run Analysis Only

If Noah-MP runs completed but you want to regenerate plots:
```bash
bash 07_calibration_validation.sh calibration_1 --skip_noahmp
```

## Integration with Calibration Workflow

This validation step (7) follows the calibration step (6):

```bash
# Step 6: Run calibration
python 06_calibration_applying_emulator.py \\
    --forcing data/raw/forcing/forcing_sample_1.nc \\
    --obs data/obs/Panama_BCI_obs_2015-07-30_2016-07-29.csv \\
    --bounds value_bounds.csv \\
    --num_calibration 200 \\
    --max_iter 500

# Step 7: Validate calibration results
bash 07_calibration_validation.sh calibration_1
```

## Batch Validation

To validate multiple calibration results:

```bash
# Create a batch validation script
for i in {1..10}; do
    echo "Validating calibration_$i"
    bash 07_calibration_validation.sh calibration_$i
done

# Or use parallel processing
parallel bash 07_calibration_validation.sh ::: calibration_{1..10}
```

## Output Summary

After completion, check the summary:
```bash
cat validation_results/calibration_1/validation_metrics.csv
```

View plots:
```bash
# On local machine with X11 forwarding
display validation_results/calibration_1/timeseries_ET.png

# Or copy to local machine
scp -r petrichor:/home/petrichor/ymwang/snap/LSTM_Emulator_4_Noahmp/validation_results/calibration_1 .
```

## Notes

- **Runtime**: Complete validation takes ~15-30 minutes per calibration (depending on Noah-MP simulation length)
- **Disk space**: Each validation uses ~100-500 MB depending on output frequency
- **Parallelization**: Can run multiple validations in parallel if system resources allow
- **Variables**: Default variables (ET, GPP, H, LE) can be customized based on your observation data

## References

- Noah-MP Documentation: https://ral.ucar.edu/model/noah-multiparameterization-land-surface-model-noah-mp-lsm
- Model Performance Metrics: Moriasi et al. (2007) - https://doi.org/10.13031/2013.23153
