# Fix Inference Script Observation Requirements

**Date:** 2025-11-16
**Task:** Modify inference script to only require main target variables in observations

## Problem

The 04_inference.py script required observations for all 31 variables when comparing predictions with observations. This was impractical since observations are typically only available for the 3 main target variables (SOIL_M, LH, HFX).

## Changes Made

### Updated Functions

1. **load_observations_from_csv()**
   - Now only loads main target variables (SOIL_M, LH, HFX)
   - Automatically detects which main variables are available in CSV
   - Returns variable names and indices for proper indexing
   - Provides clear messages about which variables are available/missing

2. **compute_metrics()**
   - Updated to work with subset of variables
   - Only computes metrics for variables that have observations
   - Parameter changed from `target_var_names` to `obs_var_names`

3. **plot_predictions()**
   - Added parameters: `obs_var_names`, `obs_indices`
   - When observations provided, only plots variables with observations
   - Properly indexes predictions to match observations

4. **plot_scatter()**
   - Updated to accept `obs_var_names` and `obs_indices`
   - Only creates scatter plots for variables with observations
   - Title updated to indicate "Main Variables"

### Main Function Updates

- Updated observation loading to unpack 4 values (added `obs_var_names`, `obs_indices`)
- Extract only predicted values for variables with observations before computing metrics
- Pass new parameters to plotting functions

### Documentation Updates

- Updated `--obs` help text to clarify only main variables needed
- Added example CSV format in epilog
- Updated examples to use correct script number (04) and paths

## New Observation File Format

CSV file only needs main target variables:
```csv
date,SOIL_M,LH,HFX
2015-07-30,0.25,120.5,45.2
2015-07-31,0.24,118.3,43.1
```

Can also include subset:
```csv
date,SOIL_M,LH
2015-07-30,0.25,120.5
2015-07-31,0.24,118.3
```

## Benefits

- More practical for real-world use cases
- Reduces data preparation burden
- Aligns with focus on main target variables
- Clear error messages when required variables missing
- Flexible - accepts any combination of main variables
