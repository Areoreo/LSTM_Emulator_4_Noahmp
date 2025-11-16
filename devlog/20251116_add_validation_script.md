# Add Detailed Time Series Validation Script

**Date:** 2025-11-16
**Task:** Add script for detailed time series validation of test samples

## Changes Made

### New Script: 03_validation.py
- Generates detailed time series comparison plots for specific test samples
- Accepts sample range arguments: `--sample START END`
- Loads train/val split indices to map validation indices to dataset indices
- Creates plots for each variable for each sample:
  - Main variables (SOIL_M, LH, HFX) in large format
  - All 31 variables in grid format
- Computes R² and RMSE for each variable
- Saves metrics summary as JSON

### Updated Training Script: 02_train_forward_comprehensive.py
- Now saves `train_val_indices.npz` with train/val split indices
- Enables reproducible validation and proper sample tracking

### Script Renumbering
- `03_inference.py` → `04_inference.py`
- `04_comprehensive_conservation_validation.py` → `05_comprehensive_conservation_validation.py`

### Documentation Updates
- README.md updated with new script workflow
- Added validation step in Quick Start guide
- Updated all script references
- Added validation output structure documentation

## New Workflow

1. **01_data_preprocessing_forward_comprehensive.py** - Data preprocessing
2. **02_train_forward_comprehensive.py** - Training (saves indices)
3. **03_validation.py** - Time series validation on test samples (NEW)
4. **04_inference.py** - Production inference
5. **05_comprehensive_conservation_validation.py** - Conservation validation

## Usage Examples

```bash
# Validate samples 0-4 from validation set
python 03_validation.py \
  --model_dir results_forward_comprehensive/AttentionLSTM_20251116_143022_dim-512_layer-2 \
  --sample 0 5

# Validate all test samples
python 03_validation.py \
  --model_dir results_forward_comprehensive/AttentionLSTM_20251116_143022_dim-512_layer-2 \
  --sample 0 -1
```

## Output Structure

```
model_dir/validation_timeseries/
├── timeseries_sample_0_main.png    # SOIL_M, LH, HFX
├── timeseries_sample_0_all.png     # All 31 variables
├── timeseries_sample_1_main.png
├── timeseries_sample_1_all.png
└── validation_metrics_summary.json
```

## Key Features
- Maps validation set indices to full dataset indices
- Shows both indices in output for traceability
- Focuses on main target variables with detailed plots
- Provides comprehensive overview with all variables
- Saves all metrics for later analysis
