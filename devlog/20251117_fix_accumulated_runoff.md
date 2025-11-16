# Fix Accumulated Runoff Variables for Water Conservation

**Date:** 2025-11-17
**Issue:** Poor water conservation (RMSE: 29 mm/day vs target <0.5 mm/day)
**Root Cause:** Model predicting accumulated runoff (monotonically increasing) instead of rates

## Changes Made

### 1. Removed CANICE and SNEQV (tropical site, always 0)
- **Config:** Removed from OUTPUT_VARS and OUTPUT_WEIGHTS
- **README:** Updated variable count 31→29
- **Conservation check:** Already defensive with .get()

### 2. Converted Accumulated Runoff to Rates

#### Config (`config_forward_comprehensive.py`)
```python
# UGDRNOFF: accumulated (mm) → UGDRNOFF_RATE: rate (mm/day)
# SFCRNOFF: accumulated (mm) → SFCRNOFF_RATE: rate (mm/day)
{
    'name': 'UGDRNOFF',  # NetCDF variable
    'output_name': 'UGDRNOFF_RATE',  # After conversion
    'aggregation': 'last',  # Last accumulated value
    'convert_accumulated_to_rate': True  # Flag
}
```

#### Preprocessing (`01_data_preprocessing_forward_comprehensive.py`)
- Lines 67-72: Load with raw name for variables needing conversion
- Lines 119-123: Build aggregation dict with raw names
- Lines 127-145: Convert accumulated → rate using np.diff()
- First day rate = first accumulated value (not 0)

#### Conservation Check (`conservation_check_comprehensive.py`)
- Lines 172-177: Use rate predictions directly (no more np.diff())
- Eliminated error amplification from differentiation

#### README
- Updated variable descriptions and formulas
- UGDRNOFF → UGDRNOFF_RATE (mm/day)

### 3. Increased Runoff Weights
- Changed from 1.5 → 2.0 (critical for water conservation)

## Why This Fixes Water Conservation

1. **Stationary data**: Rates don't grow unbounded → easier to normalize
2. **No error accumulation**: No differencing of predictions
3. **Physical patterns**: LSTM learns daily runoff variation
4. **No error propagation**: Prediction errors don't cascade

## Expected Results

- Water conservation RMSE: 29 mm/day → <1 mm/day (target: ACCEPTABLE)
- Potentially <0.5 mm/day (target: GOOD)
- Runoff predictions: realistic daily variation (not monotonic)

## Status

✅ Preprocessing script verified working (10/10 samples loaded)
⏳ Awaiting retraining and validation

## Next Steps

1. Retrain model with new preprocessed data
2. Run comprehensive conservation validation
3. Compare water conservation metrics before/after
