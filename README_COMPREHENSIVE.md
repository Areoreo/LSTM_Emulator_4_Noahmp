# Comprehensive LSTM Emulator with Full Conservation Checking

This document describes the **comprehensive** version of the LSTM emulator that predicts ALL energy and water cycle variables for complete physical conservation validation.

## Overview

### Key Improvements

1. **Expanded Output Variables**: Predicts 32+ variables covering:
   - Full energy balance (5 core variables)
   - Energy components (9 detailed flux variables)
   - Water fluxes (5 variables)
   - Water storage (7 variables including multi-layer soil moisture)
   - Temperature (5 variables)

2. **Full Conservation Checking**:
   - Complete energy balance validation
   - Complete water balance validation
   - Component-level energy checks (canopy, ground, bare ground)
   - Detailed ET and runoff breakdowns

3. **Enhanced Model Architecture**:
   - Larger hidden dimensions (2048) to handle more outputs
   - Deeper network (3 layers) for complex relationships
   - Variable-specific loss weights for prioritizing critical outputs

## File Structure

### Configuration
- `config_forward_comprehensive.py` - Comprehensive model configuration with 32+ target variables

### Core Scripts
- `01_data_preprocessing_forward_comprehensive.py` - Data preprocessing for all variables
- `02_train_forward_comprehensive.py` - Training script for comprehensive model
- `04_comprehensive_conservation_validation.py` - Full conservation validation

### Conservation Checking
- `conservation_check_comprehensive.py` - Full conservation checking module
- `conservation_check.py` - Original (partial) conservation module

### Legacy Files (3-variable model)
- `config_forward.py` - Original 3-variable configuration
- `01_data_preprocessing_forward.py` - Original preprocessing
- `02_train_forward.py` - Original training
- `03_conservation_validation.py` - Original conservation checking

## Target Variables (32 total)

### Energy Balance (5 variables)
Core energy conservation: FSA - FIRA = HFX + LH + GRDFLX

```
FSA       - Total absorbed SW radiation (W/m²)
FIRA      - Total net LW radiation to atmosphere (W/m²)
HFX       - Total sensible heat to atmosphere (W/m²)
LH        - Total latent heat to atmosphere (W/m²)
GRDFLX    - Heat flux into the soil (W/m²)
```

### Energy Components (9 variables)
Detailed energy budget breakdown:

```
SAV       - Solar radiation absorbed by canopy (W/m²)
SAG       - Solar radiation absorbed by ground (W/m²)
IRC       - Canopy net LW rad (W/m²)
SHC       - Canopy sensible heat (W/m²)
EVC       - Canopy evap heat (W/m²)
IRG       - Below-canopy ground net LW rad (W/m²)
SHG       - Below-canopy ground sensible heat (W/m²)
EVG       - Below-canopy ground evap heat (W/m²)
GHV       - Below-canopy ground heat to soil (W/m²)
```

### Water Fluxes (5 variables)
Water cycle fluxes:

```
ECAN      - Canopy water evaporation rate (mm/s)
ETRAN     - Transpiration rate (mm/s)
EDIR      - Direct from soil evaporation rate (mm/s)
UGDRNOFF  - Accumulated underground runoff (mm)
SFCRNOFF  - Accumulated surface runoff (mm)
```

### Water Storage (7 variables)
Water storage state variables:

```
SOIL_M    - Volumetric soil moisture Layer 1 (m³/m³)
SOIL_M_L2 - Volumetric soil moisture Layer 2 (m³/m³)
SOIL_M_L3 - Volumetric soil moisture Layer 3 (m³/m³)
SOIL_M_L4 - Volumetric soil moisture Layer 4 (m³/m³)
CANLIQ    - Canopy liquid water content (mm)
CANICE    - Canopy ice water content (mm)
SNEQV     - Snow water equivalent (mm)
```

### Temperature (5 variables)
Temperature state variables:

```
SOIL_T    - Soil temperature Layer 1 (K)
SOIL_T_L2 - Soil temperature Layer 2 (K)
TG        - Ground temperature (K)
TV        - Vegetation temperature (K)
TRAD      - Surface radiative temperature (K)
```

## Complete Workflow

### 1. Data Preprocessing

```bash
conda activate dfm
python 01_data_preprocessing_forward_comprehensive.py
```

**Output:** `data/processed_data_forward_comprehensive.pkl`

This extracts and normalizes all 32+ target variables from simulation outputs.

### 2. Model Training

```bash
python 02_train_forward_comprehensive.py
```

**Output:**
- `results_forward_comprehensive/best_model.pth` - Trained model weights
- `results_forward_comprehensive/config.json` - Model configuration
- `results_forward_comprehensive/training_history.png` - Loss curves
- `results_forward_comprehensive/loss_history.npz` - Training metrics

**Training Configuration:**
- Learning rate: 0.0005
- Batch size: 8 (smaller due to many outputs)
- Hidden dim: 2048 (larger for complexity)
- Layers: 3 (deeper network)
- Patience: 150 epochs

### 3. Conservation Validation

```bash
python 04_comprehensive_conservation_validation.py \
    --model_dir results_forward_comprehensive \
    --n_samples 20 \
    --sample_detail 0
```

**Arguments:**
- `--model_dir`: Directory with trained model (required)
- `--n_samples`: Number of test samples to validate (default: 20)
- `--sample_detail`: Sample index for detailed plots (optional)
- `--output_dir`: Custom output directory (default: model_dir/comprehensive_conservation)

**Outputs:**
- `comprehensive_conservation_summary.png` - Aggregated statistics
- `energy_conservation_comprehensive.png` - Detailed energy balance
- `water_conservation_comprehensive.png` - Detailed water balance
- `comprehensive_conservation_results.json` - Numerical results
- `comprehensive_conservation_report.txt` - Human-readable report

## Conservation Equations

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
- Total ET = `ECAN + ETRAN + EDIR` (converted from mm/s to mm/day)
- Total Runoff = `ΔUGDRNOFF + ΔSFCRNOFF` (daily rates from accumulated)
- Storage = `SOIL_M_all_layers + CANLIQ + CANICE + SNEQV`

## Loss Weights

The model uses variable-specific loss weights to prioritize important outputs:

```python
# Energy balance variables: weight = 2.0 (highest priority)
# Energy components: weight = 1.5
# Water fluxes: weight = 2.0
# Water storage (layer 1): weight = 2.0
# Water storage (other layers): weight = 1.5
# Temperature: weight = 1.0-1.5
```

These weights ensure the model focuses on physically critical variables while still learning all outputs.

## Interpreting Results

### Energy Conservation

**Excellent:** RMSE < 10 W/m²
- Model predictions maintain strong energy balance
- Physical consistency very high

**Good:** RMSE < 30 W/m²
- Model shows reasonable energy conservation
- Minor deviations acceptable

**Acceptable:** RMSE < 50 W/m²
- Some energy balance violations
- May need model refinement

**Poor:** RMSE > 50 W/m²
- Significant conservation violations
- Model needs retraining or architecture changes

### Water Conservation

**Excellent:** RMSE < 0.1 mm/day
- Strong water balance maintenance
- Very high physical consistency

**Good:** RMSE < 0.5 mm/day
- Good water conservation
- Acceptable for most applications

**Acceptable:** RMSE < 1.0 mm/day
- Minor water balance issues
- Consider model improvements

**Poor:** RMSE > 1.0 mm/day
- Significant water conservation violations
- Requires model refinement

## Model Comparison

### Original Model (3 variables)
- **Targets:** SOIL_M, LH, HFX
- **Parameters:** ~2M
- **Conservation:** Partial (energy only, using predicted HFX/LH)

### Comprehensive Model (32 variables)
- **Targets:** All energy/water cycle variables
- **Parameters:** ~8-10M (larger network)
- **Conservation:** Full (energy + water, all components)

## Use Cases

### 1. Research & Development
- Understanding model physics learning
- Validating emulator against conservation laws
- Identifying which processes are well/poorly captured

### 2. Parameter Sensitivity Analysis
- Predict full land surface response to parameter changes
- Analyze cascading effects through energy/water cycles
- Identify parameter importance for specific fluxes

### 3. Uncertainty Quantification
- Propagate parameter uncertainty through full model
- Assess conservation constraint violations under uncertainty
- Validate ensemble predictions

### 4. Model Benchmarking
- Compare emulator predictions against full NoahMP
- Validate conservation across wide parameter ranges
- Test emulator performance on extreme conditions

## Troubleshooting

### Issue: Training loss not decreasing

**Solutions:**
1. Reduce learning rate (try 0.0001)
2. Check data for NaN/Inf values
3. Reduce model complexity (fewer layers or smaller hidden_dim)
4. Adjust loss weights to balance variable importance

### Issue: Poor conservation for specific variable category

**Solutions:**
1. Increase loss weight for that category
2. Check if simulation data has issues for those variables
3. Add physics-informed loss terms
4. Increase model capacity (hidden_dim, num_layers)

### Issue: Memory errors during training

**Solutions:**
1. Reduce batch_size (try 4 or 2)
2. Reduce hidden_dim (try 1024 or 1536)
3. Use gradient accumulation
4. Train on GPU with more memory

### Issue: Conservation good on training, poor on test

**Solutions:**
1. Increase training data
2. Add regularization (increase dropout)
3. Use learning rate scheduling
4. Implement early stopping

## Advanced Features

### Custom Variable Sets

You can modify `config_forward_comprehensive.py` to predict different variable combinations:

```python
# Example: Focus only on energy balance
TARGET_VARIABLES = ENERGY_BALANCE_TARGETS + ENERGY_COMPONENT_TARGETS

# Example: Add custom variables
CUSTOM_TARGETS = [
    {
        'name': 'YOUR_VARIABLE',
        'aggregation': 'mean',
        'description': 'Description',
        'category': 'custom'
    }
]
TARGET_VARIABLES = ENERGY_BALANCE_TARGETS + CUSTOM_TARGETS
```

### Physics-Informed Training

Future enhancement: Add conservation constraints directly to loss function:

```python
# Pseudo-code for physics-informed loss
energy_residual = (FSA - FIRA) - (HFX + LH + GRDFLX)
physics_loss = torch.mean(energy_residual ** 2)
total_loss = prediction_loss + lambda_physics * physics_loss
```

### Multi-Site Extension

Current: Single-point emulation
Extension: Multi-site with site-specific parameters

```python
# Add site embedding to model input
# Predict site-specific responses
# Validate conservation per site
```

## Citations & References

If you use this comprehensive emulator in your research, please cite:

1. The NoahMP land surface model
2. The LSTM architecture papers
3. This emulator framework (when published)

## Support & Contributing

For issues, questions, or contributions:
- Create an issue in the repository
- Contact the development team
- Refer to main project documentation

---

**Version:** 2.0 (Comprehensive)
**Last Updated:** 2025-11-15
**Status:** Development/Testing
