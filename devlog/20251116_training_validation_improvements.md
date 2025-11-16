# Training and Validation Improvements

**Date:** 2025-11-16
**Task:** Integrate validation, improve outputs, consolidate docs

## Changes Made

### Training Script (02_train_forward_comprehensive.py)
- Added timestamped output folders per training run
- Integrated validation metrics computation (R², RMSE, PBIAS)
- Added scatter plots for all variables (main + all)
- Focus on main targets: SOIL_M, LH, HFX
- Save validation_metrics.json automatically

### Inference Script (03_inference.py)
- Updated to use config_forward_comprehensive
- Now compatible with comprehensive model

### Documentation
- Consolidated all docs into single README.md
- Removed redundant README_COMPREHENSIVE.md and COMPREHENSIVE_MODEL_SUMMARY.md
- Clear quick start guide and usage examples

### Files Removed
- 02b_validation_forward.py (integrated into training script)

## New Output Structure

Training creates timestamped folders:
```
results_forward_comprehensive/
└── AttentionLSTM_20251116_143022_dim-512_layer-2/
    ├── best_model.pth
    ├── config.json
    ├── validation_metrics.json
    ├── validation_scatter_main.png (SOIL_M, LH, HFX)
    ├── validation_scatter_all.png (all 31 vars)
    ├── training_history.png
    └── loss_history.npz
```

## Key Features
- Automatic validation during training
- Metrics for all 31 variables
- Focused validation plots for main targets
- Comprehensive scatter plots
- Everything saved automatically
