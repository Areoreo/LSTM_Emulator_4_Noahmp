# LSTM-Based Parameter Prediction from Simulation Results

This project implements an LSTM neural network to predict NoahMP land surface model parameters from simulation outputs. This is an **inverse modeling** problem where we estimate the parameters that generated observed time series data.

## ✨ Key Features

- **Flexible Variable Configuration**: Easily add or remove input variables via `config.py`
- **Automatic Model Adaptation**: Model automatically adjusts to any number of input variables
- **Multiple Aggregation Methods**: Support for mean, sum, min, max daily aggregations
- **Multi-layer Variable Support**: Handle variables with multiple layers (e.g., soil layers)
- **Temporal Features**: Optional day-of-year and month encoding
- **🎯 Parameter Loss Weights**: Focus training on specific parameters with customizable loss weights

## 🚀 Quick Start

```bash
# 1. View/edit configuration
python config.py

# 2. Preprocess data
python data_preprocessing.py

# 3. Train model
python train.py

# 4. Analyze results
python analyze_results.py
```

**See [USAGE_GUIDE.md](USAGE_GUIDE.md) for detailed instructions on adding new variables.**

**See [PARAMETER_WEIGHTS_GUIDE.md](PARAMETER_WEIGHTS_GUIDE.md) for using weighted loss to focus on specific parameters.**

## Project Overview

### Problem Statement

- **Input**: Time series of physical variables (soil moisture, latent heat, sensible heat) at daily resolution
- **Output**: 13 NoahMP model parameters that generated these simulations
- **Approach**: Use bidirectional LSTM to capture temporal patterns and predict parameters

### Data Description

#### Input Data (Time Series)

- **SOIL_M**: Volumetric soil moisture (m³/m³) - first soil layer
- **LH**: Latent heat flux (W/m²)
- **HFX**: Sensible heat flux (W/m²)
- **doy**: Day of year (normalized, 0-1)

Data shape: `(n_samples, n_variables, n_timesteps)` = `(144, 4, 366)`

#### Output Data (Parameters)

13 NoahMP parameters:

- `VCMX25_EBF`: Maximum carboxylation rate
- `HVT_EBF`, `HVB_EBF`: Vegetation height parameters
- `CWPVT_EBF`: Canopy wind parameter
- `Z0MVT_EBF`: Momentum roughness length
- `WLTSMC_CL`, `REFSMC_CL`, `MAXSMC_CL`, `SATDK_CL`: Clay soil parameters
- `WLTSMC_SCL`, `REFSMC_SCL`, `MAXSMC_SCL`, `SATDK_SCL`: Sandy clay soil parameters

Data shape: `(n_samples, n_params)` = `(144, 13)`

## Project Structure

```
LSTM_Emulator/
├── config.py                               # 🔧 Main configuration file
├── config_extended.py                      # Example config with more variables
├── data/
│   ├── raw/
│   │   ├── param/
│   │   │   └── noahmp_param_sets.txt      # Parameter sets
│   │   └── sim_results/
│   │       └── sample_{1-150}/             # Simulation outputs
│   └── processed_data.pkl                  # Preprocessed data
├── results/
│   └── LSTM_YYYYMMDD_HHMMSS/              # Training results
│       ├── best_model.pth                  # Saved model weights
│       ├── config.json                     # Model configuration
│       ├── metrics.json                    # Validation metrics
│       ├── training_history.pkl            # Training history
│       ├── training_history.png            # Loss curves
│       ├── predictions.png                 # Prediction plots
│       ├── r2_comparison.png               # R² comparison
│       ├── results_summary.txt             # Detailed report
│       └── summary_table.csv               # Metrics table
├── data_preprocessing.py                   # Data loading and preprocessing
├── lstm_model.py                           # LSTM model architectures
├── train.py                                # Training script
├── analyze_results.py                      # Results analysis
├── README.md                               # Project overview
└── USAGE_GUIDE.md                          # 📖 Detailed usage guide
```

## Installation & Setup

### Environment Setup

```bash
# Activate conda environment
conda activate dfm

# Required packages (should already be installed):
# - pytorch
# - xarray
# - netCDF4
# - numpy
# - pandas
# - matplotlib
# - scikit-learn
```

## Usage

### 0. Configuration (Optional)

All settings are controlled via `config.py`. Edit this file to:

- Add or remove input variables
- Change model architecture
- Adjust training hyperparameters

```bash
# View current configuration
python config.py
```

**See [USAGE_GUIDE.md](USAGE_GUIDE.md) for examples of adding variables like LWFORC, SWFORC, RAINRATE, T2MV, etc.**

### 1. Data Preprocessing

```bash
python data_preprocessing.py
```

This script:

- Loads simulation results from NetCDF files
- Extracts variables specified in `config.py`
- Aggregates 30-min data to daily resolution (using specified aggregation methods)
- Normalizes inputs and outputs
- Saves preprocessed data to configured output file

**Default configuration**: 3 physical variables + day of year = 4 input features

### 2. Model Training

```bash
python train.py
```

This script:

- Loads preprocessed data
- Creates train/validation split (ratio from config)
- Trains LSTM model with early stopping
- Automatically adapts to number of input variables
- Saves model weights, training history, and predictions

**All hyperparameters are controlled via `config.py`**

### 3. Results Analysis

```bash
python analyze_results.py
```

This script:

- Generates detailed performance report
- Creates R² comparison plots
- Exports summary statistics to CSV
- Prints comprehensive analysis

## Model Architecture

### LSTMParameterPredictor

```
Input: (batch_size, seq_len=366, input_dim=4)
↓
LSTM Layers (2 layers, hidden_dim=128, dropout=0.2)
↓
Last Hidden State (batch_size, 128)
↓
FC Layer (128 → 64) + ReLU + Dropout
↓
FC Layer (64 → 32) + ReLU + Dropout
↓
FC Layer (32 → 13)
↓
Output: (batch_size, 13) parameter predictions
```

**Total parameters**: 211,469

### Alternative Model Architectures

Two additional model variants are available in `lstm_model.py` for comparison:

1. **BiLSTMParameterPredictor**: Bidirectional LSTM that processes sequences in both forward and backward directions
2. **AttentionLSTMParameterPredictor**: LSTM with attention mechanism that learns to focus on the most relevant timesteps

To use a different model, change `model_type` in `config.py`:

```python
MODEL_CONFIG = {
    'model_type': 'AttentionLSTM',  # Options: 'LSTM', 'BiLSTM', 'AttentionLSTM'
    ...
}
```

## Improvements & Future Work

### Data-Related

1. **More samples**: 144 samples is limited for deep learning
2. **Additional variables**: Include more output variables (e.g., runoff, evapotranspiration)
3. **Multi-site data**: Use data from different locations/conditions
4. **Data augmentation**: Generate synthetic samples

### Model-Related

1. **Ensemble methods**: Train multiple models and average predictions
2. **Physics-informed constraints**: Incorporate physical constraints on parameters
3. **Uncertainty quantification**: Estimate prediction uncertainty
4. **Multi-head attention**: Extend attention mechanism to multi-head architecture

### Problem Formulation

1. **Parameter subset selection**: Focus on identifiable parameters only
2. **Hierarchical approach**: Predict parameter groups sequentially
3. **Multi-task learning**: Jointly predict parameters and reconstruct outputs
4. **Regularization**: Add constraints based on parameter physical ranges

## Technical Details

### Data Normalization

- **Inputs**: Standardized to zero mean, unit variance per variable
- **Outputs**: Standardized to zero mean, unit variance per parameter
- Statistics saved for inverse transformation

### Training Strategy

- **Loss function**: MSE (Mean Squared Error)
- **Optimizer**: Adam
- **Learning rate scheduler**: ReduceLROnPlateau (factor=0.5, patience=10)
- **Early stopping**: Stops if validation loss doesn't improve for 20 epochs
- **Best model**: Saved based on minimum validation loss

### Reproducibility

- Random seed: 42 (set for PyTorch and NumPy)
- Device: CUDA (GPU) if available, else CPU

## References

This is an inverse problem in hydrological modeling, related to:

- Parameter estimation in land surface models
- Data assimilation
- Model calibration

## Limitations

1. **Inverse problem difficulty**: This is an ill-posed problem where unique parameter identification is challenging
2. **Limited data**: 144 samples is relatively small for deep learning
3. **Single-point simulation**: Results may not generalize to other locations
4. **Temporal coverage**: Only one year of data per sample

## Contact & Support

For questions or issues, please refer to the project documentation or contact the development team.

---

_Generated using PyTorch 2.4.1 with CUDA support_
