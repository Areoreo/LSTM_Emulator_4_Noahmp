# LSTM Emulator - Usage Guide

## Quick Start

The system is now fully configurable! All settings are controlled via `config.py`.

### Basic Workflow

```bash
# 1. Check/modify configuration
python config.py  # View current config

# 2. Preprocess data
python data_preprocessing.py

# 3. Train model
python train.py

# 4. Analyze results
python analyze_results.py
```

## Adding New Input Variables

The easiest way to add new variables is to edit `config.py`. Here's how:

### Step 1: Check Available Variables

First, check what variables are available in your NetCDF files:

```python
import xarray as xr
ds = xr.open_dataset('data/raw/sim_results/sample_1/output/201507301730.LDASOUT_DOMAIN1')
print(list(ds.variables.keys()))
```

### Step 2: Update config.py

Edit the `INPUT_VARIABLES` list in `config.py`:

```python
INPUT_VARIABLES = [
    {
        'name': 'SOIL_M',
        'aggregation': 'mean',
        'layer': 0,  # Optional: for multi-layer variables
        'description': 'Volumetric soil moisture (m3/m3) - Layer 1'
    },
    {
        'name': 'LH',
        'aggregation': 'mean',
        'description': 'Latent heat flux (W/m2)'
    },
    # Add new variables here:
    {
        'name': 'T2MV',
        'aggregation': 'mean',
        'description': 'Temperature at 2m (K)'
    },
]
```

#### Variable Configuration Fields

- **name**: Variable name in NetCDF file (required)
- **aggregation**: How to aggregate from 30-min to daily (required)
  - `'mean'`: Daily average (for temperatures, fluxes)
  - `'sum'`: Daily total (for precipitation)
  - `'min'`: Daily minimum
  - `'max'`: Daily maximum
- **layer**: Layer index for multi-layer variables (optional)
  - Example: `layer: 0` for first soil layer
- **description**: Human-readable description (required)

### Step 3: Reprocess and Retrain

```bash
# Preprocess with new variables
python data_preprocessing.py

# Train model (automatically detects new input dimension)
python train.py
```

## Examples

### Example 1: Current Configuration (3 variables + time)

```python
# config.py
INPUT_VARIABLES = [
    {'name': 'SOIL_M', 'aggregation': 'mean', 'layer': 0,
     'description': 'Soil moisture layer 1'},
    {'name': 'LH', 'aggregation': 'mean',
     'description': 'Latent heat'},
    {'name': 'HFX', 'aggregation': 'mean',
     'description': 'Sensible heat'},
]
ADD_TIME_FEATURES = True  # Adds 'doy' (day of year)
```

**Result**: 4 input features

### Example 2: Extended Configuration (7 variables + time)

```python
# config.py or use config_extended.py
INPUT_VARIABLES = [
    {'name': 'SOIL_M', 'aggregation': 'mean', 'layer': 0,
     'description': 'Soil moisture'},
    {'name': 'LH', 'aggregation': 'mean',
     'description': 'Latent heat'},
    {'name': 'HFX', 'aggregation': 'mean',
     'description': 'Sensible heat'},
    {'name': 'LWFORC', 'aggregation': 'mean',
     'description': 'Longwave forcing'},
    {'name': 'SWFORC', 'aggregation': 'mean',
     'description': 'Shortwave forcing'},
    {'name': 'RAINRATE', 'aggregation': 'sum',  # Note: sum for precipitation
     'description': 'Daily precipitation'},
    {'name': 'T2MV', 'aggregation': 'mean',
     'description': 'Temperature at 2m'},
]
ADD_TIME_FEATURES = True
```

**Result**: 8 input features

To use the extended configuration:
```bash
# Option 1: Rename config_extended.py
mv config.py config_baseline.py
mv config_extended.py config.py

# Option 2: Copy extended config
cp config_extended.py config.py

# Then run preprocessing and training
python data_preprocessing.py
python train.py
```

### Example 3: Multiple Soil Layers

```python
INPUT_VARIABLES = [
    {'name': 'SOIL_M', 'aggregation': 'mean', 'layer': 0,
     'description': 'Soil moisture layer 1'},
    {'name': 'SOIL_M', 'aggregation': 'mean', 'layer': 1,
     'description': 'Soil moisture layer 2'},
    {'name': 'SOIL_M', 'aggregation': 'mean', 'layer': 2,
     'description': 'Soil moisture layer 3'},
    # ... other variables
]
```

### Example 4: Add Monthly Seasonality

```python
INPUT_VARIABLES = [
    # ... your variables
]

ADD_TIME_FEATURES = True   # Adds day of year (1 feature)
ADD_MONTH_FEATURE = True   # Adds one-hot encoded months (12 features)
```

## Configuration Options

### Model Configuration

```python
MODEL_CONFIG = {
    'model_type': 'LSTM',    # or 'BiLSTM'
    'hidden_dim': 128,       # LSTM hidden units
    'num_layers': 2,         # Number of LSTM layers
    'dropout': 0.1,          # Dropout rate
}
```

**Tips:**
- More input variables → consider larger `hidden_dim` (e.g., 256)
- Deeper models (more layers) can capture more complex patterns
- BiLSTM can be better at capturing temporal dependencies

### Training Configuration

```python
TRAINING_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 16,
    'num_epochs': 1000,
    'patience': 200,        # Early stopping patience
    'train_ratio': 0.8,     # 80% train, 20% validation
}
```

### Data Configuration

```python
MAX_SAMPLES = 200                                    # Max samples to process
PARAMETER_FILE = 'data/raw/param/noahmp_param_sets.txt'
SIMULATION_DIR = 'data/raw/sim_results'
OUTPUT_FILE = 'data/processed_data.pkl'             # Where to save preprocessed data
RESULTS_DIR = 'results'
```

## Available NetCDF Variables

Common variables you can use (check your specific data):

**State Variables:**
- `SOIL_M` - Soil moisture (4 layers)
- `SOIL_T` - Soil temperature
- `SNEQV` - Snow water equivalent
- `SNOWH` - Snow height
- `CANWAT` - Canopy water

**Flux Variables:**
- `LH` - Latent heat flux
- `HFX` - Sensible heat flux
- `GRDFLX` - Ground heat flux
- `SFCRNOFF` - Surface runoff
- `UGDRNOFF` - Underground drainage

**Forcing Variables:**
- `LWFORC` - Longwave radiation forcing
- `SWFORC` - Shortwave radiation forcing
- `RAINRATE` - Precipitation rate
- `QSNOW` - Snowfall rate
- `T2MV` - 2m temperature
- `Q2MV` - 2m specific humidity

**Vegetation Variables:**
- `LAI` - Leaf area index
- `FVEG` - Green vegetation fraction
- `TR` - Transpiration
- `ECAN` - Canopy evaporation

## Troubleshooting

### Variable Not Found Error

```
Warning: Variable 'VARNAME' not found in sample X
```

**Solution**: Check if the variable exists in your NetCDF files:
```bash
python3 -c "import xarray as xr; ds = xr.open_dataset('data/raw/sim_results/sample_1/output/201507301730.LDASOUT_DOMAIN1'); print(list(ds.variables.keys()))"
```

### Wrong Layer Dimension

If you get an error about layer dimensions:
1. Check the variable's dimensions in the NetCDF
2. Adjust the dimension name in `data_preprocessing.py` line 63-64
3. Common dimension names: `soil_layers_stag`, `snow_layers`

### Model Input Dimension Mismatch

This happens automatically! The model detects `input_dim` from the data:
```python
input_dim = X.shape[1]  # Number of variables (automatic)
```

No manual changes needed in model code.

## Best Practices

### 1. Start Simple
Begin with a few variables (3-5), verify they work, then add more.

### 2. Check Data Quality
```python
# After preprocessing, inspect the data
import pickle
with open('data/processed_data.pkl', 'rb') as f:
    data = pickle.load(f)
print(f"Variables: {data['variable_names']}")
print(f"Shapes: X={data['X'].shape}, y={data['y'].shape}")
```

### 3. Version Your Configs
```bash
# Save your config before experimenting
cp config.py config_baseline.py
cp config.py config_experiment1.py
```

### 4. Use Meaningful Output Names
```python
OUTPUT_FILE = 'data/processed_data_7vars.pkl'  # Descriptive name
```

### 5. Monitor Training
Watch the console output - it shows which variables are being used:
```
Input variables (8): SOIL_M, LH, HFX, LWFORC, SWFORC, RAINRATE, T2MV, doy
```

## Advanced: Custom Aggregations

If you need custom aggregation methods, edit `data_preprocessing.py`:

```python
# In load_simulation_data function, around line 90
agg_dict = {}
for var_config in variables:
    agg_method = var_config['aggregation']
    var_name = var_config['name']

    # Add custom aggregation
    if agg_method == 'custom':
        # Your custom aggregation logic here
        pass
    else:
        agg_dict[var_name] = agg_method
```

## Advanced: Parameter Loss Weights

You can now control how much each parameter contributes to the training loss! This is useful when some parameters are more important or more predictable than others.

**Quick example** - focus on well-predicted parameters:

```python
# config.py
WEIGHT_STRATEGY = 'performance_based'  # High weight for good params, low for bad
```

Or create custom weights:

```python
# config.py
PARAMETER_WEIGHTS = {
    'VCMX25_EBF': 2.0,   # High priority (well-predicted)
    'MAXSMC_CL': 2.0,    # High priority (well-predicted)
    'SATDK_SCL': 0.3,    # Low priority (poorly-predicted)
}
```

**See [PARAMETER_WEIGHTS_GUIDE.md](PARAMETER_WEIGHTS_GUIDE.md) for complete documentation!**

## Summary Checklist

- [ ] Identify which variables to use
- [ ] Update `INPUT_VARIABLES` in `config.py`
- [ ] Set aggregation method for each variable
- [ ] Optionally adjust model hyperparameters
- [ ] Optionally set parameter loss weights
- [ ] Run `python data_preprocessing.py`
- [ ] Run `python train.py`
- [ ] Analyze results with `python analyze_results.py`

## Questions?

Check the main README.md for project overview and architecture details.

For parameter weights, see PARAMETER_WEIGHTS_GUIDE.md.
