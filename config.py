"""
Example configuration with extended variables
This shows how to easily add more input variables

To use this config:
1. Rename this file to config.py (backup the original first)
2. Run data_preprocessing.py to preprocess with new variables
3. Run train.py to train with new variables
"""

# =============================================================================
# INPUT VARIABLES CONFIGURATION - EXTENDED VERSION
# =============================================================================

INPUT_VARIABLES = [
    # Original physical variables
    {
        'name': 'SOIL_M',
        'aggregation': 'mean',
        'layer': 0,  # First soil layer
        'description': 'Volumetric soil moisture (m3/m3) - Layer 1'
    },
    {
        'name': 'LH',
        'aggregation': 'mean',
        'description': 'Latent heat flux (W/m2)'
    },
    {
        'name': 'HFX',
        'aggregation': 'mean',
        'description': 'Sensible heat flux (W/m2)'
    },

    # Additional forcing variables
    {
        'name': 'LWFORC',
        'aggregation': 'mean',
        'description': 'Longwave forcing (W/m2)'
    },
    {
        'name': 'SWFORC',
        'aggregation': 'mean',
        'description': 'Shortwave forcing (W/m2)'
    },
    {
        'name': 'RAINRATE',
        'aggregation': 'sum',  # Sum for precipitation (total daily)
        'description': 'Total daily precipitation (mm/day)'
    },
    {
        'name': 'T2MV',
        'aggregation': 'mean',
        'description': 'Temperature at 2m (K)'
    },

    # Additional state variables (examples - check your NetCDF for available vars)
    # {
    #     'name': 'SNOW',
    #     'aggregation': 'mean',
    #     'description': 'Snow water equivalent (mm)'
    # },
    # {
    #     'name': 'CANWAT',
    #     'aggregation': 'mean',
    #     'description': 'Canopy water (mm)'
    # },
]

# Add temporal features
ADD_TIME_FEATURES = True  # Add day of year (normalized)
ADD_MONTH_FEATURE = False  # Add month as one-hot encoding

# =============================================================================
# DATA PREPROCESSING CONFIGURATION
# =============================================================================

MAX_SAMPLES = 200
PARAMETER_FILE = 'data/raw/param/noahmp_param_sets.txt'
SIMULATION_DIR = 'data/raw/sim_results'
OUTPUT_FILE = 'data/processed_data_extended.pkl'  # Different filename to keep original

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_CONFIG = {
    'model_type': 'BiLSTM',  # Try bidirectional LSTM with more variables
    'hidden_dim': 256,  # Larger hidden dimension for more variables
    'num_layers': 2,
    'dropout': 0.2,
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

TRAINING_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 16,
    'num_epochs': 1000,
    'patience': 200,
    'train_ratio': 0.8,
}

# =============================================================================
# PARAMETER LOSS WEIGHTS CONFIGURATION
# =============================================================================

# Control how much each parameter contributes to the loss
# Higher weight = model focuses more on predicting that parameter accurately
# Set to None to use uniform weights (all parameters equally important)

# Option 1: Uniform weights (default)
# PARAMETER_WEIGHTS = None  # All parameters have equal weight

# Option 2: Custom weights - specify weight for each parameter
# Based on previous results, focus on well-identifiable parameters
PARAMETER_WEIGHTS = {
    'VCMX25_EBF': 1.0,     # Well predicted (R²=0.80) - focus more
    'HVT_EBF': 1.0,        # Moderately predicted
    'HVB_EBF': 1,        # Poorly predicted - less focus
    'CWPVT_EBF': 1,
    'Z0MVT_EBF': 1,      # Very poorly predicted - minimal focus
    'WLTSMC_CL': 1,
    'REFSMC_CL': 1,
    'MAXSMC_CL': 1.0,      # Well predicted (R²=0.76) - focus more
    'SATDK_CL': 1,
    'WLTSMC_SCL': 0.1,     # Negative R² - minimal focus
    'REFSMC_SCL': 0.1,
    'MAXSMC_SCL': 0.1,
    'SATDK_SCL': 0.1,      # Very poorly predicted - minimal focus
}

# Option 3: Focus only on specific parameters
# PARAMETER_WEIGHTS = {
#     'VCMX25_EBF': 1.0,     # Vegetation parameter
#     'MAXSMC_CL': 1.0,      # Soil parameter
#     # All others: 0.1 (minimal weight)
# }

# Weight strategy presets
WEIGHT_STRATEGY = 'uniform'  # 'uniform', 'performance_based', 'focus_top', or 'custom'

def get_parameter_weights(param_names):
    """
    Get parameter weights based on strategy

    Args:
        param_names: List of parameter names

    Returns:
        numpy array of weights (one per parameter)
    """
    import numpy as np

    if PARAMETER_WEIGHTS is not None:
        # Custom weights provided
        weights = np.array([PARAMETER_WEIGHTS.get(name, 1.0) for name in param_names])
        return weights

    # Apply preset strategies
    if WEIGHT_STRATEGY == 'uniform':
        # All parameters equally important
        return np.ones(len(param_names))

    elif WEIGHT_STRATEGY == 'performance_based':
        # Weight based on previous performance (from initial results)
        # High weight for well-predicted params, low for poorly-predicted
        performance_weights = {
            'VCMX25_EBF': 2.0,
            'MAXSMC_CL': 2.0,
            'HVT_EBF': 1.0,
            'HVB_EBF': 0.5,
            'WLTSMC_CL': 0.5,
            'REFSMC_CL': 0.5,
            'CWPVT_EBF': 0.5,
            'REFSMC_SCL': 0.3,
            'Z0MVT_EBF': 0.3,
            'SATDK_CL': 0.3,
            'MAXSMC_SCL': 0.3,
            'WLTSMC_SCL': 0.3,
            'SATDK_SCL': 0.3,
        }
        weights = np.array([performance_weights.get(name, 1.0) for name in param_names])
        return weights

    elif WEIGHT_STRATEGY == 'focus_top':
        # Focus only on top 2 parameters, minimal weight for others
        focus_params = {'VCMX25_EBF', 'MAXSMC_CL'}
        weights = np.array([1.0 if name in focus_params else 0.1 for name in param_names])
        return weights

    else:  # 'custom' or unknown
        return np.ones(len(param_names))

# =============================================================================
# RESULTS CONFIGURATION
# =============================================================================

RESULTS_DIR = 'results'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_variable_names():
    """Get list of variable names"""
    names = [var['name'] for var in INPUT_VARIABLES]
    if ADD_TIME_FEATURES:
        names.append('doy')
    if ADD_MONTH_FEATURE:
        for i in range(12):
            names.append(f'month_{i+1}')
    return names

def get_num_variables():
    """Get total number of input variables"""
    num_vars = len(INPUT_VARIABLES)
    if ADD_TIME_FEATURES:
        num_vars += 1
    if ADD_MONTH_FEATURE:
        num_vars += 12
    return num_vars

def print_config():
    """Print current configuration"""
    print("="*80)
    print("CONFIGURATION SUMMARY (EXTENDED)")
    print("="*80)
    print(f"\nInput Variables ({len(INPUT_VARIABLES)}):")
    for i, var in enumerate(INPUT_VARIABLES, 1):
        layer_info = f" [Layer {var['layer']}]" if 'layer' in var else ""
        print(f"  {i}. {var['name']}{layer_info} - {var['aggregation']} - {var['description']}")

    if ADD_TIME_FEATURES:
        print(f"  {len(INPUT_VARIABLES)+1}. doy - Day of year (normalized)")

    if ADD_MONTH_FEATURE:
        print(f"  {len(INPUT_VARIABLES)+2}-{len(INPUT_VARIABLES)+13}. month_1 to month_12 - One-hot encoded month")

    print(f"\nTotal input features: {get_num_variables()}")

    print(f"\nData Configuration:")
    print(f"  Max samples: {MAX_SAMPLES}")
    print(f"  Parameter file: {PARAMETER_FILE}")
    print(f"  Simulation dir: {SIMULATION_DIR}")
    print(f"  Output file: {OUTPUT_FILE}")

    print(f"\nModel Configuration:")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")

    print(f"\nTraining Configuration:")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")

    print(f"\nParameter Loss Weights:")
    print(f"  Strategy: {WEIGHT_STRATEGY}")
    if PARAMETER_WEIGHTS is not None:
        print(f"  Custom weights: Yes ({len(PARAMETER_WEIGHTS)} parameters)")
    else:
        print(f"  Custom weights: No (using strategy preset)")

    print("="*80)

if __name__ == '__main__':
    print_config()
