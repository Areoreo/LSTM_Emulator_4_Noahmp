"""
Configuration for FORWARD LSTM model
Predicts time series results from parameters and forcing data

FORWARD Problem:
- Input: Parameters (9 NoahMP params) + Forcing variables (time series)
- Output: Target variables time series (SOIL_M, LH, HFX)
"""

# =============================================================================
# TARGET VARIABLES CONFIGURATION (What we want to predict)
# =============================================================================

TARGET_VARIABLES = [
    {
        'name': 'SOIL_M',
        'aggregation': 'mean',
        'layer': 0,
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
]

# =============================================================================
# FORCING VARIABLES CONFIGURATION (Time series inputs)
# =============================================================================

FORCING_VARIABLES = [
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
        'aggregation': 'sum',
        'description': 'Total daily precipitation (mm/day)'
    },
    {
        'name': 'T2MV',
        'aggregation': 'mean',
        'description': 'Temperature at 2m (K)'
    },
]

# Add temporal features
ADD_TIME_FEATURES = True  # Add day of year (normalized)
ADD_MONTH_FEATURE = False  # Add month as one-hot encoding

# =============================================================================
# DATA PREPROCESSING CONFIGURATION
# =============================================================================

MAX_SAMPLES = 250
PARAMETER_FILE = 'data/raw/param/noahmp_param_sets.txt'
SIMULATION_DIR = 'data/raw/sim_results'
OUTPUT_FILE = 'data/processed_data_forward.pkl'

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_CONFIG = {
    'model_type': 'AttentionLSTM',  # Options: 'LSTM', 'BiLSTM', 'AttentionLSTM'
    'hidden_dim': 1536,
    'num_layers': 2,
    'dropout': 0.2,
    'param_embedding_dim': 64,  # Dimension for parameter embedding
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

TRAINING_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 16,
    'num_epochs': 1000,
    'patience': 120,
    'train_ratio': 0.8,
}

# =============================================================================
# OUTPUT LOSS WEIGHTS CONFIGURATION
# =============================================================================

# Control how much each output variable contributes to the loss
# Higher weight = model focuses more on predicting that variable accurately
# Set to None to use uniform weights (all variables equally important)

OUTPUT_WEIGHTS = {
    'SOIL_M': 1.0,
    'LH': 1.0,
    'HFX': 1.0,
}

def get_output_weights(variable_names):
    """
    Get output weights based on variable names

    Args:
        variable_names: List of variable names

    Returns:
        numpy array of weights (one per variable)
    """
    import numpy as np

    if OUTPUT_WEIGHTS is not None:
        weights = np.array([OUTPUT_WEIGHTS.get(name, 1.0) for name in variable_names])
        return weights

    return np.ones(len(variable_names))

# =============================================================================
# RESULTS CONFIGURATION
# =============================================================================

RESULTS_DIR = 'results_forward'

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_forcing_variable_names():
    """Get list of forcing variable names"""
    names = [var['name'] for var in FORCING_VARIABLES]
    if ADD_TIME_FEATURES:
        names.append('doy')
    if ADD_MONTH_FEATURE:
        for i in range(12):
            names.append(f'month_{i+1}')
    return names

def get_target_variable_names():
    """Get list of target variable names"""
    return [var['name'] for var in TARGET_VARIABLES]

def get_num_forcing_variables():
    """Get total number of forcing variables"""
    num_vars = len(FORCING_VARIABLES)
    if ADD_TIME_FEATURES:
        num_vars += 1
    if ADD_MONTH_FEATURE:
        num_vars += 12
    return num_vars

def get_num_target_variables():
    """Get total number of target variables"""
    return len(TARGET_VARIABLES)

def print_config():
    """Print current configuration"""
    print("="*80)
    print("FORWARD MODEL CONFIGURATION")
    print("="*80)

    print(f"\nTarget Variables ({len(TARGET_VARIABLES)}) - What we predict:")
    for i, var in enumerate(TARGET_VARIABLES, 1):
        layer_info = f" [Layer {var['layer']}]" if 'layer' in var else ""
        print(f"  {i}. {var['name']}{layer_info} - {var['aggregation']} - {var['description']}")

    print(f"\nForcing Variables ({len(FORCING_VARIABLES)}) - Time series inputs:")
    for i, var in enumerate(FORCING_VARIABLES, 1):
        print(f"  {i}. {var['name']} - {var['aggregation']} - {var['description']}")

    if ADD_TIME_FEATURES:
        print(f"  {len(FORCING_VARIABLES)+1}. doy - Day of year (normalized)")

    if ADD_MONTH_FEATURE:
        print(f"  {len(FORCING_VARIABLES)+2}-{len(FORCING_VARIABLES)+13}. month_1 to month_12 - One-hot encoded month")

    print(f"\nTotal forcing features: {get_num_forcing_variables()}")
    print(f"Total target features: {get_num_target_variables()}")

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

    print(f"\nOutput Loss Weights:")
    if OUTPUT_WEIGHTS is not None:
        for var_name, weight in OUTPUT_WEIGHTS.items():
            print(f"  {var_name}: {weight}")
    else:
        print(f"  Uniform weights (all variables equal)")

    print("="*80)

if __name__ == '__main__':
    print_config()
