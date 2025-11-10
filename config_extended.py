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

    print("="*80)

if __name__ == '__main__':
    print_config()
