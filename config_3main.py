"""
Configuration file for LSTM parameter prediction
Easily customize input variables and model settings here
"""

# =============================================================================
# INPUT VARIABLES CONFIGURATION
# =============================================================================

# Define which variables to use as input features
# Each variable should be a dict with:
#   - 'name': Variable name in NetCDF file
#   - 'aggregation': How to aggregate from 30-min to daily ('mean', 'sum', 'min', 'max')
#   - 'description': Human-readable description
#   - 'layer': Optional layer index for multi-layer variables (e.g., SOIL_M has 4 layers)

INPUT_VARIABLES = [
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
]

# Example: Uncomment to add more variables
# ADDITIONAL_VARIABLES = [
#     {
#         'name': 'LWFORC',
#         'aggregation': 'mean',
#         'description': 'Longwave forcing (W/m2)'
#     },
#     {
#         'name': 'SWFORC',
#         'aggregation': 'mean',
#         'description': 'Shortwave forcing (W/m2)'
#     },
#     {
#         'name': 'RAINRATE',
#         'aggregation': 'sum',  # Sum for precipitation
#         'description': 'Precipitation rate (mm/day)'
#     },
#     {
#         'name': 'T2MV',
#         'aggregation': 'mean',
#         'description': 'Temperature at 2m (K)'
#     },
# ]
# INPUT_VARIABLES.extend(ADDITIONAL_VARIABLES)

# Add temporal features
ADD_TIME_FEATURES = True  # Add day of year (normalized)
ADD_MONTH_FEATURE = False  # Add month as one-hot encoding

# =============================================================================
# DATA PREPROCESSING CONFIGURATION
# =============================================================================

MAX_SAMPLES = 200  # Maximum number of samples to process
PARAMETER_FILE = 'data/raw/param/noahmp_param_sets.txt'
SIMULATION_DIR = 'data/raw/sim_results'
OUTPUT_FILE = 'data/processed_data.pkl'

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_CONFIG = {
    'model_type': 'LSTM',  # 'LSTM' or 'BiLSTM'
    'hidden_dim': 128,
    'num_layers': 2,
    'dropout': 0.1,
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
    print("CONFIGURATION SUMMARY")
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
