"""
Configuration for COMPREHENSIVE FORWARD LSTM model
Predicts ALL energy and water cycle variables for full conservation checking

COMPREHENSIVE FORWARD Problem:
- Input: Parameters (9 NoahMP params) + Forcing variables (time series)
- Output: All energy/water cycle variables for conservation validation
- Goal: Validate model follows physical conservation laws
"""

# =============================================================================
# COMPREHENSIVE TARGET VARIABLES CONFIGURATION
# =============================================================================

# Core Energy Balance Variables (for energy conservation check)
ENERGY_BALANCE_TARGETS = [
    {
        'name': 'FSA',
        'aggregation': 'mean',
        'description': 'Total absorbed SW radiation (W/m2)',
        'category': 'energy_balance'
    },
    {
        'name': 'FIRA',
        'aggregation': 'mean',
        'description': 'Total net LW radiation to atmosphere (W/m2)',
        'category': 'energy_balance'
    },
    {
        'name': 'HFX',
        'aggregation': 'mean',
        'description': 'Total sensible heat to atmosphere (W/m2)',
        'category': 'energy_balance'
    },
    {
        'name': 'LH',
        'aggregation': 'mean',
        'description': 'Total latent heat to atmosphere (W/m2)',
        'category': 'energy_balance'
    },
    {
        'name': 'GRDFLX',
        'aggregation': 'mean',
        'description': 'Heat flux into the soil (W/m2)',
        'category': 'energy_balance'
    },
]

# Energy Component Variables (for detailed energy budget)
ENERGY_COMPONENT_TARGETS = [
    {
        'name': 'SAV',
        'aggregation': 'mean',
        'description': 'Solar radiation absorbed by canopy (W/m2)',
        'category': 'energy_components'
    },
    {
        'name': 'SAG',
        'aggregation': 'mean',
        'description': 'Solar radiation absorbed by ground (W/m2)',
        'category': 'energy_components'
    },
    {
        'name': 'IRC',
        'aggregation': 'mean',
        'description': 'Canopy net LW rad (W/m2)',
        'category': 'energy_components'
    },
    {
        'name': 'SHC',
        'aggregation': 'mean',
        'description': 'Canopy sensible heat (W/m2)',
        'category': 'energy_components'
    },
    {
        'name': 'EVC',
        'aggregation': 'mean',
        'description': 'Canopy evap heat (W/m2)',
        'category': 'energy_components'
    },
    {
        'name': 'IRG',
        'aggregation': 'mean',
        'description': 'Below-canopy ground net LW rad (W/m2)',
        'category': 'energy_components'
    },
    {
        'name': 'SHG',
        'aggregation': 'mean',
        'description': 'Below-canopy ground sensible heat (W/m2)',
        'category': 'energy_components'
    },
    {
        'name': 'EVG',
        'aggregation': 'mean',
        'description': 'Below-canopy ground evap heat (W/m2)',
        'category': 'energy_components'
    },
    {
        'name': 'GHV',
        'aggregation': 'mean',
        'description': 'Below-canopy ground heat to soil (W/m2)',
        'category': 'energy_components'
    },
]

# Water Flux Variables (for water balance)
WATER_FLUX_TARGETS = [
    {
        'name': 'ECAN',
        'aggregation': 'mean',
        'description': 'Canopy water evaporation rate (mm/s)',
        'category': 'water_fluxes'
    },
    {
        'name': 'ETRAN',
        'aggregation': 'mean',
        'description': 'Transpiration rate (mm/s)',
        'category': 'water_fluxes'
    },
    {
        'name': 'EDIR',
        'aggregation': 'mean',
        'description': 'Direct from soil evaporation rate (mm/s)',
        'category': 'water_fluxes'
    },
    {
        'name': 'UGDRNOFF',
        'aggregation': 'last',  # Accumulated, so take last value
        'description': 'Accumulated underground runoff (mm)',
        'category': 'water_fluxes'
    },
    {
        'name': 'SFCRNOFF',
        'aggregation': 'last',  # Accumulated, so take last value
        'description': 'Accumulated surface runoff (mm)',
        'category': 'water_fluxes'
    },
]

# Water Storage Variables
WATER_STORAGE_TARGETS = [
    {
        'name': 'SOIL_M',
        'aggregation': 'mean',
        'layer': 0,
        'description': 'Volumetric soil moisture Layer 1 (m3/m3)',
        'category': 'water_storage'
    },
    {
        'name': 'SOIL_M',
        'aggregation': 'mean',
        'layer': 1,
        'description': 'Volumetric soil moisture Layer 2 (m3/m3)',
        'category': 'water_storage',
        'output_name': 'SOIL_M_L2'
    },
    {
        'name': 'SOIL_M',
        'aggregation': 'mean',
        'layer': 2,
        'description': 'Volumetric soil moisture Layer 3 (m3/m3)',
        'category': 'water_storage',
        'output_name': 'SOIL_M_L3'
    },
    {
        'name': 'SOIL_M',
        'aggregation': 'mean',
        'layer': 3,
        'description': 'Volumetric soil moisture Layer 4 (m3/m3)',
        'category': 'water_storage',
        'output_name': 'SOIL_M_L4'
    },
    {
        'name': 'CANLIQ',
        'aggregation': 'mean',
        'description': 'Canopy liquid water content (mm)',
        'category': 'water_storage'
    },
    {
        'name': 'CANICE',
        'aggregation': 'mean',
        'description': 'Canopy ice water content (mm)',
        'category': 'water_storage'
    },
    {
        'name': 'SNEQV',
        'aggregation': 'mean',
        'description': 'Snow water equivalent (mm)',
        'category': 'water_storage'
    },
]

# Temperature Variables (for energy/water coupling)
TEMPERATURE_TARGETS = [
    {
        'name': 'SOIL_T',
        'aggregation': 'mean',
        'layer': 0,
        'description': 'Soil temperature Layer 1 (K)',
        'category': 'temperature'
    },
    {
        'name': 'SOIL_T',
        'aggregation': 'mean',
        'layer': 1,
        'description': 'Soil temperature Layer 2 (K)',
        'category': 'temperature',
        'output_name': 'SOIL_T_L2'
    },
    {
        'name': 'TG',
        'aggregation': 'mean',
        'description': 'Ground temperature (K)',
        'category': 'temperature'
    },
    {
        'name': 'TV',
        'aggregation': 'mean',
        'description': 'Vegetation temperature (K)',
        'category': 'temperature'
    },
    {
        'name': 'TRAD',
        'aggregation': 'mean',
        'description': 'Surface radiative temperature (K)',
        'category': 'temperature'
    },
]

# Combine all target variables
TARGET_VARIABLES = (
    ENERGY_BALANCE_TARGETS +
    ENERGY_COMPONENT_TARGETS +
    WATER_FLUX_TARGETS +
    WATER_STORAGE_TARGETS +
    TEMPERATURE_TARGETS
)

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

MAX_SAMPLES = 10
PARAMETER_FILE = 'data/raw/param/noahmp_param_sets.txt'
SIMULATION_DIR = 'data/raw/sim_results'
OUTPUT_FILE = 'data/processed_data_forward_comprehensive.pkl'

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_CONFIG = {
    'model_type': 'AttentionLSTM',  # Options: 'LSTM', 'BiLSTM', 'AttentionLSTM'
    'hidden_dim': 1536,  # Increased for more outputs
    'num_layers': 3,     # Increased for complexity
    'dropout': 0.3,
    'param_embedding_dim': 128,  # Increased for richer parameter representation
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

TRAINING_CONFIG = {
    'learning_rate': 0.0005,  # Lower for stability
    'batch_size': 8,          # Smaller due to more outputs
    'num_epochs': 1000,
    'patience': 10,
    'train_ratio': 0.8,
}

# =============================================================================
# OUTPUT LOSS WEIGHTS CONFIGURATION
# =============================================================================

# Weight different variable categories differently
# Higher weight = model focuses more on that variable

OUTPUT_WEIGHTS = {
    # Energy balance variables (most critical for conservation)
    'FSA': 2.0,
    'FIRA': 2.0,
    'HFX': 2.0,
    'LH': 2.0,
    'GRDFLX': 2.0,

    # Energy components (important for detailed budget)
    'SAV': 1.5,
    'SAG': 1.5,
    'IRC': 1.5,
    'SHC': 1.5,
    'EVC': 1.5,
    'IRG': 1.5,
    'SHG': 1.5,
    'EVG': 1.5,
    'GHV': 1.5,

    # Water fluxes (critical for water balance)
    'ECAN': 2.0,
    'ETRAN': 2.0,
    'EDIR': 2.0,
    'UGDRNOFF': 1.5,
    'SFCRNOFF': 1.5,

    # Water storage (important for state)
    'SOIL_M': 2.0,
    'SOIL_M_L2': 1.5,
    'SOIL_M_L3': 1.5,
    'SOIL_M_L4': 1.5,
    'CANLIQ': 1.0,
    'CANICE': 1.0,
    'SNEQV': 1.0,

    # Temperature (important for coupling)
    'SOIL_T': 1.5,
    'SOIL_T_L2': 1.0,
    'TG': 1.5,
    'TV': 1.5,
    'TRAD': 1.5,
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

RESULTS_DIR = 'results_forward_comprehensive'

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
    """Get list of target variable names (with layer/output name handling)"""
    names = []
    for var in TARGET_VARIABLES:
        if 'output_name' in var:
            names.append(var['output_name'])
        else:
            names.append(var['name'])
    return names

def get_target_variable_categories():
    """Get category for each target variable"""
    categories = []
    for var in TARGET_VARIABLES:
        categories.append(var['category'])
    return categories

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

def get_variables_by_category(category):
    """Get all variables in a specific category"""
    return [var for var in TARGET_VARIABLES if var['category'] == category]

def print_config():
    """Print current configuration"""
    print("="*80)
    print("COMPREHENSIVE FORWARD MODEL CONFIGURATION")
    print("="*80)

    # Group by category
    categories = {
        'energy_balance': 'Energy Balance (Core)',
        'energy_components': 'Energy Components (Detailed)',
        'water_fluxes': 'Water Fluxes',
        'water_storage': 'Water Storage',
        'temperature': 'Temperature'
    }

    print(f"\nTarget Variables ({len(TARGET_VARIABLES)} total):")
    for cat_key, cat_name in categories.items():
        cat_vars = get_variables_by_category(cat_key)
        if cat_vars:
            print(f"\n  {cat_name} ({len(cat_vars)} variables):")
            for i, var in enumerate(cat_vars, 1):
                layer_info = f" [Layer {var['layer']}]" if 'layer' in var else ""
                output_name = var.get('output_name', var['name'])
                print(f"    {i}. {output_name}{layer_info} - {var['description']}")

    print(f"\nForcing Variables ({len(FORCING_VARIABLES)}):")
    for i, var in enumerate(FORCING_VARIABLES, 1):
        print(f"  {i}. {var['name']} - {var['aggregation']} - {var['description']}")

    if ADD_TIME_FEATURES:
        print(f"  {len(FORCING_VARIABLES)+1}. doy - Day of year (normalized)")

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

    print(f"\nConservation Checking:")
    print(f"  Energy balance variables: {len(ENERGY_BALANCE_TARGETS)}")
    print(f"  Energy component variables: {len(ENERGY_COMPONENT_TARGETS)}")
    print(f"  Water flux variables: {len(WATER_FLUX_TARGETS)}")
    print(f"  Water storage variables: {len(WATER_STORAGE_TARGETS)}")

    print("="*80)

if __name__ == '__main__':
    print_config()
