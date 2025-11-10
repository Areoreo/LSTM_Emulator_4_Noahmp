"""
Data preprocessing script for LSTM-based parameter prediction
Loads simulation results and parameters, aggregates to daily resolution
Flexible variable selection via config.py
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import config

def load_simulation_data(sample_idx, data_dir='data/raw/sim_results', variables=None):
    """
    Load simulation data for a single sample with flexible variable selection

    Args:
        sample_idx: Sample index (1-based)
        data_dir: Directory containing simulation results
        variables: List of variable configs (if None, uses config.INPUT_VARIABLES)

    Returns:
        DataFrame with daily aggregated variables
    """
    if variables is None:
        variables = config.INPUT_VARIABLES

    sim_path = Path(data_dir) / f'sample_{sample_idx}' / 'output' / '201507301730.LDASOUT_DOMAIN1'

    if not sim_path.exists():
        return None

    # Load with xarray
    ds = xr.open_dataset(sim_path)

    # Get time information
    # Convert byte strings and handle underscore format (e.g., 2015-07-30_17:30:00)
    time_strings = [s.decode() if isinstance(s, bytes) else s for s in ds['Times'].values]
    time_strings = [s.replace('_', ' ') for s in time_strings]
    times = pd.to_datetime(time_strings)

    # Extract variables dynamically based on config
    data_dict = {'time': times}

    for var_config in variables:
        var_name = var_config['name']

        # Check if variable exists in dataset
        if var_name not in ds:
            print(f"Warning: Variable '{var_name}' not found in sample {sample_idx}")
            ds.close()
            return None

        # Extract data
        var_data = ds[var_name]

        # Handle multi-layer variables
        if 'layer' in var_config:
            layer_dim = None
            # Find the layer dimension name
            for dim in ['soil_layers_stag', 'snow_layers']:
                if dim in var_data.dims:
                    layer_dim = dim
                    break

            if layer_dim:
                var_data = var_data.isel({layer_dim: var_config['layer']})
            else:
                print(f"Warning: Layer specified for {var_name} but no layer dimension found")

        # Squeeze spatial dimensions
        for dim in ['south_north', 'west_east']:
            if dim in var_data.dims:
                var_data = var_data.isel({dim: 0})

        data_dict[var_name] = var_data.values

    ds.close()

    # Create DataFrame with 30-min data
    df = pd.DataFrame(data_dict)

    # Aggregate to daily based on specified aggregation method
    df['date'] = df['time'].dt.date
    agg_dict = {var_config['name']: var_config['aggregation']
                for var_config in variables}

    daily_df = df.groupby('date').agg(agg_dict).reset_index()

    # Add temporal features if configured
    if config.ADD_TIME_FEATURES:
        daily_df['doy'] = pd.to_datetime(daily_df['date']).dt.dayofyear / 365.0

    if config.ADD_MONTH_FEATURE:
        months = pd.to_datetime(daily_df['date']).dt.month
        # One-hot encode months
        for month in range(1, 13):
            daily_df[f'month_{month}'] = (months == month).astype(float)

    return daily_df

def load_parameters(param_file='data/raw/param/noahmp_param_sets.txt'):
    """
    Load parameter sets

    Returns:
        DataFrame with parameter values
    """
    params = pd.read_csv(param_file, sep=r'\s+')
    return params

def preprocess_all_data(max_samples=None, output_file=None):
    """
    Process all samples and create input/output arrays
    Uses configuration from config.py

    Args:
        max_samples: Maximum number of samples to process (if None, uses config.MAX_SAMPLES)
        output_file: Output pickle file path (if None, uses config.OUTPUT_FILE)

    Returns:
        dict with X (inputs) and y (outputs)
    """
    # Use config defaults if not specified
    if max_samples is None:
        max_samples = config.MAX_SAMPLES
    if output_file is None:
        output_file = config.OUTPUT_FILE

    # Print configuration
    config.print_config()

    print("\nLoading parameters...")
    params = load_parameters(config.PARAMETER_FILE)

    print(f"\nProcessing simulation results for {max_samples} samples...")

    all_data = []
    valid_indices = []

    for idx in tqdm(range(1, max_samples + 1)):
        daily_data = load_simulation_data(idx, data_dir=config.SIMULATION_DIR)

        if daily_data is not None:
            all_data.append(daily_data)
            valid_indices.append(idx - 1)  # 0-based index for params
        else:
            print(f"Warning: Sample {idx} not found or missing variables")

    print(f"\nSuccessfully loaded {len(all_data)} samples")

    if len(all_data) == 0:
        raise ValueError("No valid samples found!")

    # Get dimensions
    n_samples = len(all_data)
    n_timesteps = len(all_data[0])
    n_variables = config.get_num_variables()
    n_params = params.shape[1]
    variable_names = config.get_variable_names()

    print(f"\nData dimensions:")
    print(f"  Samples: {n_samples}")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  Variables: {n_variables}")
    print(f"  Parameters: {n_params}")

    # Create input array: (n_samples, n_variables, n_timesteps)
    X = np.zeros((n_samples, n_variables, n_timesteps))

    # Fill array dynamically based on variable names
    for i, daily_data in enumerate(all_data):
        for j, var_name in enumerate(variable_names):
            X[i, j, :] = daily_data[var_name].values

    # Create output array: (n_samples, n_params)
    y = params.iloc[valid_indices].values

    # Normalize inputs
    X_mean = X.mean(axis=(0, 2), keepdims=True)
    X_std = X.std(axis=(0, 2), keepdims=True)
    X_normalized = (X - X_mean) / (X_std + 1e-8)

    # Normalize outputs
    y_mean = y.mean(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True)
    y_normalized = (y - y_mean) / (y_std + 1e-8)

    # Save processed data
    data_dict = {
        'X': X_normalized,
        'y': y_normalized,
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean,
        'y_std': y_std,
        'variable_names': variable_names,
        'param_names': params.columns.tolist(),
        'n_timesteps': n_timesteps,
        'valid_indices': valid_indices,
        'n_variables': n_variables
    }

    # Create output directory if needed
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(data_dict, f)

    print(f"\nData preprocessing complete!")
    print(f"Input shape: {X_normalized.shape} (samples, variables, timesteps)")
    print(f"Output shape: {y_normalized.shape} (samples, parameters)")
    print(f"Saved to: {output_file}")

    return data_dict

if __name__ == '__main__':
    data = preprocess_all_data()
