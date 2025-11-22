"""
Data preprocessing script for COMPREHENSIVE FORWARD LSTM model
Prepares data for predicting ALL energy/water cycle variables

COMPREHENSIVE FORWARD Problem:
- Input: Parameters (9) + Forcing time series (LWFORC, SWFORC, RAINRATE, T2MV, doy)
- Output: All energy/water cycle variables (32+ variables)
- Goal: Full conservation checking
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import config_forward_comprehensive as config

def load_simulation_data(sample_idx, data_dir='data/raw/sim_results'):
    """
    Load both forcing and ALL target variables for a single sample

    Args:
        sample_idx: Sample index (1-based)
        data_dir: Directory containing simulation results

    Returns:
        tuple: (forcing_df, target_df) with daily aggregated data
    """
    sim_path = Path(data_dir) / f'sample_{sample_idx}' / 'output' / '201507301730.LDASOUT_DOMAIN1'

    if not sim_path.exists():
        return None, None

    # Load with xarray
    ds = xr.open_dataset(sim_path)

    # Get time information
    time_strings = [s.decode() if isinstance(s, bytes) else s for s in ds['Times'].values]
    time_strings = [s.replace('_', ' ') for s in time_strings]
    times = pd.to_datetime(time_strings)

    # Extract forcing variables
    forcing_data = {'time': times}
    for var_config in config.FORCING_VARIABLES:
        var_name = var_config['name']

        if var_name not in ds:
            print(f"Warning: Forcing variable '{var_name}' not found in sample {sample_idx}")
            ds.close()
            return None, None

        var_data = ds[var_name]

        # Squeeze spatial dimensions
        for dim in ['south_north', 'west_east']:
            if dim in var_data.dims:
                var_data = var_data.isel({dim: 0})

        forcing_data[var_name] = var_data.values

    # Extract target variables
    target_data = {'time': times}
    for var_config in config.TARGET_VARIABLES:
        var_name = var_config['name']

        # For variables that need accumulated-to-rate conversion,
        # use raw name initially (conversion happens later)
        if var_config.get('convert_accumulated_to_rate', False):
            output_name = var_name
        else:
            output_name = var_config.get('output_name', var_name)

        if var_name not in ds:
            print(f"Warning: Target variable '{var_name}' not found in sample {sample_idx}")
            ds.close()
            return None, None

        var_data = ds[var_name]

        # Handle multi-layer variables
        if 'layer' in var_config:
            layer_dim = None
            for dim in ['soil_layers_stag', 'snow_layers']:
                if dim in var_data.dims:
                    layer_dim = dim
                    break

            if layer_dim:
                var_data = var_data.isel({layer_dim: var_config['layer']})

        # Squeeze spatial dimensions
        for dim in ['south_north', 'west_east']:
            if dim in var_data.dims:
                var_data = var_data.isel({dim: 0})

        target_data[output_name] = var_data.values

    ds.close()

    # Create DataFrames with 30-min data
    forcing_df = pd.DataFrame(forcing_data)
    target_df = pd.DataFrame(target_data)

    # Aggregate to daily
    forcing_df['date'] = forcing_df['time'].dt.date
    target_df['date'] = target_df['time'].dt.date

    # Forcing aggregation
    forcing_agg_dict = {var_config['name']: var_config['aggregation']
                        for var_config in config.FORCING_VARIABLES}
    forcing_daily = forcing_df.groupby('date').agg(forcing_agg_dict).reset_index()

    # Target aggregation
    target_agg_dict = {}
    for var_config in config.TARGET_VARIABLES:
        # Use raw name for aggregation (same as column name in DataFrame)
        # For variables with conversion, use raw name; they'll be renamed after conversion
        if var_config.get('convert_accumulated_to_rate', False):
            col_name = var_config['name']
        else:
            col_name = var_config.get('output_name', var_config['name'])
        target_agg_dict[col_name] = var_config['aggregation']

    target_daily = target_df.groupby('date').agg(target_agg_dict).reset_index()

    # Convert accumulated runoff variables to daily rates
    # UGDRNOFF and SFCRNOFF are accumulated values that increase monotonically
    # We need to compute the daily change (rate) for proper model training
    if 'UGDRNOFF' in target_daily.columns:
        ugdrnoff_accumulated = target_daily['UGDRNOFF'].values
        ugdrnoff_rate = np.diff(ugdrnoff_accumulated, prepend=ugdrnoff_accumulated[0])
        # First day rate should be the first accumulated value (not 0)
        ugdrnoff_rate[0] = ugdrnoff_accumulated[0]
        target_daily['UGDRNOFF_RATE'] = ugdrnoff_rate
        target_daily.drop('UGDRNOFF', axis=1, inplace=True)

    if 'SFCRNOFF' in target_daily.columns:
        sfcrnoff_accumulated = target_daily['SFCRNOFF'].values
        sfcrnoff_rate = np.diff(sfcrnoff_accumulated, prepend=sfcrnoff_accumulated[0])
        # First day rate should be the first accumulated value (not 0)
        sfcrnoff_rate[0] = sfcrnoff_accumulated[0]
        target_daily['SFCRNOFF_RATE'] = sfcrnoff_rate
        target_daily.drop('SFCRNOFF', axis=1, inplace=True)

    # Add temporal features to forcing
    if config.ADD_TIME_FEATURES:
        forcing_daily['doy'] = pd.to_datetime(forcing_daily['date']).dt.dayofyear / 365.0

    if config.ADD_MONTH_FEATURE:
        months = pd.to_datetime(forcing_daily['date']).dt.month
        for month in range(1, 13):
            forcing_daily[f'month_{month}'] = (months == month).astype(float)

    return forcing_daily, target_daily

def load_parameters(param_file='data/raw/param/noahmp_param_sets.txt'):
    """
    Load parameter sets

    Returns:
        DataFrame with parameter values
    """
    params = pd.read_csv(param_file, sep=r'\s+')
    return params

def normalize_data(data, method='z-score', fit_stats=None):
    """
    Normalize data using specified method

    Args:
        data: numpy array to normalize
        method: 'z-score' or 'min-max'
        fit_stats: Dict with pre-computed statistics (for inference).
                   If None, compute from data (for training)

    Returns:
        tuple: (normalized_data, stats_dict)
    """
    if method == 'z-score':
        if fit_stats is None:
            # Compute statistics from data
            mean = np.nanmean(data, axis=tuple(range(data.ndim - 1)), keepdims=True)
            std = np.nanstd(data, axis=tuple(range(data.ndim - 1)), keepdims=True)
            stats = {'mean': mean, 'std': std, 'method': 'z-score'}
        else:
            mean = fit_stats['mean']
            std = fit_stats['std']
            stats = fit_stats

        normalized = (data - mean) / (std + 1e-8)

    elif method == 'min-max':
        if fit_stats is None:
            # Compute statistics from data
            min_val = np.nanmin(data, axis=tuple(range(data.ndim - 1)), keepdims=True)
            max_val = np.nanmax(data, axis=tuple(range(data.ndim - 1)), keepdims=True)
            stats = {'min': min_val, 'max': max_val, 'method': 'min-max'}
        else:
            min_val = fit_stats['min']
            max_val = fit_stats['max']
            stats = fit_stats

        # Scale to [0, 1]
        normalized = (data - min_val) / (max_val - min_val + 1e-8)

    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'z-score' or 'min-max'")

    return normalized, stats

def denormalize_data(normalized_data, stats):
    """
    Reverse normalization using stored statistics

    Args:
        normalized_data: Normalized numpy array
        stats: Statistics dict from normalize_data()

    Returns:
        Original scale data
    """
    method = stats['method']

    if method == 'z-score':
        return normalized_data * stats['std'] + stats['mean']
    elif method == 'min-max':
        return normalized_data * (stats['max'] - stats['min']) + stats['min']
    else:
        raise ValueError(f"Unknown normalization method in stats: {method}")

def preprocess_all_data(max_samples=None, output_file=None):
    """
    Process all samples for comprehensive forward modeling

    Creates:
    - X_forcing: (n_samples, n_timesteps, n_forcing_vars) - time series of forcing
    - X_params: (n_samples, n_params) - parameters for each sample
    - y: (n_samples, n_timesteps, n_target_vars) - ALL target time series

    Args:
        max_samples: Maximum number of samples to process
        output_file: Output pickle file path

    Returns:
        dict with processed data
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
    print(f"Extracting {config.get_num_target_variables()} target variables...")

    all_forcing = []
    all_targets = []
    valid_indices = []
    failed_samples = []

    for idx in tqdm(range(1, max_samples + 1)):
        forcing_data, target_data = load_simulation_data(idx, data_dir=config.SIMULATION_DIR)

        if forcing_data is not None and target_data is not None:
            # Check for NaN values
            if forcing_data.isnull().any().any() or target_data.isnull().any().any():
                print(f"Warning: Sample {idx} contains NaN values, skipping")
                failed_samples.append(idx)
                continue

            all_forcing.append(forcing_data)
            all_targets.append(target_data)
            valid_indices.append(idx - 1)  # 0-based index for params
        else:
            failed_samples.append(idx)

    print(f"\nSuccessfully loaded {len(all_forcing)} samples")
    if failed_samples:
        print(f"Failed/skipped samples: {len(failed_samples)}")
        if len(failed_samples) <= 10:
            print(f"  Indices: {failed_samples}")

    if len(all_forcing) == 0:
        raise ValueError("No valid samples found!")

    # Get dimensions
    n_samples = len(all_forcing)
    n_timesteps = len(all_forcing[0])
    n_forcing_vars = config.get_num_forcing_variables()
    n_target_vars = config.get_num_target_variables()
    n_params = params.shape[1]

    forcing_var_names = config.get_forcing_variable_names()
    target_var_names = config.get_target_variable_names()
    target_categories = config.get_target_variable_categories()
    param_names = params.columns.tolist()

    print(f"\nData dimensions:")
    print(f"  Samples: {n_samples}")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  Forcing variables: {n_forcing_vars}")
    print(f"  Target variables: {n_target_vars}")
    print(f"  Parameters: {n_params}")

    # Create forcing array: (n_samples, n_timesteps, n_forcing_vars)
    X_forcing = np.zeros((n_samples, n_timesteps, n_forcing_vars))
    for i, forcing_data in enumerate(all_forcing):
        for j, var_name in enumerate(forcing_var_names):
            X_forcing[i, :, j] = forcing_data[var_name].values

    # Create target array: (n_samples, n_timesteps, n_target_vars)
    y = np.zeros((n_samples, n_timesteps, n_target_vars))
    for i, target_data in enumerate(all_targets):
        for j, var_name in enumerate(target_var_names):
            y[i, :, j] = target_data[var_name].values

    # Create parameter array: (n_samples, n_params)
    X_params = params.iloc[valid_indices].values

    # Apply log transformation to specified parameters
    log_transformed_params = []
    if config.LOG_TRANSFORM_PARAMS:
        print(f"\nApplying log10 transformation to parameters: {config.LOG_TRANSFORM_PARAMS}")
        for param_name in config.LOG_TRANSFORM_PARAMS:
            if param_name in param_names:
                param_idx = param_names.index(param_name)
                # Check original value range
                orig_min = X_params[:, param_idx].min()
                orig_max = X_params[:, param_idx].max()
                print(f"  {param_name}: original range [{orig_min:.2e}, {orig_max:.2e}]")

                # Apply log10 transformation
                X_params[:, param_idx] = np.log10(X_params[:, param_idx])
                log_transformed_params.append(param_name)

                # Check transformed value range
                trans_min = X_params[:, param_idx].min()
                trans_max = X_params[:, param_idx].max()
                print(f"  {param_name}: log10 range [{trans_min:.2f}, {trans_max:.2f}]")
            else:
                print(f"  Warning: Parameter '{param_name}' not found in parameter file")

    # Check for infinite or NaN values
    print("\nChecking data quality...")
    if np.any(np.isnan(X_forcing)) or np.any(np.isinf(X_forcing)):
        print("Warning: X_forcing contains NaN or Inf values")
    if np.any(np.isnan(X_params)) or np.any(np.isinf(X_params)):
        print("Warning: X_params contains NaN or Inf values")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        print("Warning: y contains NaN or Inf values")

    # Normalize data using configured method
    norm_method = config.NORMALIZATION_METHOD
    print(f"\nNormalizing data using method: {norm_method}")

    # Normalize forcing variables (per variable, across samples and time)
    X_forcing_normalized, forcing_norm_stats = normalize_data(X_forcing, method=norm_method)

    # Normalize parameters (per parameter, across samples)
    X_params_normalized, params_norm_stats = normalize_data(X_params, method=norm_method)

    # Normalize targets (per variable, across samples and time)
    y_normalized, targets_norm_stats = normalize_data(y, method=norm_method)

    # Print normalization statistics
    print("\nNormalization statistics:")
    if norm_method == 'z-score':
        print(f"  Forcing - mean range: [{forcing_norm_stats['mean'].min():.2e}, {forcing_norm_stats['mean'].max():.2e}]")
        print(f"  Forcing - std range: [{forcing_norm_stats['std'].min():.2e}, {forcing_norm_stats['std'].max():.2e}]")
        print(f"  Params - mean range: [{params_norm_stats['mean'].min():.2e}, {params_norm_stats['mean'].max():.2e}]")
        print(f"  Params - std range: [{params_norm_stats['std'].min():.2e}, {params_norm_stats['std'].max():.2e}]")
        print(f"  Targets - mean range: [{targets_norm_stats['mean'].min():.2e}, {targets_norm_stats['mean'].max():.2e}]")
        print(f"  Targets - std range: [{targets_norm_stats['std'].min():.2e}, {targets_norm_stats['std'].max():.2e}]")
    elif norm_method == 'min-max':
        print(f"  Forcing - min range: [{forcing_norm_stats['min'].min():.2e}, {forcing_norm_stats['min'].max():.2e}]")
        print(f"  Forcing - max range: [{forcing_norm_stats['max'].min():.2e}, {forcing_norm_stats['max'].max():.2e}]")
        print(f"  Params - min range: [{params_norm_stats['min'].min():.2e}, {params_norm_stats['min'].max():.2e}]")
        print(f"  Params - max range: [{params_norm_stats['max'].min():.2e}, {params_norm_stats['max'].max():.2e}]")
        print(f"  Targets - min range: [{targets_norm_stats['min'].min():.2e}, {targets_norm_stats['min'].max():.2e}]")
        print(f"  Targets - max range: [{targets_norm_stats['max'].min():.2e}, {targets_norm_stats['max'].max():.2e}]")

    # Save processed data
    data_dict = {
        'X_forcing': X_forcing_normalized,
        'X_params': X_params_normalized,
        'y': y_normalized,
        # Normalization statistics (new format)
        'forcing_norm_stats': forcing_norm_stats,
        'params_norm_stats': params_norm_stats,
        'targets_norm_stats': targets_norm_stats,
        'normalization_method': norm_method,
        # Backward compatibility (for z-score only)
        'X_forcing_mean': forcing_norm_stats.get('mean', None),
        'X_forcing_std': forcing_norm_stats.get('std', None),
        'X_params_mean': params_norm_stats.get('mean', None),
        'X_params_std': params_norm_stats.get('std', None),
        'y_mean': targets_norm_stats.get('mean', None),
        'y_std': targets_norm_stats.get('std', None),
        # Variable names and metadata
        'forcing_var_names': forcing_var_names,
        'target_var_names': target_var_names,
        'target_categories': target_categories,
        'param_names': param_names,
        'log_transformed_params': log_transformed_params,  # Track which params were log-transformed
        'n_timesteps': n_timesteps,
        'valid_indices': valid_indices,
        'failed_indices': failed_samples,
        'n_forcing_vars': n_forcing_vars,
        'n_target_vars': n_target_vars,
        'n_params': n_params
    }

    # Create output directory if needed
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(data_dict, f)

    print(f"\nData preprocessing complete!")
    print(f"Forcing input shape: {X_forcing_normalized.shape} (samples, timesteps, forcing_vars)")
    print(f"Parameter input shape: {X_params_normalized.shape} (samples, n_params)")
    print(f"Target output shape: {y_normalized.shape} (samples, timesteps, target_vars)")
    if log_transformed_params:
        print(f"Log10-transformed parameters: {log_transformed_params}")
    print(f"Saved to: {output_file}")

    # Print target variable breakdown
    print(f"\nTarget variables by category:")
    for category in ['energy_balance', 'energy_components', 'water_fluxes', 'water_storage', 'temperature']:
        count = sum(1 for c in target_categories if c == category)
        print(f"  {category}: {count} variables")

    return data_dict

if __name__ == '__main__':
    data = preprocess_all_data()
