"""
Inference script for parameter prediction from observation data
Loads observation data (Panama CSV) and forcing data, then predicts parameters
"""

import numpy as np
import pandas as pd
import torch
import pickle
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import xarray as xr
from lstm_model import LSTMParameterPredictor, BiLSTMParameterPredictor, AttentionLSTMParameterPredictor
import config


def load_observation_data(
    obs_file='data/obs/Panama_BCI_v5.1_fluxtowerdata.csv',
    start_time_local='2015-07-30 12:30:00',
    n_days=None,
    utc_offset_hours=-5
):
    """
    Load observation data from Panama CSV

    Args:
        obs_file: Path to observation CSV file
        start_time_local: Start time in local time (Panama time)
        n_days: Number of days to load (if None, loads until end of file)
        utc_offset_hours: UTC offset for local time (Panama is UTC-5)

    Returns:
        DataFrame with observation data at 30-min resolution
    """
    print(f"\nLoading observation data from {obs_file}...")

    # Read CSV
    df = pd.read_csv(obs_file)

    # Parse dates - handle both "M/D/YY H:MM" and other formats
    df['datetime_local'] = pd.to_datetime(df['date'], format='mixed')

    # Convert to UTC
    # UTC offset is negative for western hemisphere (e.g., UTC-5 means local = UTC - 5, so UTC = local + 5)
    df['datetime_utc'] = df['datetime_local'] - pd.Timedelta(hours=utc_offset_hours)

    # Filter by start time
    start_dt_local = pd.to_datetime(start_time_local)
    start_dt_utc = start_dt_local - pd.Timedelta(hours=utc_offset_hours)

    print(f"Start time (local): {start_dt_local}")
    print(f"Start time (UTC): {start_dt_utc}")

    # Filter data
    df_filtered = df[df['datetime_utc'] >= start_dt_utc].copy()

    if n_days is not None:
        end_dt_utc = start_dt_utc + pd.Timedelta(days=n_days)
        df_filtered = df_filtered[df_filtered['datetime_utc'] < end_dt_utc].copy()
        print(f"End time (UTC): {end_dt_utc}")

    # Select relevant columns
    # Map observation variables to simulation variables:
    # SWC -> SOIL_M (soil moisture)
    # LE -> LH (latent heat)
    # H -> HFX (sensible heat)

    # Rename columns to match simulation names
    obs_data = pd.DataFrame({
        'time': df_filtered['datetime_utc'],
        'SOIL_M': df_filtered['SWC'],  # Soil moisture
        'LH': df_filtered['LE'],        # Latent heat
        'HFX': df_filtered['H'],        # Sensible heat
    })

    print(f"Loaded {len(obs_data)} timesteps ({len(obs_data)/48:.1f} days)")
    print(f"Time range: {obs_data['time'].min()} to {obs_data['time'].max()}")

    # Check for missing values
    missing_counts = obs_data[['SOIL_M', 'LH', 'HFX']].isnull().sum()
    if missing_counts.any():
        print("\nWarning: Missing values detected:")
        print(missing_counts)
        print("Filling missing values with forward fill...")
        obs_data[['SOIL_M', 'LH', 'HFX']] = obs_data[['SOIL_M', 'LH', 'HFX']].fillna(method='ffill')
        obs_data[['SOIL_M', 'LH', 'HFX']] = obs_data[['SOIL_M', 'LH', 'HFX']].fillna(method='bfill')

    return obs_data


def load_forcing_data(
    sample_idx=1,
    data_dir='data/raw/sim_results',
    start_time_utc='2015-07-30 17:30:00',
    n_timesteps=None
):
    """
    Load forcing data from a sample NetCDF file
    This provides the forcing variables that are not in the observation data

    Args:
        sample_idx: Sample index to use for forcing data
        data_dir: Directory containing simulation results
        start_time_utc: Start time in UTC
        n_timesteps: Number of timesteps to load

    Returns:
        DataFrame with forcing data
    """
    print(f"\nLoading forcing data from sample {sample_idx}...")

    sim_path = Path(data_dir) / f'sample_{sample_idx}' / 'output' / '201507301730.LDASOUT_DOMAIN1'

    if not sim_path.exists():
        print(f"Warning: Forcing file not found at {sim_path}")
        return None

    # Load with xarray
    ds = xr.open_dataset(sim_path)

    # Get time information
    time_strings = [s.decode() if isinstance(s, bytes) else s for s in ds['Times'].values]
    time_strings = [s.replace('_', ' ') for s in time_strings]
    times = pd.to_datetime(time_strings)

    # Filter by start time
    start_dt = pd.to_datetime(start_time_utc)
    time_mask = times >= start_dt

    if n_timesteps is not None:
        time_mask = time_mask & (times < start_dt + pd.Timedelta(minutes=30*n_timesteps))

    times_filtered = times[time_mask]

    # Extract forcing variables based on config
    forcing_vars = {}

    # Check if using extended config with forcing variables
    for var_config in config.INPUT_VARIABLES:
        var_name = var_config['name']

        # Skip the output variables (we get these from observations)
        if var_name in ['SOIL_M', 'LH', 'HFX']:
            continue

        if var_name not in ds:
            print(f"Warning: Forcing variable '{var_name}' not found in NetCDF")
            continue

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

        forcing_vars[var_name] = var_data.values[time_mask]

    ds.close()

    # Create DataFrame
    forcing_data = pd.DataFrame({'time': times_filtered})
    for var_name, var_values in forcing_vars.items():
        forcing_data[var_name] = var_values

    print(f"Loaded {len(forcing_data)} timesteps of forcing data")
    print(f"Forcing variables: {list(forcing_vars.keys())}")

    return forcing_data


def combine_obs_and_forcing(obs_data, forcing_data):
    """
    Combine observation and forcing data

    Args:
        obs_data: DataFrame with observation data (SOIL_M, LH, HFX)
        forcing_data: DataFrame with forcing data (or None if not using forcing)

    Returns:
        DataFrame with combined data
    """
    if forcing_data is None:
        print("\nNo forcing data, using only observations")
        return obs_data

    print("\nCombining observation and forcing data...")

    # Merge on time
    combined = pd.merge(obs_data, forcing_data, on='time', how='left')

    # Fill any missing forcing values
    for col in forcing_data.columns:
        if col != 'time' and col in combined.columns:
            if combined[col].isnull().any():
                print(f"Warning: Missing values in {col}, filling with forward fill")
                combined[col] = combined[col].fillna(method='ffill').fillna(method='bfill')

    return combined


def aggregate_to_daily(data, variables):
    """
    Aggregate 30-min data to daily resolution based on config

    Args:
        data: DataFrame with 30-min data
        variables: List of variable configs from config.INPUT_VARIABLES

    Returns:
        DataFrame with daily aggregated data
    """
    print("\nAggregating to daily resolution...")

    # Create date column
    data['date'] = data['time'].dt.date

    # Build aggregation dictionary
    agg_dict = {}
    for var_config in variables:
        var_name = var_config['name']
        if var_name in data.columns:
            agg_dict[var_name] = var_config['aggregation']

    # Aggregate
    daily_df = data.groupby('date').agg(agg_dict).reset_index()

    # Add temporal features if configured
    if config.ADD_TIME_FEATURES:
        daily_df['doy'] = pd.to_datetime(daily_df['date']).dt.dayofyear / 365.0

    if config.ADD_MONTH_FEATURE:
        months = pd.to_datetime(daily_df['date']).dt.month
        for month in range(1, 13):
            daily_df[f'month_{month}'] = (months == month).astype(float)

    print(f"Aggregated to {len(daily_df)} days")

    return daily_df


def prepare_input_array(daily_df, variable_names, X_mean, X_std):
    """
    Prepare input array for model inference

    Args:
        daily_df: DataFrame with daily aggregated data
        variable_names: List of variable names (from training)
        X_mean: Mean values for normalization (from training)
        X_std: Std values for normalization (from training)

    Returns:
        Normalized input array (1, n_variables, n_timesteps)
    """
    print("\nPreparing input array...")

    n_timesteps = len(daily_df)
    n_variables = len(variable_names)

    # Create input array
    X = np.zeros((1, n_variables, n_timesteps))

    # Fill array
    for j, var_name in enumerate(variable_names):
        if var_name not in daily_df.columns:
            print(f"Warning: Variable '{var_name}' not found in data, using zeros")
            continue
        X[0, j, :] = daily_df[var_name].values

    print(f"Input shape: {X.shape} (samples, variables, timesteps)")

    # Normalize using training statistics
    print("Normalizing inputs...")
    X_normalized = (X - X_mean) / (X_std + 1e-8)

    return X_normalized


def load_model_and_data(model_dir, data_file=None):
    """
    Load trained model and preprocessing data

    Args:
        model_dir: Directory containing best_model.pth
        data_file: Path to processed data pickle (optional, for normalization stats)

    Returns:
        model, data_dict
    """
    print(f"\nLoading model from {model_dir}...")

    model_dir = Path(model_dir)

    # Load config
    with open(model_dir / 'config.json', 'r') as f:
        import json
        model_config = json.load(f)

    print("Model configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")

    # Load preprocessing data
    if data_file is None:
        data_file = config.OUTPUT_FILE

    print(f"\nLoading preprocessing data from {data_file}...")
    with open(data_file, 'rb') as f:
        data_dict = pickle.load(f)

    print(f"Variable names: {data_dict['variable_names']}")
    print(f"Parameter names: {data_dict['param_names']}")

    # Create model
    input_dim = data_dict['n_variables']
    output_dim = len(data_dict['param_names'])

    if model_config['model_type'] == 'LSTM':
        model = LSTMParameterPredictor(
            input_dim=input_dim,
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            output_dim=output_dim,
            dropout=model_config['dropout']
        )
    elif model_config['model_type'] == 'BiLSTM':
        model = BiLSTMParameterPredictor(
            input_dim=input_dim,
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            output_dim=output_dim,
            dropout=model_config['dropout']
        )
    elif model_config['model_type'] == 'AttentionLSTM':
        model = AttentionLSTMParameterPredictor(
            input_dim=input_dim,
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            output_dim=output_dim,
            dropout=model_config['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {model_config['model_type']}. Available types: 'LSTM', 'BiLSTM', 'AttentionLSTM'")

    # Load weights
    checkpoint = torch.load(model_dir / 'best_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Model loaded successfully")

    return model, data_dict


def predict_parameters(model, X_normalized, data_dict):
    """
    Predict parameters using trained model

    Args:
        model: Trained model
        X_normalized: Normalized input array (1, n_variables, n_timesteps)
        data_dict: Dictionary with normalization statistics

    Returns:
        Predicted parameters (denormalized)
    """
    print("\nRunning inference...")

    # Transpose to (1, n_timesteps, n_variables) for LSTM
    X_transposed = np.transpose(X_normalized, (0, 2, 1))
    X_tensor = torch.FloatTensor(X_transposed)

    # Predict
    with torch.no_grad():
        y_pred_normalized = model(X_tensor)
        y_pred_normalized = y_pred_normalized.cpu().numpy()

    # Denormalize
    y_mean = data_dict['y_mean']
    y_std = data_dict['y_std']
    y_pred = y_pred_normalized * (y_std + 1e-8) + y_mean

    return y_pred[0]  # Return first (and only) sample


def main():
    parser = argparse.ArgumentParser(description='Infer parameters from observation data')
    parser.add_argument('--obs_file', type=str,
                       default='data/obs/Panama_BCI_v5.1_fluxtowerdata.csv',
                       help='Path to observation CSV file')
    parser.add_argument('--start_time_local', type=str,
                       default='2015-07-30 12:30:00',
                       help='Start time in local time (Panama time)')
    parser.add_argument('--n_days', type=int, default=None,
                       help='Number of days to process (default: all available)')
    parser.add_argument('--model_dir', type=str,
                       default='results/AttentionLSTM_20251110_112216_dim-1024_layer-2',
                       help='Directory containing trained model')
    parser.add_argument('--data_file', type=str, default=None,
                       help='Path to processed data pickle (default: from config)')
    parser.add_argument('--forcing_sample', type=int, default=1,
                       help='Sample index to use for forcing data')
    parser.add_argument('--use_forcing', action='store_true',
                       help='Use forcing data from sample (default: False)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save predicted parameters (CSV)')

    args = parser.parse_args()

    print("="*80)
    print("PARAMETER INFERENCE FROM OBSERVATION DATA")
    print("="*80)

    # Load model and preprocessing data
    model, data_dict = load_model_and_data(args.model_dir, args.data_file)

    # Load observation data
    obs_data = load_observation_data(
        obs_file=args.obs_file,
        start_time_local=args.start_time_local,
        n_days=args.n_days,
        utc_offset_hours=-5
    )

    # Load forcing data if requested
    forcing_data = None
    if args.use_forcing:
        # Calculate start time in UTC for forcing
        start_dt_local = pd.to_datetime(args.start_time_local)
        start_dt_utc = start_dt_local + pd.Timedelta(hours=5)  # Panama is UTC-5, so UTC = local + 5
        start_time_utc_str = start_dt_utc.strftime('%Y-%m-%d %H:%M:%S')

        forcing_data = load_forcing_data(
            sample_idx=args.forcing_sample,
            start_time_utc=start_time_utc_str,
            n_timesteps=len(obs_data)
        )

    # Combine data
    combined_data = combine_obs_and_forcing(obs_data, forcing_data)

    # Aggregate to daily
    daily_data = aggregate_to_daily(combined_data, config.INPUT_VARIABLES)

    # Prepare input array
    X_normalized = prepare_input_array(
        daily_data,
        data_dict['variable_names'],
        data_dict['X_mean'],
        data_dict['X_std']
    )

    # Predict parameters
    params_pred = predict_parameters(model, X_normalized, data_dict)

    # Display results
    print("\n" + "="*80)
    print("PREDICTED PARAMETERS")
    print("="*80)

    param_names = data_dict['param_names']
    for i, (name, value) in enumerate(zip(param_names, params_pred)):
        print(f"{name:20s}: {value:12.6f}")

    # Save results if requested
    if args.output_file is not None:
        output_df = pd.DataFrame({
            'parameter': param_names,
            'value': params_pred
        })
        output_df.to_csv(args.output_file, index=False)
        print(f"\nResults saved to: {args.output_file}")

    print("="*80)
    print("Inference complete!")
    print("="*80)


if __name__ == '__main__':
    main()
