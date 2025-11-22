"""
Parameter Calibration using LSTM Emulator with Adam Optimizer

Calibrates NoahMP parameters by minimizing error between emulator predictions and observations
using gradient-based optimization (Adam) with multiple random starting points.

Key Features:
  - Multiple random starting points (num_calibration) for ensemble predictions
  - Fast gradient-based optimization with Adam
  - Avoids local optima through diverse initialization
  - Uncertainty quantification from multiple calibration runs
  - Uses observation data from data/obs/Panama_BCI_obs_2015-07-30_2016-07-29.csv

Input:
  1. Trained emulator model
  2. Forcing data (NetCDF)
  3. Observation data (CSV with daily resolution)
  4. Parameter bounds

Output:
  - Top N calibrated parameter sets
  - Optimization history
  - Comparison plots for each calibration result
  - Ensemble statistics and parameter distributions
"""

import torch
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime

from lstm_model_forward import LSTMForwardPredictor, BiLSTMForwardPredictor, AttentionLSTMForwardPredictor
import config_forward_comprehensive as config


def set_dropout_to_eval(model):
    """Set all dropout layers to eval mode while keeping model in train mode"""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()


def load_model(model_dir, device='cpu'):
    """Load trained emulator model"""
    model_dir = Path(model_dir)

    # Load config
    with open(model_dir / 'config.json', 'r') as f:
        config_dict = json.load(f)

    # Load data statistics (for normalization)
    data_file = Path(config.OUTPUT_FILE)
    with open(data_file, 'rb') as f:
        data_dict = pickle.load(f)

    # Create model
    model_type = config_dict['model_type']
    model_config = config_dict['model_config']

    model_kwargs = {
        'n_params': config_dict['n_params'],
        'n_forcing_vars': config_dict['n_forcing_vars'],
        'n_target_vars': config_dict['n_target_vars'],
        'hidden_dim': model_config['hidden_dim'],
        'num_layers': model_config['num_layers'],
        'param_embedding_dim': model_config['param_embedding_dim'],
        'dropout': model_config['dropout']
    }

    if model_type == 'LSTM':
        model = LSTMForwardPredictor(**model_kwargs)
    elif model_type == 'BiLSTM':
        model = BiLSTMForwardPredictor(**model_kwargs)
    elif model_type == 'AttentionLSTM':
        model = AttentionLSTMForwardPredictor(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(model_dir / 'best_model.pth', map_location=device))
    model = model.to(device)

    # Keep model in train mode for gradient computation, but disable dropout
    model.train()
    set_dropout_to_eval(model)

    return model, data_dict, config_dict


def load_forcing_data(forcing_file):
    """Load and process forcing data from NetCDF"""
    print(f"\nLoading forcing data from: {forcing_file}")

    ds = xr.open_dataset(forcing_file)
    times = pd.to_datetime(ds['time'].values)

    # Extract forcing variables
    forcing_data = {'time': times}
    for var_config in config.FORCING_VARIABLES:
        var_name = var_config['name']
        if var_name in ds:
            forcing_data[var_name] = ds[var_name].values

    ds.close()

    # Create DataFrame and aggregate to daily
    forcing_df = pd.DataFrame(forcing_data)
    forcing_df['date'] = forcing_df['time'].dt.date

    agg_dict = {}
    for var_config in config.FORCING_VARIABLES:
        var_name = var_config['name']
        if var_name in forcing_df.columns:
            agg_dict[var_name] = var_config['aggregation']

    forcing_daily = forcing_df.groupby('date').agg(agg_dict).reset_index()

    # Add temporal features
    if config.ADD_TIME_FEATURES:
        forcing_daily['doy'] = pd.to_datetime(forcing_daily['date']).dt.dayofyear / 365.0

    print(f"  Loaded {len(forcing_daily)} days of forcing data")
    print(f"  Date range: {forcing_daily['date'].iloc[0]} to {forcing_daily['date'].iloc[-1]}")

    return forcing_daily


def load_observation_data(obs_file, start_date, end_date):
    """
    Load and process observation data

    Args:
        obs_file: Path to observation CSV file
        start_date: Start date (matching forcing data)
        end_date: End date (matching forcing data)

    Returns:
        DataFrame with daily aggregated observations
    """
    print(f"\nLoading observation data from: {obs_file}")

    # Read observation data
    obs_df = pd.read_csv(obs_file)

    # Parse date - the file already has daily data
    obs_df['date'] = pd.to_datetime(obs_df['date']).dt.date

    # Filter by date range
    start_date_obj = pd.to_datetime(start_date).date()
    end_date_obj = pd.to_datetime(end_date).date()
    obs_df = obs_df[(obs_df['date'] >= start_date_obj) & (obs_df['date'] <= end_date_obj)]

    # Target columns (already in the right format: SOIL_M, LH, HFX)
    target_cols = ['LH', 'HFX', 'SOIL_M']
    available_cols = [col for col in target_cols if col in obs_df.columns]

    if not available_cols:
        raise ValueError(f"No target variables found in observation data. Looking for: {target_cols}")

    print(f"  Loaded {len(obs_df)} days of observations")
    print(f"  Available variables: {available_cols}")
    print(f"  Date range: {obs_df['date'].iloc[0]} to {obs_df['date'].iloc[-1]}")

    return obs_df, available_cols


def predict_with_emulator(model, params, forcing_tensor, data_dict, device='cpu', return_numpy=True):
    """
    Run emulator prediction

    Args:
        model: Trained emulator model
        params: Parameter array or tensor (numpy array or torch tensor)
        forcing_tensor: Forcing tensor (n_timesteps, n_forcing_vars)
        data_dict: Normalization statistics
        device: Computing device
        return_numpy: If True, return numpy array; if False, return tensor (for gradients)

    Returns:
        Predictions (tensor if return_numpy=False, numpy array otherwise)
    """
    # Convert params to tensor if needed
    if isinstance(params, np.ndarray):
        params = torch.FloatTensor(params).to(device)

    # Normalize parameters using the appropriate method
    if params.dim() == 1:
        params = params.unsqueeze(0)

    params_stats = data_dict['params_norm_stats']
    if params_stats['method'] == 'z-score':
        params_mean = torch.FloatTensor(params_stats['mean']).to(device)
        params_std = torch.FloatTensor(params_stats['std']).to(device)
        params_normalized = (params - params_mean) / (params_std + 1e-8)
    elif params_stats['method'] == 'min-max':
        params_min = torch.FloatTensor(params_stats['min']).to(device)
        params_max = torch.FloatTensor(params_stats['max']).to(device)
        params_normalized = (params - params_min) / (params_max - params_min + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {params_stats['method']}")

    # Forcing is already normalized
    if forcing_tensor.dim() == 2:
        forcing_tensor = forcing_tensor.unsqueeze(0)

    # Predict
    predictions_normalized = model(params_normalized, forcing_tensor)

    # Denormalize predictions using the appropriate method
    targets_stats = data_dict['targets_norm_stats']
    if targets_stats['method'] == 'z-score':
        y_mean = torch.FloatTensor(targets_stats['mean']).to(device)
        y_std = torch.FloatTensor(targets_stats['std']).to(device)
        predictions = predictions_normalized * y_std + y_mean
    elif targets_stats['method'] == 'min-max':
        y_min = torch.FloatTensor(targets_stats['min']).to(device)
        y_max = torch.FloatTensor(targets_stats['max']).to(device)
        predictions = predictions_normalized * (y_max - y_min) + y_min
    else:
        raise ValueError(f"Unknown normalization method: {targets_stats['method']}")

    # Return as numpy or tensor
    if return_numpy:
        return predictions[0].detach().cpu().numpy()
    else:
        return predictions[0]  # Keep as tensor for gradient flow


def calculate_loss(predictions, observations_tensor, variable_mask):
    """
    Calculate normalized RMSE loss for gradient-based optimization

    Args:
        predictions: Tensor (n_timesteps, n_vars)
        observations_tensor: Tensor (n_timesteps, n_vars)
        variable_mask: Boolean tensor indicating valid observations

    Returns:
        Loss tensor (scalar)
    """
    # Calculate MSE for each variable with valid observations
    mse_per_var = []

    for i in range(predictions.shape[1]):
        if variable_mask[i]:
            valid_obs_mask = ~torch.isnan(observations_tensor[:, i])
            if valid_obs_mask.sum() > 0:
                pred_valid = predictions[valid_obs_mask, i]
                obs_valid = observations_tensor[valid_obs_mask, i]

                # Normalized by observation range
                obs_range = obs_valid.max() - obs_valid.min() + 1e-8
                mse = torch.mean((pred_valid - obs_valid) ** 2) / (obs_range ** 2)
                mse_per_var.append(mse)

    if len(mse_per_var) == 0:
        return torch.tensor(0.0, requires_grad=True)

    # Mean across variables
    loss = torch.mean(torch.stack(mse_per_var))
    return loss


def calculate_normalized_rmse(predictions, observations, variable_names):
    """
    Calculate mean normalized RMSE across multiple variables (for reporting)

    Args:
        predictions: Array (n_timesteps, n_vars)
        observations: DataFrame with date and variable columns
        variable_names: List of variable names to evaluate

    Returns:
        Mean normalized RMSE and per-variable errors
    """
    nrmse_list = []
    var_errors = {}

    for i, var_name in enumerate(variable_names):
        if var_name not in observations.columns:
            continue

        obs_values = observations[var_name].values
        pred_values = predictions[:len(obs_values), i]

        # Remove NaN values
        valid_mask = ~np.isnan(obs_values) & ~np.isnan(pred_values)
        obs_clean = obs_values[valid_mask]
        pred_clean = pred_values[valid_mask]

        if len(obs_clean) == 0:
            continue

        # Calculate RMSE
        rmse = np.sqrt(np.mean((pred_clean - obs_clean) ** 2))

        # Normalize by observation range
        obs_range = np.max(obs_clean) - np.min(obs_clean)
        nrmse = rmse / (obs_range + 1e-8)

        nrmse_list.append(nrmse)
        var_errors[var_name] = {'rmse': rmse, 'nrmse': nrmse}

    return np.mean(nrmse_list), var_errors


def load_parameter_bounds(bounds_file, param_names):
    """
    Load parameter bounds from CSV file

    Args:
        bounds_file: Path to bounds CSV
        param_names: List of parameter names

    Returns:
        bounds array (n_params, 2)
    """
    bounds_df = pd.read_csv(bounds_file)

    bounds = []
    for param_name in param_names:
        # Remove suffix (_EBF, _CL, _SCL) to match bounds file
        base_name = param_name.split('_')[0]

        if base_name in bounds_df['variable'].values:
            row = bounds_df[bounds_df['variable'] == base_name].iloc[0]
            bounds.append([row['Lower bound'], row['Upper bound']])
        else:
            # Use wide bounds if not specified
            print(f"  Warning: No bounds for {param_name}, using wide range")
            bounds.append([0.01, 10.0])

    return np.array(bounds)


def load_default_parameters(default_param_file, param_names):
    """
    Load default Noah-MP parameters from file

    Args:
        default_param_file: Path to default parameter file
        param_names: List of parameter names to extract

    Returns:
        Array of default parameter values matching param_names order
    """
    try:
        with open(default_param_file, 'r') as f:
            lines = f.readlines()

        # Parse parameter names and values
        names_line = lines[0].strip().split()
        values_line = lines[1].strip().split()

        # Create dictionary mapping parameter names to values
        default_dict = {}
        for name, value in zip(names_line, values_line):
            try:
                default_dict[name] = float(value)
            except ValueError:
                # Handle scientific notation like 9.74E-7
                default_dict[name] = float(value.replace('E', 'e'))

        # Extract values in the order of param_names
        default_params = []
        for param_name in param_names:
            if param_name in default_dict:
                default_params.append(default_dict[param_name])
            else:
                # Parameter not found, will be handled by caller
                default_params.append(None)

        return np.array(default_params)

    except Exception as e:
        print(f"  Warning: Could not load default parameters from {default_param_file}: {e}")
        return None


def initialize_random_params(bounds, baseline_params, calibrate_indices, seed=None):
    """
    Initialize parameters randomly within bounds

    Args:
        bounds: Parameter bounds array
        baseline_params: Baseline parameter values
        calibrate_indices: Indices of parameters to calibrate
        seed: Random seed for reproducibility

    Returns:
        Randomly initialized parameter array
    """
    if seed is not None:
        np.random.seed(seed)

    params = baseline_params.copy()

    for idx in calibrate_indices:
        lower, upper = bounds[idx]
        params[idx] = np.random.uniform(lower, upper)

    return params


def run_adam_optimization(model, initial_params, forcing_tensor, observations_tensor,
                         variable_mask, calibrate_indices, bounds, data_dict,
                         lr=0.01, max_iterations=500, patience=50, device='cpu'):
    """
    Run Adam optimization from a single starting point

    Args:
        model: Trained emulator model
        initial_params: Initial parameter values
        forcing_tensor: Normalized forcing data tensor
        observations_tensor: Observation data tensor
        variable_mask: Boolean mask for available variables
        calibrate_indices: Indices of parameters to calibrate
        bounds: Parameter bounds
        data_dict: Normalization statistics
        lr: Learning rate for Adam
        max_iterations: Maximum number of iterations
        patience: Early stopping patience
        device: Computing device

    Returns:
        Optimized parameters and optimization history
    """
    # Create parameter tensor (only calibrate specified parameters)
    params_to_optimize = torch.FloatTensor(initial_params[calibrate_indices]).to(device)
    params_to_optimize.requires_grad = True

    # Fixed parameters
    fixed_params = torch.FloatTensor(initial_params).to(device)

    # Bounds for calibrated parameters
    calibrate_bounds = bounds[calibrate_indices]
    lower_bounds = torch.FloatTensor(calibrate_bounds[:, 0]).to(device)
    upper_bounds = torch.FloatTensor(calibrate_bounds[:, 1]).to(device)

    # Optimizer
    optimizer = torch.optim.Adam([params_to_optimize], lr=lr)

    # Tracking
    history = {
        'loss': [],
        'nrmse': [],
        'iteration': []
    }

    best_loss = float('inf')
    best_params = None
    patience_counter = 0

    for iteration in range(max_iterations):
        optimizer.zero_grad()

        # Construct full parameter set
        full_params = fixed_params.clone()
        full_params[calibrate_indices] = params_to_optimize

        # Forward pass (return tensor for gradient computation)
        predictions = predict_with_emulator(model, full_params, forcing_tensor, data_dict, device, return_numpy=False)

        # Calculate loss
        loss = calculate_loss(predictions, observations_tensor, variable_mask)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Project parameters to bounds
        with torch.no_grad():
            params_to_optimize.clamp_(lower_bounds, upper_bounds)

        # Track progress
        current_loss = loss.item()
        history['loss'].append(current_loss)
        history['iteration'].append(iteration)

        # Calculate NRMSE for reporting (every 50 iterations)
        if iteration % 50 == 0 or iteration == max_iterations - 1:
            with torch.no_grad():
                full_params_np = full_params.cpu().numpy()
                pred_np = predictions.cpu().numpy()
                # We need to calculate NRMSE, but we need obs_data DataFrame
                # For now, just use loss as proxy
                nrmse = np.sqrt(current_loss)
                history['nrmse'].append(nrmse)
                print(f"    Iteration {iteration:4d}: Loss = {current_loss:.6f}, NRMSE ≈ {nrmse:.6f}")

        # Early stopping
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = full_params.detach().cpu().numpy().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at iteration {iteration}")
                break

    return best_params, history


def run_calibration_adam(model_dir, forcing_file, obs_file, bounds_file,
                        calibrate_params=None, output_dir='calibration_results',
                        num_calibration=5, max_iterations=500, lr=0.01,
                        patience=50, device='cpu'):
    """
    Run multiple parameter calibrations using Adam optimizer with random starts

    Args:
        model_dir: Directory with trained model
        forcing_file: NetCDF file with forcing data
        obs_file: CSV file with observations
        bounds_file: CSV file with parameter bounds
        calibrate_params: List of parameter names to calibrate (None = all)
        output_dir: Output directory
        num_calibration: Number of random starting points
        max_iterations: Maximum optimization iterations per run
        lr: Learning rate for Adam
        patience: Early stopping patience
        device: Computing device
    """
    print("="*80)
    print("PARAMETER CALIBRATION WITH ADAM OPTIMIZER")
    print(f"Running {num_calibration} calibrations from random starting points")
    print("="*80)

    # Load model
    print("\n[1/6] Loading emulator model...")
    model, data_dict, config_dict = load_model(model_dir, device)
    print(f"  Model: {config_dict['model_type']}")

    # Load forcing data
    print("\n[2/6] Loading forcing data...")
    forcing_df = load_forcing_data(forcing_file)
    forcing_array = forcing_df[data_dict['forcing_var_names']].values

    # Normalize and convert to tensor using the appropriate method
    forcing_stats = data_dict['forcing_norm_stats']
    if forcing_stats['method'] == 'z-score':
        forcing_normalized = (forcing_array - forcing_stats['mean']) / (forcing_stats['std'] + 1e-8)
    elif forcing_stats['method'] == 'min-max':
        forcing_normalized = (forcing_array - forcing_stats['min']) / (forcing_stats['max'] - forcing_stats['min'] + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {forcing_stats['method']}")

    forcing_tensor = torch.FloatTensor(forcing_normalized).to(device)

    # Get date range from forcing
    start_date = forcing_df['date'].iloc[0]
    end_date = forcing_df['date'].iloc[-1]

    # Load observation data
    print("\n[3/6] Loading observation data...")
    obs_data, available_vars = load_observation_data(obs_file, start_date, end_date)

    # Convert observations to tensor
    n_timesteps = len(forcing_array)
    n_vars = len(data_dict['target_var_names'])
    observations_array = np.full((n_timesteps, n_vars), np.nan)
    variable_mask = np.zeros(n_vars, dtype=bool)

    for i, var_name in enumerate(data_dict['target_var_names']):
        if var_name in available_vars:
            observations_array[:len(obs_data), i] = obs_data[var_name].values
            variable_mask[i] = True

    observations_tensor = torch.FloatTensor(observations_array).to(device)
    variable_mask_tensor = torch.BoolTensor(variable_mask).to(device)

    # Setup parameters to calibrate
    print("\n[4/6] Setting up calibration parameters...")
    param_names = data_dict['param_names']

    if calibrate_params is None:
        calibrate_params = param_names
        calibrate_indices = list(range(len(param_names)))
    else:
        calibrate_indices = [i for i, name in enumerate(param_names) if name in calibrate_params]
        calibrate_params = [param_names[i] for i in calibrate_indices]

    print(f"  Total parameters: {len(param_names)}")
    print(f"  Calibrating: {len(calibrate_params)} parameters")

    # Load parameter bounds
    bounds = load_parameter_bounds(bounds_file, param_names)

    # Get baseline parameters
    # Priority: 1) Default Noah-MP params, 2) Training data mean, 3) Midpoint of bounds
    default_param_file = Path('data/raw/param/default_param.txt')
    default_params = load_default_parameters(default_param_file, param_names)

    if default_params is not None and not np.any(default_params == None):
        # First priority: Use scientifically validated default Noah-MP parameters
        baseline_params = default_params
        print(f"  Using Noah-MP default parameters as baseline")
    elif data_dict['X_params_mean'] is not None:
        # Second priority: Use mean from training data (for z-score normalization)
        baseline_params = data_dict['X_params_mean'].flatten().copy()
        print(f"  Using training data mean as baseline (default params not available)")
    else:
        # Fallback: Use midpoint of bounds
        baseline_params = np.array([(lower + upper) / 2 for lower, upper in bounds])
        print(f"  Using midpoint of bounds as baseline (default params and mean not available)")

    # Run multiple calibrations from random starting points
    print(f"\n[5/6] Running {num_calibration} calibrations from random starting points...")
    print(f"  Max iterations per run: {max_iterations}")
    print(f"  Learning rate: {lr}")
    print(f"  Early stopping patience: {patience}")

    all_calibration_results = []

    for run_idx in range(num_calibration):
        print(f"\n{'='*60}")
        print(f"Calibration Run {run_idx + 1}/{num_calibration} (seed={42 + run_idx})")
        print(f"{'='*60}")

        # Initialize random starting point
        initial_params = initialize_random_params(
            bounds, baseline_params, calibrate_indices, seed=42 + run_idx
        )

        print(f"  Random initialization complete")

        # Run optimization
        optimized_params, history = run_adam_optimization(
            model=model,
            initial_params=initial_params,
            forcing_tensor=forcing_tensor,
            observations_tensor=observations_tensor,
            variable_mask=variable_mask_tensor,
            calibrate_indices=calibrate_indices,
            bounds=bounds,
            data_dict=data_dict,
            lr=lr,
            max_iterations=max_iterations,
            patience=patience,
            device=device
        )

        # Calculate final error
        predictions_np = predict_with_emulator(model, optimized_params, forcing_tensor, data_dict, device)
        error, var_errors = calculate_normalized_rmse(predictions_np, obs_data, data_dict['target_var_names'])

        all_calibration_results.append({
            'run_id': run_idx + 1,
            'seed': 42 + run_idx,
            'initial_params': initial_params.copy(),
            'params': optimized_params.copy(),
            'calibrate_params': optimized_params[calibrate_indices].copy(),
            'error': error,
            'var_errors': var_errors,
            'history': history,
            'n_iterations': len(history['loss'])
        })

        print(f"\nRun {run_idx + 1} Complete:")
        print(f"  Final NRMSE: {error:.6f}")
        print(f"  Iterations: {len(history['loss'])}")

    # Sort results by error
    print(f"\n[6/6] Analyzing {num_calibration} calibration results...")
    top_results = sorted(all_calibration_results, key=lambda x: x['error'])

    print(f"\nCalibration results summary:")
    errors = [r['error'] for r in top_results]
    print(f"  Best NRMSE:  {min(errors):.6f}")
    print(f"  Worst NRMSE: {max(errors):.6f}")
    print(f"  Mean NRMSE:  {np.mean(errors):.6f}")
    print(f"  Std NRMSE:   {np.std(errors):.6f}")

    print(f"\nIndividual results:")
    for i, res in enumerate(top_results, 1):
        print(f"  #{i} (Run {res['run_id']}): NRMSE = {res['error']:.6f}")

    # Save results
    print("\n" + "="*80)
    print("CALIBRATION COMPLETE")
    print("="*80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save all calibration results
    for i, res in enumerate(top_results):
        result_dir = output_path / f'calibration_{i+1}'
        result_dir.mkdir(exist_ok=True)

        # Save calibrated parameters
        calib_params_df = pd.DataFrame({
            'parameter': param_names,
            'baseline': baseline_params,
            'initial': res['initial_params'],
            'calibrated': res['params'],
            'is_calibrated': ['Yes' if j in calibrate_indices else 'No' for j in range(len(param_names))]
        })
        calib_params_df.to_csv(result_dir / 'calibrated_parameters.csv', index=False)

        # Save detailed result
        result_dict = {
            'rank': i + 1,
            'final_error': float(res['error']),
            'variable_errors': {k: {'rmse': float(v['rmse']), 'nrmse': float(v['nrmse'])}
                              for k, v in res['var_errors'].items()},
            'calibrated_params': calibrate_params,
            'calibrated_values': res['calibrate_params'].tolist(),
            'n_iterations': res['n_iterations']
        }

        with open(result_dir / 'calibration_result.json', 'w') as f:
            json.dump(result_dict, f, indent=2)

        # Save optimization history
        history_df = pd.DataFrame({
            'iteration': res['history']['iteration'],
            'loss': res['history']['loss']
        })
        history_df.to_csv(result_dir / 'optimization_history.csv', index=False)

        # Generate predictions and save
        final_predictions_np = predict_with_emulator(
            model, res['params'], forcing_tensor, data_dict, device
        )

        pred_df = pd.DataFrame({'date': forcing_df['date'].values})
        for j, var_name in enumerate(data_dict['target_var_names']):
            pred_df[f'{var_name}_pred'] = final_predictions_np[:, j]
            if var_name in obs_data.columns:
                pred_df[f'{var_name}_obs'] = obs_data[var_name].values

        pred_df.to_csv(result_dir / 'predictions_comparison.csv', index=False)

        # Plot results
        plot_calibration_results(
            pred_df, data_dict['target_var_names'], available_vars,
            result_dir / 'calibration_results.png', rank=i+1, error=res['error']
        )

        # Plot optimization history
        plot_optimization_history(
            res['history'], result_dir / 'optimization_history.png'
        )

    # Save ensemble summary
    ensemble_summary = {
        'num_calibrations': num_calibration,
        'optimization_method': 'adam_random_starts',
        'max_iterations': max_iterations,
        'learning_rate': lr,
        'patience': patience,
        'statistics': {
            'nrmse_mean': float(np.mean(errors)),
            'nrmse_std': float(np.std(errors)),
            'nrmse_min': float(np.min(errors)),
            'nrmse_max': float(np.max(errors))
        },
        'results': [
            {
                'rank': i+1,
                'run_id': res['run_id'],
                'seed': res['seed'],
                'nrmse': float(res['error']),
                'n_iterations': res['n_iterations'],
                'variable_errors': {k: float(v['nrmse']) for k, v in res['var_errors'].items()}
            }
            for i, res in enumerate(top_results)
        ]
    }

    with open(output_path / 'ensemble_summary.json', 'w') as f:
        json.dump(ensemble_summary, f, indent=2)

    # Calculate and save parameter statistics
    param_statistics = calculate_parameter_statistics(
        top_results, calibrate_indices, param_names, baseline_params
    )

    with open(output_path / 'parameter_statistics.json', 'w') as f:
        json.dump(param_statistics, f, indent=2)

    # Create parameter distribution plots
    plot_parameter_distributions(
        top_results, calibrate_indices, param_names,
        output_path / 'parameter_distributions.png'
    )

    plot_parameter_correlation(
        top_results, calibrate_indices, param_names,
        output_path / 'parameter_correlations.png'
    )

    print(f"\nResults saved to: {output_path}")
    print(f"Generated {num_calibration} calibration results from random starting points")
    print(f"Parameter distribution plots created")

    return top_results, ensemble_summary


def calculate_parameter_statistics(results, calibrate_indices, param_names, baseline_params):
    """
    Calculate statistics for calibrated parameters across multiple runs

    Args:
        results: List of calibration results
        calibrate_indices: Indices of calibrated parameters
        param_names: All parameter names
        baseline_params: Baseline parameter values

    Returns:
        Dictionary with parameter statistics
    """
    # Extract calibrated parameters from all runs
    n_runs = len(results)
    n_params = len(calibrate_indices)

    param_values = np.zeros((n_runs, n_params))
    for i, res in enumerate(results):
        param_values[i, :] = res['calibrate_params']

    # Calculate statistics
    statistics = {}
    for i, param_idx in enumerate(calibrate_indices):
        param_name = param_names[param_idx]
        values = param_values[:, i]

        statistics[param_name] = {
            'baseline': float(baseline_params[param_idx]),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'cv': float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0.0,
            'all_values': values.tolist()
        }

    return {
        'num_calibrations': n_runs,
        'num_parameters': n_params,
        'parameter_names': [param_names[i] for i in calibrate_indices],
        'parameters': statistics
    }


def plot_parameter_distributions(results, calibrate_indices, param_names, save_path):
    """
    Plot distributions of calibrated parameters across multiple runs
    """
    from matplotlib.gridspec import GridSpec

    # Extract parameter values
    n_params = len(calibrate_indices)
    param_data = []
    param_labels = []

    for i, param_idx in enumerate(calibrate_indices):
        param_name = param_names[param_idx]
        values = [res['calibrate_params'][i] for res in results]
        param_data.append(values)
        param_labels.append(param_name)

    # Create figure with subplots
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)

    for i in range(n_params):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])

        values = param_data[i]
        param_name = param_labels[i]

        # Create histogram
        ax.hist(values, bins=min(20, len(values)), density=True, alpha=0.6,
                color='skyblue', edgecolor='black')

        # Add vertical lines for mean and median
        mean_val = np.mean(values)
        median_val = np.median(values)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='green', linestyle=':', linewidth=2,
                  label=f'Median: {median_val:.4f}')

        # Calculate and display statistics
        std_val = np.std(values)
        cv_val = std_val / mean_val if mean_val != 0 else 0

        ax.set_xlabel('Parameter Value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{param_name}\n(CV={cv_val:.2%}, σ={std_val:.4f})',
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Parameter Distributions - Adam Optimizer ({len(results)} runs)',
                fontsize=14, fontweight='bold', y=0.995)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Parameter distribution plot saved to: {save_path}")
    plt.close()


def plot_parameter_correlation(results, calibrate_indices, param_names, save_path):
    """
    Plot correlation matrix of calibrated parameters
    """
    # Extract parameter values
    n_params = len(calibrate_indices)
    n_runs = len(results)

    param_matrix = np.zeros((n_runs, n_params))
    for i, res in enumerate(results):
        param_matrix[i, :] = res['calibrate_params']

    # Calculate correlation matrix
    corr_matrix = np.corrcoef(param_matrix.T)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Set ticks and labels
    param_labels = [param_names[i] for i in calibrate_indices]
    ax.set_xticks(np.arange(n_params))
    ax.set_yticks(np.arange(n_params))
    ax.set_xticklabels(param_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(param_labels, fontsize=9)

    # Add correlation values as text
    for i in range(n_params):
        for j in range(n_params):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center",
                          color="black" if abs(corr_matrix[i, j]) < 0.5 else "white",
                          fontsize=8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation Coefficient', fontsize=10)

    ax.set_title(f'Parameter Correlation Matrix - Adam ({len(results)} runs)',
                fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Parameter correlation plot saved to: {save_path}")
    plt.close()


def plot_calibration_results(pred_df, target_var_names, available_vars, save_path, rank=1, error=0.0):
    """Plot comparison of predictions vs observations"""
    # Only plot available observed variables
    plot_vars = [v for v in available_vars if v in target_var_names]

    if not plot_vars:
        print(f"  Warning: No observed variables to plot")
        return

    n_vars = len(plot_vars)
    fig, axes = plt.subplots(n_vars, 1, figsize=(14, 4 * n_vars))

    if n_vars == 1:
        axes = [axes]

    for i, var_name in enumerate(plot_vars):
        ax = axes[i]

        dates = pd.to_datetime(pred_df['date'])

        # Plot predictions
        ax.plot(dates, pred_df[f'{var_name}_pred'],
               label='Calibrated Emulator (Adam)', linewidth=2, alpha=0.8)

        # Plot observations if available
        if f'{var_name}_obs' in pred_df.columns:
            obs_values = pred_df[f'{var_name}_obs'].values
            valid_mask = ~np.isnan(obs_values)
            ax.scatter(dates[valid_mask], obs_values[valid_mask],
                      label='Observations', alpha=0.6, s=20, color='red')

            # Calculate R2
            pred_values = pred_df[f'{var_name}_pred'].values[valid_mask]
            obs_clean = obs_values[valid_mask]
            ss_res = np.sum((obs_clean - pred_values) ** 2)
            ss_tot = np.sum((obs_clean - np.mean(obs_clean)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            # Calculate RMSE
            rmse = np.sqrt(np.mean((obs_clean - pred_values) ** 2))

            ax.text(0.02, 0.98, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}',
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', bbox=dict(boxstyle='round',
                   facecolor='white', alpha=0.8))

        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel(var_name, fontsize=11)
        ax.set_title(f'{var_name} - Calibrated vs Observed', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.suptitle(f'Calibration Result #{rank} - Adam Optimizer (NRMSE = {error:.4f})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_optimization_history(history, save_path):
    """Plot optimization history showing loss over iterations"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(history['iteration'], history['loss'], linewidth=2, color='blue', alpha=0.7)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss (Normalized MSE)', fontsize=12)
    ax.set_title('Adam Optimization History', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visualization

    # Add final loss value
    final_loss = history['loss'][-1]
    ax.text(0.98, 0.98, f'Final Loss: {final_loss:.6f}',
           transform=ax.transAxes, fontsize=11,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate NoahMP parameters using LSTM emulator with Adam optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Run 5 calibrations from random starting points (fast)
  python 06_calibration_applying_emulator.py \\
    --model_dir results_forward_comprehensive/AttentionLSTM_20251116_171820_dim-512_layer-2 \\
    --forcing data/raw/forcing/forcing_sample_1.nc \\
    --obs data/obs/Panama_BCI_obs_2015-07-30_2016-07-29.csv \\
    --bounds value_bounds.csv \\
    --num_calibration 5 \\
    --output calibration_results

  # Run 20 calibrations for better uncertainty quantification
  python 06_calibration_applying_emulator.py \\
    --model_dir results_forward_comprehensive/AttentionLSTM_20251116_171820_dim-512_layer-2 \\
    --forcing data/raw/forcing/forcing_sample_1.nc \\
    --obs data/obs/Panama_BCI_obs_2015-07-30_2016-07-29.csv \\
    --bounds value_bounds.csv \\
    --num_calibration 20 \\
    --max_iter 500

  # Calibrate specific parameters with custom learning rate
  python 06_calibration_applying_emulator.py \\
    --model_dir results_forward_comprehensive/AttentionLSTM_20251116_171820_dim-512_layer-2 \\
    --forcing data/raw/forcing/forcing_sample_1.nc \\
    --obs data/obs/Panama_BCI_obs_2015-07-30_2016-07-29.csv \\
    --bounds value_bounds.csv \\
    --calibrate_params VCMX25 HVT SATDK \\
    --num_calibration 10 \\
    --lr 0.005 \\
    --max_iter 300
        """
    )

    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained emulator model')
    parser.add_argument('--forcing', type=str, required=True,
                       help='NetCDF file with forcing data')
    parser.add_argument('--obs', type=str, required=True,
                       help='CSV file with observation data')
    parser.add_argument('--bounds', type=str, required=True,
                       help='CSV file with parameter bounds')
    parser.add_argument('--calibrate_params', type=str, nargs='+', default=None,
                       help='List of parameter names to calibrate (default: all)')
    parser.add_argument('--num_calibration', type=int, default=5,
                       help='Number of random starting points for calibration (default: 5)')
    parser.add_argument('--output', type=str, default='calibration_results',
                       help='Output directory')
    parser.add_argument('--max_iter', type=int, default=500,
                       help='Maximum optimization iterations per run (default: 500)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate for Adam optimizer (default: 0.01)')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience (default: 50)')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Run calibration
    run_calibration_adam(
        model_dir=args.model_dir,
        forcing_file=args.forcing,
        obs_file=args.obs,
        bounds_file=args.bounds,
        calibrate_params=args.calibrate_params,
        output_dir=args.output,
        num_calibration=args.num_calibration,
        max_iterations=args.max_iter,
        lr=args.lr,
        patience=args.patience,
        device=device
    )


if __name__ == '__main__':
    main()
