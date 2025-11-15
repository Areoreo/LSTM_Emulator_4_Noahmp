"""
Inference script for FORWARD LSTM model
Use trained model to predict time series from new parameters and forcing data
"""

import torch
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from lstm_model_forward import LSTMForwardPredictor, BiLSTMForwardPredictor, AttentionLSTMForwardPredictor
import config_forward as config


def load_model(model_dir, device='cpu'):
    """
    Load trained model from directory

    Args:
        model_dir: Directory containing model files
        device: Device to load model on

    Returns:
        model, data_dict, config_dict
    """
    model_dir = Path(model_dir)

    # Load config
    with open(model_dir / 'config.json', 'r') as f:
        config_dict = json.load(f)

    # Load data statistics (for normalization)
    data_file = config.OUTPUT_FILE
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

    # Load weights
    model.load_state_dict(torch.load(model_dir / 'best_model.pth', map_location=device))
    model = model.to(device)
    model.eval()

    return model, data_dict, config_dict


def predict(model, params, forcing, data_dict, device='cpu'):
    """
    Predict time series from parameters and forcing

    Args:
        model: Trained model
        params: Parameters array (n_params,) or (n_samples, n_params)
        forcing: Forcing array (n_timesteps, n_forcing_vars) or (n_samples, n_timesteps, n_forcing_vars)
        data_dict: Dictionary with normalization statistics
        device: Device to run prediction on

    Returns:
        Predicted time series (denormalized)
    """
    # Handle single sample case
    if params.ndim == 1:
        params = params.reshape(1, -1)
    if forcing.ndim == 2:
        forcing = forcing.reshape(1, forcing.shape[0], forcing.shape[1])

    # Normalize inputs
    params_normalized = (params - data_dict['X_params_mean']) / (data_dict['X_params_std'] + 1e-8)
    forcing_normalized = (forcing - data_dict['X_forcing_mean']) / (data_dict['X_forcing_std'] + 1e-8)

    # Convert to tensors
    params_tensor = torch.FloatTensor(params_normalized).to(device)
    forcing_tensor = torch.FloatTensor(forcing_normalized).to(device)

    # Predict
    with torch.no_grad():
        predictions_normalized = model(params_tensor, forcing_tensor)
        predictions_normalized = predictions_normalized.cpu().numpy()

    # Denormalize
    predictions = predictions_normalized * data_dict['y_std'] + data_dict['y_mean']

    return predictions


def predict_from_sample_idx(model, data_dict, sample_idx, device='cpu'):
    """
    Predict time series for a specific sample from the dataset

    Args:
        model: Trained model
        data_dict: Dictionary with data
        sample_idx: Sample index
        device: Device to run prediction on

    Returns:
        predictions, true_values, forcing, params
    """
    # Get sample data
    forcing = data_dict['X_forcing'][sample_idx]  # (n_timesteps, n_forcing_vars)
    params = data_dict['X_params'][sample_idx]    # (n_params,)
    true_values = data_dict['y'][sample_idx]      # (n_timesteps, n_target_vars)

    # Predict (already normalized in data_dict)
    params_tensor = torch.FloatTensor(params).unsqueeze(0).to(device)
    forcing_tensor = torch.FloatTensor(forcing).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions_normalized = model(params_tensor, forcing_tensor)
        predictions_normalized = predictions_normalized.cpu().numpy()[0]

    # Denormalize
    predictions = predictions_normalized * data_dict['y_std'] + data_dict['y_mean']
    true_values_denorm = true_values * data_dict['y_std'] + data_dict['y_mean']
    forcing_denorm = forcing * data_dict['X_forcing_std'] + data_dict['X_forcing_mean']
    params_denorm = params * data_dict['X_params_std'] + data_dict['X_params_mean']

    return predictions, true_values_denorm, forcing_denorm, params_denorm


def plot_prediction(predictions, true_values, target_var_names, forcing=None,
                    forcing_var_names=None, save_path=None):
    """
    Plot predicted vs true time series

    Args:
        predictions: Predicted values (n_timesteps, n_target_vars) or (1, n_timesteps, n_target_vars)
        true_values: True values (n_timesteps, n_target_vars) or (1, n_timesteps, n_target_vars)
        target_var_names: List of target variable names
        forcing: Optional forcing data to plot (n_timesteps, n_forcing_vars)
        forcing_var_names: List of forcing variable names
        save_path: Path to save plot
    """
    # Handle batch dimension
    if predictions.ndim == 3:
        predictions = predictions[0]
    if true_values.ndim == 3:
        true_values = true_values[0]
    if forcing is not None and forcing.ndim == 3:
        forcing = forcing[0]

    n_target_vars = len(target_var_names)
    n_plots = n_target_vars

    # Add forcing plots if provided
    if forcing is not None and forcing_var_names is not None:
        n_forcing_vars = min(len(forcing_var_names), 4)  # Limit to 4 forcing vars
        n_plots += n_forcing_vars
    else:
        n_forcing_vars = 0

    # Create subplots
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots))

    if n_plots == 1:
        axes = [axes]

    # Plot target variables
    for i, var_name in enumerate(target_var_names):
        ax = axes[i]

        ax.plot(true_values[:, i], label='True', linewidth=2, alpha=0.7)
        ax.plot(predictions[:, i], label='Predicted', linewidth=2, alpha=0.7, linestyle='--')

        # Compute R²
        ss_res = np.sum((true_values[:, i] - predictions[:, i]) ** 2)
        ss_tot = np.sum((true_values[:, i] - true_values[:, i].mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        ax.set_xlabel('Time (days)', fontsize=10)
        ax.set_ylabel(var_name, fontsize=10)
        ax.set_title(f'{var_name} (R² = {r2:.3f})', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Plot forcing variables
    if forcing is not None and forcing_var_names is not None:
        for i in range(n_forcing_vars):
            ax = axes[n_target_vars + i]

            ax.plot(forcing[:, i], linewidth=2, alpha=0.7, color='gray')

            ax.set_xlabel('Time (days)', fontsize=10)
            ax.set_ylabel(forcing_var_names[i], fontsize=10)
            ax.set_title(f'Forcing: {forcing_var_names[i]}', fontsize=12)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='FORWARD LSTM Inference')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained model')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Sample index to predict (default: 0)')
    parser.add_argument('--output_dir', type=str, default='inference_forward',
                        help='Directory to save predictions')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.model_dir}...")
    model, data_dict, config_dict = load_model(args.model_dir, device)

    print(f"Model type: {config_dict['model_type']}")
    print(f"Parameters: {config_dict['total_parameters']:,}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Predict for sample
    print(f"\nPredicting for sample {args.sample_idx}...")
    predictions, true_values, forcing, params = predict_from_sample_idx(
        model, data_dict, args.sample_idx, device
    )

    # Compute metrics
    target_var_names = data_dict['target_var_names']
    print("\nPrediction Metrics:")
    for i, var_name in enumerate(target_var_names):
        ss_res = np.sum((true_values[:, i] - predictions[:, i]) ** 2)
        ss_tot = np.sum((true_values[:, i] - true_values[:, i].mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        rmse = np.sqrt(np.mean((true_values[:, i] - predictions[:, i]) ** 2))

        print(f"  {var_name}: R² = {r2:.4f}, RMSE = {rmse:.4f}")

    # Plot
    print("\nGenerating plot...")
    forcing_var_names = data_dict['forcing_var_names']
    plot_prediction(
        predictions, true_values, target_var_names,
        forcing, forcing_var_names,
        save_path=output_dir / f'prediction_sample_{args.sample_idx}.png'
    )

    # Save predictions
    output_file = output_dir / f'predictions_sample_{args.sample_idx}.npz'
    np.savez(
        output_file,
        predictions=predictions,
        true_values=true_values,
        forcing=forcing,
        params=params,
        target_var_names=target_var_names,
        forcing_var_names=forcing_var_names,
        param_names=data_dict['param_names']
    )
    print(f"Predictions saved to {output_file}")

    # Save as CSV for easy viewing
    csv_file = output_dir / f'predictions_sample_{args.sample_idx}.csv'
    df_data = {}
    for i, var_name in enumerate(target_var_names):
        df_data[f'{var_name}_true'] = true_values[:, i]
        df_data[f'{var_name}_pred'] = predictions[:, i]

    df = pd.DataFrame(df_data)
    df.to_csv(csv_file, index=False)
    print(f"Predictions saved to {csv_file}")

    print("\nInference complete!")


if __name__ == '__main__':
    main()
