"""
Training script for FORWARD LSTM model
Predicts time series from parameters and forcing data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

from lstm_model_forward import LSTMForwardPredictor, BiLSTMForwardPredictor, AttentionLSTMForwardPredictor
import config_forward as config

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_data(data_file='data/processed_data_forward.pkl'):
    """Load preprocessed data"""
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data

def create_dataloaders(X_forcing, X_params, y, train_ratio=0.8, batch_size=16):
    """
    Create train and validation dataloaders

    Args:
        X_forcing: Forcing array (n_samples, n_timesteps, n_forcing_vars)
        X_params: Parameter array (n_samples, n_params)
        y: Target array (n_samples, n_timesteps, n_target_vars)
        train_ratio: Ratio of training data
        batch_size: Batch size

    Returns:
        train_loader, val_loader, train_indices, val_indices
    """
    n_samples = X_forcing.shape[0]
    n_train = int(n_samples * train_ratio)

    # Random shuffle
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Create datasets
    X_forcing_train = torch.FloatTensor(X_forcing[train_indices])
    X_params_train = torch.FloatTensor(X_params[train_indices])
    y_train = torch.FloatTensor(y[train_indices])

    X_forcing_val = torch.FloatTensor(X_forcing[val_indices])
    X_params_val = torch.FloatTensor(X_params[val_indices])
    y_val = torch.FloatTensor(y[val_indices])

    train_dataset = TensorDataset(X_forcing_train, X_params_train, y_train)
    val_dataset = TensorDataset(X_forcing_val, X_params_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_indices, val_indices

class WeightedMSELoss(nn.Module):
    """
    Weighted Mean Squared Error Loss for time series
    Allows different weights for each output variable
    """
    def __init__(self, weights=None):
        """
        Args:
            weights: Tensor of shape (n_target_vars,) with weight for each variable
                     If None, uses uniform weights (standard MSE)
        """
        super(WeightedMSELoss, self).__init__()
        self.register_buffer('weights', weights)

    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (batch_size, seq_len, n_target_vars)
            target: Ground truth (batch_size, seq_len, n_target_vars)

        Returns:
            Weighted MSE loss (scalar)
        """
        # Compute squared errors
        squared_errors = (pred - target) ** 2  # (batch_size, seq_len, n_target_vars)

        if self.weights is not None:
            # Apply weights to each variable
            # Broadcast weights to (1, 1, n_target_vars)
            weights_expanded = self.weights.view(1, 1, -1)
            weighted_errors = squared_errors * weights_expanded
            # Average over batch, time, and variables
            loss = weighted_errors.sum() / (pred.size(0) * pred.size(1) * self.weights.sum())
        else:
            # Standard MSE
            loss = squared_errors.mean()

        return loss

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    n_batches = 0

    for X_forcing_batch, X_params_batch, y_batch in loader:
        X_forcing_batch = X_forcing_batch.to(device)
        X_params_batch = X_params_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        y_pred = model(X_params_batch, X_forcing_batch)
        loss = criterion(y_pred, y_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches

def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_forcing_batch, X_params_batch, y_batch in loader:
            X_forcing_batch = X_forcing_batch.to(device)
            X_params_batch = X_params_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_params_batch, X_forcing_batch)
            loss = criterion(y_pred, y_batch)

            total_loss += loss.item()
            n_batches += 1

            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    avg_loss = total_loss / n_batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return avg_loss, all_preds, all_targets

def compute_metrics(y_true, y_pred, target_var_names):
    """
    Compute metrics for each target variable

    Args:
        y_true: True values (n_samples, n_timesteps, n_target_vars)
        y_pred: Predicted values (n_samples, n_timesteps, n_target_vars)
        target_var_names: List of target variable names

    Returns:
        Dictionary of metrics
    """
    n_target_vars = y_true.shape[2]
    metrics = {}

    for i, var_name in enumerate(target_var_names):
        # Extract time series for this variable
        y_true_var = y_true[:, :, i].flatten()
        y_pred_var = y_pred[:, :, i].flatten()

        # Compute R²
        ss_res = np.sum((y_true_var - y_pred_var) ** 2)
        ss_tot = np.sum((y_true_var - y_true_var.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # Compute RMSE
        rmse = np.sqrt(np.mean((y_true_var - y_pred_var) ** 2))

        # Compute MAE
        mae = np.mean(np.abs(y_true_var - y_pred_var))

        metrics[var_name] = {
            'R2': float(r2),
            'RMSE': float(rmse),
            'MAE': float(mae)
        }

    return metrics

def plot_training_history(history, save_path):
    """Plot training history"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(history['val_loss'], label='Val Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training History', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions(y_true, y_pred, target_var_names, save_path, n_samples=3):
    """
    Plot time series predictions vs true values

    Args:
        y_true: True values (n_samples, n_timesteps, n_target_vars)
        y_pred: Predicted values (n_samples, n_timesteps, n_target_vars)
        target_var_names: List of target variable names
        save_path: Path to save plot
        n_samples: Number of samples to plot
    """
    n_target_vars = y_true.shape[2]
    n_samples = min(n_samples, y_true.shape[0])

    fig, axes = plt.subplots(n_samples, n_target_vars, figsize=(15, 4 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        for j, var_name in enumerate(target_var_names):
            ax = axes[i, j]

            # Plot true and predicted time series
            ax.plot(y_true[i, :, j], label='True', linewidth=2, alpha=0.7)
            ax.plot(y_pred[i, :, j], label='Predicted', linewidth=2, alpha=0.7)

            ax.set_xlabel('Time (days)', fontsize=10)
            ax.set_ylabel(var_name, fontsize=10)
            ax.set_title(f'Sample {i+1} - {var_name}', fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_r2_comparison(metrics, save_path):
    """Plot R² comparison across variables"""
    var_names = list(metrics.keys())
    r2_values = [metrics[var]['R2'] for var in var_names]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['green' if r2 > 0.7 else 'orange' if r2 > 0.5 else 'red' for r2 in r2_values]
    bars = ax.bar(var_names, r2_values, color=colors, alpha=0.7, edgecolor='black')

    ax.set_xlabel('Target Variable', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Prediction Performance by Variable', fontsize=14)
    ax.axhline(y=0.7, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Good (R²>0.7)')
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Moderate (R²>0.5)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)

    # Add value labels on bars
    for bar, r2 in zip(bars, r2_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{r2:.3f}', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def train_model():
    """Main training function"""
    # Set seed for reproducibility
    set_seed(42)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    data = load_data(config.OUTPUT_FILE)

    X_forcing = data['X_forcing']
    X_params = data['X_params']
    y = data['y']
    target_var_names = data['target_var_names']
    param_names = data['param_names']

    print(f"Forcing shape: {X_forcing.shape}")
    print(f"Parameters shape: {X_params.shape}")
    print(f"Target shape: {y.shape}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, train_indices, val_indices = create_dataloaders(
        X_forcing, X_params, y,
        train_ratio=config.TRAINING_CONFIG['train_ratio'],
        batch_size=config.TRAINING_CONFIG['batch_size']
    )

    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples: {len(val_indices)}")

    # Create model
    print("\nCreating model...")
    model_type = config.MODEL_CONFIG['model_type']
    model_kwargs = {
        'n_params': data['n_params'],
        'n_forcing_vars': data['n_forcing_vars'],
        'n_target_vars': data['n_target_vars'],
        'hidden_dim': config.MODEL_CONFIG['hidden_dim'],
        'num_layers': config.MODEL_CONFIG['num_layers'],
        'param_embedding_dim': config.MODEL_CONFIG['param_embedding_dim'],
        'dropout': config.MODEL_CONFIG['dropout']
    }

    if model_type == 'LSTM':
        model = LSTMForwardPredictor(**model_kwargs)
    elif model_type == 'BiLSTM':
        model = BiLSTMForwardPredictor(**model_kwargs)
    elif model_type == 'AttentionLSTM':
        model = AttentionLSTMForwardPredictor(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model type: {model_type}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function with weights
    output_weights = config.get_output_weights(target_var_names)
    if output_weights is not None and not np.all(output_weights == 1.0):
        weights_tensor = torch.FloatTensor(output_weights).to(device)
        criterion = WeightedMSELoss(weights_tensor)
        print(f"\nUsing weighted loss:")
        for var_name, weight in zip(target_var_names, output_weights):
            print(f"  {var_name}: {weight}")
    else:
        criterion = WeightedMSELoss(None)
        print("\nUsing uniform loss (all variables equal weight)")

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.TRAINING_CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Training loop
    print("\nStarting training...")
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.TRAINING_CONFIG['patience']

    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"{model_type}_{timestamp}_dim-{config.MODEL_CONFIG['hidden_dim']}_layer-{config.MODEL_CONFIG['num_layers']}"
    results_dir = Path(config.RESULTS_DIR) / model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {results_dir}")

    for epoch in range(config.TRAINING_CONFIG['num_epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, _, _ = validate(model, val_loader, criterion, device)

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Learning rate scheduler
        scheduler.step(val_loss)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config.TRAINING_CONFIG['num_epochs']}: "
                  f"Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), results_dir / 'best_model.pth')
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Load best model
    print("\nLoading best model...")
    model.load_state_dict(torch.load(results_dir / 'best_model.pth'))

    # Final validation
    print("\nComputing final metrics...")
    val_loss, val_preds, val_targets = validate(model, val_loader, criterion, device)

    # Denormalize predictions and targets
    y_mean = data['y_mean']
    y_std = data['y_std']
    val_preds_denorm = val_preds * y_std + y_mean
    val_targets_denorm = val_targets * y_std + y_mean

    # Compute metrics
    metrics = compute_metrics(val_targets_denorm, val_preds_denorm, target_var_names)

    print("\nValidation Metrics:")
    for var_name, var_metrics in metrics.items():
        print(f"\n{var_name}:")
        for metric_name, value in var_metrics.items():
            print(f"  {metric_name}: {value:.4f}")

    # Save results
    print("\nSaving results...")

    # Save model config
    model_config = {
        'model_type': model_type,
        'model_config': config.MODEL_CONFIG,
        'training_config': config.TRAINING_CONFIG,
        'n_params': data['n_params'],
        'n_forcing_vars': data['n_forcing_vars'],
        'n_target_vars': data['n_target_vars'],
        'total_parameters': total_params,
        'trainable_parameters': trainable_params
    }

    with open(results_dir / 'config.json', 'w') as f:
        json.dump(model_config, f, indent=2)

    # Save metrics
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save training history
    with open(results_dir / 'training_history.pkl', 'wb') as f:
        pickle.dump(history, f)

    # Save validation predictions
    np.save(results_dir / 'val_predictions.npy', val_preds_denorm)
    np.save(results_dir / 'val_targets.npy', val_targets_denorm)
    np.save(results_dir / 'val_indices.npy', val_indices)

    # Plot results
    print("\nGenerating plots...")
    plot_training_history(history, results_dir / 'training_history.png')
    plot_predictions(val_targets_denorm, val_preds_denorm, target_var_names,
                     results_dir / 'predictions.png', n_samples=3)
    plot_r2_comparison(metrics, results_dir / 'r2_comparison.png')

    # Save summary
    with open(results_dir / 'results_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("FORWARD MODEL TRAINING RESULTS\n")
        f.write("="*80 + "\n\n")

        f.write(f"Model: {model_type}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Best validation loss: {best_val_loss:.6f}\n\n")

        f.write("Model Configuration:\n")
        for key, value in config.MODEL_CONFIG.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        f.write("Training Configuration:\n")
        for key, value in config.TRAINING_CONFIG.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")

        f.write("Validation Metrics:\n")
        for var_name, var_metrics in metrics.items():
            f.write(f"\n{var_name}:\n")
            for metric_name, value in var_metrics.items():
                f.write(f"  {metric_name}: {value:.4f}\n")

    print(f"\nTraining complete! Results saved to: {results_dir}")

if __name__ == '__main__':
    train_model()
