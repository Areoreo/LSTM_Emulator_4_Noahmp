"""
Training script for LSTM parameter prediction model
Uses configuration from config.py
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

from lstm_model import LSTMParameterPredictor, BiLSTMParameterPredictor
import config

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_data(data_file='data/processed_data.pkl'):
    """Load preprocessed data"""
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data

def create_dataloaders(X, y, train_ratio=0.8, batch_size=16):
    """
    Create train and validation dataloaders

    Args:
        X: Input array (n_samples, n_variables, n_timesteps)
        y: Output array (n_samples, n_params)
        train_ratio: Ratio of training data
        batch_size: Batch size

    Returns:
        train_loader, val_loader, train_indices, val_indices
    """
    n_samples = X.shape[0]
    n_train = int(n_samples * train_ratio)

    # Random shuffle
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Transpose X to (n_samples, n_timesteps, n_variables) for LSTM
    X_transposed = np.transpose(X, (0, 2, 1))

    # Create datasets
    X_train = torch.FloatTensor(X_transposed[train_indices])
    y_train = torch.FloatTensor(y[train_indices])
    X_val = torch.FloatTensor(X_transposed[val_indices])
    y_val = torch.FloatTensor(y[val_indices])

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_indices, val_indices

class WeightedMSELoss(nn.Module):
    """
    Weighted Mean Squared Error Loss
    Allows different weights for each output parameter
    """
    def __init__(self, weights=None):
        """
        Args:
            weights: Tensor of shape (n_params,) with weight for each parameter
                     If None, uses uniform weights (standard MSE)
        """
        super(WeightedMSELoss, self).__init__()
        self.register_buffer('weights', weights)

    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (batch_size, n_params)
            target: Ground truth (batch_size, n_params)

        Returns:
            Weighted MSE loss (scalar)
        """
        # Compute squared errors per parameter
        squared_errors = (pred - target) ** 2  # (batch_size, n_params)

        if self.weights is not None:
            # Apply weights to each parameter
            weighted_errors = squared_errors * self.weights.unsqueeze(0)  # Broadcast weights
            # Average over batch and parameters
            loss = weighted_errors.sum() / (pred.size(0) * self.weights.sum())
        else:
            # Standard MSE
            loss = squared_errors.mean()

        return loss

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        y_pred = model(X_batch)
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
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            total_loss += loss.item()
            n_batches += 1

            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    avg_loss = total_loss / n_batches
    predictions = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    return avg_loss, predictions, targets

def plot_training_history(train_losses, val_losses, save_path):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training History', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_predictions(predictions, targets, param_names, save_path):
    """Plot predicted vs actual parameters"""
    n_params = predictions.shape[1]
    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for i in range(n_params):
        ax = axes[i]
        ax.scatter(targets[:, i], predictions[:, i], alpha=0.5, s=20)

        # Add 1:1 line
        min_val = min(targets[:, i].min(), predictions[:, i].min())
        max_val = max(targets[:, i].max(), predictions[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

        # Calculate R²
        ss_res = np.sum((targets[:, i] - predictions[:, i]) ** 2)
        ss_tot = np.sum((targets[:, i] - targets[:, i].mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        ax.set_xlabel(f'Actual {param_names[i]}', fontsize=10)
        ax.set_ylabel(f'Predicted {param_names[i]}', fontsize=10)
        ax.set_title(f'{param_names[i]} (R²={r2:.3f})', fontsize=11)
        ax.grid(True, alpha=0.3)

    # Remove extra subplots
    for i in range(n_params, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def train_model(
    model_type=None,
    hidden_dim=None,
    num_layers=None,
    dropout=None,
    learning_rate=None,
    batch_size=None,
    num_epochs=None,
    patience=None,
    train_ratio=None,
    data_file=None,
    results_dir=None
):
    """
    Main training function
    Uses configuration from config.py if parameters not specified

    Args:
        model_type: 'LSTM' or 'BiLSTM' (default: from config)
        hidden_dim: Hidden dimension size (default: from config)
        num_layers: Number of LSTM layers (default: from config)
        dropout: Dropout rate (default: from config)
        learning_rate: Learning rate (default: from config)
        batch_size: Batch size (default: from config)
        num_epochs: Maximum number of epochs (default: from config)
        patience: Early stopping patience (default: from config)
        train_ratio: Train/validation split ratio (default: from config)
        data_file: Path to preprocessed data (default: from config)
        results_dir: Directory to save results (default: from config)
    """
    # Use config defaults if not specified
    if model_type is None:
        model_type = config.MODEL_CONFIG['model_type']
    if hidden_dim is None:
        hidden_dim = config.MODEL_CONFIG['hidden_dim']
    if num_layers is None:
        num_layers = config.MODEL_CONFIG['num_layers']
    if dropout is None:
        dropout = config.MODEL_CONFIG['dropout']
    if learning_rate is None:
        learning_rate = config.TRAINING_CONFIG['learning_rate']
    if batch_size is None:
        batch_size = config.TRAINING_CONFIG['batch_size']
    if num_epochs is None:
        num_epochs = config.TRAINING_CONFIG['num_epochs']
    if patience is None:
        patience = config.TRAINING_CONFIG['patience']
    if train_ratio is None:
        train_ratio = config.TRAINING_CONFIG['train_ratio']
    if data_file is None:
        data_file = config.OUTPUT_FILE
    if results_dir is None:
        results_dir = config.RESULTS_DIR

    # Setup
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(results_dir) / f'{model_type}_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    data = load_data(data_file)
    X = data['X']
    y = data['y']
    param_names = data['param_names']
    variable_names = data.get('variable_names', [])

    print(f"\nData shapes - X: {X.shape}, y: {y.shape}")
    print(f"Input variables ({len(variable_names)}): {', '.join(variable_names)}")
    print(f"Output parameters ({len(param_names)}): {len(param_names)} parameters")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, train_idx, val_idx = create_dataloaders(
        X, y, train_ratio=train_ratio, batch_size=batch_size
    )

    # Create model
    input_dim = X.shape[1]  # Number of variables
    output_dim = y.shape[1]  # Number of parameters

    print(f"\nBuilding {model_type} model...")
    if model_type == 'LSTM':
        model = LSTMParameterPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout
        )
    elif model_type == 'BiLSTM':
        model = BiLSTMParameterPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    # Print model info
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Get parameter loss weights from config
    param_weights = config.get_parameter_weights(param_names)
    print(f"\n=== Parameter Loss Weights ===")
    print(f"Strategy: {config.WEIGHT_STRATEGY}")

    # Show weights for each parameter
    for i, (name, weight) in enumerate(zip(param_names, param_weights)):
        if weight != 1.0:
            print(f"  {name}: {weight:.2f}")
        else:
            print(f"  {name}: {weight:.2f} (standard)")

    # Convert weights to tensor
    weight_tensor = torch.FloatTensor(param_weights).to(device)

    # Loss and optimizer
    if np.allclose(param_weights, 1.0):
        # All weights are 1.0, use standard MSE for efficiency
        criterion = nn.MSELoss()
        print("\nUsing standard MSE loss (all weights = 1.0)")
    else:
        # Use weighted MSE loss
        criterion = WeightedMSELoss(weights=weight_tensor)
        print(f"\nUsing weighted MSE loss (weight ratio: {param_weights.max():.2f}:{param_weights.min():.2f})")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_targets = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Learning rate scheduler
        scheduler.step(val_loss)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, run_dir / 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(run_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final validation
    val_loss, val_preds, val_targets = validate(model, val_loader, criterion, device)

    # Save training history
    print("Saving training history...")
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_epoch': checkpoint['epoch'],
        'config': {
            'model_type': model_type,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'weight_strategy': config.WEIGHT_STRATEGY,
        },
        'parameter_weights': {name: float(weight) for name, weight in zip(param_names, param_weights)},
    }

    with open(run_dir / 'training_history.pkl', 'wb') as f:
        pickle.dump(history, f)

    with open(run_dir / 'config.json', 'w') as f:
        json.dump(history['config'], f, indent=2)

    # Plot training history
    print("Plotting training history...")
    plot_training_history(train_losses, val_losses, run_dir / 'training_history.png')

    # Plot predictions
    print("Plotting predictions...")
    plot_predictions(val_preds, val_targets, param_names, run_dir / 'predictions.png')

    # Save predictions
    np.save(run_dir / 'val_predictions.npy', val_preds)
    np.save(run_dir / 'val_targets.npy', val_targets)
    np.save(run_dir / 'val_indices.npy', val_idx)

    # Calculate metrics
    mse = np.mean((val_preds - val_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(val_preds - val_targets))

    # Per-parameter R²
    r2_scores = []
    for i in range(val_targets.shape[1]):
        ss_res = np.sum((val_targets[:, i] - val_preds[:, i]) ** 2)
        ss_tot = np.sum((val_targets[:, i] - val_targets[:, i].mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        r2_scores.append(r2)

    # Save metrics
    metrics = {
        'val_mse': float(mse),
        'val_rmse': float(rmse),
        'val_mae': float(mae),
        'mean_r2': float(np.mean(r2_scores)),
        'r2_per_param': {name: float(r2) for name, r2 in zip(param_names, r2_scores)}
    }

    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Training Complete ===")
    print(f"Results saved to: {run_dir}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Validation RMSE: {rmse:.6f}")
    print(f"Validation MAE: {mae:.6f}")
    print(f"Mean R²: {np.mean(r2_scores):.4f}")

    return model, history, metrics

if __name__ == '__main__':
    # Train model using configuration from config.py
    print("Training model with configuration from config.py...")
    print("="*80)
    config.print_config()
    print("="*80)

    model, history, metrics = train_model()  # Uses defaults from config.py
