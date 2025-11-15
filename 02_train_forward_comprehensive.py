"""
Training script for COMPREHENSIVE FORWARD LSTM model
Predicts ALL energy/water cycle variables for full conservation checking
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
import config_forward_comprehensive as config

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_data(data_file):
    """Load preprocessed data"""
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data

def create_dataloaders(X_forcing, X_params, y, train_ratio=0.8, batch_size=16):
    """Create train and validation dataloaders"""
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
    """Weighted Mean Squared Error Loss for different variable importance"""
    def __init__(self, weights=None):
        super(WeightedMSELoss, self).__init__()
        self.register_buffer('weights', weights)

    def forward(self, pred, target):
        squared_errors = (pred - target) ** 2
        if self.weights is not None:
            weighted_errors = squared_errors * self.weights.view(1, 1, -1)
            return weighted_errors.mean()
        else:
            return squared_errors.mean()

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    n_batches = 0

    for forcing, params, target in train_loader:
        forcing = forcing.to(device)
        params = params.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(params, forcing)
        loss = criterion(output, target)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for forcing, params, target in val_loader:
            forcing = forcing.to(device)
            params = params.to(device)
            target = target.to(device)

            output = model(params, forcing)
            loss = criterion(output, target)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches

def plot_training_history(train_losses, val_losses, save_path):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History - Comprehensive Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main training function"""
    set_seed(42)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading data from {config.OUTPUT_FILE}...")
    data = load_data(config.OUTPUT_FILE)

    print(f"Data loaded:")
    print(f"  Forcing shape: {data['X_forcing'].shape}")
    print(f"  Params shape: {data['X_params'].shape}")
    print(f"  Target shape: {data['y'].shape}")
    print(f"  Number of target variables: {data['n_target_vars']}")

    # Create dataloaders
    print(f"\nCreating dataloaders...")
    train_loader, val_loader, train_indices, val_indices = create_dataloaders(
        data['X_forcing'],
        data['X_params'],
        data['y'],
        train_ratio=config.TRAINING_CONFIG['train_ratio'],
        batch_size=config.TRAINING_CONFIG['batch_size']
    )

    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

    # Create model
    print(f"\nCreating {config.MODEL_CONFIG['model_type']} model...")
    model_kwargs = {
        'n_params': data['n_params'],
        'n_forcing_vars': data['n_forcing_vars'],
        'n_target_vars': data['n_target_vars'],
        'hidden_dim': config.MODEL_CONFIG['hidden_dim'],
        'num_layers': config.MODEL_CONFIG['num_layers'],
        'param_embedding_dim': config.MODEL_CONFIG['param_embedding_dim'],
        'dropout': config.MODEL_CONFIG['dropout']
    }

    if config.MODEL_CONFIG['model_type'] == 'LSTM':
        model = LSTMForwardPredictor(**model_kwargs)
    elif config.MODEL_CONFIG['model_type'] == 'BiLSTM':
        model = BiLSTMForwardPredictor(**model_kwargs)
    elif config.MODEL_CONFIG['model_type'] == 'AttentionLSTM':
        model = AttentionLSTMForwardPredictor(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {config.MODEL_CONFIG['model_type']}")

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Create loss function with weights
    weights = config.get_output_weights(data['target_var_names'])
    weights_tensor = torch.FloatTensor(weights).to(device)
    criterion = WeightedMSELoss(weights=weights_tensor)

    print(f"\nLoss weights summary:")
    print(f"  Min weight: {weights.min():.2f}")
    print(f"  Max weight: {weights.max():.2f}")
    print(f"  Mean weight: {weights.mean():.2f}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.TRAINING_CONFIG['learning_rate'])

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, verbose=True
    )

    # Training loop
    print(f"\nStarting training...")
    print(f"  Epochs: {config.TRAINING_CONFIG['num_epochs']}")
    print(f"  Patience: {config.TRAINING_CONFIG['patience']}")
    print(f"  Learning rate: {config.TRAINING_CONFIG['learning_rate']}")

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    # Create results directory
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.TRAINING_CONFIG['num_epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config.TRAINING_CONFIG['num_epochs']}] "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            torch.save(model.state_dict(), results_dir / 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= config.TRAINING_CONFIG['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Plot training history
    plot_training_history(train_losses, val_losses, results_dir / 'training_history.png')
    print(f"Training history plot saved to {results_dir / 'training_history.png'}")

    # Save configuration
    config_dict = {
        'model_type': config.MODEL_CONFIG['model_type'],
        'model_config': config.MODEL_CONFIG,
        'training_config': config.TRAINING_CONFIG,
        'n_params': data['n_params'],
        'n_forcing_vars': data['n_forcing_vars'],
        'n_target_vars': data['n_target_vars'],
        'forcing_var_names': data['forcing_var_names'],
        'target_var_names': data['target_var_names'],
        'target_categories': data['target_categories'],
        'param_names': data['param_names'],
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'best_val_loss': best_val_loss,
        'final_epoch': epoch + 1,
        'train_samples': len(train_indices),
        'val_samples': len(val_indices),
        'train_ratio': config.TRAINING_CONFIG['train_ratio'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(results_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nConfiguration saved to {results_dir / 'config.json'}")
    print(f"Model saved to {results_dir / 'best_model.pth'}")

    # Save loss history
    np.savez(results_dir / 'loss_history.npz',
             train_losses=train_losses,
             val_losses=val_losses,
             best_val_loss=best_val_loss,
             final_epoch=epoch + 1)

    print(f"\nAll results saved to {results_dir}")

if __name__ == '__main__':
    main()
