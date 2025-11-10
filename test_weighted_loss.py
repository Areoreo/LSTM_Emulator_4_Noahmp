"""
Quick test script to verify weighted loss implementation
"""

import torch
import torch.nn as nn
import numpy as np

# Import the weighted loss from train.py
import sys
sys.path.append('.')
from train import WeightedMSELoss

def test_weighted_loss():
    """Test that weighted loss works correctly"""

    print("="*60)
    print("Testing Weighted MSE Loss Implementation")
    print("="*60)

    # Create sample predictions and targets
    batch_size = 4
    n_params = 3

    pred = torch.tensor([
        [1.0, 2.0, 3.0],
        [1.5, 2.5, 3.5],
        [0.5, 1.5, 2.5],
        [1.2, 2.2, 3.2],
    ])

    target = torch.tensor([
        [1.0, 2.0, 3.0],  # Perfect prediction
        [1.0, 2.0, 3.0],  # Error: [0.5, 0.5, 0.5]
        [1.0, 2.0, 3.0],  # Error: [-0.5, -0.5, -0.5]
        [1.0, 2.0, 3.0],  # Error: [0.2, 0.2, 0.2]
    ])

    # Test 1: Uniform weights (should match standard MSE)
    print("\n### Test 1: Uniform Weights ###")
    standard_mse = nn.MSELoss()
    weighted_mse_uniform = WeightedMSELoss(weights=torch.ones(n_params))

    loss_std = standard_mse(pred, target)
    loss_weighted = weighted_mse_uniform(pred, target)

    print(f"Standard MSE: {loss_std.item():.6f}")
    print(f"Weighted MSE (uniform): {loss_weighted.item():.6f}")
    print(f"Difference: {abs(loss_std.item() - loss_weighted.item()):.8f}")

    if abs(loss_std.item() - loss_weighted.item()) < 1e-6:
        print("✓ PASS: Uniform weights match standard MSE")
    else:
        print("✗ FAIL: Uniform weights should match standard MSE")

    # Test 2: Custom weights - emphasize first parameter
    print("\n### Test 2: Custom Weights [3.0, 1.0, 1.0] ###")
    weights = torch.tensor([3.0, 1.0, 1.0])
    weighted_mse_custom = WeightedMSELoss(weights=weights)

    loss_custom = weighted_mse_custom(pred, target)
    print(f"Weighted MSE (custom): {loss_custom.item():.6f}")

    # Manually calculate expected loss
    squared_errors = (pred - target) ** 2
    print(f"Squared errors per param (mean across batch):")
    for i in range(n_params):
        print(f"  Param {i}: {squared_errors[:, i].mean().item():.6f}")

    weighted_errors = squared_errors * weights.unsqueeze(0)
    expected_loss = weighted_errors.sum() / (batch_size * weights.sum())
    print(f"Expected weighted loss: {expected_loss.item():.6f}")
    print(f"Actual weighted loss:   {loss_custom.item():.6f}")

    if abs(loss_custom.item() - expected_loss.item()) < 1e-6:
        print("✓ PASS: Custom weights calculated correctly")
    else:
        print("✗ FAIL: Custom weights calculation error")

    # Test 3: Extreme weights - focus only on parameter 0
    print("\n### Test 3: Extreme Weights [10.0, 0.1, 0.1] ###")
    weights_extreme = torch.tensor([10.0, 0.1, 0.1])
    weighted_mse_extreme = WeightedMSELoss(weights=weights_extreme)

    loss_extreme = weighted_mse_extreme(pred, target)
    print(f"Weighted MSE (extreme focus on param 0): {loss_extreme.item():.6f}")
    print(f"Standard MSE (all equal):                {loss_std.item():.6f}")
    print(f"Ratio: {loss_extreme.item() / loss_std.item():.2f}x")

    # Test 4: None weights (should behave like uniform)
    print("\n### Test 4: None Weights (should match standard MSE) ###")
    weighted_mse_none = WeightedMSELoss(weights=None)
    loss_none = weighted_mse_none(pred, target)

    print(f"Standard MSE: {loss_std.item():.6f}")
    print(f"Weighted MSE (None): {loss_none.item():.6f}")

    if abs(loss_std.item() - loss_none.item()) < 1e-6:
        print("✓ PASS: None weights match standard MSE")
    else:
        print("✗ FAIL: None weights should match standard MSE")

    # Test 5: Gradient flow test
    print("\n### Test 5: Gradient Flow Test ###")
    pred_grad = pred.clone().requires_grad_(True)
    loss = weighted_mse_custom(pred_grad, target)
    loss.backward()

    if pred_grad.grad is not None:
        print("✓ PASS: Gradients flow correctly")
        print(f"  Max gradient magnitude: {pred_grad.grad.abs().max().item():.6f}")
    else:
        print("✗ FAIL: No gradients computed")

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)

if __name__ == '__main__':
    test_weighted_loss()
