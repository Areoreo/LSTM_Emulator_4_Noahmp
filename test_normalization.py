"""
Test script to verify normalization methods work correctly
"""
import numpy as np
import sys
import importlib.util

# Import module with numeric prefix
spec = importlib.util.spec_from_file_location(
    "preprocessing",
    "01_data_preprocessing_forward_comprehensive.py"
)
preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocessing)

normalize_data = preprocessing.normalize_data
denormalize_data = preprocessing.denormalize_data

def test_normalization():
    """Test both z-score and min-max normalization"""

    # Create test data
    np.random.seed(42)
    test_data = np.random.randn(100, 10, 5) * 10 + 50  # (samples, timesteps, features)

    print("="*80)
    print("TESTING NORMALIZATION METHODS")
    print("="*80)
    print(f"\nOriginal data shape: {test_data.shape}")
    print(f"Original data range: [{test_data.min():.2f}, {test_data.max():.2f}]")
    print(f"Original data mean: {test_data.mean():.2f}")
    print(f"Original data std: {test_data.std():.2f}")

    # Test z-score normalization
    print("\n" + "="*80)
    print("Z-SCORE NORMALIZATION")
    print("="*80)

    normalized_z, stats_z = normalize_data(test_data, method='z-score')
    print(f"\nNormalized range: [{normalized_z.min():.2f}, {normalized_z.max():.2f}]")
    print(f"Normalized mean: {normalized_z.mean():.6f} (should be ~0)")
    print(f"Normalized std: {normalized_z.std():.6f} (should be ~1)")

    # Test denormalization
    denormalized_z = denormalize_data(normalized_z, stats_z)
    print(f"\nDenormalized range: [{denormalized_z.min():.2f}, {denormalized_z.max():.2f}]")
    print(f"Denormalized mean: {denormalized_z.mean():.2f}")
    print(f"Max reconstruction error: {np.abs(denormalized_z - test_data).max():.2e}")

    assert np.allclose(denormalized_z, test_data, rtol=1e-5), "Z-score denormalization failed!"
    print("✓ Z-score normalization/denormalization PASSED")

    # Test min-max normalization
    print("\n" + "="*80)
    print("MIN-MAX NORMALIZATION")
    print("="*80)

    normalized_mm, stats_mm = normalize_data(test_data, method='min-max')
    print(f"\nNormalized range: [{normalized_mm.min():.2f}, {normalized_mm.max():.2f}] (should be [0, 1])")
    print(f"Normalized mean: {normalized_mm.mean():.6f}")

    # Test denormalization
    denormalized_mm = denormalize_data(normalized_mm, stats_mm)
    print(f"\nDenormalized range: [{denormalized_mm.min():.2f}, {denormalized_mm.max():.2f}]")
    print(f"Denormalized mean: {denormalized_mm.mean():.2f}")
    print(f"Max reconstruction error: {np.abs(denormalized_mm - test_data).max():.2e}")

    assert np.allclose(denormalized_mm, test_data, rtol=1e-5), "Min-max denormalization failed!"
    print("✓ Min-max normalization/denormalization PASSED")

    # Test with pre-computed statistics (inference mode)
    print("\n" + "="*80)
    print("TESTING INFERENCE MODE (using pre-computed stats)")
    print("="*80)

    new_data = np.random.randn(50, 10, 5) * 10 + 50  # New unseen data

    # Apply normalization using stats from training data
    normalized_new_z, _ = normalize_data(new_data, method='z-score', fit_stats=stats_z)
    normalized_new_mm, _ = normalize_data(new_data, method='min-max', fit_stats=stats_mm)

    print(f"\nNew data normalized with z-score stats:")
    print(f"  Range: [{normalized_new_z.min():.2f}, {normalized_new_z.max():.2f}]")
    print(f"  Mean: {normalized_new_z.mean():.6f}")

    print(f"\nNew data normalized with min-max stats:")
    print(f"  Range: [{normalized_new_mm.min():.2f}, {normalized_new_mm.max():.2f}]")
    print(f"  Note: May be outside [0,1] if new data has values outside training range")

    # Verify denormalization
    denorm_new_z = denormalize_data(normalized_new_z, stats_z)
    denorm_new_mm = denormalize_data(normalized_new_mm, stats_mm)

    assert np.allclose(denorm_new_z, new_data, rtol=1e-5), "Inference z-score failed!"
    assert np.allclose(denorm_new_mm, new_data, rtol=1e-5), "Inference min-max failed!"
    print("✓ Inference mode PASSED")

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)

if __name__ == '__main__':
    test_normalization()
