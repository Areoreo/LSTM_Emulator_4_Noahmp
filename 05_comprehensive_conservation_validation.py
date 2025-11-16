"""
Comprehensive Conservation Validation for Full Model

This script validates the comprehensive LSTM model that predicts ALL energy/water
cycle variables. It performs:

1. Full energy conservation checking with all predicted fluxes
2. Full water conservation checking with all water cycle variables
3. Comparison with baseline simulation conservation
4. Detailed diagnostic plots and reports

Run this AFTER training the comprehensive model.
"""

import torch
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm

from lstm_model_forward import LSTMForwardPredictor, BiLSTMForwardPredictor, AttentionLSTMForwardPredictor
import config_forward_comprehensive as config
from conservation_check_comprehensive import ComprehensiveConservationChecker, dict_from_arrays


def load_model(model_dir, device='cpu'):
    """Load trained comprehensive model"""
    model_dir = Path(model_dir)

    # Load config
    with open(model_dir / 'config.json', 'r') as f:
        config_dict = json.load(f)

    # Load data statistics
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

    model.load_state_dict(torch.load(model_dir / 'best_model.pth', map_location=device))
    model = model.to(device)
    model.eval()

    return model, data_dict, config_dict


def predict_from_sample_idx(model, data_dict, sample_idx, device='cpu'):
    """Predict all variables for a sample"""
    # Get sample data (already normalized)
    forcing = data_dict['X_forcing'][sample_idx]
    params = data_dict['X_params'][sample_idx]
    true_values = data_dict['y'][sample_idx]

    # Predict
    params_tensor = torch.FloatTensor(params).unsqueeze(0).to(device)
    forcing_tensor = torch.FloatTensor(forcing).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions_normalized = model(params_tensor, forcing_tensor)
        predictions_normalized = predictions_normalized.cpu().numpy()[0]

    # Denormalize
    predictions = predictions_normalized * data_dict['y_std'] + data_dict['y_mean']
    true_values_denorm = true_values * data_dict['y_std'] + data_dict['y_mean']
    forcing_denorm = forcing * data_dict['X_forcing_std'] + data_dict['X_forcing_mean']

    return predictions, true_values_denorm, forcing_denorm


def validate_sample_conservation(model, data_dict, sample_indices, device='cpu', verbose=True):
    """
    Validate conservation for multiple samples

    Args:
        model: Trained model
        data_dict: Data dictionary
        sample_indices: List of sample indices to validate
        device: Device to run on
        verbose: Print progress

    Returns:
        dict with aggregated conservation statistics
    """
    checker = ComprehensiveConservationChecker()

    all_energy_stats = []
    all_water_stats = []

    print("\nValidating conservation across samples...")
    for idx in tqdm(sample_indices, desc="Processing samples"):
        try:
            # Get predictions
            predictions, true_values, forcing_denorm = predict_from_sample_idx(
                model, data_dict, idx, device
            )

            # Convert to dictionaries
            pred_dict = dict_from_arrays(predictions, data_dict['target_var_names'])
            forcing_dict = dict_from_arrays(forcing_denorm, data_dict['forcing_var_names'])

            # Check energy conservation
            energy_stats = checker.check_full_energy_conservation(
                pred_dict, forcing=forcing_dict, verbose=False
            )
            all_energy_stats.append(energy_stats)

            # Check water conservation
            water_stats = checker.check_full_water_conservation(
                pred_dict, forcing=forcing_dict, verbose=False
            )
            all_water_stats.append(water_stats)

        except Exception as e:
            if verbose:
                print(f"\nWarning: Could not process sample {idx}: {e}")
            continue

    # Aggregate statistics
    results = {
        'energy': {
            'mean_residual': np.mean([s['mean_residual'] for s in all_energy_stats]),
            'std_residual': np.mean([s['std_residual'] for s in all_energy_stats]),
            'rmse': np.mean([s['rmse'] for s in all_energy_stats]),
            'relative_error_pct': np.mean([s['relative_error_pct'] for s in all_energy_stats]),
            'all_residuals': np.concatenate([s['residual_timeseries'] for s in all_energy_stats]),
        },
        'water': {
            'mean_residual': np.mean([s['mean_residual'] for s in all_water_stats]),
            'std_residual': np.mean([s['std_residual'] for s in all_water_stats]),
            'rmse': np.mean([s['rmse'] for s in all_water_stats]),
            'cumulative_precip': np.sum([s['cumulative_precip'] for s in all_water_stats]),
            'cumulative_et': np.sum([s['cumulative_et'] for s in all_water_stats]),
            'cumulative_runoff': np.sum([s['cumulative_runoff'] for s in all_water_stats]),
            'all_residuals': np.concatenate([s['residual_timeseries'] for s in all_water_stats]),
        },
        'n_samples': len(all_energy_stats),
    }

    return results, all_energy_stats, all_water_stats


def plot_aggregated_conservation(results, save_dir):
    """Plot aggregated conservation results across all samples"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Energy residual histogram
    ax = axes[0, 0]
    ax.hist(results['energy']['all_residuals'], bins=100, alpha=0.7,
            edgecolor='black', color='coral')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Perfect Conservation')
    ax.set_xlabel('Energy Residual (W/m²)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Energy Conservation (n={results["n_samples"]} samples)', fontsize=12, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Water residual histogram
    ax = axes[0, 1]
    ax.hist(results['water']['all_residuals'], bins=100, alpha=0.7,
            edgecolor='black', color='skyblue')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Perfect Conservation')
    ax.set_xlabel('Water Residual (mm/day)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Water Conservation (n={results["n_samples"]} samples)', fontsize=12, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Energy statistics table
    ax = axes[1, 0]
    ax.axis('off')
    energy_text = f"""
ENERGY CONSERVATION SUMMARY
{'='*40}

Mean Residual:     {results['energy']['mean_residual']:>8.2f} W/m²
Std Residual:      {results['energy']['std_residual']:>8.2f} W/m²
RMSE:              {results['energy']['rmse']:>8.2f} W/m²
Relative Error:    {results['energy']['relative_error_pct']:>8.2f} %

Assessment:
"""
    if results['energy']['rmse'] < 10:
        energy_text += "  ✓ EXCELLENT"
    elif results['energy']['rmse'] < 30:
        energy_text += "  ✓ GOOD"
    elif results['energy']['rmse'] < 50:
        energy_text += "  ⚠ ACCEPTABLE"
    else:
        energy_text += "  ✗ POOR"

    ax.text(0.1, 0.5, energy_text, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('Energy Conservation Statistics', fontsize=12, weight='bold', pad=20)

    # Water statistics table
    ax = axes[1, 1]
    ax.axis('off')
    water_text = f"""
WATER CONSERVATION SUMMARY
{'='*40}

Mean Residual:     {results['water']['mean_residual']:>8.4f} mm/day
Std Residual:      {results['water']['std_residual']:>8.4f} mm/day
RMSE:              {results['water']['rmse']:>8.4f} mm/day

Cumulative (mm):
  Precipitation:   {results['water']['cumulative_precip']:>8.2f}
  ET:              {results['water']['cumulative_et']:>8.2f}
  Runoff:          {results['water']['cumulative_runoff']:>8.2f}

Assessment:
"""
    if results['water']['rmse'] < 0.1:
        water_text += "  ✓ EXCELLENT"
    elif results['water']['rmse'] < 0.5:
        water_text += "  ✓ GOOD"
    elif results['water']['rmse'] < 1.0:
        water_text += "  ⚠ ACCEPTABLE"
    else:
        water_text += "  ✗ POOR"

    ax.text(0.1, 0.5, water_text, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    ax.set_title('Water Conservation Statistics', fontsize=12, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_dir / 'comprehensive_conservation_summary.png', dpi=300, bbox_inches='tight')
    print(f"\nAggregated conservation plot saved to: {save_dir / 'comprehensive_conservation_summary.png'}")

    return fig


def main():
    """Main comprehensive validation function"""
    parser = argparse.ArgumentParser(
        description='Comprehensive Conservation Validation for Full LSTM Model'
    )
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing trained comprehensive model')
    parser.add_argument('--n_samples', type=int, default=20,
                       help='Number of test samples to validate')
    parser.add_argument('--sample_detail', type=int, default=None,
                       help='Sample index for detailed conservation plots (optional)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save results')

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir is None:
        output_dir = Path(args.model_dir) / 'comprehensive_conservation'
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("COMPREHENSIVE CONSERVATION VALIDATION")
    print("="*80)
    print("Validating model with ALL energy/water cycle variables")
    print("="*80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and data
    print(f"\nLoading comprehensive model from {args.model_dir}...")
    model, data_dict, config_dict = load_model(args.model_dir, device)

    print(f"\nModel configuration:")
    print(f"  Type: {config_dict['model_type']}")
    print(f"  Parameters: {config_dict['total_parameters']:,}")
    print(f"  Target variables: {config_dict['n_target_vars']}")

    # Get test sample indices
    n_samples_total = len(data_dict['X_forcing'])
    train_size = int(config_dict['train_ratio'] * n_samples_total)
    test_indices = list(range(train_size, min(train_size + args.n_samples, n_samples_total)))

    print(f"\nValidating on {len(test_indices)} test samples (indices: {test_indices[0]}-{test_indices[-1]})")

    # Validate conservation across all samples
    print(f"\n{'='*80}")
    print("AGGREGATED CONSERVATION VALIDATION")
    print(f"{'='*80}")

    results, all_energy_stats, all_water_stats = validate_sample_conservation(
        model, data_dict, test_indices, device=device, verbose=True
    )

    # Print summary
    print(f"\n{'='*80}")
    print("CONSERVATION SUMMARY")
    print(f"{'='*80}")
    print(f"\nEnergy Conservation (across {results['n_samples']} samples):")
    print(f"  Mean residual: {results['energy']['mean_residual']:.2f} W/m²")
    print(f"  Std residual: {results['energy']['std_residual']:.2f} W/m²")
    print(f"  RMSE: {results['energy']['rmse']:.2f} W/m²")
    print(f"  Relative error: {results['energy']['relative_error_pct']:.2f}%")

    print(f"\nWater Conservation (across {results['n_samples']} samples):")
    print(f"  Mean residual: {results['water']['mean_residual']:.4f} mm/day")
    print(f"  Std residual: {results['water']['std_residual']:.4f} mm/day")
    print(f"  RMSE: {results['water']['rmse']:.4f} mm/day")
    print(f"  Cumulative precipitation: {results['water']['cumulative_precip']:.2f} mm")
    print(f"  Cumulative ET: {results['water']['cumulative_et']:.2f} mm")
    print(f"  Cumulative runoff: {results['water']['cumulative_runoff']:.2f} mm")

    # Plot aggregated results
    plot_aggregated_conservation(results, output_dir)

    # Detailed sample plot if requested
    if args.sample_detail is not None:
        print(f"\n{'='*80}")
        print(f"DETAILED CONSERVATION FOR SAMPLE {args.sample_detail}")
        print(f"{'='*80}")

        checker = ComprehensiveConservationChecker()
        predictions, true_values, forcing_denorm = predict_from_sample_idx(
            model, data_dict, args.sample_detail, device
        )

        pred_dict = dict_from_arrays(predictions, data_dict['target_var_names'])
        forcing_dict = dict_from_arrays(forcing_denorm, data_dict['forcing_var_names'])

        energy_stats = checker.check_full_energy_conservation(pred_dict, forcing_dict, verbose=True)
        water_stats = checker.check_full_water_conservation(pred_dict, forcing_dict, verbose=True)

        # Use improved visualization
        checker.plot_improved_conservation(energy_stats, water_stats, save_dir=output_dir)
        print(f"\nDetailed conservation plots saved to:")
        print(f"  - {output_dir / 'energy_conservation_improved.png'}")
        print(f"  - {output_dir / 'water_conservation_improved.png'}")

    # Save numerical results with improved format
    results_file = output_dir / 'comprehensive_conservation_results.json'

    # Determine assessment levels
    energy_assessment = 'EXCELLENT' if results['energy']['rmse'] < 10 else \
                       'GOOD' if results['energy']['rmse'] < 30 else \
                       'ACCEPTABLE' if results['energy']['rmse'] < 50 else 'POOR'

    water_assessment = 'EXCELLENT' if results['water']['rmse'] < 0.1 else \
                      'GOOD' if results['water']['rmse'] < 0.5 else \
                      'ACCEPTABLE' if results['water']['rmse'] < 1.0 else 'POOR'

    save_results = {
        'metadata': {
            'model_dir': str(args.model_dir),
            'model_type': config_dict['model_type'],
            'n_target_vars': config_dict['n_target_vars'],
            'n_samples_validated': results['n_samples'],
            'test_indices': test_indices,
        },
        'energy_conservation': {
            'description': 'Energy balance: residual = input - output',
            'equation': '(FSA - FIRA) - (HFX + LH + GRDFLX)',
            'statistics': {
                'mean_residual_Wm2': float(results['energy']['mean_residual']),
                'std_residual_Wm2': float(results['energy']['std_residual']),
                'rmse_Wm2': float(results['energy']['rmse']),
                'relative_error_percent': float(results['energy']['relative_error_pct']),
            },
            'assessment': energy_assessment,
            'thresholds': {
                'EXCELLENT': '< 10 W/m²',
                'GOOD': '< 30 W/m²',
                'ACCEPTABLE': '< 50 W/m²',
                'POOR': '>= 50 W/m²'
            }
        },
        'water_conservation': {
            'description': 'Water balance: residual = input - output - change',
            'equation': 'Precipitation - (ET + Runoff + ΔStorage)',
            'statistics': {
                'mean_residual_mm_per_day': float(results['water']['mean_residual']),
                'std_residual_mm_per_day': float(results['water']['std_residual']),
                'rmse_mm_per_day': float(results['water']['rmse']),
            },
            'cumulative_totals_mm': {
                'precipitation': float(results['water']['cumulative_precip']),
                'evapotranspiration': float(results['water']['cumulative_et']),
                'runoff': float(results['water']['cumulative_runoff']),
                'total_output': float(results['water']['cumulative_et'] + results['water']['cumulative_runoff']),
            },
            'assessment': water_assessment,
            'thresholds': {
                'EXCELLENT': '< 0.1 mm/day',
                'GOOD': '< 0.5 mm/day',
                'ACCEPTABLE': '< 1.0 mm/day',
                'POOR': '>= 1.0 mm/day'
            }
        },
    }

    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Generate detailed report with improved format
    report_file = output_dir / 'comprehensive_conservation_report.txt'
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("    COMPREHENSIVE CONSERVATION VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("MODEL INFORMATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Model directory:  {args.model_dir}\n")
        f.write(f"Model type:       {config_dict['model_type']}\n")
        f.write(f"Target variables: {config_dict['n_target_vars']}\n")
        f.write(f"Test samples:     {results['n_samples']}\n")
        f.write(f"Sample indices:   {test_indices[0]} - {test_indices[-1]}\n\n")

        f.write("="*80 + "\n")
        f.write("ENERGY CONSERVATION\n")
        f.write("="*80 + "\n\n")

        f.write("Conservation Check:\n")
        f.write("  Equation:  residual = input - output\n")
        f.write("  Formula:   (FSA - FIRA) - (HFX + LH + GRDFLX)\n\n")

        f.write("Statistics (W/m²):\n")
        f.write(f"  Mean residual:     {results['energy']['mean_residual']:>10.2f}\n")
        f.write(f"  Std residual:      {results['energy']['std_residual']:>10.2f}\n")
        f.write(f"  RMSE:              {results['energy']['rmse']:>10.2f}\n")
        f.write(f"  Relative error:    {results['energy']['relative_error_pct']:>10.2f} %\n\n")

        f.write("Assessment Criteria:\n")
        f.write("  EXCELLENT:    RMSE < 10 W/m²\n")
        f.write("  GOOD:         RMSE < 30 W/m²\n")
        f.write("  ACCEPTABLE:   RMSE < 50 W/m²\n")
        f.write("  POOR:         RMSE >= 50 W/m²\n\n")

        f.write(f"Result: {energy_assessment}\n")
        if energy_assessment == 'EXCELLENT':
            f.write("  ✓ Model maintains excellent energy conservation\n")
            f.write("  ✓ Predictions are physically consistent\n\n")
        elif energy_assessment == 'GOOD':
            f.write("  ✓ Model shows good energy conservation\n")
            f.write("  ✓ Predictions are generally physically consistent\n\n")
        elif energy_assessment == 'ACCEPTABLE':
            f.write("  ⚠ Minor deviations from perfect energy conservation\n")
            f.write("  ⚠ Consider model refinement for critical applications\n\n")
        else:
            f.write("  ✗ Significant energy conservation violations detected\n")
            f.write("  ✗ Model requires further training or architecture changes\n\n")

        f.write("="*80 + "\n")
        f.write("WATER CONSERVATION\n")
        f.write("="*80 + "\n\n")

        f.write("Conservation Check:\n")
        f.write("  Equation:  residual = input - output - change\n")
        f.write("  Formula:   Precipitation - (ET + Runoff + ΔStorage)\n\n")

        f.write("Statistics (mm/day):\n")
        f.write(f"  Mean residual:     {results['water']['mean_residual']:>10.4f}\n")
        f.write(f"  Std residual:      {results['water']['std_residual']:>10.4f}\n")
        f.write(f"  RMSE:              {results['water']['rmse']:>10.4f}\n\n")

        f.write("Cumulative Water Balance (mm):\n")
        total_output = results['water']['cumulative_et'] + results['water']['cumulative_runoff']
        f.write(f"  Precipitation (Input):     {results['water']['cumulative_precip']:>10.2f}\n")
        f.write(f"  ET (Output):               {results['water']['cumulative_et']:>10.2f}\n")
        f.write(f"  Runoff (Output):           {results['water']['cumulative_runoff']:>10.2f}\n")
        f.write(f"  Total Output:              {total_output:>10.2f}\n")
        f.write(f"  Imbalance:                 {results['water']['cumulative_precip'] - total_output:>10.2f}\n\n")

        f.write("Assessment Criteria:\n")
        f.write("  EXCELLENT:    RMSE < 0.1 mm/day\n")
        f.write("  GOOD:         RMSE < 0.5 mm/day\n")
        f.write("  ACCEPTABLE:   RMSE < 1.0 mm/day\n")
        f.write("  POOR:         RMSE >= 1.0 mm/day\n\n")

        f.write(f"Result: {water_assessment}\n")
        if water_assessment == 'EXCELLENT':
            f.write("  ✓ Model maintains excellent water conservation\n")
            f.write("  ✓ Water balance is highly accurate\n\n")
        elif water_assessment == 'GOOD':
            f.write("  ✓ Model shows good water conservation\n")
            f.write("  ✓ Water balance is generally accurate\n\n")
        elif water_assessment == 'ACCEPTABLE':
            f.write("  ⚠ Minor deviations from perfect water conservation\n")
            f.write("  ⚠ Consider model refinement for hydrological applications\n\n")
        else:
            f.write("  ✗ Significant water conservation violations detected\n")
            f.write("  ✗ Model requires further training or architecture changes\n\n")

        f.write("="*80 + "\n")
        f.write("SUMMARY & RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")

        f.write("Physical Consistency:\n")
        f.write("  Conservation laws are fundamental physical constraints that must be\n")
        f.write("  satisfied by any realistic Earth system model. This validation tests\n")
        f.write("  whether the LSTM emulator has learned to respect these constraints.\n\n")

        overall_pass = energy_assessment in ['EXCELLENT', 'GOOD'] and water_assessment in ['EXCELLENT', 'GOOD']
        if overall_pass:
            f.write("Overall Assessment: PASS\n")
            f.write("  ✓ Model demonstrates strong physical consistency\n")
            f.write("  ✓ Suitable for use in climate/hydrological applications\n")
            f.write("  ✓ Predictions respect fundamental conservation laws\n\n")
        else:
            f.write("Overall Assessment: NEEDS IMPROVEMENT\n")
            f.write("  ⚠ Model shows conservation violations\n")
            f.write("  ⚠ Recommendations:\n")
            f.write("     - Increase training data or epochs\n")
            f.write("     - Adjust loss function to emphasize conservation variables\n")
            f.write("     - Consider physics-informed neural network approaches\n")
            f.write("     - Review data preprocessing and normalization\n\n")

        f.write("Generated Files:\n")
        f.write(f"  - {results_file.name} (detailed numerical results)\n")
        f.write(f"  - comprehensive_conservation_summary.png (aggregated plots)\n")
        if args.sample_detail is not None:
            f.write(f"  - energy_conservation_improved.png (detailed energy plots)\n")
            f.write(f"  - water_conservation_improved.png (detailed water plots)\n\n")
        else:
            f.write("\n")

        f.write("="*80 + "\n")

    print(f"Report saved to: {report_file}")

    print("\n" + "="*80)
    print("COMPREHENSIVE CONSERVATION VALIDATION COMPLETE!")
    print("="*80)
    print(f"Results directory: {output_dir}")

if __name__ == '__main__':
    main()
