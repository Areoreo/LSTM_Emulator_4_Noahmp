"""
Analysis script to visualize and summarize training results
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from pathlib import Path
import pandas as pd

def load_results(results_dir):
    """Load all results from a training run"""
    results_path = Path(results_dir)

    # Load metrics
    with open(results_path / 'metrics.json', 'r') as f:
        metrics = json.load(f)

    # Load config
    with open(results_path / 'config.json', 'r') as f:
        config = json.load(f)

    # Load history
    with open(results_path / 'training_history.pkl', 'rb') as f:
        history = pickle.load(f)

    # Load predictions
    val_preds = np.load(results_path / 'val_predictions.npy')
    val_targets = np.load(results_path / 'val_targets.npy')

    return metrics, config, history, val_preds, val_targets

def create_detailed_report(results_dir, output_file='results_summary.txt'):
    """Create detailed text report"""
    metrics, config, history, val_preds, val_targets = load_results(results_dir)

    results_path = Path(results_dir)
    report_path = results_path / output_file

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LSTM PARAMETER PREDICTION - RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")

        # Model configuration
        f.write("MODEL CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        for key, value in config.items():
            f.write(f"  {key:20s}: {value}\n")

        # Parameter weights (if available)
        if 'parameter_weights' in history:
            f.write("\n" + "="*80 + "\n")
            f.write("PARAMETER LOSS WEIGHTS:\n")
            f.write("-" * 40 + "\n")
            param_weights = history['parameter_weights']

            # Check if weights are uniform
            weights_values = list(param_weights.values())
            if all(w == 1.0 for w in weights_values):
                f.write("  All parameters have equal weight (1.0)\n")
            else:
                # Sort by weight (descending)
                sorted_weights = sorted(param_weights.items(), key=lambda x: x[1], reverse=True)
                max_weight = max(weights_values)
                min_weight = min(weights_values)
                f.write(f"  Weight range: {min_weight:.2f} - {max_weight:.2f}\n")
                f.write(f"  Weight ratio: {max_weight/min_weight:.2f}:1\n\n")

                for param_name, weight in sorted_weights:
                    emphasis = ""
                    if weight >= 1.5:
                        emphasis = " [HIGH FOCUS]"
                    elif weight <= 0.5:
                        emphasis = " [LOW FOCUS]"
                    f.write(f"  {param_name:20s}: {weight:.2f}{emphasis}\n")

        # Training results
        f.write("\n" + "="*80 + "\n")
        f.write("TRAINING RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Best Epoch          : {history['best_epoch']}\n")
        f.write(f"  Final Train Loss    : {history['train_losses'][-1]:.6f}\n")
        f.write(f"  Best Val Loss       : {history['best_val_loss']:.6f}\n")
        f.write(f"  Total Epochs Run    : {len(history['train_losses'])}\n")

        # Validation metrics
        f.write("\n" + "="*80 + "\n")
        f.write("VALIDATION METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  MSE                 : {metrics['val_mse']:.6f}\n")
        f.write(f"  RMSE                : {metrics['val_rmse']:.6f}\n")
        f.write(f"  MAE                 : {metrics['val_mae']:.6f}\n")
        f.write(f"  Mean R²             : {metrics['mean_r2']:.4f}\n")

        # Per-parameter R²
        f.write("\n" + "="*80 + "\n")
        f.write("PER-PARAMETER R² SCORES:\n")
        f.write("-" * 40 + "\n")

        # Sort by R² score
        r2_items = sorted(metrics['r2_per_param'].items(), key=lambda x: x[1], reverse=True)

        for param, r2 in r2_items:
            status = "Good" if r2 > 0.7 else ("Fair" if r2 > 0.3 else ("Poor" if r2 > 0 else "Bad"))
            f.write(f"  {param:20s}: {r2:7.4f}  [{status}]\n")

        # Parameter prediction statistics
        f.write("\n" + "="*80 + "\n")
        f.write("PARAMETER PREDICTION STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Parameter':<20s} {'Mean Error':>12s} {'Std Error':>12s} {'RMSE':>12s}\n")
        f.write("-" * 60 + "\n")

        for i, (param, r2) in enumerate(r2_items):
            errors = val_preds[:, list(metrics['r2_per_param'].keys()).index(param)] - \
                    val_targets[:, list(metrics['r2_per_param'].keys()).index(param)]
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            rmse = np.sqrt(np.mean(errors**2))
            f.write(f"{param:<20s} {mean_error:12.4f} {std_error:12.4f} {rmse:12.4f}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY:\n")
        f.write("-" * 40 + "\n")
        n_good = sum(1 for r2 in metrics['r2_per_param'].values() if r2 > 0.7)
        n_fair = sum(1 for r2 in metrics['r2_per_param'].values() if 0.3 < r2 <= 0.7)
        n_poor = sum(1 for r2 in metrics['r2_per_param'].values() if 0 < r2 <= 0.3)
        n_bad = sum(1 for r2 in metrics['r2_per_param'].values() if r2 <= 0)

        f.write(f"  Good predictions (R² > 0.7)  : {n_good}\n")
        f.write(f"  Fair predictions (0.3-0.7)   : {n_fair}\n")
        f.write(f"  Poor predictions (0-0.3)     : {n_poor}\n")
        f.write(f"  Bad predictions (R² < 0)     : {n_bad}\n")
        f.write("\n")
        f.write("  This is an inverse problem, which is inherently difficult.\n")
        f.write("  Some parameters may not be identifiable from the given outputs.\n")
        f.write("="*80 + "\n")

    print(f"Detailed report saved to: {report_path}")
    return report_path

def plot_r2_comparison(results_dir, output_file='r2_comparison.png'):
    """Create bar plot comparing R² scores"""
    metrics, _, _, _, _ = load_results(results_dir)

    results_path = Path(results_dir)

    # Sort parameters by R² score
    r2_items = sorted(metrics['r2_per_param'].items(), key=lambda x: x[1], reverse=True)
    params = [item[0] for item in r2_items]
    r2_scores = [item[1] for item in r2_items]

    # Create color map (green for good, yellow for fair, red for poor/bad)
    colors = ['green' if r2 > 0.7 else 'yellowgreen' if r2 > 0.3 else 'orange' if r2 > 0 else 'red'
              for r2 in r2_scores]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(params, r2_scores, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels
    for i, (bar, r2) in enumerate(zip(bars, r2_scores)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{r2:.3f}',
                ha='left', va='center', fontsize=9, fontweight='bold')

    # Add reference lines
    ax.axvline(0.7, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Good (0.7)')
    ax.axvline(0.3, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Fair (0.3)')
    ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parameter', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Prediction Performance (R² Scores)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_path / output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"R² comparison plot saved to: {results_path / output_file}")

def create_summary_table(results_dir, output_file='summary_table.csv'):
    """Create CSV table with all metrics"""
    metrics, config, history, val_preds, val_targets = load_results(results_dir)

    results_path = Path(results_dir)

    # Create DataFrame
    data = []
    for param, r2 in metrics['r2_per_param'].items():
        idx = list(metrics['r2_per_param'].keys()).index(param)
        errors = val_preds[:, idx] - val_targets[:, idx]

        data.append({
            'Parameter': param,
            'R2': r2,
            'Mean_Error': np.mean(errors),
            'Std_Error': np.std(errors),
            'RMSE': np.sqrt(np.mean(errors**2)),
            'MAE': np.mean(np.abs(errors)),
            'Min_Error': np.min(errors),
            'Max_Error': np.max(errors)
        })

    df = pd.DataFrame(data)
    df = df.sort_values('R2', ascending=False)
    df.to_csv(results_path / output_file, index=False)

    print(f"Summary table saved to: {results_path / output_file}")
    return df

def main(results_dir):
    """Run all analysis"""
    print("="*80)
    print("ANALYZING TRAINING RESULTS")
    print("="*80)
    print(f"\nResults directory: {results_dir}\n")

    # Create detailed report
    print("Creating detailed report...")
    report_path = create_detailed_report(results_dir)

    # Create R² comparison plot
    print("Creating R² comparison plot...")
    plot_r2_comparison(results_dir)

    # Create summary table
    print("Creating summary table...")
    df = create_summary_table(results_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nTop 3 best predicted parameters:")
    for i, row in df.head(3).iterrows():
        print(f"  {row['Parameter']:20s}: R² = {row['R2']:.4f}")

    print(f"\nBottom 3 worst predicted parameters:")
    for i, row in df.tail(3).iterrows():
        print(f"  {row['Parameter']:20s}: R² = {row['R2']:.4f}")

    # Display report
    print("\n" + "="*80)
    print("FULL REPORT:")
    print("="*80)
    with open(report_path, 'r') as f:
        print(f.read())

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Find most recent results directory
        results_dirs = sorted(Path('results').glob('LSTM_*'))
        if results_dirs:
            results_dir = results_dirs[-1]
        else:
            print("No results found!")
            sys.exit(1)

    main(results_dir)
