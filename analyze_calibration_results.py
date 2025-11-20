"""
Analyze Parameter Calibration Results
Loads calibration results and creates detailed visualizations and statistics
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_calibration_results(calibration_dir):
    """Load all calibration results from directory"""
    calibration_dir = Path(calibration_dir)

    # Load summary
    with open(calibration_dir / 'ensemble_summary.json', 'r') as f:
        summary = json.load(f)

    # Load parameter statistics
    with open(calibration_dir / 'parameter_statistics.json', 'r') as f:
        param_stats = json.load(f)

    print("="*80)
    print("CALIBRATION RESULTS ANALYSIS")
    print("="*80)
    print(f"\nNumber of calibration runs: {summary['num_calibrations']}")
    print(f"Optimization method: {summary['optimization_method']}")
    print(f"\nNRMSE Statistics:")
    print(f"  Mean: {summary['statistics']['nrmse_mean']:.6f}")
    print(f"  Std:  {summary['statistics']['nrmse_std']:.6f}")
    print(f"  Min:  {summary['statistics']['nrmse_min']:.6f}")
    print(f"  Max:  {summary['statistics']['nrmse_max']:.6f}")

    return summary, param_stats


def print_parameter_statistics(param_stats):
    """Print detailed parameter statistics"""
    print("\n" + "="*80)
    print("PARAMETER UNCERTAINTY ANALYSIS")
    print("="*80)

    print(f"\nNumber of calibrated parameters: {param_stats['num_parameters']}")
    print(f"\nParameter Statistics:")
    print("-"*80)
    print(f"{'Parameter':<15} {'Mean':<12} {'Std':<12} {'CV (%)':<10} {'Min':<12} {'Max':<12}")
    print("-"*80)

    for param_name in param_stats['parameter_names']:
        stats = param_stats['parameters'][param_name]
        print(f"{param_name:<15} {stats['mean']:>11.4f} {stats['std']:>11.4f} "
              f"{stats['cv']*100:>9.2f} {stats['min']:>11.4f} {stats['max']:>11.4f}")

    print("-"*80)

    # Identify well-constrained and poorly-constrained parameters
    print("\nParameter Constraint Analysis:")
    print("-"*80)

    well_constrained = []
    moderately_constrained = []
    poorly_constrained = []

    for param_name in param_stats['parameter_names']:
        cv = param_stats['parameters'][param_name]['cv']
        if cv < 0.05:  # CV < 5%
            well_constrained.append((param_name, cv))
        elif cv < 0.15:  # CV < 15%
            moderately_constrained.append((param_name, cv))
        else:
            poorly_constrained.append((param_name, cv))

    if well_constrained:
        print("\nWell-Constrained (CV < 5%):")
        for name, cv in well_constrained:
            print(f"  {name}: CV = {cv*100:.2f}%")

    if moderately_constrained:
        print("\nModerately-Constrained (5% ≤ CV < 15%):")
        for name, cv in moderately_constrained:
            print(f"  {name}: CV = {cv*100:.2f}%")

    if poorly_constrained:
        print("\nPoorly-Constrained (CV ≥ 15%):")
        for name, cv in poorly_constrained:
            print(f"  {name}: CV = {cv*100:.2f}%")


def create_comparison_plot(summary, param_stats, output_path):
    """Create comprehensive comparison plot"""

    # Extract data
    param_names = param_stats['parameter_names']
    n_params = len(param_names)

    means = [param_stats['parameters'][name]['mean'] for name in param_names]
    stds = [param_stats['parameters'][name]['std'] for name in param_names]
    cvs = [param_stats['parameters'][name]['cv'] for name in param_names]
    baselines = [param_stats['parameters'][name]['baseline'] for name in param_names]

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Parameter values with uncertainty
    ax1 = fig.add_subplot(gs[0, :])
    x_pos = np.arange(n_params)
    ax1.errorbar(x_pos, means, yerr=stds, fmt='o', capsize=5, capthick=2,
                 markersize=8, label='Calibrated (Mean ± Std)', color='steelblue')
    ax1.scatter(x_pos, baselines, marker='x', s=100, color='red',
                label='Baseline', zorder=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(param_names, rotation=45, ha='right')
    ax1.set_ylabel('Parameter Value', fontsize=11)
    ax1.set_title('Calibrated Parameters with Uncertainty', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Coefficient of Variation
    ax2 = fig.add_subplot(gs[1, 0])
    colors = ['green' if cv < 0.05 else 'orange' if cv < 0.15 else 'red' for cv in cvs]
    bars = ax2.bar(x_pos, np.array(cvs)*100, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(5, color='green', linestyle='--', linewidth=1.5, label='Well-constrained (CV<5%)')
    ax2.axhline(15, color='orange', linestyle='--', linewidth=1.5, label='Moderately (CV<15%)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(param_names, rotation=45, ha='right')
    ax2.set_ylabel('Coefficient of Variation (%)', fontsize=11)
    ax2.set_title('Parameter Uncertainty (CV)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Relative change from baseline
    ax3 = fig.add_subplot(gs[1, 1])
    rel_changes = [(m - b) / b * 100 if b != 0 else 0 for m, b in zip(means, baselines)]
    colors_change = ['blue' if rc < 0 else 'red' for rc in rel_changes]
    bars = ax3.bar(x_pos, rel_changes, color=colors_change, alpha=0.7, edgecolor='black')
    ax3.axhline(0, color='black', linestyle='-', linewidth=1)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(param_names, rotation=45, ha='right')
    ax3.set_ylabel('Change from Baseline (%)', fontsize=11)
    ax3.set_title('Calibration Impact on Parameters', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. NRMSE distribution across runs
    ax4 = fig.add_subplot(gs[2, 0])
    nrmse_values = [r['nrmse'] for r in summary['results']]
    ax4.hist(nrmse_values, bins=min(20, len(nrmse_values)), alpha=0.7,
             color='skyblue', edgecolor='black')
    ax4.axvline(summary['statistics']['nrmse_mean'], color='red',
                linestyle='--', linewidth=2, label=f"Mean: {summary['statistics']['nrmse_mean']:.6f}")
    ax4.set_xlabel('NRMSE', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('NRMSE Distribution Across Calibration Runs', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Parameter ranges
    ax5 = fig.add_subplot(gs[2, 1])
    mins = [param_stats['parameters'][name]['min'] for name in param_names]
    maxs = [param_stats['parameters'][name]['max'] for name in param_names]
    ranges = [(ma - mi) / me * 100 if me != 0 else 0
              for mi, ma, me in zip(mins, maxs, means)]
    bars = ax5.bar(x_pos, ranges, alpha=0.7, color='purple', edgecolor='black')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(param_names, rotation=45, ha='right')
    ax5.set_ylabel('Range / Mean (%)', fontsize=11)
    ax5.set_title('Parameter Range Relative to Mean', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Calibration Analysis Summary ({summary["num_calibrations"]} runs)',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_path}")
    plt.close()


def export_to_csv(param_stats, output_path):
    """Export parameter statistics to CSV"""
    data = []
    for param_name in param_stats['parameter_names']:
        stats = param_stats['parameters'][param_name]
        data.append({
            'Parameter': param_name,
            'Baseline': stats['baseline'],
            'Mean': stats['mean'],
            'Std': stats['std'],
            'CV': stats['cv'],
            'Min': stats['min'],
            'Max': stats['max'],
            'Median': stats['median']
        })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Parameter statistics exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze parameter calibration results and create visualizations'
    )
    parser.add_argument('--calibration_dir', type=str, required=True,
                       help='Directory containing calibration results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for analysis (default: same as calibration_dir)')

    args = parser.parse_args()

    calibration_dir = Path(args.calibration_dir)
    output_dir = Path(args.output_dir) if args.output_dir else calibration_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load results
    summary, param_stats = load_calibration_results(calibration_dir)

    # Print statistics
    print_parameter_statistics(param_stats)

    # Create comprehensive comparison plot
    create_comparison_plot(summary, param_stats, output_dir / 'calibration_analysis.png')

    # Export to CSV
    export_to_csv(param_stats, output_dir / 'parameter_statistics_table.csv')

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput files saved to: {output_dir}")
    print("  - calibration_analysis.png")
    print("  - parameter_statistics_table.csv")


if __name__ == '__main__':
    main()
