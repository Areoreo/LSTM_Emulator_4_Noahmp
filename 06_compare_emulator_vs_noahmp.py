"""
Compare emulator results with original Noah-MP model results and observations
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Main target variables
MAIN_TARGET_VARS = ['SOIL_M', 'LH', 'HFX']

# Mapping from emulator variable names to Noah-MP NetCDF variable names
NOAHMP_VAR_MAPPING = {
    'SOIL_M': 'SOIL_M',  # Soil moisture (0-10cm)
    'LH': 'LH',          # Latent heat flux
    'HFX': 'HFX'         # Sensible heat flux
}


def load_noahmp_results(sample_dir):
    """
    Load Noah-MP model results from NetCDF file

    Args:
        sample_dir: Directory containing Noah-MP output

    Returns:
        DataFrame with daily aggregated results
    """
    output_dir = Path(sample_dir) / 'output'
    nc_files = sorted(list(output_dir.glob('*.LDASOUT_DOMAIN1')))

    if not nc_files:
        raise ValueError(f"No NetCDF files found in {output_dir}")

    print(f"  Loading {len(nc_files)} NetCDF file(s) from {output_dir}")

    # Load all NetCDF files
    datasets = []
    for nc_file in nc_files:
        ds = xr.open_dataset(nc_file)
        datasets.append(ds)

    # Concatenate along time dimension if multiple files
    if len(datasets) > 1:
        ds_combined = xr.concat(datasets, dim='Time')
    else:
        ds_combined = datasets[0]

    # Extract time - Noah-MP uses 'Times' variable with format 'YYYY-MM-DD_HH:MM:SS'
    times_bytes = ds_combined['Times'].values
    times_str = [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in times_bytes]
    times = pd.to_datetime(times_str, format='%Y-%m-%d_%H:%M:%S')

    # Extract target variables
    data = {'time': times}

    for emu_var, noahmp_var in NOAHMP_VAR_MAPPING.items():
        if noahmp_var in ds_combined:
            values = ds_combined[noahmp_var].values
            # Handle multi-dimensional variables
            if values.ndim == 4:
                # Shape: (time, south_north, soil_layers, west_east)
                # Take first spatial point (0,0) and first soil layer (0)
                values = values[:, 0, 0, 0]
            elif values.ndim == 3:
                # Shape: (time, south_north, west_east)
                # Take first spatial point (0,0)
                values = values[:, 0, 0]
            elif values.ndim == 2:
                # Shape: (time, something_else) - just take first column
                values = values[:, 0]
            # else: ndim == 1, use as is

            data[emu_var] = values
        else:
            print(f"  Warning: Variable '{noahmp_var}' not found in NetCDF")

    # Close datasets
    for ds in datasets:
        ds.close()

    # Create DataFrame
    df = pd.DataFrame(data)

    # Aggregate to daily mean
    df['date'] = df['time'].dt.date
    agg_dict = {var: 'mean' for var in MAIN_TARGET_VARS if var in df.columns}
    df_daily = df.groupby('date').agg(agg_dict).reset_index()

    print(f"  Loaded {len(df_daily)} days of data")

    return df_daily


def load_emulator_results(csv_file):
    """Load emulator predictions from CSV file"""
    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df


def load_observations(obs_file):
    """Load observations from CSV file"""
    df = pd.read_csv(obs_file)
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df


def compute_metrics(obs, pred, var_names):
    """
    Compute metrics comparing observations and predictions

    Args:
        obs: Observations DataFrame
        pred: Predictions DataFrame
        var_names: List of variable names to compare

    Returns:
        Dictionary of metrics per variable
    """
    metrics = {}

    for var_name in var_names:
        if var_name not in obs.columns or var_name not in pred.columns:
            print(f"  Warning: Variable '{var_name}' not found in data")
            continue

        obs_var = obs[var_name].values
        pred_var = pred[var_name].values

        # Remove NaN values
        valid = ~(np.isnan(obs_var) | np.isnan(pred_var))
        obs_var = obs_var[valid]
        pred_var = pred_var[valid]

        if len(obs_var) == 0:
            metrics[var_name] = {
                'R2': np.nan,
                'RMSE': np.nan,
                'MAE': np.nan,
                'PBIAS': np.nan,
                'n_valid': 0
            }
            continue

        # Compute R²
        ss_res = np.sum((obs_var - pred_var) ** 2)
        ss_tot = np.sum((obs_var - obs_var.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        # Compute RMSE
        rmse = np.sqrt(np.mean((obs_var - pred_var) ** 2))

        # Compute MAE
        mae = np.mean(np.abs(obs_var - pred_var))

        # Compute PBIAS (Percent Bias)
        pbias = 100 * np.sum(pred_var - obs_var) / np.sum(obs_var) if np.sum(obs_var) != 0 else np.nan

        metrics[var_name] = {
            'R2': float(r2),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'PBIAS': float(pbias),
            'n_valid': int(len(obs_var))
        }

    return metrics


def plot_comparison(df1, df2, df1_label, df2_label, var_names, sample_idx, save_path):
    """
    Plot time series comparison between two datasets

    Args:
        df1: First DataFrame (e.g., emulator)
        df2: Second DataFrame (e.g., Noah-MP or observations)
        df1_label: Label for first dataset
        df2_label: Label for second dataset
        var_names: List of variables to plot
        sample_idx: Sample index for title
        save_path: Path to save plot
    """
    n_vars = len(var_names)
    fig, axes = plt.subplots(n_vars, 1, figsize=(14, 4 * n_vars))

    if n_vars == 1:
        axes = [axes]

    for i, var_name in enumerate(var_names):
        ax = axes[i]

        if var_name in df1.columns:
            ax.plot(df1['date'], df1[var_name], linewidth=2, alpha=0.8,
                   color='blue', label=df1_label, zorder=2)

        if var_name in df2.columns:
            ax.plot(df2['date'], df2[var_name], linewidth=2, alpha=0.7,
                   color='red', label=df2_label, linestyle='--', zorder=3)

        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel(var_name, fontsize=11)
        ax.set_title(f'{var_name} Comparison (Parameter Set {sample_idx})', fontsize=12)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.suptitle(f'{df1_label} vs {df2_label}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved plot to: {save_path}")

    plt.close()


def plot_scatter(df1, df2, df1_label, df2_label, var_names, metrics, sample_idx, save_path):
    """
    Plot scatter plots comparing two datasets

    Args:
        df1: First DataFrame (x-axis)
        df2: Second DataFrame (y-axis)
        df1_label: Label for first dataset (x-axis)
        df2_label: Label for second dataset (y-axis)
        var_names: List of variables to plot
        metrics: Dictionary of metrics
        sample_idx: Sample index for title
        save_path: Path to save plot
    """
    n_vars = len(var_names)
    ncols = min(3, n_vars)
    nrows = (n_vars + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

    if n_vars == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_vars > 1 else axes

    for plot_idx, var_name in enumerate(var_names):
        ax = axes[plot_idx]

        if var_name not in df1.columns or var_name not in df2.columns:
            ax.text(0.5, 0.5, f'{var_name}\nNo data', ha='center', va='center')
            ax.set_title(var_name)
            continue

        # Get data
        x_data = df1[var_name].values
        y_data = df2[var_name].values

        # Remove NaN values
        valid = ~(np.isnan(x_data) | np.isnan(y_data))
        x_data = x_data[valid]
        y_data = y_data[valid]

        if len(x_data) > 0:
            # Scatter plot
            ax.scatter(x_data, y_data, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)

            # 1:1 line
            min_val = min(x_data.min(), y_data.min())
            max_val = max(x_data.max(), y_data.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')

            # Get metrics
            var_metrics = metrics.get(var_name, {})
            r2 = var_metrics.get('R2', np.nan)
            rmse = var_metrics.get('RMSE', np.nan)
            mae = var_metrics.get('MAE', np.nan)
            pbias = var_metrics.get('PBIAS', np.nan)

            # Add metrics text box
            textstr = f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}\nPBIAS = {pbias:.2f}%'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)

        ax.set_xlabel(f'{df1_label} {var_name}', fontsize=11)
        ax.set_ylabel(f'{df2_label} {var_name}', fontsize=11)
        ax.set_title(f'{var_name} (Sample {sample_idx})', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='lower right')
        ax.set_aspect('equal', adjustable='box')

    # Hide unused subplots
    for j in range(plot_idx + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'{df1_label} vs {df2_label}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved scatter plot to: {save_path}")

    plt.close()


def main():
    """Main comparison function"""

    # Configuration
    n_samples = 5
    noahmp_base_dir = Path('inference/samples_SP_panama_forward_calibration')
    emulator_base_dir = Path('inference')
    obs_file = 'data/obs/Panama_BCI_obs_2015-07-30_2016-07-29.csv'
    output_dir = Path('inference/comparison_results')
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load observations
    print("\n" + "="*60)
    print("LOADING OBSERVATIONS")
    print("="*60)
    obs_df = load_observations(obs_file)
    print(f"  ✓ Loaded {len(obs_df)} days of observations")

    # Process each sample
    all_metrics_emu_vs_noahmp = []
    all_metrics_noahmp_vs_obs = []

    for sample_idx in range(1, n_samples + 1):
        print("\n" + "="*60)
        print(f"PROCESSING SAMPLE {sample_idx}")
        print("="*60)

        # Load emulator results
        print("\nLoading emulator results...")
        emu_file = emulator_base_dir / f'predictions_sample_{sample_idx}.csv'
        emu_df = load_emulator_results(emu_file)
        print(f"  ✓ Loaded emulator results: {len(emu_df)} days")

        # Load Noah-MP results
        print("\nLoading Noah-MP results...")
        noahmp_dir = noahmp_base_dir / f'sample_{sample_idx}'
        noahmp_df = load_noahmp_results(noahmp_dir)

        # Merge dataframes on date
        merged_emu_noahmp = pd.merge(emu_df, noahmp_df, on='date', suffixes=('_emu', '_noahmp'))
        merged_noahmp_obs = pd.merge(noahmp_df, obs_df, on='date', suffixes=('_noahmp', '_obs'))

        # Compute metrics: Emulator vs Noah-MP
        print("\n--- Emulator vs Noah-MP Metrics ---")
        emu_cols = [f'{var}_emu' for var in MAIN_TARGET_VARS if f'{var}_emu' in merged_emu_noahmp.columns]
        noahmp_cols = [f'{var}_noahmp' for var in MAIN_TARGET_VARS if f'{var}_noahmp' in merged_emu_noahmp.columns]

        # Prepare DataFrames for metric computation
        emu_only = merged_emu_noahmp[['date'] + emu_cols].copy()
        emu_only.columns = ['date'] + [col.replace('_emu', '') for col in emu_cols]

        noahmp_only = merged_emu_noahmp[['date'] + noahmp_cols].copy()
        noahmp_only.columns = ['date'] + [col.replace('_noahmp', '') for col in noahmp_cols]

        metrics_emu_vs_noahmp = compute_metrics(noahmp_only, emu_only, MAIN_TARGET_VARS)
        all_metrics_emu_vs_noahmp.append(metrics_emu_vs_noahmp)

        for var_name, var_metrics in metrics_emu_vs_noahmp.items():
            print(f"  {var_name}:")
            print(f"    R²    = {var_metrics['R2']:.4f}")
            print(f"    RMSE  = {var_metrics['RMSE']:.4f}")
            print(f"    MAE   = {var_metrics['MAE']:.4f}")
            print(f"    PBIAS = {var_metrics['PBIAS']:.2f}%")
            print(f"    N     = {var_metrics['n_valid']}")

        # Compute metrics: Noah-MP vs Observations
        print("\n--- Noah-MP vs Observations Metrics ---")
        noahmp_cols_obs = [f'{var}_noahmp' for var in MAIN_TARGET_VARS if f'{var}_noahmp' in merged_noahmp_obs.columns]
        obs_cols = [f'{var}_obs' for var in MAIN_TARGET_VARS if f'{var}_obs' in merged_noahmp_obs.columns]

        noahmp_obs_only = merged_noahmp_obs[['date'] + noahmp_cols_obs].copy()
        noahmp_obs_only.columns = ['date'] + [col.replace('_noahmp', '') for col in noahmp_cols_obs]

        obs_only = merged_noahmp_obs[['date'] + obs_cols].copy()
        obs_only.columns = ['date'] + [col.replace('_obs', '') for col in obs_cols]

        metrics_noahmp_vs_obs = compute_metrics(obs_only, noahmp_obs_only, MAIN_TARGET_VARS)
        all_metrics_noahmp_vs_obs.append(metrics_noahmp_vs_obs)

        for var_name, var_metrics in metrics_noahmp_vs_obs.items():
            print(f"  {var_name}:")
            print(f"    R²    = {var_metrics['R2']:.4f}")
            print(f"    RMSE  = {var_metrics['RMSE']:.4f}")
            print(f"    MAE   = {var_metrics['MAE']:.4f}")
            print(f"    PBIAS = {var_metrics['PBIAS']:.2f}%")
            print(f"    N     = {var_metrics['n_valid']}")

        # Create plots
        print("\nGenerating comparison plots...")

        # Emulator vs Noah-MP time series
        plot_path = output_dir / f'emulator_vs_noahmp_sample_{sample_idx}.png'
        plot_comparison(emu_only, noahmp_only, 'Emulator', 'Noah-MP',
                       MAIN_TARGET_VARS, sample_idx, plot_path)

        # Emulator vs Noah-MP scatter
        scatter_path = output_dir / f'emulator_vs_noahmp_sample_{sample_idx}_scatter.png'
        plot_scatter(noahmp_only, emu_only, 'Noah-MP', 'Emulator',
                    MAIN_TARGET_VARS, metrics_emu_vs_noahmp, sample_idx, scatter_path)

        # Noah-MP vs Observations time series
        plot_path = output_dir / f'noahmp_vs_obs_sample_{sample_idx}.png'
        plot_comparison(noahmp_obs_only, obs_only, 'Noah-MP', 'Observations',
                       MAIN_TARGET_VARS, sample_idx, plot_path)

        # Noah-MP vs Observations scatter
        scatter_path = output_dir / f'noahmp_vs_obs_sample_{sample_idx}_scatter.png'
        plot_scatter(obs_only, noahmp_obs_only, 'Observations', 'Noah-MP',
                    MAIN_TARGET_VARS, metrics_noahmp_vs_obs, sample_idx, scatter_path)

    # Save all metrics to JSON
    print("\n" + "="*60)
    print("SAVING METRICS")
    print("="*60)

    metrics_output = {
        'emulator_vs_noahmp': all_metrics_emu_vs_noahmp,
        'noahmp_vs_observations': all_metrics_noahmp_vs_obs
    }

    metrics_file = output_dir / 'all_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    print(f"  ✓ Saved all metrics to: {metrics_file}")

    print("\n" + "="*60)
    print("✓ COMPARISON COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Metrics JSON: all_metrics.json")
    print(f"  - Time series plots: *_sample_*.png")
    print(f"  - Scatter plots: *_sample_*_scatter.png")


if __name__ == '__main__':
    main()
