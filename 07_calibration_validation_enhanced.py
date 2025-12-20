#!/usr/bin/env python3
"""
Enhanced Calibration Validation Script with Period Markers

This script compares Noah-MP model outputs with different parameter sets,
distinguishing between calibration and validation time periods in both
analysis and plotting.

Features:
- Separate metrics calculation for calibration and validation periods (RMSE, PBIAS, MAE)
- Time series plots with period labels drawn directly on the boundary
- Observation (truth) curve plotted prominently in black
- Column name mapping between observation and simulation data
- Publication-quality figures sized for A4 paper
- Comprehensive comparison of default, expert, and emulator-calibrated parameters
- Metrics summary output file
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from pathlib import Path
from datetime import datetime

# Add project directory to path
sys.path.insert(0, '/home/petrichor/ymwang/snap/Emulator-based_calibration/calibration-BCI')

# Import publication plotting settings
HAS_PLOT_SETTINGS = False
try:
    import plot_settings
    HAS_PLOT_SETTINGS = True
except (ImportError, OSError):
    pass

# Set default style
try:
    if 'seaborn-v0_8-whitegrid' in plt.style.available:
        plt.style.use('seaborn-v0_8-whitegrid')
    elif 'seaborn-whitegrid' in plt.style.available:
        plt.style.use('seaborn-whitegrid')
    else:
        plt.style.use('ggplot')
except:
    pass

import config_forward_comprehensive as config

# Get full time range for plotting
TIME_RANGE_FULL = getattr(config, 'TIME_RANGE_FULL', None)


# =============================================================================
# Column Name Mapping: Observation -> Simulation
# =============================================================================
OBS_TO_SIM_MAPPING = {
    # Fluxtower observation names -> Noah-MP simulation names
    'LE': 'LH',           # Latent heat flux
    'H': 'HFX',           # Sensible heat flux
    'SWC': 'SOIL_M',      # Soil water content / soil moisture
    'Rnet': 'RNET',       # Net radiation
    'Rs': 'SWDOWN',       # Shortwave radiation
    'tair': 'T2M',        # Air temperature
}

# Reverse mapping: Simulation -> Observation
SIM_TO_OBS_MAPPING = {v: k for k, v in OBS_TO_SIM_MAPPING.items()}


# =============================================================================
# A4 Paper Configuration for Publication-Quality Figures
# =============================================================================
A4_WIDTH_INCHES = 8.27  # 210mm
MARGIN_INCHES = 0.75
USABLE_WIDTH = A4_WIDTH_INCHES - 2 * MARGIN_INCHES  # ~6.77 inches

# Font sizes for actual print size
FONT_SIZES = {
    'title': 10,
    'subtitle': 10,
    'axis_label': 9,
    'tick_label': 8,
    'legend': 8,
    'annotation': 7,
    'period_label': 9,
}

# Colors for different model types
COLORS = {
    'observed': '#000000',      # Black
    'emulator': '#0077BB',      # Blue
    'expert': '#EE7733',        # Orange
    'default': '#009988',       # Teal
}


def setup_matplotlib():
    """Configure matplotlib for publication-quality output"""
    plt.rcParams.update({
        'font.size': FONT_SIZES['tick_label'],
        'axes.titlesize': FONT_SIZES['subtitle'],
        'axes.labelsize': FONT_SIZES['axis_label'],
        'xtick.labelsize': FONT_SIZES['tick_label'],
        'ytick.labelsize': FONT_SIZES['tick_label'],
        'legend.fontsize': FONT_SIZES['legend'],
        'figure.titlesize': FONT_SIZES['title'],
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def parse_noahmp_output(output_file):
    """Parse Noah-MP output CSV file and return DataFrame"""
    if not os.path.exists(output_file):
        return None
    try:
        df = pd.read_csv(output_file)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"Error reading {output_file}: {e}")
        return None


def read_observation_data(obs_file, time_range=None):
    """Read observation data from CSV file and apply column name mapping"""
    if not os.path.exists(obs_file):
        print(f"Warning: Observation file not found: {obs_file}")
        return None
    
    obs_df = pd.read_csv(obs_file)
    
    # Handle timestamp column
    if 'timestamp' in obs_df.columns:
        obs_df['timestamp'] = pd.to_datetime(obs_df['timestamp'])
    elif 'date' in obs_df.columns:
        obs_df['timestamp'] = pd.to_datetime(obs_df['date'])
    
    # Apply column name mapping (observation -> simulation names)
    rename_map = {}
    for obs_name, sim_name in OBS_TO_SIM_MAPPING.items():
        if obs_name in obs_df.columns and sim_name not in obs_df.columns:
            rename_map[obs_name] = sim_name
    
    if rename_map:
        obs_df = obs_df.rename(columns=rename_map)
        print(f"  Applied column mapping: {rename_map}")
    
    # Filter by time range if specified
    if time_range is not None:
        start_date = pd.to_datetime(time_range['start'])
        end_date = pd.to_datetime(time_range['end'])
        obs_df = obs_df[
            (obs_df['timestamp'] >= start_date) & 
            (obs_df['timestamp'] <= end_date)
        ].reset_index(drop=True)
    
    return obs_df


def calculate_metrics(obs, sim):
    """Calculate performance metrics between observations and simulations"""
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs_clean = obs[mask]
    sim_clean = sim[mask]
    
    if len(obs_clean) == 0:
        return {'RMSE': np.nan, 'PBIAS': np.nan, 'MAE': np.nan,
                'BIAS': np.nan, 'R2': np.nan, 'NSE': np.nan, 'N': 0}
    
    bias = np.mean(sim_clean - obs_clean)
    mae = np.mean(np.abs(sim_clean - obs_clean))
    rmse = np.sqrt(np.mean((sim_clean - obs_clean)**2))
    
    obs_mean = np.mean(obs_clean)
    obs_sum = np.sum(obs_clean)
    if abs(obs_sum) > 1e-10:
        pbias = 100 * np.sum(sim_clean - obs_clean) / obs_sum
    else:
        pbias = np.nan
    
    ss_res = np.sum((obs_clean - sim_clean)**2)
    ss_tot = np.sum((obs_clean - obs_mean)**2)
    if ss_tot > 1e-10:
        r2 = 1 - (ss_res / ss_tot)
        nse = 1 - (ss_res / ss_tot)
    else:
        r2 = np.nan
        nse = np.nan
    
    return {'RMSE': rmse, 'PBIAS': pbias, 'MAE': mae,
            'BIAS': bias, 'R2': r2, 'NSE': nse, 'N': len(obs_clean)}


def calculate_metrics_by_period(obs_df, sim_df, variable, cal_start, cal_end, val_start, val_end):
    """
    Calculate metrics separately for calibration and validation periods
    """
    results = {}
    
    # Check if variable exists in both dataframes
    if variable not in obs_df.columns:
        print(f"    Warning: Variable '{variable}' not found in observation data")
        return None
    if variable not in sim_df.columns:
        print(f"    Warning: Variable '{variable}' not found in simulation data")
        return None
    
    # Merge dataframes on timestamp
    merged = pd.merge(
        obs_df[['timestamp', variable]].rename(columns={variable: 'obs'}),
        sim_df[['timestamp', variable]].rename(columns={variable: 'sim'}),
        on='timestamp',
        how='inner'
    )
    
    if merged.empty:
        print(f"    Warning: No matching timestamps for variable '{variable}'")
        return None
    
    # Calibration period
    cal_mask = (merged['timestamp'] >= pd.to_datetime(cal_start)) & \
               (merged['timestamp'] <= pd.to_datetime(cal_end))
    cal_data = merged[cal_mask]
    results['calibration'] = calculate_metrics(cal_data['obs'].values, cal_data['sim'].values)
    
    # Validation period
    val_mask = (merged['timestamp'] >= pd.to_datetime(val_start)) & \
               (merged['timestamp'] <= pd.to_datetime(val_end))
    val_data = merged[val_mask]
    results['validation'] = calculate_metrics(val_data['obs'].values, val_data['sim'].values)
    
    # Full period
    results['full'] = calculate_metrics(merged['obs'].values, merged['sim'].values)
    
    return results


def plot_timeseries_with_periods(obs_df, sim_dfs, variable, output_file,
                                  cal_start, cal_end, val_start, val_end,
                                  labels=None):
    """
    Plot time series comparison with calibration/validation period labels
    
    Features:
    - Background shading for different periods
    - Vertical line at period boundary with period labels drawn directly on plot
    - Period labels drawn at left/right of boundary (NOT in legend)
    - Legend only shows line types (Observed, Emulator, Expert, Default)
    - Observation (truth) curve plotted prominently in black
    """
    setup_matplotlib()
    
    # Create figure sized for A4 width
    fig, ax = plt.subplots(figsize=(USABLE_WIDTH, USABLE_WIDTH * 0.45))
    
    # Convert dates
    cal_start_dt = pd.to_datetime(cal_start)
    cal_end_dt = pd.to_datetime(cal_end)
    val_start_dt = pd.to_datetime(val_start)
    val_end_dt = pd.to_datetime(val_end)
    
    # Check if observation variable exists
    obs_var_exists = obs_df is not None and variable in obs_df.columns
    
    # Get time range from data
    all_times = []
    if obs_df is not None and 'timestamp' in obs_df.columns:
        all_times.extend(obs_df['timestamp'].tolist())
    for sim_df in sim_dfs.values():
        if sim_df is not None and 'timestamp' in sim_df.columns:
            all_times.extend(sim_df['timestamp'].tolist())
    
    if not all_times:
        print(f"Warning: No time data available for {variable}")
        plt.close()
        return
    
    time_min = min(all_times)
    time_max = max(all_times)
    
    # Add background shading for periods (no label - not in legend)
    ax.axvspan(cal_start_dt, cal_end_dt, alpha=0.15, color='blue', zorder=0)
    ax.axvspan(val_start_dt, val_end_dt, alpha=0.15, color='orange', zorder=0)
    
    # Add vertical line at period boundary (no label - not in legend)
    boundary_date = cal_end_dt + (val_start_dt - cal_end_dt) / 2
    ax.axvline(x=boundary_date, color='gray', linestyle='--', linewidth=1.5, zorder=1)
    
    # Plot observations (truth curve) - aggregate to daily for cleaner visualization
    # Filter to TIME_RANGE_FULL to match simulation period
    if obs_var_exists:
        obs_plot = obs_df[['timestamp', variable]].dropna()
        
        # Apply time range filter if TIME_RANGE_FULL is defined
        if TIME_RANGE_FULL is not None:
            full_start = pd.to_datetime(TIME_RANGE_FULL['start'])
            full_end = pd.to_datetime(TIME_RANGE_FULL['end'])
            obs_plot = obs_plot[
                (obs_plot['timestamp'] >= full_start) & 
                (obs_plot['timestamp'] <= full_end)
            ]
        
        # Aggregate to daily means for cleaner time series plot
        obs_daily = aggregate_to_daily(obs_plot.copy(), variable)
        
        if obs_daily is not None and not obs_daily.empty:
            ax.plot(obs_daily['timestamp'], obs_daily[variable], 
                    color=COLORS['observed'], linewidth=1.0,
                    label='Observed', linestyle='-', zorder=10)
            print(f'    Plotted observation (daily): {len(obs_daily)} points')
        else:
            print(f'    Warning: No valid observation data for {variable}')
    else:
        print(f'    Warning: Variable {variable} not found in observation data')
    
    # Plot simulations - aggregate to daily means for consistency with observations
    linestyles = ['-', '--', '-.']
    for i, (name, sim_df) in enumerate(sim_dfs.items()):
        if sim_df is not None and variable in sim_df.columns:
            label = labels.get(name, name.capitalize()) if labels else name.capitalize()
            color = COLORS.get(name, f'C{i}')
            # Aggregate simulation data to daily means
            sim_daily = aggregate_to_daily(sim_df[['timestamp', variable]].copy(), variable)
            if sim_daily is not None and not sim_daily.empty:
                ax.plot(sim_daily['timestamp'], sim_daily[variable],
                       color=color, linewidth=1,
                       label=label, linestyle=linestyles[i % len(linestyles)],
                       alpha=0.85, zorder=2)
                print(f'    Plotted {name} simulation (daily): {len(sim_daily)} points')
    
    # Get y limits for positioning period labels
    y_min, y_max = ax.get_ylim()
    y_pos = y_max - (y_max - y_min) * 0.05  # Position near top
    
    # Draw period labels directly on the plot (NOT in legend)
    ax.text(boundary_date, y_pos, 'Calibration Period  ', 
            ha='right', va='top', fontsize=FONT_SIZES['period_label'], 
            fontweight='bold', color='#444444', alpha=0.9)
    ax.text(boundary_date, y_pos, '  Validation Period', 
            ha='left', va='top', fontsize=FONT_SIZES['period_label'], 
            fontweight='bold', color='#444444', alpha=0.9)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=FONT_SIZES['axis_label'])
    ax.set_ylabel(variable, fontsize=FONT_SIZES['axis_label'])
    ax.set_title(f'{variable} - Model Comparison', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Legend - only line types, no period markers
    ax.legend(loc='lower right', fontsize=FONT_SIZES['legend'], 
              framealpha=0.9, ncol=4)
    
    # Grid
    ax.grid(True, alpha=0.3, zorder=0)
    
    plt.tight_layout()
    
    # Save in multiple formats
    fig.savefig(f"{output_file}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{output_file}.pdf", dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}.png, {output_file}.pdf")
    
    plt.close()


def plot_metrics_comparison(all_metrics, variable, output_file):
    """
    Plot metrics comparison bar chart for calibration vs validation periods
    
    Shows performance metrics (RMSE, PBIAS, MAE) for each model and period
    """
    setup_matplotlib()
    
    # Prepare data
    models = ['default', 'expert', 'emulator']
    metrics_to_plot = ['RMSE', 'PBIAS', 'MAE']
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(1, 3, figsize=(USABLE_WIDTH, USABLE_WIDTH * 0.35))
    
    x = np.arange(len(models))
    width = 0.35
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # Calibration period values
        cal_values = []
        for model in models:
            if model in all_metrics and all_metrics[model] is not None and 'calibration' in all_metrics[model]:
                cal_values.append(all_metrics[model]['calibration'].get(metric, np.nan))
            else:
                cal_values.append(np.nan)
        
        # Validation period values
        val_values = []
        for model in models:
            if model in all_metrics and all_metrics[model] is not None and 'validation' in all_metrics[model]:
                val_values.append(all_metrics[model]['validation'].get(metric, np.nan))
            else:
                val_values.append(np.nan)
        
        # Plot bars
        bars1 = ax.bar(x - width/2, cal_values, width, label='Calibration', 
                       color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, val_values, width, label='Validation',
                       color='darkorange', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    if metric == 'PBIAS':
                        label_text = f'{height:.1f}%'
                    else:
                        label_text = f'{height:.3f}'
                    ax.annotate(label_text,
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3), textcoords="offset points",
                              ha='center', va='bottom', fontsize=6)
        
        metric_labels = {
            'RMSE': 'RMSE',
            'PBIAS': 'PBIAS (%)',
            'MAE': 'MAE'
        }
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=FONT_SIZES['axis_label'])
        ax.set_title(metric_labels.get(metric, metric), fontsize=FONT_SIZES['subtitle'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Default', 'Expert', 'Emulator'], fontsize=FONT_SIZES['tick_label'])
        
        if idx == 0:
            ax.legend(fontsize=FONT_SIZES['legend'], loc='upper right')
        
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    fig.suptitle(f'{variable} - Performance Metrics by Period', 
                fontsize=FONT_SIZES['title'], fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    fig.savefig(f"{output_file}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{output_file}.pdf", dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}.png, {output_file}.pdf")
    
    plt.close()


def plot_scatter_by_period(obs_df, sim_df, variable, output_file, 
                           cal_start, cal_end, val_start, val_end, 
                           model_name='Simulated'):
    """
    Plot scatter comparison with points colored by period
    """
    setup_matplotlib()
    
    if obs_df is None or sim_df is None:
        return
    
    if variable not in obs_df.columns or variable not in sim_df.columns:
        return
    
    # Merge data
    merged = pd.merge(
        obs_df[['timestamp', variable]].rename(columns={variable: 'obs'}),
        sim_df[['timestamp', variable]].rename(columns={variable: 'sim'}),
        on='timestamp',
        how='inner'
    )
    
    if merged.empty:
        return
    
    # Remove NaN values
    merged = merged.dropna()
    
    # Create masks for periods
    cal_mask = (merged['timestamp'] >= pd.to_datetime(cal_start)) & \
               (merged['timestamp'] <= pd.to_datetime(cal_end))
    val_mask = (merged['timestamp'] >= pd.to_datetime(val_start)) & \
               (merged['timestamp'] <= pd.to_datetime(val_end))
    
    cal_data = merged[cal_mask]
    val_data = merged[val_mask]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(USABLE_WIDTH * 0.6, USABLE_WIDTH * 0.6))
    
    # Plot calibration period
    if not cal_data.empty:
        ax.scatter(cal_data['obs'], cal_data['sim'], 
                   alpha=0.6, s=15, c='steelblue', label='Calibration', edgecolors='none')
    
    # Plot validation period
    if not val_data.empty:
        ax.scatter(val_data['obs'], val_data['sim'],
                   alpha=0.6, s=15, c='darkorange', label='Validation', edgecolors='none')
    
    # 1:1 line
    all_values = np.concatenate([merged['obs'].values, merged['sim'].values])
    if len(all_values) > 0:
        lims = [np.nanmin(all_values), np.nanmax(all_values)]
        margin = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, 'k--', linewidth=1, label='1:1 line', zorder=0)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    
    # Calculate and display metrics for each period (RMSE, PBIAS, MAE)
    cal_metrics = calculate_metrics(cal_data['obs'].values, cal_data['sim'].values) if not cal_data.empty else {'RMSE': np.nan, 'PBIAS': np.nan, 'MAE': np.nan}
    val_metrics = calculate_metrics(val_data['obs'].values, val_data['sim'].values) if not val_data.empty else {'RMSE': np.nan, 'PBIAS': np.nan, 'MAE': np.nan}
    
    stats_text = f"Calibration: RMSE={cal_metrics['RMSE']:.3f}, PBIAS={cal_metrics['PBIAS']:.1f}%, MAE={cal_metrics['MAE']:.3f}\n"
    stats_text += f"Validation:  RMSE={val_metrics['RMSE']:.3f}, PBIAS={val_metrics['PBIAS']:.1f}%, MAE={val_metrics['MAE']:.3f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=FONT_SIZES['annotation'],
            va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(f'Observed {variable}', fontsize=FONT_SIZES['axis_label'])
    ax.set_ylabel(f'{model_name} {variable}', fontsize=FONT_SIZES['axis_label'])
    ax.set_title(f'{variable} - {model_name}', fontsize=FONT_SIZES['subtitle'], fontweight='bold')
    ax.legend(fontsize=FONT_SIZES['legend'], loc='lower right')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig.savefig(f"{output_file}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{output_file}.pdf", dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}.png, {output_file}.pdf")
    
    plt.close()

def aggregate_to_daily(df, variable):
    """Aggregate data to daily means for cleaner time series plotting"""
    if df is None or "timestamp" not in df.columns or variable not in df.columns:
        return None
    
    df_copy = df[["timestamp", variable]].copy()
    df_copy["date"] = df_copy["timestamp"].dt.date
    daily = df_copy.groupby("date")[variable].mean().reset_index()
    daily["timestamp"] = pd.to_datetime(daily["date"])
    daily = daily.drop(columns=["date"])
    return daily


def calculate_seasonal_cycle(df, variable):
    """
    Calculate multi-year monthly climatology (seasonal cycle)
    
    Process:
    1. For each year, calculate monthly averages
    2. For each month, calculate the average across all years
    
    Returns DataFrame with columns: month, value, std
    """
    if df is None or "timestamp" not in df.columns or variable not in df.columns:
        return None
    
    df_copy = df[["timestamp", variable]].dropna().copy()
    df_copy["year"] = df_copy["timestamp"].dt.year
    df_copy["month"] = df_copy["timestamp"].dt.month
    
    # Step 1: Monthly average for each year
    monthly_by_year = df_copy.groupby(["year", "month"])[variable].mean().reset_index()
    
    # Step 2: Average across years for each month
    seasonal = monthly_by_year.groupby("month")[variable].agg(["mean", "std"]).reset_index()
    seasonal.columns = ["month", "value", "std"]
    
    return seasonal


def calculate_diurnal_cycle(df, variable):
    """
    Calculate multi-day hourly climatology (diurnal cycle)
    
    Process:
    1. For each day, calculate hourly averages (if sub-hourly data)
    2. For each hour, calculate the average across all days
    
    Returns DataFrame with columns: hour, value, std
    """
    if df is None or "timestamp" not in df.columns or variable not in df.columns:
        return None
    
    df_copy = df[["timestamp", variable]].dropna().copy()
    df_copy["date"] = df_copy["timestamp"].dt.date
    df_copy["hour"] = df_copy["timestamp"].dt.hour
    
    # Step 1: Hourly average for each day (handles sub-hourly data like 30-min)
    hourly_by_day = df_copy.groupby(["date", "hour"])[variable].mean().reset_index()
    
    # Step 2: Average across days for each hour
    diurnal = hourly_by_day.groupby("hour")[variable].agg(["mean", "std"]).reset_index()
    diurnal.columns = ["hour", "value", "std"]
    
    return diurnal


def plot_seasonal_cycle(obs_df, sim_dfs, variable, output_file, labels=None):
    """Plot seasonal cycle (multi-year monthly climatology) comparison"""
    setup_matplotlib()
    
    fig, ax = plt.subplots(figsize=(USABLE_WIDTH, USABLE_WIDTH * 0.5))
    
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    # Plot observation seasonal cycle
    if obs_df is not None and variable in obs_df.columns:
        obs_seasonal = calculate_seasonal_cycle(obs_df, variable)
        if obs_seasonal is not None and not obs_seasonal.empty:
            ax.plot(obs_seasonal["month"], obs_seasonal["value"],
                    color=COLORS["observed"], linewidth=2.0, marker="o", markersize=6,
                    label="Observed", linestyle="-", zorder=10)
            ax.fill_between(obs_seasonal["month"], 
                           obs_seasonal["value"] - obs_seasonal["std"],
                           obs_seasonal["value"] + obs_seasonal["std"],
                           color=COLORS["observed"], alpha=0.15)
    
    # Plot simulation seasonal cycles
    linestyles = ["-", "--", "-."]
    markers = ["s", "^", "D"]
    for i, (name, sim_df) in enumerate(sim_dfs.items()):
        if sim_df is not None and variable in sim_df.columns:
            sim_seasonal = calculate_seasonal_cycle(sim_df, variable)
            if sim_seasonal is not None and not sim_seasonal.empty:
                label = labels.get(name, name.capitalize()) if labels else name.capitalize()
                color = COLORS.get(name, f"C{i}")
                ax.plot(sim_seasonal["month"], sim_seasonal["value"],
                       color=color, linewidth=1.5, marker=markers[i % len(markers)], markersize=5,
                       label=label, linestyle=linestyles[i % len(linestyles)], alpha=0.85)
                ax.fill_between(sim_seasonal["month"],
                               sim_seasonal["value"] - sim_seasonal["std"],
                               sim_seasonal["value"] + sim_seasonal["std"],
                               color=color, alpha=0.1)
    
    ax.set_xlabel("Month", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel(variable, fontsize=FONT_SIZES["axis_label"])
    ax.set_title(f"{variable} - Seasonal Cycle (Multi-year Monthly Mean)", 
                 fontsize=FONT_SIZES["subtitle"], fontweight="bold")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names, fontsize=FONT_SIZES["tick_label"])
    ax.legend(loc="best", fontsize=FONT_SIZES["legend"], framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig.savefig(f"{output_file}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_file}.pdf", dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}.png, {output_file}.pdf")
    
    plt.close()


def plot_diurnal_cycle(obs_df, sim_dfs, variable, output_file, labels=None):
    """Plot diurnal cycle (multi-day hourly climatology) comparison"""
    setup_matplotlib()
    
    fig, ax = plt.subplots(figsize=(USABLE_WIDTH, USABLE_WIDTH * 0.5))
    
    # Plot observation diurnal cycle
    if obs_df is not None and variable in obs_df.columns:
        obs_diurnal = calculate_diurnal_cycle(obs_df, variable)
        if obs_diurnal is not None and not obs_diurnal.empty:
            ax.plot(obs_diurnal["hour"], obs_diurnal["value"],
                    color=COLORS["observed"], linewidth=2.0, marker="o", markersize=5,
                    label="Observed", linestyle="-", zorder=10)
            ax.fill_between(obs_diurnal["hour"], 
                           obs_diurnal["value"] - obs_diurnal["std"],
                           obs_diurnal["value"] + obs_diurnal["std"],
                           color=COLORS["observed"], alpha=0.15)
    
    # Plot simulation diurnal cycles
    linestyles = ["-", "--", "-."]
    markers = ["s", "^", "D"]
    for i, (name, sim_df) in enumerate(sim_dfs.items()):
        if sim_df is not None and variable in sim_df.columns:
            sim_diurnal = calculate_diurnal_cycle(sim_df, variable)
            if sim_diurnal is not None and not sim_diurnal.empty:
                label = labels.get(name, name.capitalize()) if labels else name.capitalize()
                color = COLORS.get(name, f"C{i}")
                ax.plot(sim_diurnal["hour"], sim_diurnal["value"],
                       color=color, linewidth=1.5, marker=markers[i % len(markers)], markersize=4,
                       label=label, linestyle=linestyles[i % len(linestyles)], alpha=0.85)
                ax.fill_between(sim_diurnal["hour"],
                               sim_diurnal["value"] - sim_diurnal["std"],
                               sim_diurnal["value"] + sim_diurnal["std"],
                               color=color, alpha=0.1)
    
    ax.set_xlabel("Hour of Day (Local Time)", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel(variable, fontsize=FONT_SIZES["axis_label"])
    ax.set_title(f"{variable} - Diurnal Cycle (Multi-day Hourly Mean)", 
                 fontsize=FONT_SIZES["subtitle"], fontweight="bold")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlim(-0.5, 23.5)
    ax.legend(loc="best", fontsize=FONT_SIZES["legend"], framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig.savefig(f"{output_file}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_file}.pdf", dpi=300, bbox_inches="tight")
    print(f"  Saved: {output_file}.png, {output_file}.pdf")
    
    plt.close()




def save_metrics_summary(all_results, output_dir, cal_start, cal_end, val_start, val_end):
    """
    Save comprehensive metrics summary to text file
    """
    summary_file = output_dir / 'metrics_summary.txt'
    
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CALIBRATION/VALIDATION METRICS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Time Ranges:\n")
        f.write(f"  Calibration: {cal_start} to {cal_end}\n")
        f.write(f"  Validation:  {val_start} to {val_end}\n\n")
        
        # Convert to DataFrame for easier processing
        results_df = pd.DataFrame(all_results)
        
        for variable in results_df['variable'].unique():
            f.write(f"\n{'='*80}\n")
            f.write(f"Variable: {variable}\n")
            f.write(f"{'='*80}\n")
            
            var_df = results_df[results_df['variable'] == variable]
            
            for period in ['calibration', 'validation']:
                period_df = var_df[var_df['period'] == period]
                if len(period_df) == 0:
                    continue
                
                f.write(f"\n--- {period.capitalize()} Period ---\n")
                f.write(f"{'Model':<20} {'RMSE':>10} {'PBIAS':>10} {'MAE':>10} {'R2':>10} {'N':>8}\n")
                f.write("-" * 70 + "\n")
                
                for _, row in period_df.iterrows():
                    model_name = row['model'].capitalize()
                    f.write(f"{model_name:<20} {row['RMSE']:>10.4f} {row['PBIAS']:>9.2f}% {row['MAE']:>10.4f} {row['R2']:>10.4f} {int(row['N']):>8}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("END OF SUMMARY\n")
        f.write(f"{'='*80}\n")
    
    print(f"  Saved: {summary_file}")
    return summary_file


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced calibration validation with period markers',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--obs', type=str, required=True,
                       help='Observation CSV file')
    parser.add_argument('--emulator_output', type=str, required=True,
                       help='Emulator-calibrated Noah-MP output CSV')
    parser.add_argument('--expert_output', type=str, required=True,
                       help='Expert-calibrated Noah-MP output CSV')
    parser.add_argument('--default_output', type=str, required=True,
                       help='Default parameter Noah-MP output CSV')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--variables', type=str, nargs='+', default=['SOIL_M', 'LH', 'HFX'],
                       help='Variables to validate')
    
    args = parser.parse_args()
    
    # Read time ranges from config (no command line override needed)
    cal_start = config.TIME_RANGE_CALIBRATION['start']
    cal_end = config.TIME_RANGE_CALIBRATION['end']
    val_start = config.TIME_RANGE_VALIDATION['start']
    val_end = config.TIME_RANGE_VALIDATION['end']
    
    print("="*70)
    print("Enhanced Calibration Validation")
    print("="*70)
    print(f"Calibration period: {cal_start} to {cal_end}")
    print(f"Validation period:  {val_start} to {val_end}")
    print(f"Variables: {args.variables}")
    print("="*70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    obs_df = read_observation_data(args.obs)
    emulator_df = parse_noahmp_output(args.emulator_output)
    expert_df = parse_noahmp_output(args.expert_output)
    default_df = parse_noahmp_output(args.default_output)
    
    if obs_df is None:
        print("Error: Cannot load observation data")
        return 1
    
    print(f"  Observation data loaded: {len(obs_df)} records")
    print(f"    Available columns: {list(obs_df.columns)}")
    if 'timestamp' in obs_df.columns:
        print(f"    Time range: {obs_df['timestamp'].min()} to {obs_df['timestamp'].max()}")
    
    sim_dfs = {
        'emulator': emulator_df,
        'expert': expert_df,
        'default': default_df
    }
    
    labels = {
        'emulator': 'Emulator-calibrated',
        'expert': 'Expert-calibrated',
        'default': 'Default'
    }
    
    # Process each variable
    all_results = []
    
    for variable in args.variables:
        print(f"\nProcessing {variable}...")
        
        # Check if variable exists in observation data
        if variable not in obs_df.columns:
            print(f"  WARNING: Variable '{variable}' not found in observation data. Skipping...")
            continue
        
        # Calculate metrics by period for each model
        all_metrics = {}
        for model_name, sim_df in sim_dfs.items():
            if sim_df is not None and variable in sim_df.columns:
                metrics_result = calculate_metrics_by_period(
                    obs_df, sim_df, variable,
                    cal_start, cal_end, val_start, val_end
                )
                
                if metrics_result is not None:
                    all_metrics[model_name] = metrics_result
                    
                    # Store results
                    for period in ['calibration', 'validation', 'full']:
                        metrics = metrics_result[period]
                        all_results.append({
                            'variable': variable,
                            'model': model_name,
                            'period': period,
                            **metrics
                        })
                    
                    # Print summary for this model
                    cal_m = metrics_result['calibration']
                    val_m = metrics_result['validation']
                    print(f"  {model_name.capitalize()}:")
                    print(f"    Calibration - RMSE: {cal_m['RMSE']:.4f}, PBIAS: {cal_m['PBIAS']:.2f}%, MAE: {cal_m['MAE']:.4f}, N: {cal_m['N']}")
                    print(f"    Validation  - RMSE: {val_m['RMSE']:.4f}, PBIAS: {val_m['PBIAS']:.2f}%, MAE: {val_m['MAE']:.4f}, N: {val_m['N']}")
        
        # Generate plots
        # 1. Time series with period labels (aggregated to daily)
        plot_timeseries_with_periods(
            obs_df, sim_dfs, variable,
            str(output_dir / f'timeseries_{variable}'),
            cal_start, cal_end, val_start, val_end,
            labels=labels
        )
        
        # 2. Seasonal cycle plot (multi-year monthly mean)
        plot_seasonal_cycle(
            obs_df, sim_dfs, variable,
            str(output_dir / f'seasonal_{variable}'),
            labels=labels
        )
        
        # 3. Diurnal cycle plot (multi-day hourly mean)
        plot_diurnal_cycle(
            obs_df, sim_dfs, variable,
            str(output_dir / f'diurnal_{variable}'),
            labels=labels
        )
        
        # 4. Metrics comparison bar chart (RMSE, PBIAS, MAE)
        if all_metrics:
            plot_metrics_comparison(
                all_metrics, variable,
                str(output_dir / f'metrics_{variable}')
            )
        
        # 3. Scatter plots by period for each model
        for model_name, sim_df in sim_dfs.items():
            if sim_df is not None and variable in sim_df.columns and variable in obs_df.columns:
                plot_scatter_by_period(
                    obs_df, sim_df, variable,
                    str(output_dir / f'scatter_{variable}_{model_name}'),
                    cal_start, cal_end, val_start, val_end,
                    model_name=labels.get(model_name, model_name)
                )
    
    # Save metrics to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Reorder columns for clarity
        col_order = ['variable', 'model', 'period', 'RMSE', 'PBIAS', 'MAE', 'BIAS', 'R2', 'NSE', 'N']
        available_cols = [c for c in col_order if c in results_df.columns]
        results_df = results_df[available_cols]
        
        csv_file = output_dir / 'validation_metrics_by_period.csv'
        results_df.to_csv(csv_file, index=False)
        print(f"\nMetrics saved to: {csv_file}")
        
        # Save text summary
        save_metrics_summary(all_results, output_dir, cal_start, cal_end, val_start, val_end)
        
        # Print summary
        print("\n" + "="*70)
        print("VALIDATION RESULTS SUMMARY")
        print("="*70)
        
        for variable in args.variables:
            if variable not in results_df['variable'].values:
                continue
            print(f"\n{variable}:")
            var_df = results_df[results_df['variable'] == variable]
            for model in ['default', 'expert', 'emulator']:
                model_df = var_df[var_df['model'] == model]
                if not model_df.empty:
                    cal_row = model_df[model_df['period'] == 'calibration'].iloc[0] if len(model_df[model_df['period'] == 'calibration']) > 0 else None
                    val_row = model_df[model_df['period'] == 'validation'].iloc[0] if len(model_df[model_df['period'] == 'validation']) > 0 else None
                    
                    print(f"  {model.capitalize():20s}")
                    if cal_row is not None:
                        print(f"    Calibration:  RMSE={cal_row['RMSE']:.4f}, PBIAS={cal_row['PBIAS']:.2f}%, MAE={cal_row['MAE']:.4f}")
                    if val_row is not None:
                        print(f"    Validation:   RMSE={val_row['RMSE']:.4f}, PBIAS={val_row['PBIAS']:.2f}%, MAE={val_row['MAE']:.4f}")
    else:
        print("\nWARNING: No metrics were calculated. Check variable names and data availability.")
    
    print("\n" + "="*70)
    print("Validation completed!")
    print(f"Results saved to: {output_dir}")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
