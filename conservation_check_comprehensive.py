"""
Comprehensive Conservation Check Module for Full Model

This module performs detailed energy and water conservation checks using
ALL predicted variables from the comprehensive LSTM emulator.

Checks:
1. Full energy conservation with all component fluxes
2. Full water conservation with all water cycle variables
3. Component-level validation (canopy, ground, bare ground)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ComprehensiveConservationChecker:
    """
    Checks full energy and water conservation using all predicted variables
    """

    def __init__(self, timestep_hours=24):  # Daily data
        """
        Initialize checker

        Args:
            timestep_hours: Timestep in hours (24 for daily data)
        """
        self.timestep_hours = timestep_hours
        self.timestep_seconds = timestep_hours * 3600

    def check_full_energy_conservation(self, predictions, forcing=None, verbose=True):
        """
        Check full energy conservation using all predicted energy variables

        Simplified Conservation Check: residual = input - output
        - Input: FSA - FIRA (Net radiation)
        - Output: HFX + LH + GRDFLX (Turbulent + Ground fluxes)
        - Residual: (FSA - FIRA) - (HFX + LH + GRDFLX)

        Args:
            predictions: Dict with predicted variables (as numpy arrays)
            forcing: Dict with forcing data (optional)
            verbose: Print detailed results

        Returns:
            dict with conservation statistics including input/output components
        """
        # Input components
        FSA = predictions['FSA']  # Total absorbed SW radiation
        FIRA = predictions['FIRA']  # Total net LW radiation to atmosphere

        # Output components
        HFX = predictions['HFX']  # Total sensible heat
        LH = predictions['LH']  # Total latent heat
        GRDFLX = predictions['GRDFLX']  # Heat flux into soil

        # Conservation check: residual = input - output
        total_input = FSA - FIRA
        total_output = HFX + LH + GRDFLX
        energy_residual = total_input - total_output

        # Store component time series for visualization
        input_components = {
            'FSA': FSA,
            'FIRA': -FIRA  # Negative because it's subtracted
        }
        output_components = {
            'HFX': HFX,
            'LH': LH,
            'GRDFLX': GRDFLX
        }

        # Component checks (if available)
        component_checks = {}

        # Check canopy energy balance: SAV = IRC + SHC + EVC
        if all(k in predictions for k in ['SAV', 'IRC', 'SHC', 'EVC']):
            canopy_input = predictions['SAV']
            canopy_output = predictions['IRC'] + predictions['SHC'] + predictions['EVC']
            canopy_residual = canopy_input - canopy_output
            component_checks['canopy'] = {
                'mean_residual': np.mean(canopy_residual),
                'std_residual': np.std(canopy_residual),
                'residual': canopy_residual
            }

        # Check ground energy balance: SAG = IRG + SHG + EVG + GHV
        if all(k in predictions for k in ['SAG', 'IRG', 'SHG', 'EVG', 'GHV']):
            ground_input = predictions['SAG']
            ground_output = predictions['IRG'] + predictions['SHG'] + predictions['EVG'] + predictions['GHV']
            ground_residual = ground_input - ground_output
            component_checks['ground'] = {
                'mean_residual': np.mean(ground_residual),
                'std_residual': np.std(ground_residual),
                'residual': ground_residual
            }

        # Statistics
        stats = {
            'mean_input': np.mean(total_input),
            'mean_output': np.mean(total_output),
            'mean_residual': np.mean(energy_residual),
            'std_residual': np.std(energy_residual),
            'max_abs_residual': np.max(np.abs(energy_residual)),
            'rmse': np.sqrt(np.mean(energy_residual**2)),
            'relative_error_pct': 100 * np.mean(np.abs(energy_residual)) / (np.abs(np.mean(total_input)) + 1e-10),
            'residual_timeseries': energy_residual,
            'input_timeseries': total_input,
            'output_timeseries': total_output,
            'input_components': input_components,
            'output_components': output_components,
            'component_checks': component_checks
        }

        if verbose:
            print("="*80)
            print("ENERGY CONSERVATION CHECK")
            print("="*80)
            print(f"Conservation: residual = input - output")
            print(f"\nMean values (W/m²):")
            print(f"  Input (FSA - FIRA): {stats['mean_input']:.2f}")
            print(f"  Output (HFX + LH + GRDFLX): {stats['mean_output']:.2f}")
            print(f"\nConservation residual:")
            print(f"  Mean: {stats['mean_residual']:.2f} W/m²")
            print(f"  Std: {stats['std_residual']:.2f} W/m²")
            print(f"  Max absolute: {stats['max_abs_residual']:.2f} W/m²")
            print(f"  RMSE: {stats['rmse']:.2f} W/m²")
            print(f"  Relative error: {stats['relative_error_pct']:.2f}%")

            if component_checks:
                print(f"\nComponent-level checks:")
                for comp_name, comp_stats in component_checks.items():
                    print(f"  {comp_name.capitalize()} energy balance:")
                    print(f"    Mean residual: {comp_stats['mean_residual']:.2f} W/m²")
                    print(f"    Std residual: {comp_stats['std_residual']:.2f} W/m²")

            print("="*80)

        return stats

    def check_full_water_conservation(self, predictions, forcing, verbose=True):
        """
        Check full water conservation using all predicted water variables

        Simplified Conservation Check: residual = input - output - change
        - Input: Precipitation
        - Output: ET + Runoff
        - Change: ΔStorage
        - Residual: Precipitation - (ET + Runoff + ΔStorage)

        Args:
            predictions: Dict with predicted variables
            forcing: Dict with forcing data (must include RAINRATE)
            verbose: Print detailed results

        Returns:
            dict with conservation statistics including input/output components
        """
        # Input: Precipitation (mm/day)
        precip = forcing['RAINRATE']

        # Output: ET components (mm/s -> mm/day)
        ecan = predictions['ECAN'] * self.timestep_seconds
        etran = predictions['ETRAN'] * self.timestep_seconds
        edir = predictions['EDIR'] * self.timestep_seconds
        total_et = ecan + etran + edir

        # Output: Runoff rates (mm/day) - now predicted directly as rates
        # Note: Variable names changed from UGDRNOFF/SFCRNOFF (accumulated)
        # to UGDRNOFF_RATE/SFCRNOFF_RATE (daily rates) during preprocessing
        ugdrnoff_rate = predictions['UGDRNOFF_RATE']
        sfcrnoff_rate = predictions['SFCRNOFF_RATE']
        total_runoff = ugdrnoff_rate + sfcrnoff_rate

        # Change: Storage variables (mm)
        canliq = predictions.get('CANLIQ', np.zeros_like(precip))
        canice = predictions.get('CANICE', np.zeros_like(precip))
        sneqv = predictions.get('SNEQV', np.zeros_like(precip))

        total_storage = canliq + canice + sneqv

        # Storage change (mm/day)
        storage_change = np.diff(total_storage, prepend=total_storage[0])

        # Conservation check: residual = input - output - change
        total_input = precip
        total_output = total_et + total_runoff
        water_residual = total_input - total_output - storage_change

        # Store component time series for visualization
        input_components = {
            'Precipitation': precip
        }
        output_components = {
            'ET (Canopy)': ecan,
            'ET (Transpiration)': etran,
            'ET (Soil)': edir,
            'Runoff (Underground)': ugdrnoff_rate,
            'Runoff (Surface)': sfcrnoff_rate
        }
        change_components = {
            'ΔStorage (Canopy Liquid)': np.diff(canliq, prepend=canliq[0]),
            'ΔStorage (Canopy Ice)': np.diff(canice, prepend=canice[0]),
            'ΔStorage (Snow)': np.diff(sneqv, prepend=sneqv[0])
        }

        # Statistics
        stats = {
            'cumulative_input': np.sum(total_input),
            'cumulative_output': np.sum(total_output),
            'cumulative_change': np.sum(storage_change),
            'cumulative_precip': np.sum(precip),
            'cumulative_et': np.sum(total_et),
            'cumulative_runoff': np.sum(total_runoff),
            'total_storage_change': total_storage[-1] - total_storage[0],
            'mean_input': np.mean(total_input),
            'mean_output': np.mean(total_output),
            'mean_change': np.mean(storage_change),
            'mean_precip': np.mean(precip),
            'mean_et': np.mean(total_et),
            'mean_runoff': np.mean(total_runoff),
            'mean_residual': np.mean(water_residual),
            'std_residual': np.std(water_residual),
            'max_abs_residual': np.max(np.abs(water_residual)),
            'rmse': np.sqrt(np.mean(water_residual**2)),
            'residual_timeseries': water_residual,
            'input_timeseries': total_input,
            'output_timeseries': total_output,
            'change_timeseries': storage_change,
            'input_components': input_components,
            'output_components': output_components,
            'change_components': change_components,
            'et_components': {
                'canopy_evap': np.sum(ecan),
                'transpiration': np.sum(etran),
                'soil_evap': np.sum(edir)
            },
            'runoff_components': {
                'underground': np.sum(ugdrnoff_rate),
                'surface': np.sum(sfcrnoff_rate)
            }
        }

        if verbose:
            print("="*80)
            print("WATER CONSERVATION CHECK")
            print("="*80)
            print(f"Conservation: residual = input - output - change")
            print(f"\nMean daily rates (mm/day):")
            print(f"  Input (Precipitation): {stats['mean_input']:.4f}")
            print(f"  Output (ET + Runoff): {stats['mean_output']:.4f}")
            print(f"    - ET: {stats['mean_et']:.4f}")
            print(f"    - Runoff: {stats['mean_runoff']:.4f}")
            print(f"  Change (ΔStorage): {stats['mean_change']:.4f}")
            print(f"\nCumulative totals (mm):")
            print(f"  Input: {stats['cumulative_input']:.2f}")
            print(f"  Output: {stats['cumulative_output']:.2f}")
            print(f"    - ET: {stats['cumulative_et']:.2f}")
            print(f"      * Canopy evaporation: {stats['et_components']['canopy_evap']:.2f}")
            print(f"      * Transpiration: {stats['et_components']['transpiration']:.2f}")
            print(f"      * Soil evaporation: {stats['et_components']['soil_evap']:.2f}")
            print(f"    - Runoff: {stats['cumulative_runoff']:.2f}")
            print(f"      * Underground: {stats['runoff_components']['underground']:.2f}")
            print(f"      * Surface: {stats['runoff_components']['surface']:.2f}")
            print(f"  Change: {stats['cumulative_change']:.2f}")
            print(f"\nConservation residual:")
            print(f"  Mean: {stats['mean_residual']:.4f} mm/day")
            print(f"  Std: {stats['std_residual']:.4f} mm/day")
            print(f"  Max absolute: {stats['max_abs_residual']:.4f} mm/day")
            print(f"  RMSE: {stats['rmse']:.4f} mm/day")
            print("="*80)

        return stats

    def plot_improved_conservation(self, energy_stats, water_stats, save_dir=None):
        """
        Create improved conservation plots with area plots and residuals

        Creates two grouped figures (energy and water), each with 3 subplots:
        1. Area plot for input components with total output line
        2. Area plot for output components with total input line
        3. Residuals plot

        Args:
            energy_stats: Output from check_full_energy_conservation
            water_stats: Output from check_full_water_conservation
            save_dir: Directory to save plots

        Returns:
            list of figure objects
        """
        figures = []

        # ===== ENERGY CONSERVATION FIGURE =====
        fig_energy, axes = plt.subplots(3, 1, figsize=(14, 12))

        # Plot 1: Input components area plot with total output line
        ax = axes[0]
        time_steps = np.arange(len(energy_stats['input_timeseries']))

        # Stack input components for area plot
        input_comps = energy_stats['input_components']
        input_values = np.array([input_comps[k] for k in input_comps.keys()])
        input_labels = list(input_comps.keys())

        # Create positive stacked area (only positive values)
        positive_mask = input_values > 0
        cumulative = np.zeros(len(time_steps))
        colors_input = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

        for i, (values, label) in enumerate(zip(input_values, input_labels)):
            positive_vals = np.maximum(values, 0)
            ax.fill_between(time_steps, cumulative, cumulative + positive_vals,
                           alpha=0.7, label=label, color=colors_input[i % len(colors_input)])
            cumulative += positive_vals

        # Add negative components separately
        cumulative_neg = np.zeros(len(time_steps))
        for i, (values, label) in enumerate(zip(input_values, input_labels)):
            negative_vals = np.minimum(values, 0)
            if np.any(negative_vals < 0):
                ax.fill_between(time_steps, cumulative_neg, cumulative_neg + negative_vals,
                               alpha=0.7, label=label + ' (neg)', color=colors_input[i % len(colors_input)],
                               hatch='///')
                cumulative_neg += negative_vals

        # Add total output line
        ax.plot(time_steps, energy_stats['output_timeseries'],
               'r-', linewidth=2, label='Total Output', alpha=0.8)

        ax.set_ylabel('Energy Flux (W/m²)', fontsize=11)
        ax.set_title('Energy Input Components vs Total Output', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Plot 2: Output components area plot with total input line
        ax = axes[1]

        # Stack output components for area plot
        output_comps = energy_stats['output_components']
        output_values = np.array([output_comps[k] for k in output_comps.keys()])
        output_labels = list(output_comps.keys())

        cumulative = np.zeros(len(time_steps))
        colors_output = ['#ffb366', '#ff6666', '#cc99ff']

        for i, (values, label) in enumerate(zip(output_values, output_labels)):
            ax.fill_between(time_steps, cumulative, cumulative + values,
                           alpha=0.7, label=label, color=colors_output[i % len(colors_output)])
            cumulative += values

        # Add total input line
        ax.plot(time_steps, energy_stats['input_timeseries'],
               'b-', linewidth=2, label='Total Input', alpha=0.8)

        ax.set_ylabel('Energy Flux (W/m²)', fontsize=11)
        ax.set_title('Energy Output Components vs Total Input', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Plot 3: Residuals
        ax = axes[2]
        residual = energy_stats['residual_timeseries']

        ax.plot(time_steps, residual, 'k-', linewidth=1.5, alpha=0.7, label='Residual')
        ax.fill_between(time_steps, 0, residual, where=(residual >= 0),
                       alpha=0.3, color='green', label='Positive')
        ax.fill_between(time_steps, 0, residual, where=(residual < 0),
                       alpha=0.3, color='red', label='Negative')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)

        # Add statistics text box
        stats_text = f'RMSE: {energy_stats["rmse"]:.2f} W/m²\n'
        stats_text += f'Mean: {energy_stats["mean_residual"]:.2f} W/m²\n'
        stats_text += f'Std: {energy_stats["std_residual"]:.2f} W/m²'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Time Steps', fontsize=11)
        ax.set_ylabel('Residual (W/m²)', fontsize=11)
        ax.set_title('Energy Conservation Residuals (Input - Output)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        figures.append(fig_energy)

        if save_dir:
            fig_energy.savefig(Path(save_dir) / 'energy_conservation_improved.png',
                             dpi=300, bbox_inches='tight')

        # ===== WATER CONSERVATION FIGURE =====
        fig_water, axes = plt.subplots(3, 1, figsize=(14, 12))

        # Plot 1: Input components area plot with total output+change line
        ax = axes[0]
        time_steps = np.arange(len(water_stats['input_timeseries']))

        # Stack input components
        input_comps = water_stats['input_components']
        cumulative = np.zeros(len(time_steps))
        colors_input = ['#66b3ff', '#99ccff', '#cceeff']

        for i, (label, values) in enumerate(input_comps.items()):
            ax.fill_between(time_steps, cumulative, cumulative + values,
                           alpha=0.7, label=label, color=colors_input[i % len(colors_input)])
            cumulative += values

        # Add total output+change line
        total_out_change = water_stats['output_timeseries'] + water_stats['change_timeseries']
        ax.plot(time_steps, total_out_change,
               'r-', linewidth=2, label='Total Output + ΔStorage', alpha=0.8)

        ax.set_ylabel('Water Flux (mm/day)', fontsize=11)
        ax.set_title('Water Input vs Total Output + ΔStorage', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Plot 2: Output+Change components area plot with total input line
        ax = axes[1]

        # Stack output components
        output_comps = water_stats['output_components']
        cumulative = np.zeros(len(time_steps))
        colors_output = ['#90ee90', '#98fb98', '#00ff00', '#32cd32', '#228b22']

        for i, (label, values) in enumerate(output_comps.items()):
            ax.fill_between(time_steps, cumulative, cumulative + values,
                           alpha=0.7, label=label, color=colors_output[i % len(colors_output)])
            cumulative += values

        # Add change components
        change_comps = water_stats['change_components']
        colors_change = ['#ffb366', '#ff9966', '#ff7f50']

        for i, (label, values) in enumerate(change_comps.items()):
            ax.fill_between(time_steps, cumulative, cumulative + values,
                           alpha=0.7, label=label, color=colors_change[i % len(colors_change)])
            cumulative += values

        # Add total input line
        ax.plot(time_steps, water_stats['input_timeseries'],
               'b-', linewidth=2, label='Total Input', alpha=0.8)

        ax.set_ylabel('Water Flux (mm/day)', fontsize=11)
        ax.set_title('Water Output + ΔStorage Components vs Total Input', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)

        # Plot 3: Residuals
        ax = axes[2]
        residual = water_stats['residual_timeseries']

        ax.plot(time_steps, residual, 'k-', linewidth=1.5, alpha=0.7, label='Residual')
        ax.fill_between(time_steps, 0, residual, where=(residual >= 0),
                       alpha=0.3, color='green', label='Positive')
        ax.fill_between(time_steps, 0, residual, where=(residual < 0),
                       alpha=0.3, color='red', label='Negative')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)

        # Add statistics text box
        stats_text = f'RMSE: {water_stats["rmse"]:.4f} mm/day\n'
        stats_text += f'Mean: {water_stats["mean_residual"]:.4f} mm/day\n'
        stats_text += f'Std: {water_stats["std_residual"]:.4f} mm/day'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        ax.set_xlabel('Time Steps', fontsize=11)
        ax.set_ylabel('Residual (mm/day)', fontsize=11)
        ax.set_title('Water Conservation Residuals (Input - Output - ΔStorage)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        figures.append(fig_water)

        if save_dir:
            fig_water.savefig(Path(save_dir) / 'water_conservation_improved.png',
                            dpi=300, bbox_inches='tight')

        return figures

    def plot_comprehensive_conservation(self, energy_stats, water_stats, save_dir=None):
        """
        Create comprehensive conservation diagnostic plots

        Args:
            energy_stats: Output from check_full_energy_conservation
            water_stats: Output from check_full_water_conservation
            save_dir: Directory to save plots

        Returns:
            list of figure objects
        """
        figures = []

        # Figure 1: Energy conservation
        fig1, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Energy balance time series
        ax = axes[0, 0]
        ax.plot(energy_stats['net_radiation_timeseries'], label='Net Radiation', alpha=0.7)
        ax.set_ylabel('W/m²')
        ax.set_title('Net Radiation (FSA - FIRA)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Energy residual time series
        ax = axes[0, 1]
        ax.plot(energy_stats['residual_timeseries'], color='red', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_ylabel('W/m²')
        ax.set_title(f'Energy Residual (Mean: {energy_stats["mean_residual"]:.2f} W/m²)')
        ax.grid(True, alpha=0.3)

        # Energy residual histogram
        ax = axes[1, 0]
        ax.hist(energy_stats['residual_timeseries'], bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Energy Residual (W/m²)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Energy Residuals')
        ax.grid(True, alpha=0.3, axis='y')

        # Conservation assessment
        ax = axes[1, 1]
        ax.axis('off')
        assessment_text = f"""
ENERGY CONSERVATION ASSESSMENT

Mean Residual: {energy_stats['mean_residual']:.2f} W/m²
Std Residual: {energy_stats['std_residual']:.2f} W/m²
RMSE: {energy_stats['rmse']:.2f} W/m²
Relative Error: {energy_stats['relative_error_pct']:.2f}%

Assessment:
"""
        if energy_stats['rmse'] < 10:
            assessment_text += "✓ EXCELLENT conservation"
        elif energy_stats['rmse'] < 30:
            assessment_text += "✓ GOOD conservation"
        elif energy_stats['rmse'] < 50:
            assessment_text += "⚠ ACCEPTABLE conservation"
        else:
            assessment_text += "✗ POOR conservation"

        ax.text(0.1, 0.5, assessment_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        figures.append(fig1)

        if save_dir:
            fig1.savefig(Path(save_dir) / 'energy_conservation_comprehensive.png',
                        dpi=300, bbox_inches='tight')

        # Figure 2: Water conservation
        fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Water balance components
        ax = axes[0, 0]
        components = ['Precipitation', 'ET', 'Runoff']
        values = [water_stats['cumulative_precip'], water_stats['cumulative_et'],
                 water_stats['cumulative_runoff']]
        colors = ['blue', 'green', 'red']
        ax.bar(components, values, alpha=0.7, color=colors, edgecolor='black')
        ax.set_ylabel('Cumulative (mm)')
        ax.set_title('Water Balance Components')
        ax.grid(True, alpha=0.3, axis='y')

        # ET breakdown
        ax = axes[0, 1]
        et_labels = ['Canopy\nEvap', 'Trans-\npiration', 'Soil\nEvap']
        et_values = [water_stats['et_components']['canopy_evap'],
                    water_stats['et_components']['transpiration'],
                    water_stats['et_components']['soil_evap']]
        ax.bar(et_labels, et_values, alpha=0.7, color=['lightblue', 'lightgreen', 'tan'],
              edgecolor='black')
        ax.set_ylabel('Cumulative ET (mm)')
        ax.set_title('ET Component Breakdown')
        ax.grid(True, alpha=0.3, axis='y')

        # Water residual time series
        ax = axes[1, 0]
        ax.plot(water_stats['residual_timeseries'], color='red', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('mm/day')
        ax.set_title(f'Water Residual (Mean: {water_stats["mean_residual"]:.4f} mm/day)')
        ax.grid(True, alpha=0.3)

        # Water conservation assessment
        ax = axes[1, 1]
        ax.axis('off')
        water_assessment = f"""
WATER CONSERVATION ASSESSMENT

Mean Residual: {water_stats['mean_residual']:.4f} mm/day
Std Residual: {water_stats['std_residual']:.4f} mm/day
RMSE: {water_stats['rmse']:.4f} mm/day

Cumulative Balance (mm):
  In: {water_stats['cumulative_precip']:.2f}
  Out: {water_stats['cumulative_et'] + water_stats['cumulative_runoff']:.2f}
  ΔStorage: {water_stats['total_storage_change']:.2f}

Assessment:
"""
        if water_stats['rmse'] < 0.1:
            water_assessment += "✓ EXCELLENT conservation"
        elif water_stats['rmse'] < 0.5:
            water_assessment += "✓ GOOD conservation"
        elif water_stats['rmse'] < 1.0:
            water_assessment += "⚠ ACCEPTABLE conservation"
        else:
            water_assessment += "✗ POOR conservation"

        ax.text(0.1, 0.5, water_assessment, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.tight_layout()
        figures.append(fig2)

        if save_dir:
            fig2.savefig(Path(save_dir) / 'water_conservation_comprehensive.png',
                        dpi=300, bbox_inches='tight')

        return figures


def dict_from_arrays(predictions_array, target_var_names):
    """
    Convert predictions array to dictionary

    Args:
        predictions_array: (n_timesteps, n_vars) array
        target_var_names: List of variable names

    Returns:
        dict mapping var_name to array
    """
    if predictions_array.ndim == 3:
        predictions_array = predictions_array[0]  # Remove batch dimension

    return {name: predictions_array[:, i] for i, name in enumerate(target_var_names)}


if __name__ == '__main__':
    print("Comprehensive Conservation Check Module")
    print("="*80)
    print("This module checks full energy and water conservation using")
    print("ALL predicted variables from the comprehensive LSTM emulator.")
    print("\nUsage:")
    print("  from conservation_check_comprehensive import ComprehensiveConservationChecker")
    print("  checker = ComprehensiveConservationChecker()")
    print("  energy_stats = checker.check_full_energy_conservation(predictions)")
    print("  water_stats = checker.check_full_water_conservation(predictions, forcing)")
    print("="*80)
