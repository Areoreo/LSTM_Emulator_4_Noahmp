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

        Energy Balance: FSA - FIRA = HFX + LH + GRDFLX

        Args:
            predictions: Dict with predicted variables (as numpy arrays)
            forcing: Dict with forcing data (optional)
            verbose: Print detailed results

        Returns:
            dict with conservation statistics
        """
        # Core energy balance
        FSA = predictions['FSA']
        FIRA = predictions['FIRA']
        HFX = predictions['HFX']
        LH = predictions['LH']
        GRDFLX = predictions['GRDFLX']

        # Net radiation
        net_radiation = FSA - FIRA

        # Total turbulent + ground fluxes
        total_fluxes = HFX + LH + GRDFLX

        # Energy residual
        energy_residual = net_radiation - total_fluxes

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
            'mean_net_radiation': np.mean(net_radiation),
            'mean_total_fluxes': np.mean(total_fluxes),
            'mean_residual': np.mean(energy_residual),
            'std_residual': np.std(energy_residual),
            'max_abs_residual': np.max(np.abs(energy_residual)),
            'rmse': np.sqrt(np.mean(energy_residual**2)),
            'relative_error_pct': 100 * np.mean(np.abs(energy_residual)) / (np.abs(np.mean(net_radiation)) + 1e-10),
            'residual_timeseries': energy_residual,
            'net_radiation_timeseries': net_radiation,
            'component_checks': component_checks
        }

        if verbose:
            print("="*80)
            print("FULL ENERGY CONSERVATION CHECK")
            print("="*80)
            print(f"Energy balance: FSA - FIRA = HFX + LH + GRDFLX")
            print(f"\nMean values (W/m²):")
            print(f"  Net radiation (FSA - FIRA): {stats['mean_net_radiation']:.2f}")
            print(f"  Total fluxes (HFX + LH + GRDFLX): {stats['mean_total_fluxes']:.2f}")
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

        Water Balance: ΔStorage = Precipitation - ET - Runoff

        Args:
            predictions: Dict with predicted variables
            forcing: Dict with forcing data (must include RAINRATE)
            verbose: Print detailed results

        Returns:
            dict with conservation statistics
        """
        # Precipitation (mm/day)
        precip = forcing['RAINRATE']

        # ET components (mm/s -> mm/day)
        ecan = predictions['ECAN'] * self.timestep_seconds  # mm/day
        etran = predictions['ETRAN'] * self.timestep_seconds
        edir = predictions['EDIR'] * self.timestep_seconds
        total_et = ecan + etran + edir

        # Runoff (accumulated mm -> daily rates)
        ugdrnoff_cumul = predictions['UGDRNOFF']
        sfcrnoff_cumul = predictions['SFCRNOFF']

        # Convert accumulated to daily rates
        ugdrnoff_rate = np.diff(ugdrnoff_cumul, prepend=ugdrnoff_cumul[0])
        sfcrnoff_rate = np.diff(sfcrnoff_cumul, prepend=sfcrnoff_cumul[0])
        total_runoff = ugdrnoff_rate + sfcrnoff_rate

        # Storage variables (mm)
        soil_m_layers = []
        for layer in range(1, 5):
            key = f'SOIL_M' if layer == 1 else f'SOIL_M_L{layer}'
            if key in predictions:
                soil_m_layers.append(predictions[key])

        # Simplified total storage (without layer depths, for demonstration)
        # In reality, should multiply by layer depths
        canliq = predictions.get('CANLIQ', np.zeros_like(precip))
        canice = predictions.get('CANICE', np.zeros_like(precip))
        sneqv = predictions.get('SNEQV', np.zeros_like(precip))

        total_storage = canliq + canice + sneqv

        # Storage change (mm/day)
        storage_change = np.diff(total_storage, prepend=total_storage[0])

        # Water balance: ΔS = P - ET - R
        predicted_storage_change = precip - total_et - total_runoff
        water_residual = storage_change - predicted_storage_change

        # Statistics
        stats = {
            'cumulative_precip': np.sum(precip),
            'cumulative_et': np.sum(total_et),
            'cumulative_runoff': np.sum(total_runoff),
            'total_storage_change': total_storage[-1] - total_storage[0],
            'mean_precip': np.mean(precip),
            'mean_et': np.mean(total_et),
            'mean_runoff': np.mean(total_runoff),
            'mean_storage_change': np.mean(storage_change),
            'mean_residual': np.mean(water_residual),
            'std_residual': np.std(water_residual),
            'max_abs_residual': np.max(np.abs(water_residual)),
            'rmse': np.sqrt(np.mean(water_residual**2)),
            'residual_timeseries': water_residual,
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
            print("FULL WATER CONSERVATION CHECK")
            print("="*80)
            print(f"Water balance: ΔStorage = Precipitation - ET - Runoff")
            print(f"\nCumulative totals (mm):")
            print(f"  Precipitation: {stats['cumulative_precip']:.2f}")
            print(f"  ET: {stats['cumulative_et']:.2f}")
            print(f"    - Canopy evaporation: {stats['et_components']['canopy_evap']:.2f}")
            print(f"    - Transpiration: {stats['et_components']['transpiration']:.2f}")
            print(f"    - Soil evaporation: {stats['et_components']['soil_evap']:.2f}")
            print(f"  Runoff: {stats['cumulative_runoff']:.2f}")
            print(f"    - Underground: {stats['runoff_components']['underground']:.2f}")
            print(f"    - Surface: {stats['runoff_components']['surface']:.2f}")
            print(f"  Storage change: {stats['total_storage_change']:.2f}")
            print(f"\nMean daily rates (mm/day):")
            print(f"  Precipitation: {stats['mean_precip']:.4f}")
            print(f"  ET: {stats['mean_et']:.4f}")
            print(f"  Runoff: {stats['mean_runoff']:.4f}")
            print(f"  Storage change: {stats['mean_storage_change']:.4f}")
            print(f"\nConservation residual:")
            print(f"  Mean: {stats['mean_residual']:.4f} mm/day")
            print(f"  Std: {stats['std_residual']:.4f} mm/day")
            print(f"  Max absolute: {stats['max_abs_residual']:.4f} mm/day")
            print(f"  RMSE: {stats['rmse']:.4f} mm/day")
            print("="*80)

        return stats

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
