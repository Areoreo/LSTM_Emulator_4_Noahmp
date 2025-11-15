"""
Extract forcing data from simulation results and save as standalone NetCDF file
This creates a template forcing file that can be used for inference
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import config_forward as config


def extract_forcing_data(sample_idx, output_file, data_dir='data/raw/sim_results'):
    """
    Extract forcing variables from a simulation sample and save as NetCDF

    Args:
        sample_idx: Sample index (1-based)
        output_file: Output NetCDF file path
        data_dir: Directory containing simulation results
    """
    sim_path = Path(data_dir) / f'sample_{sample_idx}' / 'output' / '201507301730.LDASOUT_DOMAIN1'

    if not sim_path.exists():
        raise FileNotFoundError(f"Simulation file not found: {sim_path}")

    print(f"Loading simulation from: {sim_path}")

    # Load with xarray
    ds = xr.open_dataset(sim_path)

    # Get time information
    time_strings = [s.decode() if isinstance(s, bytes) else s for s in ds['Times'].values]
    time_strings = [s.replace('_', ' ') for s in time_strings]
    times = pd.to_datetime(time_strings)

    print(f"\nTime range: {times[0]} to {times[-1]}")
    print(f"Number of timesteps: {len(times)}")

    # Extract forcing variables
    forcing_vars = {}
    print("\nExtracting forcing variables:")

    for var_config in config.FORCING_VARIABLES:
        var_name = var_config['name']

        if var_name not in ds:
            print(f"  Warning: Variable '{var_name}' not found in dataset")
            continue

        var_data = ds[var_name]

        # Squeeze spatial dimensions (assuming single point)
        for dim in ['south_north', 'west_east']:
            if dim in var_data.dims:
                var_data = var_data.isel({dim: 0})

        forcing_vars[var_name] = var_data
        print(f"  ✓ {var_name}: {var_data.shape}")

    ds.close()

    # Create new dataset with only forcing variables
    forcing_ds = xr.Dataset(
        data_vars={name: (['time'], data.values) for name, data in forcing_vars.items()},
        coords={'time': times}
    )

    # Add metadata
    forcing_ds.attrs['title'] = 'Forcing data for LSTM forward model'
    forcing_ds.attrs['source_sample'] = f'sample_{sample_idx}'
    forcing_ds.attrs['created_by'] = 'extract_forcing_data.py'
    forcing_ds.attrs['description'] = 'Atmospheric forcing variables extracted from NoahMP simulation output'

    # Add variable descriptions
    for var_config in config.FORCING_VARIABLES:
        var_name = var_config['name']
        if var_name in forcing_ds:
            forcing_ds[var_name].attrs['description'] = var_config['description']
            forcing_ds[var_name].attrs['aggregation'] = var_config['aggregation']

    # Save to NetCDF
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    forcing_ds.to_netcdf(output_path)
    print(f"\n✓ Forcing data saved to: {output_path}")

    # Print summary
    print("\nForcing dataset summary:")
    print(forcing_ds)

    return forcing_ds


def main():
    parser = argparse.ArgumentParser(
        description='Extract forcing data from simulation results'
    )
    parser.add_argument('--sample_idx', type=int, default=1,
                       help='Sample index to extract from (default: 1)')
    parser.add_argument('--output', type=str,
                       default='data/raw/forcing/forcing_sample_1.nc',
                       help='Output NetCDF file path')
    parser.add_argument('--data_dir', type=str,
                       default='data/raw/sim_results',
                       help='Directory containing simulation results')

    args = parser.parse_args()

    extract_forcing_data(args.sample_idx, args.output, args.data_dir)


if __name__ == '__main__':
    main()
