"""
Utility script to validate input files for inference
Checks forcing NetCDF and parameter text files for correctness
"""

import xarray as xr
import pandas as pd
import argparse
from pathlib import Path
import config_forward as config


def validate_forcing_file(forcing_file):
    """
    Validate forcing NetCDF file

    Args:
        forcing_file: Path to forcing NetCDF file

    Returns:
        bool: True if valid, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"VALIDATING FORCING FILE: {forcing_file}")
    print('='*60)

    try:
        # Load NetCDF
        ds = xr.open_dataset(forcing_file)

        # Check dimensions
        if 'time' not in ds.dims:
            print("  ✗ ERROR: 'time' dimension not found")
            return False
        print(f"  ✓ Time dimension found: {len(ds['time'])} timesteps")

        # Check required variables
        required_vars = [var['name'] for var in config.FORCING_VARIABLES]
        missing_vars = []

        for var_name in required_vars:
            if var_name not in ds:
                missing_vars.append(var_name)
                print(f"  ✗ ERROR: Variable '{var_name}' not found")
            else:
                print(f"  ✓ Variable '{var_name}' found: shape {ds[var_name].shape}")

        if missing_vars:
            print(f"\n  ✗ VALIDATION FAILED: Missing variables: {missing_vars}")
            return False

        # Check data quality
        print("\n  Data Quality Checks:")
        for var_name in required_vars:
            data = ds[var_name].values
            n_nan = pd.isna(data).sum()
            n_inf = (~pd.isna(data) & ~pd.isnf(data)).sum() if hasattr(pd, 'isnf') else 0

            if n_nan > 0:
                print(f"    ⚠ WARNING: {var_name} has {n_nan} NaN values")
            if n_inf > 0:
                print(f"    ⚠ WARNING: {var_name} has {n_inf} Inf values")

            if n_nan == 0 and n_inf == 0:
                print(f"    ✓ {var_name}: No NaN or Inf values")
                print(f"      Range: [{data.min():.2f}, {data.max():.2f}]")

        ds.close()

        print(f"\n  ✓ VALIDATION PASSED")
        return True

    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}")
        return False


def validate_parameter_file(param_file):
    """
    Validate parameter text file

    Args:
        param_file: Path to parameter text file

    Returns:
        bool: True if valid, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"VALIDATING PARAMETER FILE: {param_file}")
    print('='*60)

    try:
        # Read parameter file
        params_df = pd.read_csv(param_file, sep=r'\s+')

        print(f"  ✓ File loaded successfully")
        print(f"  Number of parameter sets: {len(params_df)}")

        # Expected parameters (from training data)
        expected_params = [
            'VCMX25_EBF', 'HVT_EBF', 'HVB_EBF', 'CWPVT_EBF', 'Z0MVT_EBF',
            'WLTSMC_CL', 'REFSMC_CL', 'MAXSMC_CL', 'SATDK_CL',
            'WLTSMC_SCL', 'REFSMC_SCL', 'MAXSMC_SCL', 'SATDK_SCL'
        ]

        # Check columns
        missing_params = set(expected_params) - set(params_df.columns)
        extra_params = set(params_df.columns) - set(expected_params)

        if missing_params:
            print(f"  ✗ ERROR: Missing parameters: {missing_params}")
            return False

        if extra_params:
            print(f"  ⚠ WARNING: Extra parameters (will be ignored): {extra_params}")

        print(f"  ✓ All required parameters present")

        # Check for NaN/Inf values
        print("\n  Data Quality Checks:")
        has_issues = False

        for param in expected_params:
            if param not in params_df.columns:
                continue

            n_nan = params_df[param].isna().sum()
            n_inf = (~params_df[param].isna() & ~params_df[param].between(-1e10, 1e10)).sum()

            if n_nan > 0:
                print(f"    ✗ ERROR: {param} has {n_nan} NaN values")
                has_issues = True
            if n_inf > 0:
                print(f"    ⚠ WARNING: {param} has {n_inf} potentially Inf values")

            if n_nan == 0 and n_inf == 0:
                values = params_df[param].values
                print(f"    ✓ {param}: Range [{values.min():.6f}, {values.max():.6f}]")

        if has_issues:
            print(f"\n  ✗ VALIDATION FAILED: Data quality issues detected")
            return False

        # Show first few rows
        print("\n  First parameter set:")
        print(params_df.iloc[0][expected_params].to_string())

        print(f"\n  ✓ VALIDATION PASSED")
        return True

    except Exception as e:
        print(f"  ✗ ERROR: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Validate input files for LSTM forward model inference'
    )
    parser.add_argument('--forcing', type=str,
                       help='NetCDF file with forcing data')
    parser.add_argument('--params', type=str,
                       help='Text file with parameters')

    args = parser.parse_args()

    if not args.forcing and not args.params:
        print("Error: At least one of --forcing or --params must be specified")
        parser.print_help()
        return

    results = []

    if args.forcing:
        forcing_valid = validate_forcing_file(args.forcing)
        results.append(('Forcing file', forcing_valid))

    if args.params:
        params_valid = validate_parameter_file(args.params)
        results.append(('Parameter file', params_valid))

    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print('='*60)

    all_valid = True
    for name, valid in results:
        status = "✓ VALID" if valid else "✗ INVALID"
        print(f"  {name}: {status}")
        if not valid:
            all_valid = False

    if all_valid:
        print(f"\n✓ All files are valid and ready for inference!")
    else:
        print(f"\n✗ Some files have issues. Please fix them before running inference.")

    return all_valid


if __name__ == '__main__':
    main()
