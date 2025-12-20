#!/usr/bin/env python3
"""
Convert calibrated parameters from CSV to TBL generator format
"""

import sys
import argparse
import pandas as pd
import numpy as np


# Define minimum values for critical parameters that cannot be zero or negative
# to prevent numerical instabilities (division by zero, log of zero, etc.)
PARAMETER_CONSTRAINTS = {
    'Z0MVT': 0.001,   # Momentum roughness length - cannot be zero (causes log(0))
    'HVT': 0.01,      # Canopy top height - should be positive
    'HVB': 0.001,     # Canopy bottom height - should be positive
    'DLEAF': 0.001,   # Leaf dimension - should be positive
    'RC': 0.001,      # Stomatal resistance - should be positive
    'CWPVT': 0.001,   # Canopy wind parameter - should be positive
}


def apply_constraints(param_name, value):
    """
    Apply minimum value constraints to parameters
    
    Parameters:
    - param_name: Parameter name (e.g., 'Z0MVT_EBF')
    - value: Calibrated value
    
    Returns:
    - Constrained value
    """
    # Extract base parameter name (remove suffix like _EBF, _CL, etc.)
    base_param = param_name.split('_')[0]
    
    if base_param in PARAMETER_CONSTRAINTS:
        min_val = PARAMETER_CONSTRAINTS[base_param]
        if value < min_val:
            print(f"  WARNING: {param_name} = {value:.6f} is below minimum {min_val:.6f}")
            print(f"           Constraining to {min_val:.6f} to avoid numerical issues")
            return min_val
    
    return value


def convert_params(input_csv, output_txt):
    """
    Convert calibrated_parameters.csv to space-delimited format
    
    Handles log-transformed parameters (SATDK) by applying inverse transformation.
    Also applies minimum value constraints for critical parameters.
    
    Parameters:
    - input_csv: Path to calibrated_parameters.csv
    - output_txt: Path to output text file
    """
    try:
        df = pd.read_csv(input_csv)
        
        # Filter only calibrated parameters
        calibrated = df[df['is_calibrated'] == 'Yes']
        
        # Create output in space-delimited format with header
        params = calibrated['parameter'].tolist()
        values = calibrated['calibrated'].tolist()
        
        # Apply inverse transformation for log-transformed parameters
        # and apply constraints
        transformed_values = []
        for param, value in zip(params, values):
            # SATDK parameters are log10-transformed, need to convert back
            if 'SATDK' in param:
                actual_value = 10 ** value
                transformed_values.append(actual_value)
                print(f"  Transformed {param}: log({value:.6f}) -> {actual_value:.6e}")
            else:
                # Apply minimum value constraints
                constrained_value = apply_constraints(param, value)
                transformed_values.append(constrained_value)
                if constrained_value != value:
                    print(f"  Constrained {param}: {value:.6f} -> {constrained_value:.6f}")
        
        # Write header and values
        with open(output_txt, 'w') as f:
            f.write(' '.join(params) + '\n')
            f.write(' '.join([f'{v:.6e}' if 'SATDK' in params[i] else f'{v:.6f}' if isinstance(v, float) else str(v) 
                             for i, v in enumerate(transformed_values)]) + '\n')
        
        print(f"Converted {len(params)} calibrated parameters")
        return 0
        
    except Exception as e:
        print(f"Error converting parameters: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(description='Convert calibrated parameters CSV to TBL format')
    parser.add_argument('--input', required=True, help='Input calibrated_parameters.csv file')
    parser.add_argument('--output', required=True, help='Output text file')
    
    args = parser.parse_args()
    
    return convert_params(args.input, args.output)


if __name__ == '__main__':
    sys.exit(main())
