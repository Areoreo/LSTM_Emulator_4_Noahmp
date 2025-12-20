import numpy as np
import csv

# Read the parameter information
var_info = []
with open('var_info_matrix.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        var_info.append(row)

# Read value bounds
value_bounds = []
with open('value_bounds.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        value_bounds.append(row)

# Create a dictionary for bounds lookup
bounds_dict = {}
for row in value_bounds:
    bounds_dict[row['variable']] = {
        'lower': float(row['Lower bound']),
        'upper': float(row['Upper bound'])
    }

# Number of samples
n_samples = 1000

# Get parameter list
param_list = []
for row in var_info:
    param_list.append({
        'column_name': row['column_name'],
        'variable': row['variable'],
        'section': row['section'],
        'type': int(row['type'])
    })

# Number of parameters
n_params = len(param_list)

# Set random seed for reproducibility
np.random.seed(42)

# Generate Latin Hypercube Samples manually
def latin_hypercube_sampling(n_samples, n_params):
    """Generate Latin Hypercube samples in [0, 1]^n_params"""
    samples = np.zeros((n_samples, n_params))

    for i in range(n_params):
        # Divide [0,1] into n_samples intervals
        intervals = np.arange(n_samples) / n_samples
        # Add random offset within each interval
        samples[:, i] = intervals + np.random.uniform(0, 1/n_samples, n_samples)
        # Shuffle to decorrelate
        np.random.shuffle(samples[:, i])

    return samples

# Generate samples normalized to [0, 1]
samples_normalized = latin_hypercube_sampling(n_samples, n_params)

# Scale samples to actual parameter bounds
samples_scaled = np.zeros((n_samples, n_params))

for i, param in enumerate(param_list):
    var_name = param['variable']
    lower = bounds_dict[var_name]['lower']
    upper = bounds_dict[var_name]['upper']

    # Scale from [0,1] to [lower, upper]
    samples_scaled[:, i] = samples_normalized[:, i] * (upper - lower) + lower

# Create output file with proper formatting
output_filename = 'noahmp_1000samples_new.txt'

with open(output_filename, 'w') as f:
    # Write header
    header = ' '.join([p['column_name'] for p in param_list])
    f.write(header + '\n')

    # Write samples
    for sample in samples_scaled:
        line = ' '.join([str(val) for val in sample])
        f.write(line + '\n')

print(f"Generated {n_samples} parameter combinations")
print(f"Output saved to: {output_filename}")
print(f"\nParameter columns ({n_params} total):")
for i, param in enumerate(param_list):
    var_name = param['variable']
    bounds = bounds_dict[var_name]
    print(f"{i+1}. {param['column_name']} ({var_name}): [{bounds['lower']}, {bounds['upper']}]")
