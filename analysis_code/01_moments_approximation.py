"""
This script computes the moments of the projected normal distribution
using both sampling from the distribution and the Taylor approximation
for each sample of the parameters. The results are saved in a dictionary
and stored in a .pt file.
"""
import argparse
import os
import sys
import time
import itertools

import torch
import yaml
from projnormal.models import ProjectedNormal
from projnormal import param_sampling

# Load configuration file
run_mode = 'script'
#run_mode = 'interactive'
if run_mode == 'interactive':
    file_name = './parameters/par_approx_3d.yaml'
    config = yaml.safe_load(open(file_name, 'r'))
elif run_mode == 'script':
    parser = argparse.ArgumentParser(
        description='Run analysis with specified configuration file.'
    )
    parser.add_argument(
        'config_path', type=str, help='Path to the configuration YAML file.'
    )
    args = parser.parse_args()
    # Load the YAML file
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

# Simulation parameters
n_dim_list = config['simulation_parameters']['n_dim_list']
eigval_list = config['simulation_parameters']['eigval_list']
eigvec_list = config['simulation_parameters']['eigvec_list']
sigma_list = config['simulation_parameters']['sigma_list']
empirical_samples = config['simulation_parameters']['empirical_samples']
n_simulations = config['simulation_parameters']['n_simulations']

# Create saving directory
results_dir = config['saving_dir']['results_dir']
os.makedirs(results_dir, exist_ok=True)

##############
# GET APPROXIMATION ERRORS
##############

start = time.time()

for n_dim, eigval, eigvec in itertools.product(
    n_dim_list, eigval_list, eigvec_list
):

    n_scales = len(sigma_list)

    # Initialize dictionary with nested lists to save results
    field_names = ['mean_y_true', 'mean_y_taylor',
                   'covariance_y_true', 'covariance_y_taylor',
                   'sm_y_true', 'sm_y_taylor',
                   'mean_x', 'covariance_x']
    results = {field: [[None for _ in range(n_simulations)] for _ in range(n_scales)]
               for field in field_names}
    results['sigma'] = sigma_list

    for v, sigma_scale in enumerate(sigma_list):
        # loop over variance scales
        var_scale = sigma_scale**2 / torch.tensor(n_dim)

        for r in range(n_simulations):
            progress_str = (
                f'cov_type: {eigval}, {eigvec} '
                f'n_dim: {n_dim}, sigma_scale: {sigma_scale}, rep: {r}'
            )
            print(progress_str)

            # Get parameters
            results['mean_x'][v][r] = param_sampling.make_mean(
              n_dim=n_dim, shape='gaussian'
            )
            results['covariance_x'][v][r] = param_sampling.make_spdm(
                n_dim=n_dim, eigvals=eigval, eigvecs=eigvec
            ) * var_scale

            # Initialize the projected normal
            prnorm = ProjectedNormal(
                mean_x=results['mean_x'][v][r],
                covariance_x=results['covariance_x'][v][r],
            )

            # Get empirical moment estimates and unpack
            with torch.no_grad():
                moments_empirical = prnorm.moments_empirical(
                    n_samples=empirical_samples
                )
            results['mean_y_true'][v][r] = moments_empirical['mean']
            results['covariance_y_true'][v][r] = moments_empirical['covariance']
            results['sm_y_true'][v][r] = moments_empirical['second_moment']

            # Get the Taylor approximation moments and unpack
            with torch.no_grad():
                moments_taylor = prnorm.moments()
            results['mean_y_taylor'][v][r] = moments_taylor['mean']
            results['covariance_y_taylor'][v][r] = moments_taylor['covariance']
            results['sm_y_taylor'][v][r] = moments_taylor['second_moment']

    # Save results
    torch.save(
        results,
        results_dir + f'results_eigvals_{eigval}_eigvecs_{eigvec}_n_{n_dim}.pt',
    )

print(f'Time taken: {time.time() - start:.2f} seconds')
