"""
This script fits the projected normal to the empirical moments of the data
using moment matching. The results are saved in a dictionary and stored in a
PyTorch file.
"""
import argparse
import os
import sys
import time
import itertools

import torch
import yaml
import projnormal.distribution.general as png
from projnormal.models import ProjectedNormal
from projnormal import param_sampling

# Set the data type
DTYPE = torch.float32

#run_mode = 'interactive'
run_mode = 'script'
if run_mode == 'interactive':
    file_name = './parameters/par_mm_3d.yaml'
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
sigma_list = config['simulation_parameters']['sigma_list']
eigval_list = config['simulation_parameters']['eigval_list']
eigvec_list = config['simulation_parameters']['eigvec_list']
empirical_samples = config['simulation_parameters']['empirical_samples']
n_simulations = config['simulation_parameters']['n_simulations']

# Fitting parameters
n_iter = config['fitting_parameters']['n_iter']
lr = config['fitting_parameters']['lr']
lr_decay_period = config['fitting_parameters']['lr_decay_period']
lr_gamma = config['fitting_parameters']['lr_gamma']
lr_gamma_cycle = config['fitting_parameters']['lr_gamma_cycle']
n_cycles = config['fitting_parameters']['n_cycles']

# Results saving directory
results_dir = config['saving_dir']['results_dir']
saveFig = True
os.makedirs(results_dir, exist_ok=True)


##############
# FIT THE PROJECTED NORMAL TO THE EMPIRICAL MOMENTS
##############

start = time.time()
for n_dim, eigval, eigvec in itertools.product(
    n_dim_list, eigval_list, eigvec_list
):

    n_scales = len(sigma_list)

    # Initialize results dictionary
    field_names = ['mean_y_true', 'mean_y_fit_taylor', 'mean_y_fit_true',
                   'covariance_y_true', 'covariance_y_fit_taylor', 'covariance_y_fit_true',
                   'sm_y_true', 'sm_y_fit_taylor', 'sm_y_fit_true',
                   'mean_x', 'covariance_x', 'mean_x_fit', 'covariance_x_fit',
                   'loss']

    results = {field: [[None for _ in range(n_simulations)] for _ in range(n_scales)]
               for field in field_names}
    results['sigma'] = sigma_list


    for v, sigma_scale in enumerate(sigma_list):
        # loop over variance scales
        var_scale = sigma_scale**2

        r = 0
        while r < n_simulations:
            progress_str = (
                f'cov_type: {eigval}, {eigvec} '
                f'n_dim: {n_dim}, sigma_scale: {sigma_scale}, rep: {r}'
            )
            print(progress_str)

            # SIMULATE DATA
            # Sample parameters and store
            results['mean_x'][v][r] = param_sampling.make_mean(
              n_dim=n_dim, shape='gaussian'
            )
            results['covariance_x'][v][r] = param_sampling.make_spdm(
                n_dim=n_dim, eigvals=eigval, eigvecs=eigvec
            ) * var_scale
            # Obtain moments and store
            moments_empirical = png.sampling.empirical_moments(
                n_samples=empirical_samples,
                mean_x=results['mean_x'][v][r],
                covariance_x=results['covariance_x'][v][r],
            )
            results['mean_y_true'][v][r] = moments_empirical['mean']
            results['covariance_y_true'][v][r] = moments_empirical['covariance']
            results['sm_y_true'][v][r] = moments_empirical['second_moment']

            # FIT TO DATA
            # Initialize the object
            prnorm = ProjectedNormal(n_dim=n_dim)
            # Initialize to guess parameters
            prnorm.moment_init(moments_empirical)
            # Fit 
            loss_dict = prnorm.moment_match(
              data_moments=moments_empirical,
              max_epochs=n_iter,
              lr=lr,
              optimizer='NAdam',
              show_progress=False,
              return_loss=True,
              n_cycles=n_cycles,
              cycle_gamma=lr_gamma_cycle,
              gamma=lr_gamma,
              step_size=lr_decay_period,
            )

            with torch.no_grad():
                # Fitted parameters
                results['mean_x_fit'][v][r] = prnorm.mean_x
                results['covariance_x_fit'][v][r] = prnorm.covariance_x.detach()

                # Fitted taylor Y moments
                moments_taylor = prnorm.moments()
                results['mean_y_fit_taylor'][v][r] = moments_taylor['mean']
                results['covariance_y_fit_taylor'][v][r] = moments_taylor['covariance']
                results['sm_y_fit_taylor'][v][r] = moments_taylor['second_moment']

                # Fitted true Y moments
                moments_fit = prnorm.moments_empirical(n_samples=empirical_samples)
                results['mean_y_fit_true'][v][r] = moments_fit['mean']
                results['covariance_y_fit_true'][v][r] = moments_fit['covariance']
                results['sm_y_fit_true'][v][r] = moments_fit['second_moment']

                # Loss
                results['loss'][v][r] = loss_dict['loss']

            # Check that the fitted parameters are not nan
            if (
                not torch.isnan(results['mean_x_fit'][v][r]).any()
                and not torch.isnan(results['covariance_x_fit'][v][r]).any()
            ):
                r += 1

    # Save results
    torch.save(
        results,
        results_dir + f'results_eigvals_{eigval}_eigvecs_{eigvec}_n_{n_dim}.pt',
    )

print(f'Time taken: {time.time() - start:.2f} seconds')
