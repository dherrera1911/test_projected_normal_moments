"""
This script fits the projected normal to the empirical moments of the data
using moment matching. The results are saved in a dictionary and stored in a
PyTorch file.
"""
import argparse
import sys
import time
import itertools
import os

import torch
import yaml

import projnormal.formulas.projected_normal as png
from projnormal.classes import ProjNormal
from projnormal import param_sampling


COV_MULT = 10

def mse_loss_weighted(momentsA, momentsB):
    """ Compute the Euclidean distance between the observed and model moments. """
    distance_means_sq = torch.sum((momentsA["mean"]*10 - momentsB["mean"]*10)**2)
    distance_sm_sq = torch.sum(
      (momentsA["covariance"]*10*COV_MULT - momentsB["covariance"]*10*COV_MULT)**2
    )
    return distance_means_sq + distance_sm_sq


# Set the data type
DTYPE = torch.float32

config = yaml.safe_load(open('./parameters/moment_match.yaml', 'r'))
saving_dirs = yaml.safe_load(open('./parameters/saving_dirs.yaml', 'r'))


def main():

    # Simulation parameters
    N_DIM_LIST = [3, 24]
    SIGMA_LIST = config['simulation_parameters']['sigma_list']
    EIGVAL_LIST = config['simulation_parameters']['eigval_list']
    EIGVEC_LIST = config['simulation_parameters']['eigvec_list']
    EMPIRICAL_SAMPLES = 10000
    N_SIMULATIONS = config['simulation_parameters']['n_simulations']

    # Fitting parameters
    N_ITER = config['fitting_parameters']['n_iter']
    LR = config['fitting_parameters']['lr']
    LR_DECAY_PERIOD = config['fitting_parameters']['lr_decay_period']
    LR_GAMMA = config['fitting_parameters']['lr_gamma']
    LR_GAMMA_CYCLE = config['fitting_parameters']['lr_gamma_cycle']
    N_CYCLES = config['fitting_parameters']['n_cycles']

    N_DIM_LIST = config['simulation_parameters']['n_dim_list']
    SAVING_DIR = saving_dirs['mm_vs_ml']

    # Create saving directory
    os.makedirs(SAVING_DIR, exist_ok=True)

    ##############
    # FIT THE PROJECTED NORMAL TO THE EMPIRICAL MOMENTS
    ##############

    start = time.time()
    for n_dim, eigval, eigvec in itertools.product(
        N_DIM_LIST, EIGVAL_LIST, EIGVEC_LIST
    ):

        n_scales = len(SIGMA_LIST)

        # Initialize results dictionary
        field_names = ['mean_y_true', 'mean_y_fit_taylor', 'mean_y_fit_true',
                       'covariance_y_true', 'covariance_y_fit_taylor', 'covariance_y_fit_true',
                       'sm_y_true', 'sm_y_fit_taylor', 'sm_y_fit_true',
                       'mean_x', 'covariance_x', 'mean_x_fit', 'covariance_x_fit',
                       'mean_x_ml', 'covariance_x_ml', 'loss', 'loss_ml']

        results = {field: [[None for _ in range(N_SIMULATIONS)] for _ in range(n_scales)]
                   for field in field_names}
        results['sigma'] = SIGMA_LIST

        for v, sigma_scale in enumerate(SIGMA_LIST):
            # loop over variance scales
            var_scale = sigma_scale**2 / torch.tensor(n_dim)

            r = 0
            while r < N_SIMULATIONS:
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
                # Get samples from distribution
                samples = png.sample(
                    mean_x=results['mean_x'][v][r],
                    covariance_x=results['covariance_x'][v][r],
                    n_samples=EMPIRICAL_SAMPLES,
                )
                results['mean_y_true'][v][r] = samples.mean(dim=0)
                results['covariance_y_true'][v][r] = samples.T.cov()
                results['sm_y_true'][v][r] = samples.T @ samples / EMPIRICAL_SAMPLES

                # FIT TO DATA - Moment matching
                # Initialize the object
                moments_empirical = {
                  'mean': results['mean_y_true'][v][r],
                  'covariance': results['covariance_y_true'][v][r]
                }
                prnorm = ProjNormal(n_dim=n_dim)
                # Initialize to guess parameters
                prnorm.moment_init(moments_empirical)
                # Fit with moment matching
                loss_dict = prnorm.moment_match(
                  data_moments=moments_empirical,
                  max_epochs=N_ITER,
                  lr=LR,
                  optimizer='NAdam',
                  show_progress=False,
                  return_loss=True,
                  n_cycles=N_CYCLES,
                  cycle_gamma=LR_GAMMA_CYCLE,
                  gamma=LR_GAMMA,
                  step_size=LR_DECAY_PERIOD,
                  loss_fun=mse_loss_weighted,
                )

                # FIT TO DATA - Maximum Likelihood
                prnorm_ml = ProjNormal(n_dim=n_dim)
                # Initialize to guess parameters
                prnorm_ml.moment_init(moments_empirical)
                prnorm_ml.covariance_x = prnorm_ml.covariance_x + torch.eye(n_dim)*1e-1
                # Fit with moment matching
                loss_dict_ml = prnorm_ml.max_likelihood(
                  y=samples,
                  max_epochs=N_ITER,
                  lr=LR,
                  optimizer='NAdam',
                  show_progress=False,
                  return_loss=True,
                  n_cycles=N_CYCLES,
                  cycle_gamma=LR_GAMMA_CYCLE,
                  gamma=LR_GAMMA,
                  step_size=LR_DECAY_PERIOD,
                )

                with torch.no_grad():
                    # Fitted parameters
                    results['mean_x_fit'][v][r] = prnorm.mean_x
                    results['covariance_x_fit'][v][r] = prnorm.covariance_x
                    results['mean_x_ml'][v][r] = prnorm_ml.mean_x
                    results['covariance_x_ml'][v][r] = prnorm_ml.covariance_x

                    # Fitted taylor Y moments
                    moments_taylor = prnorm.moments()
                    results['mean_y_fit_taylor'][v][r] = moments_taylor['mean']
                    results['covariance_y_fit_taylor'][v][r] = moments_taylor['covariance']
                    results['sm_y_fit_taylor'][v][r] = moments_taylor['second_moment']

                    # Fitted true Y moments
                    moments_fit = prnorm.moments_empirical(n_samples=EMPIRICAL_SAMPLES)
                    results['mean_y_fit_true'][v][r] = moments_fit['mean']
                    results['covariance_y_fit_true'][v][r] = moments_fit['covariance']
                    results['sm_y_fit_true'][v][r] = moments_fit['second_moment']

                    # Loss
                    results['loss'][v][r] = loss_dict['loss']
                    results['loss_ml'][v][r] = loss_dict_ml['loss']

                # Check that the fitted parameters are not nan
                if (
                    not torch.isnan(results['mean_x_fit'][v][r]).any()
                    and not torch.isnan(results['covariance_x_fit'][v][r]).any()
                ):
                    r += 1

        # Save results
        torch.save(
            results,
            SAVING_DIR + f'results_eigvals_{eigval}_eigvecs_{eigvec}_n_{n_dim}.pt',
        )

    print(f'Time taken: {time.time() - start:.2f} seconds')


if __name__ == "__main__":
    main()
