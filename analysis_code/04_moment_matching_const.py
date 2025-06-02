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

import projnormal.distribution.const as pnc
from projnormal.models import ProjNormalConst
from projnormal import param_sampling
from projnormal import quadratic_forms as qf

COV_MULT = 2

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
    N_DIM_LIST = config['simulation_parameters']['n_dim_list']
    SIGMA_LIST = config['simulation_parameters']['sigma_list']
    EIGVAL_LIST = config['simulation_parameters']['eigval_list']
    EIGVEC_LIST = config['simulation_parameters']['eigvec_list']
    EMPIRICAL_SAMPLES = config['simulation_parameters']['empirical_samples']
    N_SIMULATIONS = config['simulation_parameters']['n_simulations']

    # Fitting parameters
    N_ITER = config['fitting_parameters']['n_iter']
    LR = config['fitting_parameters']['lr']
    LR_DECAY_PERIOD = config['fitting_parameters']['lr_decay_period']
    LR_GAMMA = config['fitting_parameters']['lr_gamma']
    LR_GAMMA_CYCLE = config['fitting_parameters']['lr_gamma_cycle']
    N_CYCLES = config['fitting_parameters']['n_cycles']

    N_DIM_LIST = config['simulation_parameters']['n_dim_list']
    SAVING_DIR = saving_dirs['moment_match_const']

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
                       'mean_x', 'covariance_x', 'const', 'const_mult',
                       'mean_x_fit', 'covariance_x_fit', 'const_fit', 'loss']

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

                expected_norm_sq = qf.moments.mean(
                    results['mean_x'][v][r], results['covariance_x'][v][r]
                )
                multiplier = -torch.log(torch.rand(1))
                results['const_mult'][v][r] = multiplier
                results['const'][v][r] = expected_norm_sq * multiplier

                # Obtain moments and store
                moments_empirical = pnc.sampling.empirical_moments(
                    n_samples=EMPIRICAL_SAMPLES,
                    mean_x=results['mean_x'][v][r],
                    covariance_x=results['covariance_x'][v][r],
                    const=results['const'][v][r],
                )
                results['mean_y_true'][v][r] = moments_empirical['mean']
                results['covariance_y_true'][v][r] = moments_empirical['covariance']
                results['sm_y_true'][v][r] = moments_empirical['second_moment']

                # FIT TO DATA
                # Initialize the object
                prnorm = ProjNormalConst(n_dim=n_dim)
                # Initialize to guess parameters
                prnorm.moment_init(moments_empirical)
                # Fit 
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

                with torch.no_grad():
                    # Fitted parameters
                    results['mean_x_fit'][v][r] = prnorm.mean_x
                    results['covariance_x_fit'][v][r] = prnorm.covariance_x
                    results['const_fit'][v][r] = prnorm.const

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
