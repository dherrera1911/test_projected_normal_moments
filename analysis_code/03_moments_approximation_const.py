"""
This script computes the moments of the projected normal distribution
using both sampling from the distribution and the Taylor approximation
for each sample of the parameters. The results are saved in a dictionary
and stored in a .pt file.
"""
import argparse
import sys
import time
import itertools
import os

import torch
import yaml

from projnormal.models import ProjNormalConst
from projnormal import param_sampling
from projnormal import quadratic_forms as qf


config = yaml.safe_load(open('./parameters/approx_moment.yaml', 'r'))
saving_dirs = yaml.safe_load(open('./parameters/saving_dirs.yaml', 'r'))

def main():

    # Load parameters
    EIGVAL_LIST = config['eigval_list']
    EIGVEC_LIST = config['eigvec_list']
    SIGMA_LIST = config['sigma_list']
    EMPIRICAL_SAMPLES = config['empirical_samples']
    N_SIMULATIONS = config['n_simulations']

    N_DIM_LIST = config['n_dim_list']
    SAVING_DIR = saving_dirs['approx_const']

    # Create saving directory
    os.makedirs(SAVING_DIR, exist_ok=True)

    ##############
    # GET APPROXIMATION ERRORS
    ##############

    start = time.time()

    for n_dim, eigval, eigvec in itertools.product(
        N_DIM_LIST, EIGVAL_LIST, EIGVEC_LIST
    ):

        n_scales = len(SIGMA_LIST)

        # Initialize dictionary with nested lists to save results
        field_names = ['mean_y_true', 'mean_y_taylor',
                       'covariance_y_true', 'covariance_y_taylor',
                       'sm_y_true', 'sm_y_taylor',
                       'mean_x', 'covariance_x', 'const', 'const_mult']
        results = {field: [[None for _ in range(N_SIMULATIONS)] for _ in range(n_scales)]
                   for field in field_names}
        results['sigma'] = SIGMA_LIST

        for v, sigma_scale in enumerate(SIGMA_LIST):
            # loop over variance scales
            var_scale = sigma_scale**2 / torch.tensor(n_dim)

            for r in range(N_SIMULATIONS):
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

                expected_norm_sq = qf.moments.mean(
                    results['mean_x'][v][r], results['covariance_x'][v][r]
                )
                multiplier = -torch.log(torch.rand(1))
                results['const_mult'][v][r] = multiplier
                results['const'][v][r] = expected_norm_sq * multiplier

                # Initialize the projected normal
                prnorm = ProjNormalConst(
                    mean_x=results['mean_x'][v][r],
                    covariance_x=results['covariance_x'][v][r],
                    const=results['const'][v][r],
                )

                # Get empirical moment estimates and unpack
                with torch.no_grad():
                    moments_empirical = prnorm.moments_empirical(
                        n_samples=EMPIRICAL_SAMPLES
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
            SAVING_DIR + f'results_eigvals_{eigval}_eigvecs_{eigvec}_n_{n_dim}.pt',
        )

    print(f'Time taken: {time.time() - start:.2f} seconds')


if __name__ == "__main__":
    main()
