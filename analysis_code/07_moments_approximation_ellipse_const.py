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

from projnormal.models import ProjNormalEllipseConst
from projnormal import param_sampling
from projnormal.ellipse_linalg import make_B_matrix
from projnormal import quadratic_forms as qf


config = yaml.safe_load(open('./parameters/approx_moment.yaml', 'r'))
saving_dirs = yaml.safe_load(open('./parameters/saving_dirs.yaml', 'r'))
N_DIRS = 2 # Number of independent ellipse directions

def main(dimension='3d'):

    # Load parameters
    EIGVAL_LIST = config['eigval_list']
    EIGVEC_LIST = config['eigvec_list']
    SIGMA_LIST = config['sigma_list']
    EMPIRICAL_SAMPLES = config['empirical_samples']
    N_SIMULATIONS = config['n_simulations']

    if dimension=='3d':
        N_DIM_LIST = [3]
        SAVING_DIR = saving_dirs['approx_ellipse_const_3d']
    elif dimension=='nd':
        N_DIM_LIST = config['n_dim_list']
        SAVING_DIR = saving_dirs['approx_ellipse_const_nd']

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
                       'mean_x', 'covariance_x',
                       'B_eigvals', 'B_eigvecs', 'B_rad_sq', 'B',
                       'const', 'const_mult']
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

                results['B_rad_sq'][v][r] = torch.rand(1) * 1.5 + 0.5
                results['B_eigvals'][v][r] = -torch.log(torch.rand(N_DIRS)) + 0.1
                results['B_eigvecs'][v][r] = param_sampling.make_ortho_vectors(n_dim, N_DIRS)
                results['B'][v][r] = make_B_matrix(
                  eigvals=results['B_eigvals'][v][r],
                  eigvecs=results['B_eigvecs'][v][r],
                  rad_sq=results['B_rad_sq'][v][r]
                )

                expected_norm_sq = qf.moments.mean(
                    results['mean_x'][v][r], results['covariance_x'][v][r]
                )
                multiplier = -torch.log(torch.rand(1))
                results['const_mult'][v][r] = multiplier
                results['const'][v][r] = expected_norm_sq * multiplier

                # Initialize the projected normal
                prnorm = ProjNormalEllipseConst(
                    mean_x=results['mean_x'][v][r],
                    covariance_x=results['covariance_x'][v][r],
                    const=results['const'][v][r],
                    B_eigvals=results['B_eigvals'][v][r],
                    B_eigvecs=results['B_eigvecs'][v][r],
                    B_rad_sq=results['B_rad_sq'][v][r],
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
    # 1. Create an argument parser
    parser = argparse.ArgumentParser(
        description="Run analysis using either 3D or ND configurations."
    )
    parser.add_argument(
        "--dimension",
        choices=["3d", "nd"],
        help="Which version to run (3d or nd)."
    )

    # 2. Parse arguments
    args = parser.parse_args()

    # 3. If user is in an interactive session but hasn't supplied --dimension,
    #    we can prompt them. If they are running a script with no dimension,
    #    you can choose a sensible default (or also prompt).
    if sys.stdin.isatty():
        # Running in a TTY (e.g., normal terminal), so let's prompt if needed
        dimension = args.dimension or input("Enter dimension (3d or nd): ").strip()
    else:
        # Non-interactive: no prompt. Could just default to "nd" or "3d".
        dimension = args.dimension if args.dimension else "3d"

    # 4. Run the main logic
    main(dimension)
