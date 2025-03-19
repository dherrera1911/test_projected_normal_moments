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

import projnormal.distribution.ellipse as pne
from projnormal.models import ProjNormalEllipse
from projnormal.ellipse_linalg import make_B_matrix
from projnormal import param_sampling

# Set the data type
DTYPE = torch.float32

config = yaml.safe_load(open('./parameters/moment_match.yaml', 'r'))
saving_dirs = yaml.safe_load(open('./parameters/saving_dirs.yaml', 'r'))
N_DIRS = 2


def main(dimension='3d'):

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

    if dimension=='3d':
        N_DIM_LIST = [3]
        SAVING_DIR = saving_dirs['moment_match_ellipse_3d']
    elif dimension=='nd':
        N_DIM_LIST = config['simulation_parameters']['n_dim_list']
        SAVING_DIR = saving_dirs['moment_match_ellipse_nd']

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
                       'B_eigvals', 'B_eigvecs', 'B_rad_sq', 'B',
                       'B_eigvals_fit', 'B_eigvecs_fit', 'B_rad_sq_fit', 'B_fit',
                       'loss']

        results = {field: [[None for _ in range(N_SIMULATIONS)] for _ in range(n_scales)]
                   for field in field_names}
        results['sigma'] = SIGMA_LIST

        for v, sigma_scale in enumerate(SIGMA_LIST):
            # loop over variance scales
            var_scale = sigma_scale**2

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

                results['B_rad_sq'][v][r] = torch.rand(1) * 1.5 + 0.5
                results['B_eigvals'][v][r] = -torch.log(torch.rand(N_DIRS)) + 0.1
                results['B_eigvecs'][v][r] = param_sampling.make_ortho_vectors(n_dim, N_DIRS)
                results['B'][v][r] = make_B_matrix(
                  eigvals=results['B_eigvals'][v][r],
                  eigvecs=results['B_eigvecs'][v][r],
                  rad_sq=results['B_rad_sq'][v][r]
                )

                # Obtain moments and store
                moments_empirical = pne.sampling.empirical_moments(
                    mean_x=results['mean_x'][v][r],
                    covariance_x=results['covariance_x'][v][r],
                    B=results['B'][v][r],
                    n_samples=EMPIRICAL_SAMPLES,
                )
                results['mean_y_true'][v][r] = moments_empirical['mean']
                results['covariance_y_true'][v][r] = moments_empirical['covariance']
                results['sm_y_true'][v][r] = moments_empirical['second_moment']

                # FIT TO DATA
                # Initialize the object
                prnorm = ProjNormalEllipse(
                  n_dim=n_dim,
                  n_dirs=N_DIRS
                )
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
                )

                with torch.no_grad():
                    # Fitted parameters
                    results['mean_x_fit'][v][r] = prnorm.mean_x
                    results['covariance_x_fit'][v][r] = prnorm.covariance_x.detach()
                    results['B_rad_sq_fit'][v][r] = prnorm.ellipse.rad_sq
                    results['B_eigvals_fit'][v][r] = prnorm.ellipse.eigvals
                    results['B_eigvecs_fit'][v][r] = prnorm.ellipse.eigvecs
                    results['B_fit'][v][r] = prnorm.B

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
