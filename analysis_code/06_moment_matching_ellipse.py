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
from projnormal.models import ProjNormalEllipseIso
from projnormal import param_sampling

COV_MULT = 10

def make_B_matrix(B_coefs, B_vecs, B_diag):
    term1 = torch.eye(B_vecs.shape[-1]) * B_diag
    term2 = torch.einsum('ki,k,kj->ij', B_vecs, B_coefs, B_vecs)
    return term1 + term2


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
N_DIRS = 1


def main():

    # Simulation parameters
    N_DIM_LIST = config['simulation_parameters']['n_dim_list']
    SIGMA_LIST = config['simulation_parameters']['sigma_list']
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
    SAVING_DIR = saving_dirs['moment_match_ellipse']

    # Create saving directory
    os.makedirs(SAVING_DIR, exist_ok=True)

    ##############
    # FIT THE PROJECTED NORMAL TO THE EMPIRICAL MOMENTS
    ##############

    start = time.time()
    for n_dim in N_DIM_LIST:

        n_scales = len(SIGMA_LIST)

        # Initialize results dictionary
        field_names = ['mean_y_true', 'mean_y_fit_taylor', 'mean_y_fit_true',
                       'covariance_y_true', 'covariance_y_fit_taylor', 'covariance_y_fit_true',
                       'sm_y_true', 'sm_y_fit_taylor', 'sm_y_fit_true',
                       'mean_x', 'covariance_x', 'mean_x_fit', 'covariance_x_fit',
                       'B_coefs', 'B_vecs', 'B_diag', 'B', 'B_fit', 'loss']

        results = {field: [[None for _ in range(N_SIMULATIONS)] for _ in range(n_scales)]
                   for field in field_names}
        results['sigma'] = SIGMA_LIST

        for v, sigma_scale in enumerate(SIGMA_LIST):
            # loop over variance scales
            var_scale = sigma_scale**2 / torch.tensor(n_dim)

            r = 0
            while r < N_SIMULATIONS:
                progress_str = (
                    f'n_dim: {n_dim}, sigma_scale: {sigma_scale}, rep: {r}'
                )
                print(progress_str)

                # SIMULATE DATA
                # Sample parameters and store
                results['mean_x'][v][r] = param_sampling.make_mean(
                  n_dim=n_dim, shape='gaussian'
                ).to(dtype=DTYPE)
                results['covariance_x'][v][r] = torch.eye(n_dim, dtype=DTYPE) * var_scale

                results['B_diag'][v][r] = torch.as_tensor(1.0)
                results['B_coefs'][v][r] = -torch.log(torch.rand(N_DIRS)) * 4 + 2.0
                results['B_vecs'][v][r] = param_sampling.make_ortho_vectors(n_dim, N_DIRS)
                results['B'][v][r] = make_B_matrix(
                  B_coefs=results['B_coefs'][v][r],
                  B_vecs=results['B_vecs'][v][r],
                  B_diag=results['B_diag'][v][r]
                ).to(dtype=DTYPE)

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

                moments_empirical['mean'] = torch.as_tensor(
                  moments_empirical['mean'], dtype=DTYPE
                )
                moments_empirical['covariance'] = torch.as_tensor(
                  moments_empirical['covariance'], dtype=DTYPE
                )

                # FIT TO DATA
                # Initialize the object
                _, vec_init = torch.linalg.eigh(moments_empirical['covariance'])
                vec_init = vec_init[:, :N_DIRS].T

                prnorm = ProjNormalEllipseIso(
                  n_dim=n_dim,
                  n_dirs=N_DIRS,
                  B_sqrt_vecs=vec_init,
                )
                prnorm.to(dtype=DTYPE)

                # Initialize to guess parameters
                prnorm.moment_init(moments_empirical)

                converged = False
                count = 0
                while not converged:
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

#                    fig, ax = plt.subplots(1, 2)
#                    ax[0].imshow(prnorm.B.detach())
#                    ax[1].imshow(results['B'][v][r])
#                    plt.show()
#
#                    plt.plot(results['B_vecs'][v][r].detach().T)
#                    plt.plot(prnorm.ellipse.sqrt_vecs.detach().T)
#                    plt.show()

                    last_loss = loss_dict['loss'][-1]
                    if last_loss < 1e-7 or count >= 4:
                        converged = True
                    else:
                        count += 1

                with torch.no_grad():
                    # Fitted parameters
                    results['mean_x_fit'][v][r] = prnorm.mean_x
                    results['covariance_x_fit'][v][r] = prnorm.covariance_x.detach()
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
            SAVING_DIR + f'results_n_{n_dim}_mult_{COV_MULT}.pt',
        )

    print(f'Time taken: {time.time() - start:.2f} seconds')


if __name__ == "__main__":
    main()
