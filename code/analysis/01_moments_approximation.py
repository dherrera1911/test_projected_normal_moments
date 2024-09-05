##################
#
# For the case of the sphere embedded in nD, obtain the empirical
# moments of the projected normal, and the Taylor approximation
# to the moments. Save the results.
#
# Additionally, a second empirical estimate of the moments can be
# computed to estimate the error in the empirical moments for
# a given number of samples.
#
##################

import os
import time
import argparse
import sys
import torch
from projected_normal.prnorm_class import ProjectedNormal
import yaml

sys.path.append("../")
from analysis_functions import sample_parameters, list_2d

# TO RUN FROM THE COMMAND LINE, UNCOMMENT THE FOLLOWING LINES
# Set up command-line argument parsing
parser = argparse.ArgumentParser(
    description="Run analysis with specified configuration file."
)
parser.add_argument(
    "config_path", type=str, help="Path to the configuration YAML file."
)
args = parser.parse_args()
# Load the YAML file
with open(args.config_path, "r") as file:
    config = yaml.safe_load(file)
###

### TO RUN INTERACTIVE, UNCOMMENT THE FOLLOWING LINES
#fileName = './parameters/par_approx_3d.yaml'
#config = yaml.safe_load(open(fileName, 'r'))
###

# Simulation parameters
sigma_list = config["simulation_parameters"]["sigma_list"]
cov_type_list = config["simulation_parameters"]["cov_type_list"]
empirical_samples = config["simulation_parameters"]["empirical_samples"]
n_simulations = config["simulation_parameters"]["n_simulations"]
n_dim_list = config["simulation_parameters"]["n_dim_list"]

# Create saving directory
results_dir = config["saving_dir"]["results_dir"]
os.makedirs(results_dir, exist_ok=True)

##############
# GET APPROXIMATION ERRORS
##############

start = time.time()
for c, covariance_type in enumerate(cov_type_list):
    # loop over covariance types
    for n, n_dim in enumerate(n_dim_list):
        # loop over dimensions

        n_scales = len(sigma_list)
        # Initialize the dictionary with nested lists to save the moments
        results = {
            "gamma_true": list_2d(n_scales, n_simulations),
            "gamma_taylor": list_2d(n_scales, n_simulations),
            "psi_true": list_2d(n_scales, n_simulations),
            "psi_taylor": list_2d(n_scales, n_simulations),
            "sm_true": list_2d(n_scales, n_simulations),
            "sm_taylor": list_2d(n_scales, n_simulations),
            "mu": list_2d(n_scales, n_simulations),
            "covariance": list_2d(n_scales, n_simulations),
            "scales": sigma_list,
        }

        for v, sigma_scale in enumerate(sigma_list):
            # loop over variance scales
            var_scale = sigma_scale / torch.tensor(n_dim / 3.0)

            for r in range(n_simulations):
                progress_str = f"covariance_type: {covariance_type}, " \
                  f"n_dim: {n_dim}, var_scale: {var_scale}, rep: {r}"
                print(progress_str)

                # Get parameters
                results['mu'][v][r], results['covariance'][v][r] = sample_parameters(
                    n_dim=n_dim, covariance_type=covariance_type
                )
                results['covariance'][v][r] = results['covariance'][v][r] * var_scale

                # Initialize the projected normal
                prnorm = ProjectedNormal(
                    n_dim=n_dim,
                    mu=results['mu'][v][r],
                    covariance=results['covariance'][v][r],
                    requires_grad=False,
                )

                # Get empirical moment estimates and unpack
                empirical_moments = prnorm.moments_empirical(
                    n_samples=empirical_samples
                )
                results['gamma_true'][v][r] = empirical_moments['gamma']
                results['psi_true'][v][r] = empirical_moments['psi']
                results['sm_true'][v][r] = empirical_moments['second_moment']

                # Get the Taylor approximation moments and unpack
                taylor_moments = prnorm.moments_approx()
                results['gamma_taylor'][v][r] = taylor_moments['gamma']
                results['psi_taylor'][v][r] = taylor_moments['psi']
                results['sm_taylor'][v][r] = taylor_moments['second_moment']

        #### CHANGE SAVING SO TO SAVE THE SINGLE DICTIONARY
        torch.save(
            results,
            results_dir + f"results_{covariance_type}_n_{n_dim}.pt",
        )

print(f"Time taken: {time.time() - start:.2f} seconds")
