"""
Plot the taylor approximation to the moments of the n-dimensional projected normal.
"""
import os
import sys
import torch
import projnormal as pn
import matplotlib.pyplot as plt

from functions_plotting_stats import (
  plot_means,
  plot_samples,
  plot_covariances,
  plot_error_stats,
)
from functions_results_processing import (
  list_2_tensor_results,
  error_rel,
  error_cos,
  error_stats,
)

plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})

####################################
# 1) SETUP: DIRECTORIES, PARAMS
####################################
DATA_DIR = '../results/model_outputs/03_approximation_const/'
SAVE_DIR = '../results/plots/03_approximation_const/'
os.makedirs(SAVE_DIR, exist_ok=True)

# PARAMETERS OF SIMULATIONS TO LOAD
eigvals = 'exponential'
eigvecs = 'random'
n_dim_list = [3, 6, 12, 24, 48]

####################################
# 2) LOAD THE SIMULATION RESULTS
####################################
results = []
for n_dim in n_dim_list:
    filename = DATA_DIR + f'results_eigvals_{eigvals}_eigvecs_{eigvecs}_n_{n_dim}.pt'
    results_n_dim = torch.load(filename, weights_only=True)
    results_n_dim = list_2_tensor_results(results_n_dim)

    results_n_dim['mean_y_error'] = error_rel(
      results_n_dim['mean_y_true'], results_n_dim['mean_y_taylor']
    )
    results_n_dim['covariance_y_error'] = error_rel(
      results_n_dim['covariance_y_true'], results_n_dim['covariance_y_taylor']
    )
    results_n_dim['mean_y_cos'] = error_cos(
      results_n_dim['mean_y_true'], results_n_dim['mean_y_taylor']
    )
    results_n_dim['covariance_y_cos'] = error_cos(
      results_n_dim['covariance_y_true'], results_n_dim['covariance_y_taylor']
    )
    results.append(results_n_dim)


####################################
# 3) PLOT INDIVIDUAL EXAMPLES OF THE APPROXIMATIONS
####################################

# Number of examples to plot for each parameter combination
n_examples = 3
# samples to plot for illustration
n_samples = 100

for n, n_dim in enumerate(n_dim_list):
    sigma_scale_vec = results[n]['sigma']

    for v, sigma in enumerate(sigma_scale_vec):

        for e in range(n_examples):

            prnorm = pn.models.ProjNormalConst(
              mean_x=results[n]['mean_x'][v,e],
              covariance_x=results[n]['covariance_x'][v,e],
              const=results[n]['const'][v,e],
            )

            ax = plot_samples(
              prnorm=prnorm, n_samples=n_samples
            )
            plot_means(
              results[n], plot_type='approx', ind=(v, e), ax=ax, cos_sim=True,
            )
            plt.savefig(SAVE_DIR + f'01_mean_example_{eigvals}_{eigvecs}_dim_{n_dim}_'\
                f'sigma{sigma}_{e}.pdf', bbox_inches='tight')
            plt.close()

            # PLOT THE APPROXIMATED AND TRUE COVARIANCE
            plot_covariances(
              results[n], plot_type='approx', ind=(v, e), cos_sim=True,
            )
            plt.savefig(SAVE_DIR + f'02_covariance_example_{eigvals}_{eigvecs}_dim_{n_dim}_'\
                f'sigma{sigma}_{e}.pdf', bbox_inches='tight')
            plt.close()


####################################
# 4) PLOT THE ERRORS AS A FUNCTION OF DIMENSION AND VARIANCE SCALE
####################################

error_stats_mean = error_stats(
    torch.stack(
      [result['mean_y_error'] for result in results]
    )
)
error_stats_cov = error_stats(
    torch.stack(
      [result['covariance_y_error'] for result in results]
    )
)
cos_stats_mean = error_stats(
    torch.stack(
      [result['mean_y_cos'] for result in results]
    )
)
cos_stats_cov = error_stats(
    torch.stack(
      [result['covariance_y_cos'] for result in results]
    )
)

# Plot gamma error
plot_error_stats(
  error_dict=error_stats_mean,
  error_label=r'$\mathrm{Error}_{\gamma}$ (%)',
  n_dim_list=n_dim_list,
  sigma_vec=results[0]['sigma'],
)
plt.savefig(SAVE_DIR + f'03_gamma_{eigvals}_{eigvecs}.pdf', bbox_inches='tight')
plt.close()

# Plot psi error
plot_error_stats(
  error_dict=error_stats_cov,
  error_label=r'$\mathrm{Error}_{\Psi}$ (%)',
  n_dim_list=n_dim_list,
  sigma_vec=results[0]['sigma'],
)
plt.savefig(SAVE_DIR + f'03_psi_{eigvals}_{eigvecs}.pdf', bbox_inches='tight')
plt.close()

# Plot gamma error
plot_error_stats(
  error_dict=cos_stats_mean,
  error_label='Cosine similarity',
  n_dim_list=n_dim_list,
  sigma_vec=results[0]['sigma'],
  ymax=1.0005,
  logscale=False,
)
plt.savefig(SAVE_DIR + f'04_gamma_{eigvals}_{eigvecs}.pdf', bbox_inches='tight')
plt.close()

# Plot psi error
plot_error_stats(
  error_dict=cos_stats_cov,
  error_label='Cosine similarity',
  n_dim_list=n_dim_list,
  sigma_vec=results[0]['sigma'],
  ymax=1.0005,
  logscale=False,
)
plt.savefig(SAVE_DIR + f'04_psi_{eigvals}_{eigvecs}.pdf', bbox_inches='tight')
plt.close()
