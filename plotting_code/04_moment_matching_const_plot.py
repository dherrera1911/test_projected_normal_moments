"""
Plot the moment matching results for the n-dimensional projected normal.
"""
import os
import sys
import torch
import numpy as np
import projnormal as pn
import matplotlib.pyplot as plt

from functions_plotting_stats import (
  plot_means,
  plot_samples,
  plot_covariances,
  plot_error_stats,
  plot_error_scatters,
)
from functions_results_processing import (
  list_2_tensor_results,
  error_rel,
  error_cos,
  error_stats,
  remove_mean_component,
)

# Plot the mean error as a function of the variance scale
plt.rcParams.update({'font.size': 14, 'font.family': 'Nimbus Sans'})


####################################
# 1) SETUP: DIRECTORIES, PARAMS
####################################
DATA_DIR = '../results/model_outputs/08_nd_moment_matching_const/'
SAVE_DIR = '../results/plots/08_nd_moment_matching_const/'
os.makedirs(SAVE_DIR, exist_ok=True)

# Choose your eigenvalue, eigenvector scenarios
eigvals = 'exponential'
eigvecs = 'random'

# List of dimensions to load
n_dim_list = [3, 6, 12, 24, 48]


####################################
# 2) LOAD THE FIT RESULTS
####################################

results = []
for n_dim in n_dim_list:
    filename = DATA_DIR + f'results_eigvals_{eigvals}_eigvecs_{eigvecs}_n_{n_dim}.pt'
    results_n_dim = torch.load(filename, weights_only=True)
    results_n_dim = list_2_tensor_results(results_n_dim)

    results_n_dim['mean_y_error'] = error_rel(
      results_n_dim['mean_y_fit_true'], results_n_dim['mean_y_fit_taylor']
    )
    results_n_dim['covariance_y_error'] = error_rel(
      results_n_dim['covariance_y_fit_true'], results_n_dim['covariance_y_fit_taylor']
    )

    results_n_dim['mean_x_error'] = error_rel(
      results_n_dim['mean_x'], results_n_dim['mean_x_fit']
    )
    results_n_dim['covariance_x_error'] = error_rel(
      results_n_dim['covariance_x'], results_n_dim['covariance_x_fit']
    )

    results_n_dim['covariance_x_ort'] = remove_mean_component(
      results_n_dim['mean_x'], results_n_dim['covariance_x']
    )
    results_n_dim['covariance_x_fit_ort'] = remove_mean_component(
      results_n_dim['mean_x'], results_n_dim['covariance_x_fit']
    )

    results_n_dim['covariance_x_ort_error'] = error_rel(
      results_n_dim['covariance_x_ort'], results_n_dim['covariance_x_fit_ort']
    )

    results_n_dim['const_error'] = error_rel(
      results_n_dim['const'].squeeze(), results_n_dim['const_fit']
    )

    results_n_dim['mean_x_cos'] = error_cos(
      results_n_dim['mean_x'], results_n_dim['mean_x_fit']
    )
    results_n_dim['covariance_x_cos'] = error_cos(
      results_n_dim['covariance_x'], results_n_dim['covariance_x_fit']
    )

    results.append(results_n_dim)


####################################
# 3) PLOT INDIVIDUAL EXAMPLES OF THE FITS
####################################

# Number of examples to plot for each parameter combination
n_examples = 3

for n, n_dim in enumerate(n_dim_list):

    for v, sigma in enumerate(results[n]['sigma']):

        for e in range(n_examples):

            # PLOT THE FITTED MEAN OF X
            plot_means(
              results[n], plot_type='fit', ind=(v, e), cos_sim=True,
            )
            filename = f'01_mu_fit_{n_dim}_{eigvals}_{eigvecs}_sigma_{sigma}_{e}.pdf'
            plt.savefig(SAVE_DIR + filename, bbox_inches='tight')
            plt.close()


            # PLOT THE FITTED COVARIANCE OF X
            plot_covariances(
              results[n], plot_type='fit', ind=(v, e), cos_sim=True,
            )
            filename = f'02_covariance_fit_{n_dim}_{eigvals}_{eigvecs}_sigma_{sigma}_{e}.pdf'
            plt.savefig(SAVE_DIR + filename, bbox_inches='tight')
            plt.close()

            # PLOT THE FITTED COVARIANCE OF X ORTHOGONAL
            plot_covariances(
              results[n], plot_type='ort', ind=(v, e), cos_sim=True,
            )
            filename = f'03_covariance_ort_fit_{n_dim}_{eigvals}_{eigvecs}_sigma_{sigma}_{e}.pdf'
            plt.savefig(SAVE_DIR + filename, bbox_inches='tight')
            plt.close()

####################################
# 4) PLOT THE ERROR CURVES
####################################

mean_x_error = error_stats(
    torch.stack(
      [result['mean_x_error'] for result in results]
    )
)
cov_x_error = error_stats(
    torch.stack(
      [result['covariance_x_error'] for result in results]
    )
)
cov_x_error_ort = error_stats(
    torch.stack(
      [result['covariance_x_ort_error'] for result in results]
    )
)
mean_y_error_taylor = error_stats(
    torch.stack(
      [result['mean_y_error'] for result in results]
    )
)
cov_y_error_taylor = error_stats(
    torch.stack(
      [result['covariance_y_error'] for result in results]
    )
)
mean_x_cos = error_stats(
    torch.stack(
      [result['mean_x_cos'] for result in results]
    )
)
cov_x_cos = error_stats(
    torch.stack(
      [result['covariance_x_cos'] for result in results]
    )
)
const_error = error_stats(
    torch.stack(
      [result['const_error'] for result in results]
    )
)


# Plot mean_x error
plot_error_stats(
  error_dict=mean_x_error,
  error_label=r'$\mathrm{Error}_{\mu}$ (%)',
  n_dim_list=n_dim_list,
  sigma_vec=results[0]['sigma'],
)
plt.savefig(SAVE_DIR + f'04_mu_{eigvals}_{eigvecs}.pdf', bbox_inches='tight')
plt.close()


# Plot covariance_x error
plot_error_stats(
  error_dict=cov_x_error,
  error_label=r'$\mathrm{Error}_{\Sigma}$ (%)',
  n_dim_list=n_dim_list,
  sigma_vec=results[0]['sigma'],
)
plt.savefig(SAVE_DIR + f'04_sigma_{eigvals}_{eigvecs}.pdf', bbox_inches='tight')
plt.close()


# Plot const error
plot_error_stats(
  error_dict=const_error,
  error_label=r'$\mathrm{Error}_{c}$ (%)',
  n_dim_list=n_dim_list,
  sigma_vec=results[0]['sigma'],
)
plt.savefig(SAVE_DIR + f'04_const_{eigvals}_{eigvecs}.pdf', bbox_inches='tight')
plt.close()


# Plot gamma error
plot_error_stats(
  error_dict=mean_x_cos,
  error_label=r'Cosine similarity ($\mu$)',
  n_dim_list=n_dim_list,
  sigma_vec=results[0]['sigma'],
  ymax=1.0005,
  logscale=False,
)
plt.savefig(SAVE_DIR + f'05_cos_mu_{eigvals}_{eigvecs}.pdf', bbox_inches='tight')
plt.close()

# Plot psi error
plot_error_stats(
  error_dict=cov_x_cos,
  error_label=r'Cosine similarity ($\Sigma$)',
  n_dim_list=n_dim_list,
  sigma_vec=results[0]['sigma'],
  ymax=1.0005,
  logscale=False,
)
plt.savefig(SAVE_DIR + f'05_cos_sigma_{eigvals}_{eigvecs}.pdf', bbox_inches='tight')
plt.close()


# Plot covariance_x orthogonal error
plot_error_stats(
  error_dict=cov_x_error_ort,
  error_label=r'$\mathrm{Error}_{\Sigma}$ (%)',
  n_dim_list=n_dim_list,
  sigma_vec=results[0]['sigma'],
)
plt.savefig(SAVE_DIR + f'06_sigma_ort_{eigvals}_{eigvecs}.pdf', bbox_inches='tight')
plt.close()

# Plot covariance_x vs covariance_x orthogonal error
plot_error_scatters(
  results=results,
  x_key='covariance_x_error',
  y_key='covariance_x_ort_error',
  n_dim_list=n_dim_list,
  labels=[
    r'$\mathrm{Error}_{\Sigma}$ (%)',
    r'$\mathrm{Error}_{\Sigma}$ Orthogonal (%)'
  ]
)
plt.plot([0, 100], [0, 100], 'k--', linewidth=3)
plt.savefig(SAVE_DIR + f'07_scatter_ort_{eigvals}_{eigvecs}.pdf', bbox_inches='tight')
plt.close()

# Plot error against approximation error
plot_error_scatters(
  results=results,
  x_key='covariance_y_error',
  y_key='covariance_x_error',
  n_dim_list=n_dim_list,
  labels=[
    r'$\mathrm{Error}_{\Psi}$ (%)',
    r'$\mathrm{Error}_{\Sigma}$ (%)'
  ]
)
plt.savefig(SAVE_DIR + f'08_scatter_fit_vs_approx_{eigvals}_{eigvecs}.pdf', bbox_inches='tight')
plt.close()


# Plot covariance_x vs covariance_x orthogonal error
plot_error_scatters(
  results=results,
  x_key='const_error',
  y_key='covariance_x_error',
  n_dim_list=n_dim_list,
  labels=[
    r'$\mathrm{Error}_{c}$ (%)',
    r'$\mathrm{Error}_{\Sigma}$ (%)'
  ]
)
plt.savefig(SAVE_DIR + f'08_scatter_const_{eigvals}_{eigvecs}.pdf', bbox_inches='tight')
plt.close()


plot_error_scatters(
  results=results,
  x_key='const',
  y_key='covariance_x_error',
  n_dim_list=n_dim_list,
  labels=[
    r'$\mathrm{Error}_{c}$ (%)',
    r'$\mathrm{Error}_{\Sigma}$ (%)'
  ]
)
plt.savefig(SAVE_DIR + f'9_scatter_const_{eigvals}_{eigvecs}.pdf', bbox_inches='tight')
plt.close()

