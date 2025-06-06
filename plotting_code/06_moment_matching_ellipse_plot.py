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
  error_rel2,
  error_cos,
  error_stats,
  remove_mean_component,
)

# Plot the mean error as a function of the variance scale
plt.rcParams.update({'font.size': 14, 'font.family': 'Nimbus Sans'})

####################################
# 1) SETUP: DIRECTORIES, PARAMS
####################################
DATA_DIR = '../results/model_outputs/06_moment_matching_ellipse'
SAVE_DIR = '../results/plots/06_moment_matching_ellipse/'
os.makedirs(SAVE_DIR, exist_ok=True)

# List of dimensions to load
n_dim_list = [3, 6, 12, 24, 48]

####################################
# 2) LOAD THE FIT RESULTS
####################################

multipliers = [10, 50]


results_list = []
for mult in multipliers:
    results = []
    for n_dim in n_dim_list:
        filename = DATA_DIR + f'/results_n_{n_dim}_mult_{mult}.pt'

        results_n_dim = torch.load(filename, weights_only=True)
        results_n_dim = list_2_tensor_results(results_n_dim)

        results_n_dim['mean_y_error'] = error_rel2(
          results_n_dim['mean_y_fit_true'], results_n_dim['mean_y_fit_taylor']
        )
        results_n_dim['covariance_y_error'] = error_rel2(
          results_n_dim['covariance_y_fit_true'], results_n_dim['covariance_y_fit_taylor']
        )

        results_n_dim['mean_x_error'] = error_rel2(
          results_n_dim['mean_x'], results_n_dim['mean_x_fit']
        )
        results_n_dim['covariance_x_error'] = error_rel2(
          results_n_dim['covariance_x'], results_n_dim['covariance_x_fit']
        )
        results_n_dim['B_error'] = error_rel2(
          results_n_dim['B'], results_n_dim['B_fit']
        )

        results_n_dim['mean_x_cos'] = error_cos(
          results_n_dim['mean_x'], results_n_dim['mean_x_fit']
        )
        results_n_dim['covariance_x_cos'] = error_cos(
          results_n_dim['covariance_x'], results_n_dim['covariance_x_fit']
        )
        results_n_dim['B_cos'] = error_cos(
          results_n_dim['B'], results_n_dim['B_fit']
        )

        mean_x = results_n_dim['mean_x']
        cov_x = results_n_dim['covariance_x']
        x_sm = torch.einsum('...i,...j->...ij', mean_x, mean_x) + cov_x
        B_vec = results_n_dim['B_vecs'].squeeze()

        results_n_dim['B_vec_cos'] = torch.einsum(
          '...i,...ij,...j->...', B_vec, x_sm, B_vec
        )
        results.append(results_n_dim)
    results_list.append(results)


# For each dimension and variance number, choose the multiplier with
# the lowest error
results = results_list[0]
keys = list(results[0].keys())
keys.remove('sigma')
keys.remove('loss')

for n in range(len(n_dim_list)):
    for m in range(len(multipliers)):
        B_error = results[n]['B_error'].mean(dim=1)
        B_error_mult = results_list[m][n]['B_error'].mean(dim=1)
        substitute_inds = torch.where(B_error > B_error_mult)
        for key in keys:
            results[n][key][substitute_inds] = results_list[m][n][key][substitute_inds]


####################################
# 3) PLOT INDIVIDUAL EXAMPLES OF THE FITS
####################################

# Number of examples to plot for each parameter combination
n_examples = 4

for n, n_dim in enumerate(n_dim_list):

    for v, sigma in enumerate(results[n]['sigma']):

        for e in range(n_examples):

            # PLOT THE FITTED MEAN OF X
            plot_means(
              results[n], plot_type='fit', ind=(v, e), cos_sim=True,
            )
            filename = f'01_mu_fit_{n_dim}_sigma_{sigma}_{e}.pdf'
            plt.savefig(SAVE_DIR + filename, bbox_inches='tight')
            plt.close()

            # PLOT THE FITTED COVARIANCE OF X
            plot_covariances(
              results[n], plot_type='fit', ind=(v, e), cos_sim=True,
            )
            filename = f'02_covariance_fit_{n_dim}_sigma_{sigma}_{e}.pdf'
            plt.savefig(SAVE_DIR + filename, bbox_inches='tight')
            plt.close()

            # PLOT THE B MATRIX
            plot_covariances(
              results[n], plot_type='B', ind=(v, e), cos_sim=True,
              color_name='viridis',
            )
            #filename = f'03_B_fit_{n_dim}_sigma_{sigma}_{e}.pdf'
            filename = f'03_B_fit_{n_dim}_sigma_{sigma}_{e}.pdf'
            plt.savefig(SAVE_DIR + filename, bbox_inches='tight')
            plt.close()

            # PLOT THE TARGET AND FITTED COVARIANCE OF Y
#            plot_covariances(
#              results[n], plot_type='fit_approx', ind=(v, e), cos_sim=True,
#            )
#            filename = f'03_covariance_target_{n_dim}_sigma_{sigma}_{e}.pdf'
#            plt.savefig(SAVE_DIR + filename, bbox_inches='tight')
#            plt.close()


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
B_error = error_stats(
    torch.stack(
      [result['B_error'] for result in results]
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
B_cos = error_stats(
    torch.stack(
      [result['B_cos'] for result in results]
    )
)


ymin = 0.001

# Plot mean_x error
plot_error_stats(
  error_dict=mean_x_error,
  error_label=r'$\mathrm{Error}_{\mu}$ (%)',
  n_dim_list=n_dim_list,
  sigma_vec=results[0]['sigma'],
  ymin=ymin,
)
plt.savefig(SAVE_DIR + f'04_mu.pdf', bbox_inches='tight')
plt.close()

# Plot covariance_x error
plot_error_stats(
  error_dict=cov_x_error,
  error_label=r'$\mathrm{Error}_{\Sigma}$ (%)',
  n_dim_list=n_dim_list,
  sigma_vec=results[0]['sigma'],
  ymin=ymin,
)
plt.savefig(SAVE_DIR + f'04_sigma.pdf', bbox_inches='tight')
plt.close()

# Plot covariance_x error
plot_error_stats(
  error_dict=B_error,
  error_label=r'$\mathrm{Error}_{B}$ (%)',
  n_dim_list=n_dim_list,
  sigma_vec=results[0]['sigma'],
  ymin=ymin,
)
plt.savefig(SAVE_DIR + f'04_B.pdf', bbox_inches='tight')
plt.close()


ymin_cos = 0.9
ymax_cos = 1.0005

# Plot gamma error
plot_error_stats(
  error_dict=mean_x_cos,
  error_label=r'Cosine similarity ($\mu$)',
  n_dim_list=n_dim_list,
  sigma_vec=results[0]['sigma'],
  ymax=ymax_cos,
  ymin=ymin_cos,
  logscale=False,
)
plt.savefig(SAVE_DIR + f'05_cos_mu.pdf', bbox_inches='tight')
plt.close()

# Plot psi error
plot_error_stats(
  error_dict=cov_x_cos,
  error_label=r'Cosine similarity ($\Sigma$)',
  n_dim_list=n_dim_list,
  sigma_vec=results[0]['sigma'],
  ymax=ymax_cos,
  ymin=ymin_cos,
  logscale=False,
)
plt.savefig(SAVE_DIR + f'05_cos_sigma.pdf', bbox_inches='tight')
plt.close()

# Plot B error
plot_error_stats(
  error_dict=B_cos,
  error_label=r'Cosine similarity ($B$)',
  n_dim_list=n_dim_list,
  sigma_vec=results[0]['sigma'],
  ymax=ymax_cos,
  ymin=ymin_cos,
  logscale=False,
)
plt.savefig(SAVE_DIR + f'05_cos_B.pdf', bbox_inches='tight')
plt.close()


#n = 3
#s = 3
#plt.scatter(
#  torch.abs(results[n]['B_vec_cos'].reshape(-1)),
#  results[n]['B_cos'].reshape(-1),
#)
#plt.show()
#
#
#n = 3
#s = 3
#plt.scatter(
#  torch.abs(results[n]['B_vec_cos'].reshape(-1)),
#  results[n]['B_error'].reshape(-1),
#)
#plt.show()
#
#
#
#
#
#fig, ax = plt.subplots(1, 2, figsize=(8, 6))
#n=2
#v=1
#r=2
#ax[0].imshow(results[n]['covariance_y_true'][v][r])
#ax[1].imshow(results[n]['covariance_y_fit_taylor'][v][r])
#plt.show()
#
#plt.plot(results[n]['loss'][v][r])
#plt.yscale('log')
#plt.show()
#
