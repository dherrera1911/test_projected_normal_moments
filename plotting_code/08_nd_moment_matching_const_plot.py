"""
Plot the moment matching results for the n-dimensional projected normal.
"""

import os
import sys
import torch
import numpy as np
import projnormal as pn
import matplotlib.pyplot as plt

# If you have custom plotting functions (same as in your other scripts)
from plotting_functions import (
    plot_means,
    draw_covariance_images,
    add_colorbar
)

####################################
# 0) HELPER FUNCTIONS
####################################

def list_2_tensor(tensor_list):
    """
    Convert a list of lists of Tensors to a 4D Tensor if covariances,
    or 3D if means. (Same as in your other scripts.)
    """
    return torch.stack([torch.stack(tensor_list[i]) for i in range(len(tensor_list))])

def error_rel(x, y):
    """
    Computes a relative error (in %) for vector (dim=3) or matrix (dim=4) data.
    E.g., 2*||x - y|| / (||x|| + ||y||).
    """
    if x.dim() == 3:
        error = 2.0 * torch.norm(x - y, dim=2) \
                / (torch.norm(x, dim=2) + torch.norm(y, dim=2))
    elif x.dim() == 4:
        error = 2.0 * torch.norm(x - y, dim=(2,3)) \
                / (torch.norm(x, dim=(2,3)) + torch.norm(y, dim=(2,3)))
    return error * 100.0

def error_mat(x, y):
    """
    Alternative matrix error. Adjust if needed.
    """
    diff = x - y
    dist = torch.linalg.matrix_norm(diff, ord=2)
    denom = 0.5 * (torch.linalg.matrix_norm(x, ord=2)
                   + torch.linalg.matrix_norm(y, ord=2))
    return (dist / denom) * 100.0


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
mean_x = []
mean_x_fit = []

mean_x = []
mean_x_fit = []
cov_x = []
cov_x_fit = []

const = []
const_mult = []
const_fit = []

cov_y_fit_taylor = []
cov_y_fit_true = []
cov_y = []

mean_y_fit_taylor = []
mean_y_fit_true = []
mean_y = []

mean_x_error = []
cov_x_error = []
const_error = []
mean_y_error_taylor = []
cov_y_error_taylor = []

sigma_scale_vec = []
var_scale_vec = []

for n_dim in n_dim_list:
    filename = DATA_DIR + f'results_eigvals_{eigvals}_eigvecs_{eigvecs}_n_{n_dim}.pt'
    results = torch.load(filename, weights_only=True)
    # Distribution parameters
    mean_x.append(
      list_2_tensor(results['mean_x'])
    )
    mean_x_fit.append(
      list_2_tensor(results['mean_x_fit'])
    )
    cov_x.append(
      list_2_tensor(results['covariance_x'])
    )
    cov_x_fit.append(
      list_2_tensor(results['covariance_x_fit'])
    )
    const.append(
      list_2_tensor(results['const']).squeeze()
    )
    const_mult.append(
      list_2_tensor(results['const_mult']).squeeze()
    )
    const_fit.append(
      list_2_tensor(results['const_fit'])
    )
    # Distributin moments
    mean_y.append(
      list_2_tensor(results['mean_y_true'])
    )
    mean_y_fit_taylor.append(
      list_2_tensor(results['mean_y_fit_taylor'])
    )
    mean_y_fit_true.append(
      list_2_tensor(results['mean_y_fit_true'])
    )
    cov_y.append(
      list_2_tensor(results['covariance_y_true'])
    )
    cov_y_fit_taylor.append(
      list_2_tensor(results['covariance_y_fit_taylor'])
    )
    cov_y_fit_true.append(
      list_2_tensor(results['covariance_y_fit_true'])
    )
    sigma_scale_vec.append(
      torch.as_tensor(results['sigma'])
    )
    var_scale_vec.append(
      sigma_scale_vec[-1] ** 2 / n_dim
    )
    # Errors for parameters
    mean_x_error.append(error_rel(mean_x[-1], mean_x_fit[-1]))
    cov_x_error.append(error_rel(cov_x[-1], cov_x_fit[-1]))
    const_error.append(
      torch.abs(const[-1] - const_fit[-1]) * 2 / (const_mult[-1] + const_fit[-1]) * 100.0
    )
    # Error for moments
    mean_y_error_taylor.append(
      error_rel(mean_y_fit_true[-1], mean_y_fit_taylor[-1])
    )
    cov_y_error_taylor.append(
      error_rel(cov_y_fit_true[-1], cov_y_fit_taylor[-1])
    )


####################################
# 3) PLOT INDIVIDUAL EXAMPLES OF THE FITS
####################################

# Number of examples to plot for each parameter combination
n_examples = 5

for n, n_dim in enumerate(n_dim_list):
    sigma_scale_vec_n = sigma_scale_vec[n]

    for v, sigma in enumerate(sigma_scale_vec_n):

        for e in range(n_examples):

            for p in range(2):

                if p == 0:
                    # Plot Y approximation ellipses
                    labels = ['True', 'Approximation']
                    mean = [mean_y_fit_true[n][v,e], mean_y_fit_taylor[n][v,e]]
                    cov = [cov_y_fit_true[n][v,e], cov_y_fit_taylor[n][v,e]]
                    error = [mean_y_error_taylor[n][v,e], cov_y_error_taylor[n][v,e]]
                    title = f'Y approximation, $\sigma^2$={sigma**2:.2f}'
                    filename = f'01_Y_approximation_{eigvals}_{eigvecs}_sigma_{sigma}_{e}.png'
                    prefix = r'$y$'
                else:
                    # Plot X fit ellipses
                    labels = ['True', 'Fit']
                    mean = [mean_x[n][v,e], mean_x_fit[n][v,e]]
                    cov = [cov_x[n][v,e], cov_x_fit[n][v,e]]
                    error = [mean_x_error[n][v,e], cov_x_error[n][v,e]]
                    title = f'X fit ellipses, $\sigma^2$={sigma**2:.2f}'
                    filename = f'02_fit_ellipses_{eigvals}_{eigvecs}_sigma_{sigma}_{e}.png'
                    prefix = r'$x$'

                ### PLOT GAMMA (MEAN) VECTORS
                plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})

                fig, ax = plt.subplots(figsize=(3.5, 3.5))
                # Plot the true and approximated means
                plot_means(
                  axes=ax, mean_list=mean, color_list=['blue', 'red'],
                  name_list=labels, linewidth=2
                )
                ax.legend(
                  loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.18), fontsize='small'
                )

                # Print the approximation errors on the plot (on top of a white rectangle)
                rect = plt.Rectangle(
                  (0.52, 0.9), 0.48, 0.09, fill=True, color='white', alpha=1,
                  transform=ax.transAxes, zorder=1000
                )
                ax.add_patch(rect)
                ax.text(
                  0.75, 0.93, r'Error:' f' {error[0]:.2f}%',
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax.transAxes, zorder=1001
                )

                # Set the labels and save the plot
                lim = torch.max(torch.abs(torch.stack(mean))) * 1.2
                plt.tight_layout()
                plt.ylabel('Value')
                plt.xlabel('Dimension')
                plt.ylim([-lim, lim])
                plt.savefig(SAVE_DIR + f'01_mean_example_{eigvals}_{eigvecs}_dim_{n_dim}_'\
                    f'sigma{sigma}_{e}.png', bbox_inches='tight')
                plt.close()

                ### DRAW COVARIANCE MATRICES IMAGES
                # Setup
                plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
                fig, ax = plt.subplots(1, 2, figsize=(6, 3))
                cmap = plt.get_cmap('bwr')

                # Get maximum absolute value of the covariance matrices
                max_val = torch.max(torch.abs(torch.stack(cov))) * 1.1
                min_val = -max_val
                color_bounds = [min_val, max_val]

                # Draw the covariance matrices
                draw_covariance_images(
                  axes=ax, cov_list=cov, label_list=labels, cmap=cmap
                )
                add_colorbar(
                  ax=ax[-1], color_bounds=color_bounds, cmap=cmap, label='Value',
                  fontsize=12, width=0.015, loc=0.997
                )

                # Print the approximation errors on the plot (on top of a white rectangle)
                rect = plt.Rectangle(
                  (0.52, 0.9), 0.48, 0.09, fill=True, color='white',
                  alpha=1, zorder=1000, transform=ax[-1].transAxes
                )
                ax[-1].add_patch(rect)
                ax[-1].text(
                  0.75, 0.93, r'Error' f'={error[1]:.2f}%',
                  horizontalalignment='center', verticalalignment='center',
                  transform=ax[-1].transAxes, zorder=1001
                )

                # Save the plot
                plt.tight_layout()
                plt.savefig(SAVE_DIR + f'02_covariance_example_{eigvals}_{eigvecs}_dim_{n_dim}_'\
                    f'sigma{sigma}_{e}.png', bbox_inches='tight')
                plt.close()


####################################
# 4) PLOT THE ERROR CURVES
####################################

# Plot the mean error as a function of the variance scale
plt.rcParams.update({'font.size': 14, 'font.family': 'Nimbus Sans'})

mean_x_error = torch.stack(mean_x_error)
cov_x_error = torch.stack(cov_x_error)
const_error = torch.stack(const_error)
mean_y_error = torch.stack(mean_y_error_taylor)
cov_y_error = torch.stack(cov_y_error_taylor)

error_stats_x_mean = {
  'median': mean_x_error.median(dim=-1).values,
  'q1': mean_x_error.quantile(0.25, dim=-1),
  'q3': mean_x_error.quantile(0.75, dim=-1)
}
error_stats_x_cov = {
  'median': cov_x_error.median(dim=-1).values,
  'q1': cov_x_error.quantile(0.25, dim=-1),
  'q3': cov_x_error.quantile(0.75, dim=-1)
}
error_stats_const = {
  'median': const_error.median(dim=-1).values,
  'q1': const_error.quantile(0.25, dim=-1),
  'q3': const_error.quantile(0.75, dim=-1)
}

error_stats_y_mean = {
  'median': mean_y_error.median(dim=-1).values,
  'q1': mean_y_error.quantile(0.25, dim=-1),
  'q3': mean_y_error.quantile(0.75, dim=-1)
}
error_stats_y_cov = {
  'median': cov_y_error.median(dim=-1).values,
  'q1': cov_y_error.quantile(0.25, dim=-1),
  'q3': cov_y_error.quantile(0.75, dim=-1)
}


error_stats = [
  error_stats_x_mean,
  error_stats_x_cov,
  error_stats_const,
]
error_name = [
  'Mean',
  'Covariance',
  'Constant',
]
error_label = [
  r'$\mathrm{Error}_{\mu}$ (%)',
  r'$\mathrm{Error}_{\Sigma}$ (%)',
  r'$\mathrm{Error}_{c}$ (%)',
]
colors = plt.cm.tab10(torch.arange(len(n_dim_list)).numpy())

for e, plt_stats in enumerate(error_stats):
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
    for n, n_dim in enumerate(n_dim_list):
        x_plt = sigma_scale_vec[n]

        # Plot median error with bars showing quartiles
        yerr = torch.stack([plt_stats['q1'][n], plt_stats['q3'][n]], dim=0)
        # Plot each n_val in its own color and label
        ax.errorbar(x_plt,
                    plt_stats['median'][n],
                    yerr=yerr,
                    fmt='o-',
                    color=colors[n],
                    label=f'n={n_dim}')

        if e==2 or e==3:
            ax.set_ylim([0.001, 100])

        ax.set_xlabel('Variance scale (s)')
        ax.set_ylabel(error_label[e])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.25),
                  fontsize='small')
    plt.tight_layout()
    plt.savefig(SAVE_DIR + f'03_{error_name[e]}_{eigvals}_{eigvecs}.png',
                bbox_inches='tight')
    plt.close()

