"""
Plot the taylor approximation to the moments of the n-dimensional projected normal.
"""
import os
import sys
import torch
import numpy as np
import projnormal as pn
import matplotlib.pyplot as plt

from plotting_functions import (
  plot_means,
  draw_covariance_images,
  add_colorbar
)

# Extract the results
def list_2_tensor(tensor_list):
    return torch.stack([torch.stack(tensor_list[i]) for i in range(len(tensor_list))])

def error_rel(x, y):
    if x.dim()==3:
        error = 2 * torch.norm(x - y, dim=2) \
            / (torch.norm(x, dim=2) + torch.norm(y, dim=2))
    elif x.dim()==4:
        error = 2 * torch.norm(x - y, dim=(2,3)) \
            / (torch.norm(x, dim=(2,3)) + torch.norm(y, dim=(2,3)))
    return error * 100

def error_mat(x, y):
    diff = x - y
    dist = torch.linalg.matrix_norm(diff, ord=2)
    dist = torch.sqrt(dist)
    return dist

####################################
# 1) SETUP: DIRECTORIES, PARAMS
####################################
DATA_DIR = '../results/model_outputs/02_nd_approximation/'
SAVE_DIR = '../results/plots/02_nd_approximation_performance/'
os.makedirs(SAVE_DIR, exist_ok=True)

# PARAMETERS OF SIMULATIONS TO LOAD
eigvals = 'uniform'
eigvecs = 'random'
n_dim_list = [3, 6, 12, 24, 48]

####################################
# 2) LOAD THE SIMULATION RESULTS
####################################
mean_x = []
mean_y = []
mean_y_taylor = []
covariance_x = []
covariance_y = []
covariance_y_taylor = []
mean_y_error = []
covariance_y_error = []
sigma_scale = []
var_scale = []

for n_dim in n_dim_list:
    filename = DATA_DIR + f'results_eigvals_{eigvals}_eigvecs_{eigvecs}_n_{n_dim}.pt'
    results = torch.load(filename, weights_only=True)
    mean_x.append(
      list_2_tensor(results['mean_x'])
    )
    covariance_x.append(
      list_2_tensor(results['covariance_x'])
    )
    covariance_y_taylor.append(
      list_2_tensor(results['covariance_y_taylor'])
    )
    mean_y_taylor.append(
      list_2_tensor(results['mean_y_taylor'])
    )
    covariance_y.append(
      list_2_tensor(results['covariance_y_true'])
    )
    mean_y.append(
      list_2_tensor(results['mean_y_true'])
    )
    sigma_scale.append(
      torch.as_tensor(results['sigma'])
    )
    var_scale.append(
      sigma_scale[-1] ** 2 / n_dim
    )
    mean_y_error.append(
      error_rel(mean_y[-1], mean_y_taylor[-1])
    )
    covariance_y_error.append(
      error_rel(covariance_y[-1], covariance_y_taylor[-1])
    )
    # Compute errors
    #covariance_y_error.append(
    #  error_mat(covariance_y, covariance_y_taylor)
    #)


####################################
# 3) PLOT INDIVIDUAL EXAMPLES OF THE APPROXIMATIONS
####################################

# Number of examples to plot for each parameter combination
n_examples = 5
# samples to plot for illustration
n_points = 100

for n, n_dim in enumerate(n_dim_list):
    sigma_scale_vec = sigma_scale[n]

    for v, sigma in enumerate(sigma_scale_vec):

        for e in range(n_examples):

            prnorm = pn.models.ProjNormal(
              mean_x=mean_x[n][v,e],
              covariance_x=covariance_x[n][v,e]
            )

            # Sample from the projected normal
            with torch.no_grad():
                samples = prnorm.sample(n_samples=n_points)

            ### PLOT GAMMA (MEAN) VECTORS
            plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})

            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            # Plot the samples
            for i in range(n_points):
                plot_means(
                  axes=ax, mean_list=[samples[i]], color_list=['black'],
                  name_list=['_nolegend_'], alpha=0.2
                )
            # Plot the true and approximated means
            plot_means(
              axes=ax, mean_list=[mean_y_taylor[n][v,e], mean_y[n][v,e]],
              color_list=['blue', 'red'], name_list=['Approximation', 'Empirical'],
              linewidth=2
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
              0.75, 0.93, r'$\mathrm{Error}_{\gamma}$:' f' {mean_y_error[n][v,e]:.2f}%',
              horizontalalignment='center', verticalalignment='center',
              transform=ax.transAxes, zorder=1001
            )

            # Set the labels and save the plot
            plt.tight_layout()
            plt.ylabel('Value')
            plt.xlabel('Dimension')
            plt.ylim([-0.85, 0.85])
            plt.savefig(SAVE_DIR + f'01_mean_example_{eigvals}_{eigvecs}_dim_{n_dim}_'\
                f'sigma{sigma}_{e}.png', bbox_inches='tight')
            plt.close()


            ### DRAW COVARIANCE MATRICES IMAGES

            # Setup
            plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
            fig, ax = plt.subplots(1, 2, figsize=(6, 3))
            cmap = plt.get_cmap('bwr')

            # Format data, make labels list and find color scale bounds
            label_list = [r'$\Psi_{T}$', r'$\Psi_{T} - \Psi$']
            cov_list = [
              covariance_y[n][v,e], covariance_y[n][v,e] - covariance_y_taylor[n][v,e]
            ]

            # Get maximum absolute value of the covariance matrices
            max_val = torch.max(torch.abs(torch.stack(cov_list))) * 1.1
            min_val = -max_val
            color_bounds = [min_val, max_val]

            # Draw the covariance matrices
            draw_covariance_images(
              axes=ax, cov_list=cov_list, label_list=label_list, cmap=cmap
            )

            add_colorbar(ax=ax[-1], color_bounds=color_bounds, cmap=cmap, label='Value',
                         fontsize=12, width=0.015, loc=0.997)

            # Print the approximation errors on the plot (on top of a white rectangle)
            rect = plt.Rectangle(
              (0.52, 0.9), 0.48, 0.09, fill=True, color='white',
              alpha=1, zorder=1000, transform=ax[-1].transAxes
            )
            ax[-1].add_patch(rect)
            ax[-1].text(
              0.75, 0.93, r'$\mathrm{Error}_{\Psi}$' f'={covariance_y_error[n][v,e]:.2f}%',
              horizontalalignment='center', verticalalignment='center',
              transform=ax[-1].transAxes, zorder=1001
            )

            # Save the plot
            plt.tight_layout()
            plt.savefig(SAVE_DIR + f'02_covariance_example_{eigvals}_{eigvecs}_dim_{n_dim}_'\
                f'sigma{sigma}_{e}.png', bbox_inches='tight')
            plt.close()


####################################
# 4) PLOT THE ERRORS AS A FUNCTION OF DIMENSION AND VARIANCE SCALE
####################################

# Plot the mean error as a function of the variance scale
plt.rcParams.update({'font.size': 14, 'font.family': 'Nimbus Sans'})

mean_y_error = torch.stack(mean_y_error)
covariance_y_error = torch.stack(covariance_y_error)

error_stats_mean = {
  'median': mean_y_error.median(dim=-1).values,
  'q1': mean_y_error.quantile(0.25, dim=-1),
  'q3': mean_y_error.quantile(0.75, dim=-1)
}
error_stats_cov = {
  'median': covariance_y_error.median(dim=-1).values,
  'q1': covariance_y_error.quantile(0.25, dim=-1),
  'q3': covariance_y_error.quantile(0.75, dim=-1)
}

error_stats = [error_stats_mean, error_stats_cov]
error_name = ['Mean', 'Covariance']
error_label = [r'$\mathrm{Error}_{\gamma}$ (%)', r'$\mathrm{Error}_{\Psi}$ (%)']
colors = plt.cm.tab10(torch.arange(len(n_dim_list)).numpy())

for e, plt_stats in enumerate(error_stats):
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
    for n, n_dim in enumerate(n_dim_list):
        x_plt = sigma_scale[n]

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

