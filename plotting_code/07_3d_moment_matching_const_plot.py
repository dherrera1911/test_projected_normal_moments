"""
Plot the moment matching results for the 3D projected normal.
"""
import os
import sys
import torch
import numpy as np
import projnormal as pn
import matplotlib.pyplot as plt

from plotting_functions import plot_ellipses_grid, plot_scatter_grid, set_grid_limits, add_grid_labels

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
    denom = (torch.linalg.matrix_norm(x, ord=2) + \
             torch.linalg.matrix_norm(y, ord=2)) * 0.5
    return dist / denom * 100

# Directories to save plots and load data
DATA_DIR = '../results/model_outputs/07_3d_moment_matching_const/'
SAVE_DIR = '../results/plots/07_3d_moment_matching_const/'
os.makedirs(SAVE_DIR, exist_ok=True)

# PARAMETERS OF SIMULATIONS TO LOAD
eigvals = 'uniform'
eigvecs = 'random'


##############
# 1) LOAD DATA, COMPARE ERROR, PLOT EXAMPLES
##############

filename = DATA_DIR + f'results_eigvals_{eigvals}_eigvecs_{eigvecs}_n_3.pt'
results = torch.load(filename, weights_only=True)
n_dim = 3

mean_x = list_2_tensor(results['mean_x'])
mean_x_fit = list_2_tensor(results['mean_x_fit'])
cov_x = list_2_tensor(results['covariance_x'])
cov_x_fit = list_2_tensor(results['covariance_x_fit'])
const = list_2_tensor(results['const']).squeeze()
const_fit = list_2_tensor(results['const_fit'])
const_mult = list_2_tensor(results['const_mult']).squeeze()

cov_y_fit_taylor = list_2_tensor(results['covariance_y_fit_taylor'])
cov_y_fit_true = list_2_tensor(results['covariance_y_fit_true'])
cov_y = list_2_tensor(results['covariance_y_true'])

mean_y_fit_taylor = list_2_tensor(results['mean_y_fit_taylor'])
mean_y_fit_true = list_2_tensor(results['mean_y_fit_true'])
mean_y = list_2_tensor(results['mean_y_true'])

sigma_scale = torch.as_tensor(results['sigma'])
var_scale_vec = sigma_scale ** 2 / n_dim

# Compute errors
mean_x_error = error_rel(mean_x, mean_x_fit)
cov_x_error = error_rel(cov_x, cov_x_fit)
const_error = torch.abs(const - const_fit) * 2 / (const + const_fit) * 100

# Compute errors for y
mean_y_error_taylor = error_rel(mean_y_fit_true, mean_y_fit_taylor)
cov_y_error_taylor = error_rel(cov_y_fit_true, cov_y_fit_taylor)

# Compute loss
mean_y_loss = error_rel(mean_y, mean_y_fit_taylor)
cov_y_loss = error_rel(cov_y, cov_y_fit_taylor)


##############
# 2) PLOT ELLIPSES SHOWING APPROXIMATIONS TO INDIVIDUAL EXAMPLES
##############

# Number of examples to plot for each parameter combination
n_examples = 8

for v, sigma in enumerate(sigma_scale):

    for e in range(n_examples):

        for p in range(2):

            if p == 0:
                # Plot Y approximation ellipses
                labels = ['Approximation', 'True']
                mean = [mean_y_fit_taylor[v][e], mean_y_fit_true[v][e]]
                cov = [cov_y_fit_taylor[v][e], cov_y_fit_true[v][e]]
                error = [mean_y_error_taylor[v,e], cov_y_error_taylor[v,e]]
                title = f'Y approximation ellipses, $\sigma^2$={sigma**2:.2f}'
                filename = f'01_approximation_ellipses_{eigvals}_{eigvecs}_sigma_{sigma}_{e}.png'
                prefix = r'$y$'
            else:
                # Plot X fit ellipses
                labels = ['Fit', 'True']
                mean = [mean_x_fit[v][e], mean_x[v][e]]
                cov = [cov_x_fit[v][e], cov_x[v][e]]
                error = [mean_x_error[v,e], cov_x_error[v,e]]
                title = f'X fit ellipses, $\sigma^2$={sigma**2:.2f}'
                filename = f'02_fit_ellipses_{eigvals}_{eigvecs}_sigma_{sigma}_{e}.png'
                prefix = r'$x$'

            plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
            fig, ax = plt.subplots(2, 2, figsize=(3.5, 3.5))
            # Draw unit circle
            plot_ellipses_grid(
              axes=ax, mean=torch.zeros(n_dim), cov=torch.eye(n_dim)/4,
              color='dimgrey', plot_center=False
            )
            # Draw taylor fitted
            plot_ellipses_grid(
              axes=ax, mean=mean[0], cov=cov[0],
              color='blue', plot_center=True, label=labels[0]
            )
            # Draw true fitted
            plot_ellipses_grid(
              axes=ax, mean=mean[1], cov=cov[1],
              color='red', plot_center=True, label=labels[1]
            )
            handles, labels = ax[0,0].get_legend_handles_labels()
            fig.legend(
              handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.11)
            )

            # Set limits
            lim = 2.0
            set_grid_limits(axes=ax, xlims=[-lim, lim], ylims=[-lim, lim])
            add_grid_labels(axes=ax, prefix=prefix)
            # Print error value in ax[0,1]
            ax[0,1].text(
              0.5, 0.5, r'$\mathrm{Error}_{\gamma}$:' f' {error[0]:.2f}%\n' \
              r'$\mathrm{Error}_{\Psi}$:' f' {error[1]:.2f}%',
              horizontalalignment='center', verticalalignment='center'
            )

            plt.tight_layout()
            plt.savefig(SAVE_DIR + filename, bbox_inches='tight')
            plt.close()


##############
# 3) PLOT THE ERROR AS A FUNCTION OF THE VARIANCE SCALE
##############

# Plot the mean error as a function of the variance scale
plt.rcParams.update({'font.size': 14, 'font.family': 'Nimbus Sans'})

error_stats_mean = {
  'median': mean_x_error.median(dim=1).values,
  'q1': mean_x_error.quantile(0.25, dim=1),
  'q3': mean_x_error.quantile(0.75, dim=1)
}
error_stats_cov = {
  'median': cov_x_error.median(dim=1).values,
  'q1': cov_x_error.quantile(0.25, dim=1),
  'q3': cov_x_error.quantile(0.75, dim=1)
}
error_stats_const = {
  'median': const_error.median(dim=1).values,
  'q1': const_error.quantile(0.25, dim=1),
  'q3': const_error.quantile(0.75, dim=1)
}

error_stats = [error_stats_mean, error_stats_cov, error_stats_const]
error_name = ['Mean', 'Covariance', 'Constant']
error_label = [
  r'$\mathrm{Error}_{\mu}$ (%)',
  r'$\mathrm{Error}_{\Sigma}$ (%)',
  r'$\mathrm{Error}_{\mathrm{const}}$ (%)'
]

for e, plt_stats in enumerate(error_stats):
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))

    # Plot median error with bars showing quartiles
    yerr = torch.stack([plt_stats['q1'], plt_stats['q3']], dim=0)
    x_plt = torch.tensor(var_scale_vec)
    ax.errorbar(x_plt, plt_stats['median'], yerr=yerr, fmt='o-')

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


##############
# 4) PLOT THE FIT ERROR VS TAYLOR APPROXIMATION ERROR
##############

fit_error = [mean_x_error, cov_x_error]
approx_error = [mean_y_error_taylor, cov_y_error_taylor]
error_name = ['Mean', 'Covariance']
label_x = [r'$\gamma$', r'$\Psi$']
label_y = [r'$\mu$', r'$\Sigma$']

for e, (fit, approx) in enumerate(zip(fit_error, approx_error)):
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))

    for i in range(len(sigma_scale)):
        ax.scatter(approx[i], fit[i], label=f'$\sigma^2$={sigma_scale[i]**2:.2f}')

    ax.set_xlabel(label_x[e] + ' approximation error (%)')
    ax.set_ylabel(label_y[e] + 'error (%)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.25),
              fontsize='small')
    plt.tight_layout()
    plt.savefig(SAVE_DIR + f'04_{error_name[e]}_{eigvals}_{eigvecs}.png',
                bbox_inches='tight')
    plt.close()


##############
# 5) PLOT THE FIT ERROR VS CONSTANT MULTIPLIER
##############

fit_error = [mean_x_error, cov_x_error, const_error]
error_name = ['Mean', 'Covariance', 'Constant']
label_y = [r'$\mu$', r'$\Sigma$', '$\mathrm{const}$']

for e, fit in enumerate(fit_error):
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))

    for i in range(len(sigma_scale)):
        ax.scatter(const_mult[i], fit[i], label=f'$\sigma^2$={sigma_scale[i]**2:.2f}')

    ax.set_xlabel('Constant multiplier')
    ax.set_ylabel(label_y[e] + 'error (%)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.25),
              fontsize='small')
    plt.tight_layout()
    plt.savefig(SAVE_DIR + f'05_{error_name[e]}_{eigvals}_{eigvecs}.png',
                bbox_inches='tight')
    plt.close()

