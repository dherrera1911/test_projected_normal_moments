"""
Plot the taylor approximation to the moments of the 3D projected normal.
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
    dist = torch.sqrt(dist)
    return dist


####################################
# 1) SETUP: DIRECTORIES, PARAMS
####################################
DATA_DIR = '../results/model_outputs/01_3d_approximation/'
SAVE_DIR = '../results/plots/01_3d_approximation_performance/'
os.makedirs(SAVE_DIR, exist_ok=True)

# PARAMETERS OF SIMULATIONS TO LOAD
eigvals = 'exponential'
eigvecs = 'random'

####################################
# 2) LOAD THE SIMULATION RESULTS
####################################

filename = DATA_DIR + f'results_eigvals_{eigvals}_eigvecs_{eigvecs}_n_3.pt'
results = torch.load(filename, weights_only=True)
n_dim = 3

mean_x = list_2_tensor(results['mean_x'])
covariance_x = list_2_tensor(results['covariance_x'])
covariance_y_taylor = list_2_tensor(results['covariance_y_taylor'])
mean_y_taylor = list_2_tensor(results['mean_y_taylor'])
covariance_y = list_2_tensor(results['covariance_y_true'])
mean_y = list_2_tensor(results['mean_y_true'])
sigma_scale = torch.as_tensor(results['sigma'])
var_scale_vec = sigma_scale ** 2 / n_dim

# Compute errors
mean_y_error = error_rel(mean_y, mean_y_taylor)
covariance_y_error = error_rel(covariance_y, covariance_y_taylor)
#covariance_y_error = error_mat(covariance_y, covariance_y_taylor)


####################################
# 3) PLOT ELLIPSES SHOWING APPROXIMATIONS TO INDIVIDUAL EXAMPLES
####################################
# Number of examples to plot for each parameter combination
n_examples = 5
# samples to plot for illustration
n_points = 200

for v in range(len(var_scale_vec)):
    var_scale = var_scale_vec[v]

    for e in range(n_examples):
        # Initialize the projected normal
        prnorm = pn.models.ProjNormal(
          mean_x=mean_x[v][e],
          covariance_x=covariance_x[v][e]
        )

        # Sample from the projected normal
        with torch.no_grad():
            samples = prnorm.sample(n_samples=n_points)

        # PLOT ELLIPSES
        plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
        fig, ax = plt.subplots(2, 2, figsize=(3.5, 3.5))
        plot_ellipses_grid(
          axes=ax, mean=torch.zeros(n_dim), cov=torch.eye(n_dim)/4,
          color='dimgrey', plot_center=False
        )
        plot_scatter_grid(axes=ax, data=samples)
        plot_ellipses_grid(
          axes=ax, mean=mean_y_taylor[v][e], cov=covariance_y_taylor[v][e],
          color='blue', plot_center=True, label='Approximation'
        )
        plot_ellipses_grid(
          axes=ax, mean=mean_y[v][e], cov=covariance_y[v][e],
          color='red', plot_center=True, label='True'
        )

        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2,
                   bbox_to_anchor=(0.5, 1.11))

        # Set limits
        set_grid_limits(axes=ax, xlims=[-1.5, 1.5], ylims=[-1.5, 1.5])
        add_grid_labels(axes=ax, prefix=r'$y$')
        # Print error value in ax[0,1]
        ax[0,1].text(0.5, 0.5, r'$\mathrm{Error}_{\gamma}$:' f' {mean_y_error[v,e]:.2f}%\n'
                               r'$\mathrm{Error}_{\Psi}$:' f' {covariance_y_error[v,e]:.2f}%',
                      horizontalalignment='center', verticalalignment='center')
        plt.tight_layout()
        plt.savefig(SAVE_DIR + f'1_approximation_ellipses_var_{eigvals}_{eigvecs}'\
            f'var{int(var_scale*100)}_{e}.png', bbox_inches='tight')
        plt.close()


####################################
# 4) PLOT THE MEAN ERROR AS A FUNCTION OF THE VARIANCE SCALE
####################################
# Plot the mean error as a function of the variance scale
plt.rcParams.update({'font.size': 14, 'font.family': 'Nimbus Sans'})

error_stats_mean = {
  'median': mean_y_error.median(dim=1).values,
  'q1': mean_y_error.quantile(0.25, dim=1),
  'q3': mean_y_error.quantile(0.75, dim=1)
}
error_stats_cov = {
  'median': covariance_y_error.median(dim=1).values,
  'q1': covariance_y_error.quantile(0.25, dim=1),
  'q3': covariance_y_error.quantile(0.75, dim=1)
}

error_stats = [error_stats_mean, error_stats_cov]
error_name = ['Mean', 'Covariance']
error_label = [r'$\mathrm{Error}_{\gamma}$ (%)', r'$\mathrm{Error}_{\Psi}$ (%)']

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
    plt.savefig(SAVE_DIR + f'3_{error_name[e]}_{eigvals}_{eigvecs}.png',
                bbox_inches='tight')
    plt.close()
