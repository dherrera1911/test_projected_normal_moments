import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import patches, colors, cm
from matplotlib.cm import ScalarMappable


#### MAIN FUNCTIONS

def plot_means(results, plot_type='approx', ind=None, ax=None, cos_sim=False):
    """
    Plot the mean obtained with approximation or fitting, and compare
    to the true value.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.2,3.2))

    if ind is None:
        ind = torch.randint(0, results['covariance_x'].shape[0], (1,)).item()

    if plot_type == 'approx':
        mean1 = results['mean_y_true'][ind]
        mean2 = results['mean_y_taylor'][ind]
        names = ['True', 'Approx']
    elif plot_type == 'fit':
        mean1 = results['mean_x'][ind]
        mean2 = results['mean_x_fit'][ind]
        names = ['True', 'Fit']

    n_dim = torch.as_tensor(mean1.shape[0])

    # Plot the true and approximated means
    plot_lines(axes=ax, mean_list=[mean1, mean2], color_list=['blue', 'red'],
      name_list=names, linewidth=3)

    ax.legend(
      loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.25),
    )

    # Print the approximation errors on the plot (on top of a white rectangle)
    norm_error = 2 * 100 * torch.norm(mean1 - mean2) \
        / (torch.norm(mean1) + torch.norm(mean2))
    add_text_box(ax, text=f'Error: {norm_error:.2f}%', line=0)

    if cos_sim:
        cos_sim = torch.nn.functional.cosine_similarity(mean1, mean2, dim=0)
        add_text_box(
          ax, text=f'cosine: {cos_sim:.3f}', line=1
        )

    # Set the labels and save the plot
    plt.tight_layout()
    plt.ylabel('Value')
    plt.xticks([0, n_dim-1], [1, int(n_dim)])
    # Set tick labels
    plt.xlabel('Dimension')
    plt.ylim(torch.tensor([-0.85, 0.85]) * 3/torch.sqrt(torch.tensor(mean1.shape[0])))

    return ax


def plot_covariances(results, plot_type='fit', ind=None, ax=None, cos_sim=False):
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    cmap = plt.get_cmap('bwr')

    if ind is None:
        ind = torch.randint(0, results['covariance_x'].shape[0], (1,)).item()

    if plot_type == 'approx':
        cov1 = results['covariance_y_true'][ind]
        cov2 = results['covariance_y_taylor'][ind]
        names = ['True', 'Approx']

    elif plot_type == 'fit':
        cov1 = results['covariance_x'][ind]
        cov2 = results['covariance_x_fit'][ind]
        names = ['True', 'Fit']

    elif plot_type == 'ort':
        cov1 = results['covariance_x_ort'][ind]
        cov2 = results['covariance_x_fit_ort'][ind]
        names = ['True', 'Fit']

    cov_list = [cov1, cov2]

    # Draw the covariance matrices
    draw_covariance_images(
      axes=ax, cov_list=cov_list, label_list=names, cmap=cmap
    )

    # Add colorbar
    max_val = torch.max(torch.abs(torch.stack(cov_list))) * 1.1
    min_val = -max_val
    color_bounds = [min_val, max_val]

    add_colorbar(
      ax=ax[-1], color_bounds=color_bounds, cmap=cmap, label='Value',
      fontsize=12, width=0.018, loc=0.997
    )

    # Calcualte errors
    norm_error = 2 * 100 * torch.norm(cov1 - cov2, dim=(-1,-2)) \
            / (torch.norm(cov1, dim=(-1,-2)) + torch.norm(cov2, dim=(-1,-2)))
    add_text_box(ax[-1], text=f'Error: {norm_error:.2f}%', line=0)
    if cos_sim:
        cos_sim = torch.nn.functional.cosine_similarity(
          cov1.view(-1), cov2.view(-1), dim=-1
        )
        add_text_box(
          ax[-1], text=f'cosine: {cos_sim:.3f}', line=1
        )

    plt.tight_layout()

    return ax


def plot_error_stats(error_dict, error_label, n_dim_list, sigma_vec, ymin=None,
                     ymax=110, logscale=True):
    """
    Plot the error statistics.
    """
    colors = plt.cm.viridis(torch.linspace(0, 1, len(n_dim_list)).numpy())

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))

    # Jitter factors
    jit = torch.linspace(0.9, 1.1, len(n_dim_list))
    for n, n_dim in enumerate(n_dim_list):
        # Plot median error with bars showing quartiles
        yerr = torch.stack([error_dict['q1'][n],
                            error_dict['q3'][n]], dim=0)
        yerr = torch.abs(yerr - error_dict['median'][n][None,:])

        ax.errorbar(
          torch.as_tensor(sigma_vec) * jit[n],
          error_dict['median'][n],
          yerr=yerr,
          label=f'$n={n_dim}$',
          fmt='o-',
          markersize=7,
          color=colors[n],
        )

    if ymin is None:
        min_val = torch.min(error_dict['q1'])
        ylim = [min_val, ymax]
    if logscale:
        ylim[0] = ylim[0] * 0.1

    ax.set_xlabel('Variance scale (s)')
    ax.set_ylabel(error_label)
    if logscale:
        ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(ylim)
    ax.set_xticks(sigma_vec, sigma_vec)
    ax.legend(loc='upper center', ncol=len(n_dim_list),
              bbox_to_anchor=(0.5, 1.25), fontsize=12)

    plt.tight_layout()
    return ax


def plot_error_scatters(results, x_key, y_key, n_dim_list, labels=None):
    """
    For each dimension, plot the error in the y_label vs the
    error in the x_label. The points are colored according to their
    dimension. The scales of covariance "s" are not indicated.
    """
    if labels is None:
        labels = [x_key, y_key]
    colors = plt.cm.viridis(torch.linspace(0, 1, len(n_dim_list)).numpy())

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
    for n, n_dim in enumerate(n_dim_list):
        # Get the data for the current dimension
        x_data = results[n][x_key].view(-1)
        y_data = results[n][y_key].view(-1)

        # Scatter plot with color according to dimension
        scatter = ax.scatter(
            x_data, y_data,
            c=colors[n],
            label=f'$n={n_dim}$',
            s=50,
            alpha=0.7
        )

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(loc='upper center', ncol=len(n_dim_list),
              bbox_to_anchor=(0.5, 1.25), fontsize=12)

    plt.tight_layout()
    return ax


#### HELPER FUNCTIONS

def plot_samples(prnorm, n_samples, ax=None):
    """
    Plot samples from the distribution.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5,3.5))
    samples = prnorm.sample(n_samples=n_samples)
    plot_lines(
      axes=ax, mean_list=samples, color_list=['grey']*n_samples,
      name_list=['_nolegend_']*n_samples, alpha=0.2
    )
    return ax


def plot_lines(axes, mean_list, color_list, name_list=None, styleList=None,
               linewidth=2, alpha=1):
    """ Plot the means in meanList as lines, with aesthetics and
    labels given in color_list, styleList and name_list."""
    if name_list is None:
        name_list = ['_N'] * len(mean_list)
    if styleList is None:
        styleList = ['-'] * len(mean_list)
    for i in range(len(mean_list)):
        axes.plot(mean_list[i], color=color_list[i], linestyle=styleList[i],
                  label=name_list[i], alpha=alpha, linewidth=linewidth)


def draw_covariance_images(axes, cov_list, label_list=None, cmap=plt.cm.viridis):
    """
    Draw the covariance matrices as images.
    -----------------
    Arguments:
    -----------------
      - axes: Axis handle on which to draw the values.
      - cov_list: List of length n containing arrays of 
          shape (c,c) 
      - xVal: List of x axis values for each element in covariances.
      - color: Color of the scatter plot.
      - label: Label for the scatter plot.
      - size: Size of the scatter plot points.
    """
    # Size of the covariance matrices
    c = cov_list[0].shape[0]
    n = len(cov_list) # Number of covariance matrices

    max_val = torch.max(torch.abs(torch.stack(cov_list))) * 1.1
    min_val = -max_val
    color_bounds = [min_val, max_val]

    for k in range(n):
        # Draw the covariance matrix as an image
        if label_list is not None:
            title_str = label_list[k]
        axes[k].imshow(cov_list[k], vmin=color_bounds[0], vmax=color_bounds[1],
                            cmap=cmap)
        axes[k].set_title(title_str)
        # Remove ticks
        axes[k].set_xticks([])
        axes[k].set_yticks([])


def add_colorbar(ax, color_bounds, cmap=plt.cm.viridis, label='',
                 ticks=None, orientation='vertical', fontsize=22,
                 width=0.025, loc=0.95):
    """
    Add a color bar to the axes.
    -----------------
    Arguments:
    -----------------
      - ax: Axis handle on which to add colorbar.
      - cmap: Color map to use.
      - color_bounds: Min and max values to use for the color variable.
          It can also be list of arrays of values, in which case the minimum
          and maximum values are used.
      - label: Label for the color bar.
      - ticks: Specific tick marks to place on the color bar.
    """
    # Get color map
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    if len(color_bounds) > 2:
        color_bounds = [color_bounds.min(), color_bounds.max()]
    # Get fig from ax
    fig = ax.get_figure()
    norm = mcolors.Normalize(vmin=color_bounds[0], vmax=color_bounds[1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Determine position of color bar based on orientation
    # Modify these to change position
    if orientation == 'horizontal':
        cax_pos = [0.2, loc, 0.67, width]
    else:  # vertical
        cax_pos = [loc, 0.11, width, 0.8]  # Adjust as needed
    cax = fig.add_axes(cax_pos)
    cbar = fig.colorbar(sm, cax=cax, ticks=ticks, orientation=orientation)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.set_title(label, loc='center', fontsize=fontsize, pad=10)
    cbar.ax.yaxis.set_label_coords(7, 1)


def add_text_box(ax, text, line=0):
    if line==0:
        rect_loc = (0.43, 0.88)
        text_loc = 0.93
    elif line==1:
        rect_loc = (0.43, 0.77)
        text_loc = 0.82
    # Print the approximation errors on the plot (on top of a white rectangle)
    rect = plt.Rectangle(
      rect_loc, 0.53, 0.11, fill=True, color='white', alpha=1,
      transform=ax.transAxes, zorder=1000
    )
    ax.add_patch(rect)
    ax.text(
      0.69, text_loc, text,
      horizontalalignment='center', verticalalignment='center',
      transform=ax.transAxes, zorder=1001,
      fontsize=14
    )

