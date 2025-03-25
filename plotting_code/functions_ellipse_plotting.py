import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import patches, colors, cm
from matplotlib.cm import ScalarMappable


#####################################
####### Plotting ellipses of the distributions
#####################################

# Function to draw single ellipse form mean and covariance
def plot_ellipse(mean, cov, ax, color='black', label=None, linestyle=None):
    """ Draw an ellipse on the input axis
    -----------------
    Arguments:
    -----------------
      - mean: Tensor with the mean of the Gaussian distribution.
        length 2 tensor
      - cov: Tensor with the covariance matrix of the Gaussian
          distribution. 2x2 tensor
      - ax: Axis handle on which to draw the ellipse.
      - color: Color of the ellipse.
    """
    if linestyle is None:
        linestyle='-'
    # Get eigenvalues and eigenvectors
    eigval, eigvec = torch.linalg.eigh(cov)
    # Get the angle of the ellipse
    angle = torch.atan2(eigvec[1, 0], eigvec[0, 0])
    # Get the length of the semi-axes
    scale = torch.sqrt(eigval)
    # Plot the ellipse
    ellipse = patches.Ellipse(xy=mean, width=scale[0]*4, height=scale[1]*4,
                              angle=angle*180/np.pi, color=color, linestyle=linestyle)
    ellipse.set_facecolor('none')
    ellipse.set_linewidth(2)
    ax.add_patch(ellipse)
    if label is not None:
        ax.plot([], [], color=color, linestyle=linestyle, label=label)


def plot_ellipses_grid(axes, mean, cov, color='black', plot_center=False, label=None,
                       linestyle=None):
    """
    For a variable wth more than 2 dimensions, plot a grid with the
    ellipses of the 2D projections of the distribution onto the
    cardinal planes.
    ----------------
    Arguments:
    ----------------
      - axes: List of axes handle on which to draw the values. Must
          be a 2D array of axes, with shape (n-1, n-1) where n is the
          number of dimensions of mean.
      - mean: Tensor with the mean of distribution. Shape (n)
      - cov: Tensor with the covariance matrix of the distribution.
          Shape (n, n)
      - color: Color of the ellipse
      - plot_center: If True, plot the center of the ellipse
      - label: Label for the ellipse
      - linestyle: Style of the ellipse
    """
    # Get number of plots
    n_axes = axes.shape
    # Iterate over the lower triangle of the covariance matrices
    for r in range(n_axes[0]):
        for c in range(n_axes[1]):
            # Check that there's an ax on which to plot
            if c <= r:
                # Note, column goes first because ellipse takes (x, y)
                mean_ellipse = torch.tensor([mean[c], mean[r+1]])
                # Get the covariance matrix for the 2D distribution
                cov_ellipse = torch.tensor([[cov[c,c], cov[r+1, c]],
                                           [cov[c, r+1], cov[r+1, r+1]]])
                # Plot the data
                plot_ellipse(mean=mean_ellipse, cov=cov_ellipse, ax=axes[r, c],
                             color=color, label=label, linestyle=linestyle)
                axes[r,c].autoscale_view()
                if plot_center:
                    axes[r,c].scatter(mean[c], mean[r+1], color=color, s=20)
                axes[r,c].set_xticks([]) # Remove ticks
                axes[r,c].set_yticks([])
            else: # Remove redundant plots
                axes[r, c].axis('off')


def plot_scatter_grid(axes, data):
    """
    For data with more than 2 dimensions, plot a grid with the
    scatter plots of the 2D projections of the data onto the
    cardinal planes.
    ----------------
    Arguments:
    ----------------
      - axes: List of axes handle on which to draw the values. Must
          be a 2D array of axes, with shape (n-1, n-1) where n is the
          number of dimensions of the data.
      - data: Array to plot. (samples, n)
    """
    # Get number of plots
    n_axes = axes.shape
    # Iterate over the lower triangle of the covariance matrices
    for c in range(n_axes[1]):
        # Check that there's an ax on which to plot
        for r in range(n_axes[0]):
            if c <= r:
                # Extract the data
                x = data[:,c]
                y = data[:,r+1]
                # Plot the data
                scatter = axes[r, c].scatter(x, y, color='black', s=5, alpha=0.3)
            else: # Remove redundant plots
                axes[r, c].axis('off')


def set_grid_limits(axes, xlims, ylims):
    """
    Set the limits of the grid of axes.
    -----------------
    Arguments:
    -----------------
      - axes: List of axes handle on which to draw the values. Must
          be a 2D array of axes, with shape (m, m)
      - xlims: List with the limits of the x axis for the grid
      - ylims: List with the limits of the y axis for the grid
    """
    # Get number of plots
    n_axes = axes.shape
    # Iterate over the lower triangle of the covariance matrices
    for r in range(n_axes[0]):
        for c in range(n_axes[1]):
            # Check that there's an ax on which to plot
            if c <= r:
                # Set the limits of the axes
                axes[r, c].set_xlim(xlims)
                axes[r, c].set_ylim(ylims)


def add_grid_labels(axes, prefix=""):
    """
    Add a label to the axes of the grid indicating the variable.
    Only adds the y labels to the first column and the x labels to the last row.
    -----------------
    Arguments:
    -----------------
      - axes: List of axes handle on which to draw the values. Must
          be a 2D array of axes, with shape (m, m)
      - prefix: Prefix to add to the labels
    """
    # Get number of plots
    n_axes = axes.shape
    # Iterate over the lower triangle of the covariance matrices
    for r in range(n_axes[0]):
        for c in range(n_axes[1]):
            # Check that there's an ax on which to plot
            if c <= r:
                # Set the limits of the axes
                if c == 0:
                    axes[r, c].set_ylabel(prefix + str(r+2))
                if r == n_axes[0]-1:
                    axes[r, c].set_xlabel(prefix + str(c+1))


def plot_3d_approximation_ellipses(axes, mean_list, cov_list, color_list=None,
                                   name_list=None, styleList=None,
                                   limits=[-1.5, 1.5]):
    """
    With the above script as template, this function plots several 3D ellipses
    in the same plot. A legend indicating the name of each ellipse is added
    above the grid
    -----------------
    Arguments:
    -----------------
      - ax: Axis handle on which to draw the ellipses.
      - mean_list: List of tensors with the mean of the Gaussian distribution.
      - cov_list: List of tensors with the covariance matrix of the Gaussian
      - color_list: List of colors for the ellipses.
      - name_list: List of names for the ellipses.
    """
    # Get the number of ellipses to plot
    nEllipses = len(mean_list)
    # If no color is provided, use a default color
    if color_list is None:
        color_list = ['black'] * nEllipses
    if styleList is None:
        styleList = [None] * nEllipses
    # If no name is provided, use a default name
    if name_list is None:
        name_list = ['' + str(i) for i in range(nEllipses)]
    # Plot the ellipses
    for i in range(nEllipses):
        plot_ellipses_grid(axes=axes, mu=mean_list[i], cov=cov_list[i],
                           color=color_list[i], linestyle=styleList[i],
                           label=name_list[i], plot_center=True)
    n_dim = len(mean_list[0])

    plot_ellipses_grid(
      axes=axes, mu=torch.zeros(n_dim), cov=torch.eye(n_dim)/4,
      color='black', plot_center=False
    )

    set_grid_limits(axes=axes, xlims=limits, ylims=limits)
    add_grid_labels(axes=axes, prefix='Y')
