import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import patches, colors, cm
from matplotlib.cm import ScalarMappable
import einops as eo
import scipy


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
    eigVal, eigVec = torch.linalg.eigh(cov)
    # Get the angle of the ellipse
    angle = torch.atan2(eigVec[1, 0], eigVec[0, 0])
    # Get the length of the semi-axes
    scale = torch.sqrt(eigVal)
    # Plot the ellipse
    ellipse = patches.Ellipse(xy=mean, width=scale[0]*4, height=scale[1]*4,
                              angle=angle*180/np.pi, color=color, linestyle=linestyle)
    ellipse.set_facecolor('none')
    ellipse.set_linewidth(2)
    ax.add_patch(ellipse)
    if label is not None:
        ax.plot([], [], color=color, linestyle=linestyle, label=label)


def plot_ellipses_grid(axes, mu, cov, color='black', withCenter=False, label=None,
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
          number of dimensions of mu.
      - mu: Tensor with the mean of distribution. Shape (n)
      - cov: Tensor with the covariance matrix of the distribution.
          Shape (n, n)
      - color: Color of the ellipse
      - withCenter: If True, plot the center of the ellipse
      - label: Label for the ellipse
      - linestyle: Style of the ellipse
    """
    # Get number of plots
    nAxes = axes.shape
    # Iterate over the lower triangle of the covariance matrices
    for r in range(nAxes[0]):
        for c in range(nAxes[1]):
            # Check that there's an ax on which to plot
            if c <= r:
                # Note, column goes first because ellipse takes (x, y)
                muEllipse = torch.tensor([mu[c], mu[r+1]])
                # Get the covariance matrix for the 2D distribution
                covEllipse = torch.tensor([[cov[c,c], cov[r+1, c]],
                                           [cov[c, r+1], cov[r+1, r+1]]])
                # Plot the data
                plot_ellipse(mean=muEllipse, cov=covEllipse, ax=axes[r, c],
                             color=color, label=label, linestyle=linestyle)
                axes[r,c].autoscale_view()
                if withCenter:
                    axes[r,c].scatter(mu[c], mu[r+1], color=color, s=20)
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
    nAxes = axes.shape
    # Iterate over the lower triangle of the covariance matrices
    for c in range(nAxes[1]):
        # Check that there's an ax on which to plot
        for r in range(nAxes[0]):
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
    nAxes = axes.shape
    # Iterate over the lower triangle of the covariance matrices
    for r in range(nAxes[0]):
        for c in range(nAxes[1]):
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
    nAxes = axes.shape
    # Iterate over the lower triangle of the covariance matrices
    for r in range(nAxes[0]):
        for c in range(nAxes[1]):
            # Check that there's an ax on which to plot
            if c <= r:
                # Set the limits of the axes
                if c == 0:
                    axes[r, c].set_ylabel(prefix + str(r+2))
                if r == nAxes[0]-1:
                    axes[r, c].set_xlabel(prefix + str(c+1))


def plot_3d_approximation_ellipses(axes, muList, covList, colorList=None,
                                   nameList=None, styleList=None,
                                   limits=[-1.5, 1.5]):
    """ 
    With the above script as template, this function plots several 3D ellipses
    in the same plot. A legend indicating the name of each ellipse is added
    above the grid
    -----------------
    Arguments:
    -----------------
      - ax: Axis handle on which to draw the ellipses.
      - muList: List of tensors with the mean of the Gaussian distribution.
      - covList: List of tensors with the covariance matrix of the Gaussian
      - colorList: List of colors for the ellipses.
      - nameList: List of names for the ellipses.
    """
    # Get the number of ellipses to plot
    nEllipses = len(muList)
    # If no color is provided, use a default color
    if colorList is None:
        colorList = ['black'] * nEllipses
    if styleList is None:
        styleList = [None] * nEllipses
    # If no name is provided, use a default name
    if nameList is None:
        nameList = ['' + str(i) for i in range(nEllipses)]
    # Plot the ellipses
    for i in range(nEllipses):
        plot_ellipses_grid(axes=axes, mu=muList[i], cov=covList[i],
                           color=colorList[i], linestyle=styleList[i],
                           label=nameList[i], withCenter=True)
    nDim = len(muList[0])
    plot_ellipses_grid(axes=axes, mu=torch.zeros(nDim), cov=torch.eye(nDim)/4,
                       color='black', withCenter=False)
    set_grid_limits(axes=axes, xlims=limits, ylims=limits)
    add_grid_labels(axes=axes, prefix='Y')


#####################################
####### Plotting high D results
#####################################

def plot_means(axes, muList, colorList, nameList=None, styleList=None,
               linewidth=2, alpha=1):
    """ Plot the means in meanList as lines, with aesthetics and
    labels given in colorList, styleList and nameList."""
    if nameList is None:
        nameList = ['_N'] * len(muList)
    if styleList is None:
        styleList = ['-'] * len(muList)
    for i in range(len(muList)):
        axes.plot(muList[i], color=colorList[i], linestyle=styleList[i],
                  label=nameList[i], alpha=alpha, linewidth=linewidth)


def draw_covariance_images(axes, covList, labelList=None, sharedScale=False,
                           cmap=plt.cm.viridis, symmetricScale=True):
    """
    Draw the covariance matrices as images.
    -----------------
    Arguments:
    -----------------
      - axes: Axis handle on which to draw the values.
      - covList: List of length n containing arrays of 
          shape (c,c) 
      - xVal: List of x axis values for each element in covariances.
      - color: Color of the scatter plot.
      - label: Label for the scatter plot.
      - size: Size of the scatter plot points.
    """
    # Size of the covariance matrices
    c = covList[0].shape[0]
    n = len(covList) # Number of covariance matrices
    # If sharedScale is True, we will use the same color scale for all images
    if sharedScale:
        covMin, covMax = list_min_max(covList)
        cBounds = np.array([covMin, covMax])
    for k in range(n):
        # Draw the covariance matrix as an image
        if not sharedScale:
            cBounds = [covariance[k].min(), covariance[k].max()]
        if symmetricScale:
            cBounds = np.max(np.abs(cBounds)) * np.array([-1, 1])
        if labelList is not None:
            titleStr = labelList[k] 
        axes[k].imshow(covList[k], vmin=cBounds[0], vmax=cBounds[1],
                            cmap=cmap)
        axes[k].set_title(titleStr)
        # Remove ticks
        axes[k].set_xticks([])
        axes[k].set_yticks([])


def list_min_max(arrayList):
    """ Get the minimum and maximum values of a list of arrays."""
    minVal = min([arr.min() for arr in arrayList])
    maxVal = max([arr.max() for arr in arrayList])
    return minVal, maxVal


def add_colorbar(ax, cBounds, cmap=plt.cm.viridis, label='',
                 ticks=None, orientation='vertical', fontsize=22,
                 width=0.025, loc=0.95):
    """
    Add a color bar to the axes.
    -----------------
    Arguments:
    -----------------
      - ax: Axis handle on which to add colorbar.
      - cmap: Color map to use.
      - cBounds: Min and max values to use for the color variable.
          It can also be list of arrays of values, in which case the minimum
          and maximum values are used.
      - label: Label for the color bar.
      - ticks: Specific tick marks to place on the color bar.
    """
    # Get color map
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    if len(cBounds) > 2:
        cBounds = [cBounds.min(), cBounds.max()]
    # Get fig from ax
    fig = ax.get_figure()
    norm = mcolors.Normalize(vmin=cBounds[0], vmax=cBounds[1])
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

