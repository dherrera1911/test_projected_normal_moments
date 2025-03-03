##################
#
# For the 3D case, we will test the approximation of the moments of a
# projected normal distribution.
#
##################

import torch
import numpy as np
import numpy.linalg as la
import projected_normal as pn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import os
import sys
sys.path.append('../')
from analysis_functions import *
from plotting_functions import *

# Directories to save plots and load data
plotDir = '../../plots/02_nd_approximation_performance/'
os.makedirs(plotDir, exist_ok=True)
dataDir = '../../results/02_nd_approximation/'

# PARAMETERS OF SIMULATION (data to load)
# Covariance types, numbers of dimensions and variances
covTypeVec = ['uncorrelated', 'correlated', 'symmetric']
nDimList = [3, 5, 10, 25, 50, 100]
varScaleVec = [0.0625, 0.25, 1, 4]

##############
# 1) LOAD THE SIMULATION RESULTS
##############

# Dictionaries to load the results
gammaTrue = {key:{} for key in covTypeVec}
gammaTaylor = {key:{} for key in covTypeVec}
psiTrue = {key:{} for key in covTypeVec}
psiTaylor = {key:{} for key in covTypeVec}
mu = {key:{} for key in covTypeVec}
cov = {key:{} for key in covTypeVec}

for n in range(len(nDimList)):
    for c in range(len(covTypeVec)):
        covType = covTypeVec[c]
        nDim = nDimList[n]
        gammaTrue[covType][n] = np.load(dataDir + f'gammaTrue_{covType}_n_{nDim}.npy')
        gammaTaylor[covType][n] = np.load(dataDir + f'gammaTaylor_{covType}_n_{nDim}.npy')
        psiTrue[covType][n] = np.load(dataDir + f'psiTrue_{covType}_n_{nDim}.npy')
        psiTaylor[covType][n] = np.load(dataDir + f'psiTaylor_{covType}_n_{nDim}.npy')
        mu[covType][n] = np.load(dataDir + f'mu_{covType}_n_{nDim}.npy')
        cov[covType][n] = np.load(dataDir + f'cov_{covType}_n_{nDim}.npy')

##############
# 2) PLOT INDIVIDUAL EXAMPLES OF THE APPROXIMATIONS
##############

# Number of examples to plot for each parameter combination
nExamples = 3
# samples to plot for illustration
nSamples = 200

for c in range(len(covTypeVec)):

    for n in range(len(nDimList)):
        # Parameters for this simulation
        covType = covTypeVec[c]
        nDim = nDimList[n]

        for v in range(len(varScaleVec)):
            varScale = varScaleVec[v]

            for e in range(nExamples):

                # Initialize the projected normal
                prnorm = pn.ProjNorm(
                    nDim=nDim, muInit=mu[covType][n][v,:,e],
                    covInit=cov[covType][n][v,:,:,e], requires_grad=False)
                # Sample from the projected normal
                samples = prnorm.sample(nSamples=nSamples)

                # Get moments to plot
                gammaTruePlt = gammaTrue[covType][n][v,:,e]
                gammaAppPlt = gammaTaylor[covType][n][v,:,e]
                psiTruePlt = psiTrue[covType][n][v,:,:,e]
                psiAppPlt = psiTaylor[covType][n][v,:,:,e]
                # Get the approximation error
                gammaErr = la.norm(gammaTruePlt - gammaAppPlt) / la.norm(gammaTruePlt) * 100
                psiErr = la.norm(psiTruePlt - psiAppPlt) / la.norm(psiTruePlt) * 100

                ### PLOT GAMMA (MEAN) VECTORS
                plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
                fig, ax = plt.subplots(figsize=(3.5, 3.5))
                # Plot the samples
                for i in range(100):
                    plot_means(axes=ax, muList=[samples[i]], colorList=['black'],
                               nameList=['_nolegend_'], alpha=0.2)
                # Plot the true and approximated means
                plot_means(axes=ax, muList=[gammaAppPlt, gammaTruePlt],
                           colorList=['blue', 'red'],
                           nameList=['Approximation', 'Empirical'], linewidth=2)
                ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.18),
                          fontsize='small')
                # Print the approximation errors on the plot (on top of a white rectangle)
                rect = plt.Rectangle((0.52, 0.9), 0.48, 0.09, fill=True, color='white',
                                     alpha=1, transform=ax.transAxes, zorder=1000)
                ax.add_patch(rect)
                ax.text(0.75, 0.93,
                    r'$\mathrm{Error}_{\gamma}$:' f' {gammaErr:.2f}%',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, zorder=1001)
                # Set the labels and save the plot
                plt.tight_layout()
                plt.ylabel('Value')
                plt.xlabel('Dimension')
                plt.ylim([-0.85, 0.85])
                plt.savefig(plotDir + f'1_mean_approximation_nDim_{nDim}_{covType}_'\
                    f'var{int(varScaleVec[v]*100)}_{e}.png', bbox_inches='tight')
                plt.close()

                ### DRAW COVARIANCE MATRICES IMAGES
                # Setup
                plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
                fig, ax = plt.subplots(1, 2, figsize=(6, 3))
                cmap = plt.get_cmap('bwr')
                # Format data, make labels list and find color scale bounds
                labelList = [r'$\Psi_{T}$', r'$\Psi_{T} - \Psi$']
                covList = [psiAppPlt, psiAppPlt - psiTruePlt]
                minVal, maxVal = list_min_max(arrayList=covList)
                cBounds = np.max(np.abs([minVal, maxVal])) * np.array([-1, 1])
                # Draw the covariance matrices
                draw_covariance_images(axes=ax, covList=covList, labelList=labelList,
                                       sharedScale=True, cmap=cmap)
                add_colorbar(ax=ax[-1], cBounds=cBounds, cmap=cmap, label='Value',
                             fontsize=12, width=0.015, loc=0.997)
                # Print the approximation errors on the plot (on top of a white rectangle)
                rect = plt.Rectangle((0.52, 0.9), 0.48, 0.09, fill=True, color='white',
                                     alpha=1, zorder=1000, transform=ax[-1].transAxes)
                ax[-1].add_patch(rect)
                ax[-1].text(0.75, 0.93,
                    r'$\mathrm{Error}_{\Psi}$' f'={psiErr:.2f}%',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax[-1].transAxes, zorder=1001)
                # Save the plot
                plt.tight_layout()
                plt.savefig(plotDir + f'2_covariance_approximation_nDim_{nDim}_'\
                    f'{covType}_' f'var{int(varScaleVec[v]*100)}.png', bbox_inches='tight')
                plt.close()


##############
# 2) PLOT THE ERRORS AS A FUNCTION OF DIMENSION AND VARIANCE SCALE
##############

### COMPUTE THE ERROR STATISTICS
gammaErrStats = {}
psiErrStats = {}
gammaErrRelStats = {}
psiErrRelStats = {}

for c in range(len(covTypeVec)):
    covType = covTypeVec[c]
    gammaErr = []
    psiErr = []
    gammaErrRel = []
    psiErrRel = []
    for n in range(len(nDimList)):
        # Get errors of approximation and put into lists
        gammaErr.append(la.norm(gammaTrue[covType][n] - gammaTaylor[covType][n], axis=1))
        psiErr.append(la.norm(psiTrue[covType][n] - psiTaylor[covType][n], axis=(1,2)))
        gammaErrRel.append(gammaErr[-1] / la.norm(gammaTrue[covType][n], axis=1) * 100)
        psiErrRel.append(psiErr[-1] / la.norm(psiTrue[covType][n], axis=(1,2)) * 100)

    # Turn into numpy arrays
    gammaErr = np.transpose(np.array(gammaErr), (1,0,2))
    psiErr = np.transpose(np.array(psiErr), (1,0,2))
    gammaErrRel = np.transpose(np.array(gammaErrRel), (1,0,2))
    psiErrRel = np.transpose(np.array(psiErrRel), (1,0,2))

    # Compute error statistics
    gammaErrStats[covType] = error_stats(gammaErr)
    psiErrStats[covType] = error_stats(psiErr)
    gammaErrRelStats[covType] = error_stats(gammaErrRel)
    psiErrRelStats[covType] = error_stats(psiErrRel)


### PLOT ERROR VS DIMENSION 

# Set the plotting parameters
plt.rcParams.update({'font.size': 14, 'font.family': 'Nimbus Sans'})

# Put errors in list and make list with corresponding labels
errType = [gammaErrStats, psiErrStats, gammaErrRelStats, psiErrRelStats]
errName = ['Mean', 'Covariance', 'Mean_relative', 'Covariance_relative']
errLabel = [r'$||\gamma - \gamma_T||$', r'$||\Sigma - \Sigma_T||$',
            r'$\mathrm{Error}_{\gamma}$ (%)',
            r'$\mathrm{Error}_{\Psi}$ (%)']
typeColor = {'uncorrelated': 'turquoise', 'correlated': 'orange', 'symmetric': 'blueviolet'}
covTypeCap = ['Uncorrelated', 'Correlated', 'Symmetric']

for e in range(len(errType)):
    for v in range(len(varScaleVec)):
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
        for c in range(len(covTypeVec)):
            # Plot the different covariance types in same plot
            covType = covTypeVec[c]

            # Make array with CI limits and with x axis values (with offset to avoid overlap)
            yerr = torch.stack((errType[e][covType]['q1'][v], errType[e][covType]['q3'][v]))
            xPlt = np.array(nDimList) * (1.1)**(c)

            # Plot the error
            ax.errorbar(xPlt, errType[e][covType]['median'][v],
                        yerr=yerr, fmt='o-', label=covTypeCap[c],
                        color=typeColor[covType])

        # Set the labels, legend and save the plot
        if e==2 or e==3:
            ax.set_ylim([0.01, 100])
        ax.set_xlabel('Number of dimensions')
        ax.set_ylabel(errLabel[e])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.25),
                  fontsize='small')
        plt.tight_layout()
        plt.savefig(plotDir + f'3_{errName[e]}_vs_ndim_{varScaleVec[v]}.png',
                    bbox_inches='tight')
        plt.close()


