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

saveFig = True
plotDir = '../../plots/01_3d_approximation_performance/'
os.makedirs(plotDir, exist_ok=True)

# set seed
np.random.seed(1911)
nDim = 3

# Directory with results of the simulations
dataDir = '../../results/01_3d_approximation/'

##############
# 1) LOAD THE SIMULATION RESULTS
##############

covTypeVec = ['uncorrelated', 'correlated', 'symmetric']
# variances used in the simulation. varScaleVec length matches first dim
varScaleVec = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
gammaTrue = {}
gammaTaylor = {}
psiTrue = {}
psiTaylor = {}
mu = {}
cov = {}
for c in range(len(covTypeVec)):
    covType = covTypeVec[c]
    gammaTrue[covType] = np.load(dataDir + f'gammaTrue_{covType}.npy')
    gammaTaylor[covType] = np.load(dataDir + f'gammaTaylor_{covType}.npy')
    psiTrue[covType] = np.load(dataDir + f'psiTrue_{covType}.npy')
    psiTaylor[covType] = np.load(dataDir + f'psiTaylor_{covType}.npy')
    mu[covType] = np.load(dataDir + f'mu_{covType}.npy')
    cov[covType] = np.load(dataDir + f'cov_{covType}.npy')


##############
# 2) PLOT ELLIPSES SHOWING APPROXIMATIONS TO INDIVIDUAL EXAMPLES
##############

# Number of examples to plot for each parameter combination
nExamples = 3
# samples to plot for illustration
nSamples = 200

for c in range(len(covTypeVec)):
    covType = covTypeVec[c]
    for v in range(len(varScaleVec)):
        for e in range(nExamples):
            varScale = varScaleVec[v]
            # Initialize the projected normal
            prnorm = pn.ProjNorm(nDim=nDim, muInit=mu[covType][v,:,e],
                                 covInit=cov[covType][v,:,:,e],
                                 requires_grad=False)
            # Sample from the projected normal
            samples = prnorm.sample(nSamples=nSamples)

            # Get moments to plot
            gammaTruePlt = gammaTrue[covType][v,:,e]
            gammaAppPlt = gammaTaylor[covType][v,:,e]
            psiTruePlt = psiTrue[covType][v,:,:,e]
            psiAppPlt = psiTaylor[covType][v,:,:,e]

            # Plot ellipses
            plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
            fig, ax = plt.subplots(2, 2, figsize=(3.5, 3.5))
            plot_ellipses_grid(axes=ax, mu=torch.zeros(nDim), cov=torch.eye(nDim)/4,
                               color='dimgrey', withCenter=False)
            plot_scatter_grid(axes=ax, data=samples)
            plot_ellipses_grid(axes=ax, mu=gammaAppPlt, cov=psiAppPlt, color='blue',
                               withCenter=True, label='Approximation')
            plot_ellipses_grid(axes=ax, mu=gammaTruePlt, cov=psiTruePlt, color='red',
                               withCenter=True, label='Empirical')
            handles, labels = ax[0,0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=2,
                       bbox_to_anchor=(0.5, 1.11))
            # Set limits
            set_grid_limits(axes=ax, xlims=[-1.5, 1.5], ylims=[-1.5, 1.5])
            add_grid_labels(axes=ax, prefix=r'$y$')
            # Print error value in ax[0,1]
            gammaErr = la.norm(gammaTruePlt - gammaAppPlt) / la.norm(gammaTruePlt) * 100
            psiErr = la.norm(psiTruePlt - psiAppPlt) / la.norm(psiTruePlt) * 100
            ax[0,1].text(0.5, 0.5, r'$\mathrm{Error}_{\gamma}$:' f' {gammaErr:.2f}%\n'
                                   r'$\mathrm{Error}_{\Psi}$:' f' {psiErr:.2f}%',
                          horizontalalignment='center', verticalalignment='center')
            plt.tight_layout()
            if saveFig:
                plt.savefig(plotDir + f'1_approximation_ellipses_var_{covType}_'\
                    f'var{int(varScale*100)}_{e}.png', bbox_inches='tight')
                plt.close()
            else:
                plt.show()


##############
# 2) PLOT THE DISTRIBUTION OF THE APPROXIMATION ERRORS
##############

gammaErrStats = {}
psiErrStats = {}
gammaErrRelStats = {}
psiErrRelStats = {}

for c in range(len(covTypeVec)):
    covType = covTypeVec[c]
    # Get errors of approximation
    gammaErr = la.norm(gammaTrue[covType] - gammaTaylor[covType], axis=1)
    psiErr = la.norm(psiTrue[covType] - psiTaylor[covType], axis=(1,2))
    gammaErrRel = gammaErr / la.norm(gammaTrue[covType], axis=1) * 100
    psiErrRel = psiErr / la.norm(psiTrue[covType], axis=(1,2)) * 100

    # Compute error statistics
    gammaErrStats[covType] = error_stats(gammaErr)
    psiErrStats[covType] = error_stats(psiErr)
    gammaErrRelStats[covType] = error_stats(gammaErrRel)
    psiErrRelStats[covType] = error_stats(psiErrRel)

    # Plot the histrogram of the relative errors
    plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
    for j in range(2):
        fig, ax = plt.subplots(2, len(varScaleVec), figsize=(16, 5))
        if j==0:
            gammaErrPlt = gammaErr
            psiErrPlt = psiErr
            title = 'Absolute'
            labelEnd = ''
        else:
            gammaErrPlt = gammaErrRel
            psiErrPlt = psiErrRel
            title = 'Relative'
            labelEnd = ' (%)'
        for v in range(len(varScaleVec)):
            # Make histograms show frequency, not counts
            ax[0,v].hist(gammaErrPlt[v], bins=20, color='black')
            ax[1,v].hist(psiErrPlt[v], bins=20, color='black')
            ax[0,v].set_title(f'Variance scale: {varScaleVec[v]}')
            ax[0,v].set_xlabel('Mean SE' + labelEnd)
            ax[1,v].set_xlabel('Covariance SE' + labelEnd)
            ax[0,v].set_ylabel('Count')
            ax[1,v].set_ylabel('Count')
        plt.tight_layout()

        if saveFig:
            plt.savefig(plotDir + f'2_approximation_error_{covType}_{title}.png',
                        bbox_inches='tight')
            plt.close()
        else:
            plt.show()


##############
# 3) PLOT THE MEAN ERROR AS A FUNCTION OF THE VARIANCE SCALE
# (USE THE MEAN ERRORS COMPUTED IN THE PREVIOUS SECTION)
##############

# Plot the mean error as a function of the variance scale
plt.rcParams.update({'font.size': 14, 'font.family': 'Nimbus Sans'})

errStats = [gammaErrStats, psiErrStats, gammaErrRelStats, psiErrRelStats]
#errStats_E = [gammaErrStats_E, psiErrStats_E, gammaErrRelStats_E, psiErrRelStats_E]
errName = ['Mean', 'Covariance', 'Mean_relative', 'Covariance_relative']
errLabel = [r'$||\gamma - \gamma_T||$', r'$||\Psi - \Psi_T||$',
            r'$\mathrm{Error}_{\gamma}$ (%)',
            r'$\mathrm{Error}_{\Psi}$ (%)']
typeColor = {'uncorrelated': 'turquoise', 'correlated': 'orange', 'symmetric': 'blueviolet'}
covTypeCap = ['Uncorrelated', 'Correlated', 'Symmetric']

for e in range(len(errStats)):
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
    for c in range(len(covTypeVec)):
        covType = covTypeVec[c]
        # Plot median error with bars showing quartiles
        yerr = torch.stack((errStats[e][covType]['q1'], errStats[e][covType]['q3']))
        xPlt = torch.tensor(varScaleVec) * (1.1) **c
        ax.errorbar(xPlt, errStats[e][covType]['median'],
                    yerr=yerr, fmt='o-', label=covTypeCap[c], color=typeColor[covType])
        # Plot empirical error
        #ax.plot(varScaleVec, errStats_E[e][covType]['median'], 'o--')
    if e==2 or e==3:
        ax.set_ylim([0.01, 100])
    ax.set_xlabel('Variance scale (s)')
    ax.set_ylabel(errLabel[e])
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Put legend on top left
    #ax.legend(loc='upper left')
    # Put legend on top of the plot, outside
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.25),
              fontsize='small')
    plt.tight_layout()
    if saveFig:
        plt.savefig(plotDir + f'3_{errName[e]}_vs_var_scale.png',
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()


