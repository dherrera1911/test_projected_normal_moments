#################
#
# Simulate data coming from a projected normal.
# Fit a projected normal to the data through moment
# matching and compare the results to the true
# parameters of the projected normal.
#
##################

import torch
import numpy as np
import numpy.linalg as la
import projected_normal as pn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
sys.path.append('../')
from analysis_functions import *
from plotting_functions import *
import time
import copy

saveFig = True
plotDir = '../../plots/03_3d_moment_matching/'
os.makedirs(plotDir, exist_ok=True)

# set seed
np.random.seed(1911)
# Parameters of simulation
nDim = 3

##############
# 1) LOAD DATA, COMPARE ERROR, PLOT EXAMPLES
##############

# Get errors
def error_rel(x, y, axis=1):
    return np.linalg.norm(x - y, axis=axis) / np.linalg.norm(x,axis=axis) * 100

# Parameters of simulation
varScaleVec = [0.0625, 0.125, 0.25, 0.5, 1.0] # must match number of dim 0 in data
covTypeVec = ['uncorrelated', 'correlated', 'symmetric']

gammaLoss = {}
gammaApproxError = {}
psiLoss = {}
psiApproxError = {}
trainLoss = {}
muError = {}
covError = {}
asymmetryIndex = {} # Rename to anisotropy index

# Load data
nExamples = 3
for c, covType in enumerate(covTypeVec):

    # Load data
    dataDir = f'../../results/03_3d_moment_matching/'
    gammaTrue = np.load(dataDir + f'gammaTrue_{covType}_n3.npy')
    gammaApprox = np.load(dataDir + f'gammaApprox_{covType}_n3.npy')
    gammaTaylor = np.load(dataDir + f'gammaTaylor_{covType}_n3.npy')
    psiTrue = np.load(dataDir + f'psiTrue_{covType}_n3.npy')
    psiApprox = np.load(dataDir + f'psiApprox_{covType}_n3.npy')
    psiTaylor = np.load(dataDir + f'psiTaylor_{covType}_n3.npy')
    muTrue = np.load(dataDir + f'muTrue_{covType}_n3.npy')
    muFit = np.load(dataDir + f'muFit_{covType}_n3.npy')
    covTrue = np.load(dataDir + f'covTrue_{covType}_n3.npy')
    covFit = np.load(dataDir + f'covFit_{covType}_n3.npy')
    loss = np.load(dataDir + f'lossArray_{covType}_n3.npy')

    # Get asymmetry index
    eigvals = np.linalg.eigvals(covTrue.transpose(0,3,1,2))
    eigMin = np.min(eigvals, axis=2)
    eigMax = np.max(eigvals, axis=2)
    asymmetryIndex[covType] = eigMax /  eigMin

    # Compute the errors for the parameters and the loss
    covType = covTypeVec[c]
    gammaLoss[covType] = error_rel(gammaTrue, gammaTaylor, axis=1)
    gammaApproxError[covType] = error_rel(gammaApprox, gammaTaylor, axis=1)
    psiLoss[covType] = error_rel(psiTrue, psiTaylor, axis=(1,2))
    psiApproxError[covType] = error_rel(psiApprox, psiTaylor, axis=(1,2))
    muError[covType] = error_rel(muTrue, muFit, axis=1)
    covError[covType] = error_rel(covTrue, covFit, axis=(1,2))
    trainLoss[covType] = loss[:,-1,:].squeeze()

    # Plot some individual examples
    for v, varScale in enumerate(varScaleVec):
        for e in range(nExamples):

            # Get moments to plot
            gammaTruePlt = gammaTrue[v,:,e]
            gammaAppPlt = gammaApprox[v,:,e]
            psiTruePlt = psiTrue[v,:,:,e]
            psiAppPlt = psiApprox[v,:,:,e]
            muTruePlt = muTrue[v,:,e]
            muFitPlt = muFit[v,:,e]
            covTruePlt = covTrue[v,:,:,e]
            covFitPlt = covFit[v,:,:,e]
            # Compute errors
            gammaErr = la.norm(gammaTruePlt - gammaAppPlt) / la.norm(gammaTruePlt) * 100
            psiErr = la.norm(psiTruePlt - psiAppPlt) / la.norm(psiTruePlt) * 100
            muErr = la.norm(muTruePlt - muFitPlt) / la.norm(muTruePlt) * 100
            covErr = la.norm(covTruePlt - covFitPlt) / la.norm(covTruePlt) * 100


            # PLOT ELLIPSES OF PROJECTED NORMAL MOMENTS
            plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
            fig, ax = plt.subplots(2, 2, figsize=(5, 5))
            # Set up variables to plot
            muList = [gammaTruePlt, gammaAppPlt]
            covList = [psiTruePlt, psiAppPlt]
            colorList = ['orangered', 'royalblue']
            styleList = ['-', '-']
            nameList = ['Observed', 'Model output']
            # Plot ellipses
            plot_3d_approximation_ellipses(axes=ax, muList=muList, covList=covList,
                                           colorList=colorList, styleList=styleList,
                                           nameList=nameList, limits=[-1.5, 1.5])
            # Add legend
            handles, labels = ax[0,0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=2,
                       bbox_to_anchor=(0.5, 1.11))
            # Print error value in ax[0,1]
            gammaErr = gammaLoss[covType][v,e]
            psiErr = psiLoss[covType][v,e]
            ax[0,1].text(0.5, 0.5, r'$\mathrm{Loss}_{\gamma}$:' f' {gammaErr:.2f}%\n'
                               r'$\mathrm{Loss}_{\Psi}$:' f' {psiErr:.2f}%',
                               horizontalalignment='center', verticalalignment='center')
            # Save plot
            plt.tight_layout()
            plt.savefig(plotDir + f'1_{covType}_'\
                f'var_{int(varScale*100)}_{e}_loss.png', bbox_inches='tight')
            plt.close()

            # Plot ellipses of fitted parameters
            plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
            fig, ax = plt.subplots(2, 2, figsize=(5, 5))
            # Set up variables to plot
            muList = [muTruePlt, muFitPlt]
            covList = [covTruePlt, covFitPlt]
            colorList = ['orangered', 'royalblue']
            styleList = ['-', '-']
            nameList = ['True', 'Fit']
            plot_3d_approximation_ellipses(axes=ax, muList=muList, covList=covList,
                                           colorList=colorList, styleList=styleList,
                                           nameList=nameList, limits=[-2, 2])
            # Add legend
            handles, labels = ax[0,0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=2,
                       bbox_to_anchor=(0.5, 1.11))
            # Print error value in ax[0,1]
            muErr = muError[covType][v,e]
            covErr = covError[covType][v,e]
            ax[0,1].text(0.5, 0.5, r'$\mathrm{Error}_{\mu}$:' f' {muErr:.2f}%\n'
                               r'$\mathrm{Error}_{\Sigma}$:' f' {covErr:.2f}%',
                               horizontalalignment='center', verticalalignment='center')
            # Save plot
            plt.tight_layout()
            plt.savefig(plotDir + f'1_{covType}_'\
                f'var_{int(varScale*100)}_{e}_params.png', bbox_inches='tight')
            plt.close()


### 2) Plot the loss for the different variances and covariance types
plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
typeColor = {'uncorrelated': 'turquoise', 'correlated': 'orange', 'symmetric': 'blueviolet'}
covTypeCap = ['Uncorrelated', 'Correlated', 'Symmetric']

varScaleVec = np.array(varScaleVec)
errorList = [gammaLoss, psiLoss]
yLabels = [r'$Loss_{\gamma}$ (%)', r'$Loss_{\Psi}$ (%)']
fileId = ['gamma', 'psi']
for t in range(2):
    # Plot gamma error
    fig, ax = plt.subplots(figsize=(5, 5))
    for c in range(len(covTypeVec)):
        covType = covTypeVec[c]
        # Get error statistis
        errorStats = error_stats(errorList[t][covType])
        yerr = torch.stack((errorStats['q1'], errorStats['q3']))
        xPlt = varScaleVec * (1 + 0.1 * (c-1))
        ax.errorbar(xPlt, errorStats['median'], yerr=yerr, fmt='o-',
                    label=covTypeCap[c], color=typeColor[covType])
        ax.set_xlabel('Variance scale')
        ax.set_ylabel(yLabels[t])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim([10**(-4), 10**(2)])
    # Put legend on top outside of the plot
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15),
              fontsize='small')
    plt.tight_layout()
    plt.savefig(plotDir + f'2_loss_vs_var_{fileId[t]}.png',
                bbox_inches='tight')
    plt.close()


### 3) Plot the fit error the different variances and covariance types
varScaleVec = np.array(varScaleVec)
errorList = [muError, covError]
yLabels = [r'$Error_{\mu}$ (%)', r'$Error_{\Sigma}$ (%)']
fileId = ['mu', 'sigma']
for t in range(2):
    # Plot gamma error
    fig, ax = plt.subplots(figsize=(5, 5))
    for c in range(len(covTypeVec)):
        covType = covTypeVec[c]
        # Get error statistis
        errorStats = error_stats(errorList[t][covType])
        yerr = torch.stack((errorStats['q1'], errorStats['q3']))
        xPlt = varScaleVec * (1 + 0.1 * (c-1))
        ax.errorbar(xPlt, errorStats['median'], yerr=yerr, fmt='o-',
                    label=covTypeCap[c], color=typeColor[covType])
        ax.set_xlabel('Variance scale')
        ax.set_ylabel(yLabels[t])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim([10**(-4), 10**(2)])
    # Put legend on top outside of the plot
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15),
              fontsize='small')
    plt.tight_layout()
    plt.savefig(plotDir + f'3_loss_vs_var_{fileId[t]}.png',
                bbox_inches='tight')
    plt.close()


### 4) Plot dependence of fitting error on other parameters

colors = plt.cm.viridis(np.linspace(0, 1, len(varScaleVec)))

# 4.1 Plot fit error vs approximation error
plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})

yError = [muError, covError]
yErrorLabel = [r'$Error_{\mu}$', r'$Error_{\Sigma}$']
fileId = ['mu', 'sigma']
for t in range(2):
    for c in range(len(covTypeVec)):
        covType = covTypeVec[c]
        for v in range(len(varScaleVec)):
            varLabel = f'{varScaleVec[v]:.2f}'
            color = colors[v]
            xError = (gammaApproxError[covType][v,:] + psiApproxError[covType][v,:])/2
            plt.plot(xError, yError[t][covType][v,:], 'o', color=color,
                     label=varLabel, markersize=5, alpha=0.7)
        # Add color legend outside the plot to the right
        plt.legend(title='Variance scale', bbox_to_anchor=(1.05, 1), loc='upper left')
        # Make log scales
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$Error_{\gamma + \Sigma}$ (%)')
        plt.ylabel(yErrorLabel[t])
    plt.tight_layout()
    if saveFig:
        plt.savefig(plotDir + f'4_fit_vs_approx_error_{fileId[t]}.png',
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# 4.2 Plot fit error vs loss
yError = [muError, covError]
yErrorLabel = [r'$Error_{\mu}$', r'$Error_{\Sigma}$']
fileId = ['mu', 'sigma']
lossDict = psiLoss
lossName = r'$Loss_{\Sigma}$ (%)'
for t in range(2):
    for c in range(len(covTypeVec)):
        covType = covTypeVec[c]
        for v in range(len(varScaleVec)):
            varLabel = f'{varScaleVec[v]:.2f}'
            color = colors[v]
            plt.plot(lossDict[covType][v], yError[t][covType][v,:], 'o', color=color,
                     label=varLabel, markersize=5, alpha=0.7)
        # Add color legend outside the plot to the right
        plt.legend(title='Variance scale', bbox_to_anchor=(1.05, 1), loc='upper left')
        # Make log scales
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(lossName)
        plt.ylabel(yErrorLabel[t])
    plt.tight_layout()
    if saveFig:
        plt.savefig(plotDir + f'4_fit_vs_loss_{fileId[t]}.png',
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# 4.3 Plot fit error vs asymmetry
yError = [muError, covError]
yErrorLabel = [r'$Error_{\mu}$', r'$Error_{\Sigma}$']
fileId = ['mu', 'sigma']
for t in range(2):
    for c in range(len(covTypeVec)):
        covType = covTypeVec[c]
        for v in range(len(varScaleVec)):
            varLabel = f'{varScaleVec[v]:.2f}'
            color = colors[v]
            plt.plot(asymmetryIndex[covType][v], yError[t][covType][v,:], 'o', color=color,
                     label=varLabel, markersize=5, alpha=0.7)
        # Add color legend outside the plot to the right
        plt.legend(title='Variance scale', bbox_to_anchor=(1.05, 1), loc='upper left')
        # Make log scales
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'Anisotropy')
        plt.ylabel(yErrorLabel[t])
    plt.tight_layout()
    if saveFig:
        plt.savefig(plotDir + f'4_fit_vs_anisotropy_{fileId[t]}.png',
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()


