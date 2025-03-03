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
plotDir = '../../plots/04_nd_moment_matching/'
os.makedirs(plotDir, exist_ok=True)
dataDir = f'../../results/04_nd_moment_matching/'

# set seed
np.random.seed(1911)

##############
# 1) LOAD DATA, COMPARE ERROR, PLOT EXAMPLES
##############

# Get errors
def error_rel(x, y, axis=1):
    return np.linalg.norm(x - y, axis=axis) / np.linalg.norm(x,axis=axis) * 100

# Parameters of simulation
varScaleVec = [0.0625, 0.25, 1.0]
nDimVec = [3, 5, 10, 25, 50, 100]
covTypeVec = ['uncorrelated']

### PLOT INDIVIDUAL EXAMPLES OF THE FIT
nExamples = 2
for c, covType in enumerate(covTypeVec):
    for n, nDim in enumerate(nDimVec):

        # Load the fitted models
        gammaTrue = np.load(dataDir + f'gammaTrue_{covType}_n{nDim}.npy')
        gammaApprox = np.load(dataDir + f'gammaApprox_{covType}_n{nDim}.npy')
        gammaTaylor = np.load(dataDir + f'gammaTaylor_{covType}_n{nDim}.npy')
        psiTrue = np.load(dataDir + f'psiTrue_{covType}_n{nDim}.npy')
        psiApprox = np.load(dataDir + f'psiApprox_{covType}_n{nDim}.npy')
        psiTaylor = np.load(dataDir + f'psiTaylor_{covType}_n{nDim}.npy')
        muTrue = np.load(dataDir + f'muTrue_{covType}_n{nDim}.npy')
        muFit = np.load(dataDir + f'muFit_{covType}_n{nDim}.npy')
        covTrue = np.load(dataDir + f'covTrue_{covType}_n{nDim}.npy')
        covFit = np.load(dataDir + f'covFit_{covType}_n{nDim}.npy')
        loss = np.load(dataDir + f'lossArray_{covType}_n{nDim}.npy')

        for v, varScale in enumerate(varScaleVec):
            for e in range(nExamples):

                # Get moments to plot
                gammaTruePlt = gammaTrue[v,:,e]
                gammaAppPlt = gammaTaylor[v,:,e]
                psiTruePlt = psiTrue[v,:,:,e]
                psiAppPlt = psiTaylor[v,:,:,e]
                muTruePlt = muTrue[v,:,e]
                muFitPlt = muFit[v,:,e]
                covTruePlt = covTrue[v,:,:,e]
                covFitPlt = covFit[v,:,:,e]

                # Get the fit loss and parameter errors
                gammaErr = la.norm(gammaTruePlt - gammaAppPlt) / la.norm(gammaTruePlt) * 100
                psiErr = la.norm(psiTruePlt - psiAppPlt) / la.norm(psiTruePlt) * 100
                muErr = la.norm(muTruePlt - muFitPlt) / la.norm(muTruePlt) * 100
                covErr = la.norm(covTruePlt - covFitPlt) / la.norm(covTruePlt) * 100

                ### PLOT GAMMA (MEAN) VECTORS
                # Setup
                plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
                fig, ax = plt.subplots(figsize=(3.5, 3.5))
                # Format data, make labels list and find color scale bounds
                gammaList = [gammaTruePlt, gammaAppPlt]
                colorList = ['orangered', 'royalblue']

                # Plot the true and approximated means
                plot_means(axes=ax, muList=gammaList, colorList=colorList,
                           nameList=['Empirical', 'Fitted'], linewidth=2)
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

                # Set layout and save the plot
                plt.tight_layout()
                plt.ylabel('Value')
                plt.xlabel('Dimension')
                plt.ylim([-0.85, 0.85])
                plt.savefig(plotDir + f'1_gamma_fit_nDim_{nDim}_{covType}_'\
                    f'var{int(varScale*100)}_{e}.png', bbox_inches='tight')
                plt.close()

                ### DRAW PSI MATRICES IMAGES
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
                plt.savefig(plotDir + f'2_psi_fit_nDim_{nDim}_{covType}_' \
                            f'var{int(varScale*100)}_{e}.png', bbox_inches='tight')
                plt.close()


                ### PLOT MU (PARAMETER MEAN) VECTORS
                # Setup
                plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
                fig, ax = plt.subplots(figsize=(3.5, 3.5))
                # Format data, make labels list and find color scale bounds
                muList = [muTruePlt, muFitPlt]
                colorList = ['orangered', 'royalblue']

                # Plot the true and approximated means
                plot_means(axes=ax, muList=muList, colorList=colorList,
                           nameList=['True', 'Fitted'], linewidth=2)
                ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.18),
                          fontsize='small')
                # Print the approximation errors on the plot (on top of a white rectangle)
                rect = plt.Rectangle((0.52, 0.9), 0.48, 0.09, fill=True, color='white',
                                     alpha=1, transform=ax.transAxes, zorder=1000)
                ax.add_patch(rect)
                ax.text(0.75, 0.93,
                    r'$\mathrm{Error}_{\mu}$:' f' {muErr:.2f}%',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, zorder=1001)

                # Set the labels and save the plot
                plt.tight_layout()
                plt.ylabel('Value')
                plt.xlabel('Dimension')
                plt.ylim([-0.85, 0.85])
                plt.savefig(plotDir + f'3_mu_fit_nDim_{nDim}_{covType}_'\
                    f'var{int(varScale*100)}_{e}.png', bbox_inches='tight')
                plt.close()

                ### DRAW SIGMA MATRICES IMAGES
                # Setup
                plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
                fig, ax = plt.subplots(1, 2, figsize=(6, 3))
                cmap = plt.get_cmap('bwr')

                # Format data, make labels list and find color scale bounds
                labelList = [r'$\Sigma_{F}$', r'$\Sigma_{F} - \Sigma$']
                covList = [covFitPlt, covFitPlt - covTruePlt]
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
                    r'$\mathrm{Error}_{\Sigma}$' f'={covErr:.2f}%',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax[-1].transAxes, zorder=1001)

                # Save the plot
                plt.tight_layout()
                plt.savefig(plotDir + f'4_sigma_fit_nDim_{nDim}_{covType}_' \
                    f'var{int(varScaleVec[v]*100)}_{e}.png', bbox_inches='tight')
                plt.close()


### COMPUTE THE ERROR STATISTICS TO PLOT

gammaLoss = {key:[] for key in covTypeVec}
gammaApproxError = {key:[] for key in covTypeVec}
psiLoss = {key:[] for key in covTypeVec}
psiApproxError = {key:[] for key in covTypeVec}
trainLoss = {key:[] for key in covTypeVec}
muError = {key:[] for key in covTypeVec}
covError = {key:[] for key in covTypeVec}

gammaLossStats = {}
gammaApproxErrorStats = {}
psiLossStats = {}
psiApproxErrorStats = {}
trainLossStats = {}
muErrorStats = {}
covErrorStats = {}


# Load data
for c, covType in enumerate(covTypeVec):
    for n, nDim in enumerate(nDimVec):
        # Load the fitted models
        gammaTrue = np.load(dataDir + f'gammaTrue_{covType}_n{nDim}.npy')
        gammaApprox = np.load(dataDir + f'gammaApprox_{covType}_n{nDim}.npy')
        gammaTaylor = np.load(dataDir + f'gammaTaylor_{covType}_n{nDim}.npy')
        psiTrue = np.load(dataDir + f'psiTrue_{covType}_n{nDim}.npy')
        psiApprox = np.load(dataDir + f'psiApprox_{covType}_n{nDim}.npy')
        psiTaylor = np.load(dataDir + f'psiTaylor_{covType}_n{nDim}.npy')
        muTrue = np.load(dataDir + f'muTrue_{covType}_n{nDim}.npy')
        muFit = np.load(dataDir + f'muFit_{covType}_n{nDim}.npy')
        covTrue = np.load(dataDir + f'covTrue_{covType}_n{nDim}.npy')
        covFit = np.load(dataDir + f'covFit_{covType}_n{nDim}.npy')
        loss = np.load(dataDir + f'lossArray_{covType}_n{nDim}.npy')

        # Compute the error of the fitting
        covType = covTypeVec[c]
        gammaLoss[covType].append(error_rel(gammaTrue, gammaTaylor, axis=1))
        gammaApproxError[covType].append(error_rel(gammaApprox, gammaTaylor, axis=1))
        psiLoss[covType].append(error_rel(psiTrue, psiTaylor, axis=(1,2)))
        psiApproxError[covType].append(error_rel(psiApprox, psiTaylor, axis=(1,2)))
        muError[covType].append(error_rel(muTrue, muFit, axis=1))
        covError[covType].append(error_rel(covTrue, covFit, axis=(1,2)))
        trainLoss[covType].append(loss[:,-1,:].squeeze())

    # Turn the list of arrays into an array
    gammaLoss[covType] = np.array(gammaLoss[covType])
    gammaApproxError[covType] = np.array(gammaApproxError[covType])
    psiLoss[covType] = np.array(psiLoss[covType])
    psiApproxError[covType] = np.array(psiApproxError[covType])
    muError[covType] = np.array(muError[covType])
    covError[covType] = np.array(covError[covType])
    trainLoss[covType] = np.array(trainLoss[covType])

    # Compute the statistics of the errors
    gammaLossStats[covType] = error_stats(gammaLoss[covType])
    gammaApproxErrorStats[covType] = error_stats(gammaApproxError[covType])
    psiLossStats[covType] = error_stats(psiLoss[covType])
    psiApproxErrorStats[covType] = error_stats(psiApproxError[covType])
    muErrorStats[covType] = error_stats(muError[covType])
    covErrorStats[covType] = error_stats(covError[covType])
    trainLossStats[covType] = error_stats(trainLoss[covType])


### 2) Plot the loss for the different variances and covariance types

plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
#typeColor = {'uncorrelated': 'turquoise', 'correlated': 'orange', 'symmetric': 'blueviolet'}
#covTypeCap = ['Uncorrelated', 'Correlated', 'Symmetric']
typeColor = {'uncorrelated': 'turquoise'}
covTypeCap = ['Uncorrelated']

errorList = [gammaLossStats, psiLossStats]
yLabels = [r'$Loss_{\gamma}$ (%)', r'$Loss_{\Psi}$ (%)']
fileId = ['gamma', 'psi']
for t in range(2):
    # Plot gamma error
    for v, varScale in enumerate(varScaleVec):
        fig, ax = plt.subplots(figsize=(5, 5))
        for c, covType in enumerate(covTypeVec):
            # Get error statistis
            errorStats = errorList[t][covType]
            yerr = torch.stack((errorStats['q1'][:,v], errorStats['q3'][:,v]))
            ymed = errorStats['median'][:,v]
            xPlt = np.array(nDimVec) * (1 + 0.1 * (c-1)) # x axis, nDim
            ax.errorbar(xPlt, ymed, yerr=yerr, fmt='o-',
                        label=covTypeCap[c], color=typeColor[covType])
            ax.set_xlabel('Dimension')
            ax.set_ylabel(yLabels[t])
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim([10**(-4), 10**(2)])
        # Put legend on top outside of the plot
        ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15),
                  fontsize='small')
        plt.tight_layout()
        plt.savefig(plotDir + f'5_loss_vs_dim_var_{varScale}_{fileId[t]}.png',
                    bbox_inches='tight')
        plt.close()


### 3) Plot the fit error the different variances and covariance types
errorList = [muErrorStats, covErrorStats]
yLabels = [r'$Error_{\mu}$ (%)', r'$Error_{\Sigma}$ (%)']
fileId = ['mu', 'sigma']

for t in range(2):
    # Plot mu error
    for v, varScale in enumerate(varScaleVec):
        fig, ax = plt.subplots(figsize=(5, 5))
        for c, covType in enumerate(covTypeVec):
            # Get error statistis
            errorStats = errorList[t][covType]
            yerr = torch.stack((errorStats['q1'][:,v], errorStats['q3'][:,v]))
            ymed = errorStats['median'][:,v]
            xPlt = np.array(nDimVec) * (1 + 0.1 * (c-1)) # x axis, nDim
            ax.errorbar(xPlt, ymed, yerr=yerr, fmt='o-',
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
        plt.savefig(plotDir + f'6_error_vs_dim_var_{varScale}_{fileId[t]}.png',
                    bbox_inches='tight')
        plt.close()


### 4) Plot dependence of fitting error on other parameters

colors = plt.cm.viridis(np.linspace(0, 1, len(nDimVec)))

# 4.1 Plot fit error vs approximation error
plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})

errorList = [muError, covError]
yErrorLabel = [r'$Error_{\mu}$', r'$Error_{\Sigma}$']
fileId = ['mu', 'sigma']
for t in range(2):
    for c, covType in enumerate(covTypeVec):
        for v, varScale in enumerate(varScaleVec):
            for n, nDim in enumerate(nDimVec):
                dimLabel = f'{nDim}'
                color = colors[n]
                xError = (gammaApproxError[covType][n,v,:] + psiApproxError[covType][n,v,:])/2
                yError = errorList[t][covType][n,v,:]
                plt.plot(xError, yError, 'o', color=color,
                         label=dimLabel, markersize=5, alpha=0.7)
            # Add color legend outside the plot to the right
            plt.legend(title='Variance scale', bbox_to_anchor=(1.05, 1), loc='upper left')
            # Make log scales
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$Error_{\gamma + \Psi}$ (%)')
            plt.ylabel(yErrorLabel[t])
            plt.tight_layout()
            plt.savefig(plotDir + f'7_fit_vs_approx_error_var_{varScale}_{fileId[t]}.png',
                        bbox_inches='tight')
            plt.close()


# 4.2 Plot fit error vs loss
errorList = [muError, covError]
yErrorLabel = [r'$Error_{\mu}$', r'$Error_{\Sigma}$']
fileId = ['mu', 'sigma']
lossDict = psiLoss
lossName = r'$Loss_{\Sigma}$ (%)'
for t in range(2):
    for c, covType in enumerate(covTypeVec):
        for v, varScale in enumerate(varScaleVec):
            for n, nDim in enumerate(nDimVec):
                dimLabel = f'{nDim}'
                color = colors[n]
                xLoss = lossDict[covType][n,v,:]
                yError = errorList[t][covType][n,v,:]
                plt.plot(xLoss, yError, 'o', color=color,
                         label=dimLabel, markersize=5, alpha=0.7)
            # Add color legend outside the plot to the right
            plt.legend(title='Variance scale', bbox_to_anchor=(1.05, 1), loc='upper left')
            # Make log scales
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(lossName)
            plt.ylabel(yErrorLabel[t])
            plt.tight_layout()
            plt.savefig(plotDir + f'8_fit_vs_loss_var_{varScale}_{fileId[t]}.png',
                        bbox_inches='tight')
            plt.close()


# 4.3 Plot fit error vs asymmetry
#yError = [muError, covError]
#yErrorLabel = [r'$Error_{\mu}$', r'$Error_{\Sigma}$']
#fileId = ['mu', 'sigma']
#for t in range(2):
#    for c in range(len(covTypeVec)):
#        covType = covTypeVec[c]
#        for v in range(len(varScaleVec)):
#            varLabel = f'{varScaleVec[v]:.2f}'
#            color = colors[v]
#            plt.plot(asymmetryIndex[covType][v], yError[t][covType][v,:], 'o', color=color,
#                     label=varLabel, markersize=5, alpha=0.7)
#        # Add color legend outside the plot to the right
#        plt.legend(title='Variance scale', bbox_to_anchor=(1.05, 1), loc='upper left')
#        # Make log scales
#        plt.xscale('log')
#        plt.yscale('log')
#        plt.xlabel(r'Anisotropy')
#        plt.ylabel(yErrorLabel[t])
#    plt.tight_layout()
#    if saveFig:
#        plt.savefig(plotDir + f'4_fit_vs_anisotropy_{fileId[t]}.png',
#                    bbox_inches='tight')
#        plt.close()
#    else:
#        plt.show()
#

