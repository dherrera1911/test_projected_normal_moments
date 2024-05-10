##################
#
# For the 3D case, we will test the approximation of the moments of a
# projected normal distribution.
#
##################

import torch
import numpy as np
import qr_library as qr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import os
import sys
sys.path.append('./code/')
from analysis_functions import *
from plotting_functions import *

saveFig = True
plotDir = './plots/nd_approximation_performance/'
os.makedirs(plotDir, exist_ok=True)

# set seed
np.random.seed(1911)
nDimList = [3, 5, 10, 25, 50, 100]

##############
# 1) TEST THE APPROXIMATION FOR A SINGLE EXAMPLE
##############

# Parameters of simulation
varScaleVec = [0.0625, 0.25, 1, 4]
nSamples = 1000000
covTypeVec = ['uncorrelated', 'correlated', 'symmetric']
corrMagnitude = 1

for nDim in nDimList:
    for c in range(len(covTypeVec)):
        covType = covTypeVec[c]
        for v in range(len(varScaleVec)):
            varScale = varScaleVec[v] / torch.tensor(nDim/3.0)
            # Plot a single example of projected normal and its approximation
            mu, covariance = sample_parameters(nDim, covType=covType,
                                               corrMagnitude=corrMagnitude)
            covariance = covariance * varScale
            # Initialize the projected normal
            prnorm = qr.ProjNorm(nDim=nDim, muInit=mu, covInit=covariance,
                                 requires_grad=False)
            # Get the Taylor approximation moments
            meanT, covT = prnorm.get_moments()
            # Get empirical moment estimates
            meanE, covE = prnorm.empirical_moments(nSamples=nSamples)
            # Sample from the projected normal
            samples = prnorm.sample(nSamples=100)

            # Plot mean vectors
            plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            for i in range(100):
                plot_means(axes=ax, muList=[samples[i]], colorList=['black'],
                           nameList=['_nolegend_'], alpha=0.2)
            plot_means(axes=ax, muList=[meanT, meanE], colorList=['blue', 'red'],
                       nameList=['Approximation', 'Empirical'], linewidth=2)
            ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.18),
                      fontsize='small')
            # Get and print error
            muErr = torch.norm(meanT - meanE) / torch.norm(meanE) * 100
            # Put white square underneath the text
            rect = plt.Rectangle((0.52, 0.9), 0.48, 0.09, fill=True, color='white',
                                 alpha=1, transform=ax.transAxes, zorder=1000)
            ax.add_patch(rect)
            # Print the text on the upper right corner
            ax.text(0.75, 0.93,
                r'$\mathrm{Error}_{\gamma}$:' f' {muErr:.2f}%',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, zorder=1001)
            plt.tight_layout()
            plt.ylabel('Value')
            plt.xlabel('Dimension')
            plt.ylim([-0.85, 0.85])
            if saveFig:
                plt.savefig(plotDir + f'1_mean_approximation_nDim_{nDim}_{covType}_'\
                    f'var{int(varScaleVec[v]*100)}.png', bbox_inches='tight')
                plt.close()
            else:
                plt.show()

            # Draw covariance matrices images
            # 3 matrices
            #plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
            #fig, ax = plt.subplots(1, 3, figsize=(8, 3))
            #cmap = plt.get_cmap('bwr')
            #covErr = torch.norm(covT - covE) / torch.norm(covE) * 100
            #labelList = [r'$\Psi_T$', r'$\Psi$',
            #              'Difference \n' r'($\mathrm{Error}_{\Psi}$' f'={covErr:.2f}%)']
            #covList = [covT, covE, covT-covE]
            #minVal, maxVal = list_min_max(arrayList=covList)
            #cBounds = np.max(np.abs([minVal, maxVal])) * np.array([-1, 1])
            #draw_covariance_images(axes=ax, covList=covList, labelList=labelList,
            #                       sharedScale=True, cmap=cmap)
            #add_colorbar(ax=ax[-1], cBounds=cBounds, cmap=cmap, label='Value',
            #             fontsize=12, width=0.015, loc=0.997)
            #plt.tight_layout()
            # 2 matrices
            plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
            fig, ax = plt.subplots(1, 2, figsize=(6, 3))
            cmap = plt.get_cmap('bwr')
            covErr = torch.norm(covT - covE) / torch.norm(covE) * 100
            labelList = [r'$\Psi_{T}$', r'$\Psi_{T} - \Psi$']
            covList = [covT, covT-covE]
            minVal, maxVal = list_min_max(arrayList=covList)
            cBounds = np.max(np.abs([minVal, maxVal])) * np.array([-1, 1])
            draw_covariance_images(axes=ax, covList=covList, labelList=labelList,
                                   sharedScale=True, cmap=cmap)
            add_colorbar(ax=ax[-1], cBounds=cBounds, cmap=cmap, label='Value',
                         fontsize=12, width=0.015, loc=0.997)
            # Put white square underneath the text
            rect = plt.Rectangle((0.52, 0.9), 0.48, 0.09, fill=True, color='white',
                                 alpha=1, zorder=1000, transform=ax[-1].transAxes)
            ax[-1].add_patch(rect)
            ax[-1].text(0.75, 0.93,
                r'$\mathrm{Error}_{\Psi}$' f'={covErr:.2f}%',
                horizontalalignment='center', verticalalignment='center',
                transform=ax[-1].transAxes, zorder=1001)
            plt.tight_layout()
            if saveFig:
                plt.savefig(plotDir + f'2_covariance_approximation_nDim_{nDim}_'\
                    f'{covType}_' f'var{int(varScaleVec[v]*100)}.png', bbox_inches='tight')
                plt.close()
            else:
                plt.show()

##############
# 2) PLOT THE DISTRIBUTION OF THE APPROXIMATION ERRORS
##############

# Parameters of simulation
nDimList = [3, 5, 10, 25, 50, 100]
varScaleVec = [0.0625, 0.25, 1.0, 4.0]
covTypeVec = ['uncorrelated', 'correlated', 'symmetric']
nSamples = 10**6
nReps = 500

muErrType = {}
covErrType = {}
muErrRelType = {}
covErrRelType = {}
#muErrType_E = {}
#covErrType_E = {}
#muErrRelType_E = {}
#covErrRelType_E = {}

for c in range(len(covTypeVec)):
    muErr = torch.zeros(len(varScaleVec), len(nDimList), nReps)
    covErr = torch.zeros(len(varScaleVec), len(nDimList), nReps)
    muErrRel = torch.zeros(len(varScaleVec), len(nDimList), nReps)
    covErrRel = torch.zeros(len(varScaleVec), len(nDimList), nReps)
    for n in range(len(nDimList)):
        nDim = nDimList[n]
        covType = covTypeVec[c]
    #    muErr_E = torch.zeros(len(varScaleVec), nReps)
    #    covErr_E = torch.zeros(len(varScaleVec), nReps)
    #    muErrRel_E = torch.zeros(len(varScaleVec), nReps)
    #    covErrRel_E = torch.zeros(len(varScaleVec), nReps)
        for v in range(len(varScaleVec)):
            varScale = varScaleVec[v] / torch.tensor(nDim/3.0)
            for i in range(nReps):
                mu, covariance = sample_parameters(nDim, covType=covType,
                                                   corrMagnitude=corrMagnitude)
                covariance = covariance * varScale
                # Initialize the projected normal
                prnorm = qr.ProjNorm(nDim=nDim, muInit=mu, covInit=covariance, requires_grad=False)
                # Get the Taylor approximation moments
                meanT, covT = prnorm.get_moments()
                # Get empirical moment estimates
                meanE, covE = prnorm.empirical_moments(nSamples=nSamples)
                # Get second empirical estimates to get baseline error
    #            meanE2, covE2 = prnorm.empirical_moments(nSamples=nSamples)
                # Compute the error
                muErr[v, n, i] = torch.norm(meanT - meanE)
                covErr[v, n, i] = torch.norm(covT - covE)
                # Compute the relative error
                muErrRel[v, n, i] = muErr[v, n, i] / torch.norm(meanE) * 100
                covErrRel[v, n, i] = covErr[v, n, i] / torch.norm(covE) * 100
                # Compute empirical error
    #            muErr_E[v, i] = (meanE - meanE2).pow(2).sum()
    #            covErr_E[v, i] = (covE - covE2).pow(2).sum()
    #            # Compute the relative error
    #            muErrRel_E[v, i] = muErr_E[v, i] / meanE.pow(2).sum() * 100
    #            covErrRel_E[v, i] = covErr_E[v, i] / covE.pow(2).sum() * 100

    # Save the error samples
    np.save(plotDir + f'2_mu_error_{covType}.npy', muErr.numpy())
    np.save(plotDir + f'2_cov_error_{covType}.npy', covErr.numpy())
    np.save(plotDir + f'2_mu_error_rel_{covType}.npy', muErrRel.numpy())
    np.save(plotDir + f'2_cov_error_rel_{covType}.npy', covErrRel.numpy())


# Compute error statistics
for c in range(len(covTypeVec)):
    covType = covTypeVec[c]
    # Load the error samples
    muErr = torch.tensor(np.load(plotDir + f'2_mu_error_{covTypeVec[c]}.npy'))
    covErr = torch.tensor(np.load(plotDir + f'2_cov_error_{covTypeVec[c]}.npy'))
    muErrRel = torch.tensor(np.load(plotDir + f'2_mu_error_rel_{covTypeVec[c]}.npy'))
    covErrRel = torch.tensor(np.load(plotDir + f'2_cov_error_rel_{covTypeVec[c]}.npy'))
    # Compute the error statistics
    muErrType[covType] = error_stats(muErr)
    covErrType[covType] = error_stats(covErr)
    muErrRelType[covType] = error_stats(muErrRel)
    covErrRelType[covType] = error_stats(covErrRel)
#    muErrType_E[covType] = error_stats(muErr_E)
#    covErrType_E[covType] = error_stats(covErr_E)
#    muErrRelType_E[covType] = error_stats(muErrRel_E)
#    covErrRelType_E[covType] = error_stats(covErrRel_E)


##############
# 3) PLOT THE MEAN ERROR AS A FUNCTION OF THE VARIANCE SCALE
# (USE THE MEAN ERRORS COMPUTED IN THE PREVIOUS SECTION)
##############

# Plot the mean error as a function of the variance scale
plt.rcParams.update({'font.size': 14, 'font.family': 'Nimbus Sans'})

errType = [muErrType, covErrType, muErrRelType, covErrRelType]
#errType_E = [muErrType_E, covErrType_E, muErrRelType_E, covErrRelType_E]
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
            covType = covTypeVec[c]
            # Plot median error with bars showing quartiles
            yerr = torch.stack((errType[e][covType]['q1'][v], errType[e][covType]['q3'][v]))
            xPlt = np.array(nDimList) * (1.1)**(c)
            ax.errorbar(xPlt, errType[e][covType]['median'][v],
                        yerr=yerr, fmt='o-', label=covTypeCap[c],
                        color=typeColor[covType])
            # Plot empirical error
            #ax.plot(varScaleVec, errType_E[e][covType]['median'], 'o--')
        if e==2 or e==3:
            ax.set_ylim([0.01, 100])
        ax.set_xlabel('Number of dimensions')
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
            plt.savefig(plotDir + f'3_{errName[e]}_vs_ndim_{varScaleVec[v]}.png',
                        bbox_inches='tight')
            plt.close()
        else:
            plt.show()


