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
plotDir = './plots/3d_approximation_performance/'
os.makedirs(plotDir, exist_ok=True)

# set seed
np.random.seed(1911)
nDim = 3

##############
# 1) TEST THE APPROXIMATION FOR A SINGLE EXAMPLE
##############

# Parameters of simulation
varScaleVec = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
varScaleVec = [0.25, 1, 4]
nSamples = 100000
covTypeVec = ['uncorrelated', 'correlated', 'symmetric']
corrMagnitude = 0.5

for c in range(len(covTypeVec)):
    covType = covTypeVec[c]
    for v in range(len(varScaleVec)):
        varScale = varScaleVec[v]
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
        samples = prnorm.sample(nSamples=200)

        # Plot ellipses
        plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
        fig, ax = plt.subplots(2, 2, figsize=(3.5, 3.5))
        plot_ellipses_grid(axes=ax, mu=torch.zeros(nDim), cov=torch.eye(nDim)/4,
                           color='dimgrey', withCenter=False)
        plot_scatter_grid(axes=ax, data=samples)
        plot_ellipses_grid(axes=ax, mu=meanT, cov=covT, color='blue', withCenter=True,
                           label='Approximation')
        plot_ellipses_grid(axes=ax, mu=meanE, cov=covE, color='red', withCenter=True,
                           label='Empirical')
        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2,
                   bbox_to_anchor=(0.5, 1.11))
        # Set limits
        set_grid_limits(axes=ax, xlims=[-1.5, 1.5], ylims=[-1.5, 1.5])
        add_grid_labels(axes=ax, prefix=r'$y$')
        # Print error value in ax[0,1]
        muErr = torch.norm(meanT - meanE) / torch.norm(meanE) * 100
        covErr = torch.norm(covT - covE) / torch.norm(covE) * 100
        ax[0,1].text(0.5, 0.5, r'$\mathrm{Error}_{\gamma}$:' f' {muErr:.2f}%\n'
                               r'$\mathrm{Error}_{\Sigma}$:' f' {covErr:.2f}%',
                      horizontalalignment='center', verticalalignment='center')
        plt.tight_layout()
        if saveFig:
            plt.savefig(plotDir + f'1_approximation_ellipses_var_{covType}_'\
                f'var{int(varScale*100)}.png', bbox_inches='tight')
            plt.close()
        else:
            plt.show()


##############
# 2) PLOT THE DISTRIBUTION OF THE APPROXIMATION ERRORS
##############

# Parameters of simulation
varScaleVec = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
covTypeVec = ['uncorrelated', 'correlated', 'symmetric']
nSamples = 10**6
nReps = 500

muErrType = {}
covErrType = {}
muErrRelType = {}
covErrRelType = {}
muErrType_E = {}
covErrType_E = {}
muErrRelType_E = {}
covErrRelType_E = {}


for c in range(len(covTypeVec)):
    covType = covTypeVec[c]
    muErr = torch.zeros(len(varScaleVec), nReps)
    covErr = torch.zeros(len(varScaleVec), nReps)
    muErrRel = torch.zeros(len(varScaleVec), nReps)
    covErrRel = torch.zeros(len(varScaleVec), nReps)
#    muErr_E = torch.zeros(len(varScaleVec), nReps)
#    covErr_E = torch.zeros(len(varScaleVec), nReps)
#    muErrRel_E = torch.zeros(len(varScaleVec), nReps)
#    covErrRel_E = torch.zeros(len(varScaleVec), nReps)
    for v in range(len(varScaleVec)):
        varScale = varScaleVec[v]
        for i in range(nReps):
            # Plot a single example of projected normal and its approximation
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
            muErr[v, i] = torch.norm(meanT - meanE)
            covErr[v, i] = torch.norm(covT - covE)
            # Compute the relative error
            muErrRel[v, i] = muErr[v, i] / torch.norm(meanE) * 100
            covErrRel[v, i] = covErr[v, i] / torch.norm(covE) * 100
            # Compute empirical error
#            muErr_E[v, i] = (meanE - meanE2).pow(2).sum()
#            covErr_E[v, i] = (covE - covE2).pow(2).sum()
#            # Compute the relative error
#            muErrRel_E[v, i] = muErr_E[v, i] / meanE.pow(2).sum() * 100
#            covErrRel_E[v, i] = covErr_E[v, i] / covE.pow(2).sum() * 100
    # Save the error samples into a csv file
    columnNames = [f'{int(varScaleVec[i]*100)}' for i in range(len(varScaleVec))]
    muErrDf = pd.DataFrame(muErr.numpy().T)
    muErrDf.columns = columnNames
    muErrDf.to_csv(plotDir + f'2_mu_error_{covType}.csv', index=False)
    covErrDf = pd.DataFrame(covErr.numpy().T)
    covErrDf.columns = columnNames
    covErrDf.to_csv(plotDir + f'2_cov_error_{covType}.csv', index=False)
    muErrRelDf = pd.DataFrame(muErrRel.numpy().T)
    muErrRelDf.columns = columnNames
    muErrRelDf.to_csv(plotDir + f'2_mu_error_rel_{covType}.csv', index=False)
    covErrRelDf = pd.DataFrame(covErrRel.numpy().T)
    covErrRelDf.columns = columnNames
    covErrRelDf.to_csv(plotDir + f'2_cov_error_rel_{covType}.csv', index=False)
    # Compute error statistics
    muErrType[covType] = error_stats(muErr)
    covErrType[covType] = error_stats(covErr)
    muErrRelType[covType] = error_stats(muErrRel)
    covErrRelType[covType] = error_stats(covErrRel)
#    muErrType_E[covType] = error_stats(muErr_E)
#    covErrType_E[covType] = error_stats(covErr_E)
#    muErrRelType_E[covType] = error_stats(muErrRel_E)
#    covErrRelType_E[covType] = error_stats(covErrRel_E)

    # Plot the histrogram of the relative errors
    plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
    for j in range(2):
        fig, ax = plt.subplots(2, len(varScaleVec), figsize=(16, 5))
        if j==0:
            muErrPlt = muErr
            covErrPlt = covErr
            title = 'Absolute'
            labelEnd = ''
        else:
            muErrPlt = muErrRel
            covErrPlt = covErrRel
            title = 'Relative'
            labelEnd = ' (%)'
        for v in range(len(varScaleVec)):
            # Make histograms show frequency, not counts
            ax[0,v].hist(muErrPlt[v].numpy(), bins=20, color='black')
            ax[1,v].hist(covErrPlt[v].numpy(), bins=20, color='black')
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

errType = [muErrType, covErrType, muErrRelType, covErrRelType]
#errType_E = [muErrType_E, covErrType_E, muErrRelType_E, covErrRelType_E]
errName = ['Mean', 'Covariance', 'Mean_relative', 'Covariance_relative']
errLabel = [r'$||\gamma - \gamma_T||$', r'$||\Sigma - \Sigma_T||$',
            r'$\mathrm{Error}_{\gamma}$ (%)',
            r'$\mathrm{Error}_{\Psi}$ (%)']
typeColor = {'uncorrelated': 'turquoise', 'correlated': 'orange', 'symmetric': 'blueviolet'}
covTypeCap = ['Uncorrelated', 'Correlated', 'Symmetric']

for e in range(len(errType)):
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
    for c in range(len(covTypeVec)):
        covType = covTypeVec[c]
        # Plot median error with bars showing quartiles
        yerr = torch.stack((errType[e][covType]['q1'], errType[e][covType]['q3']))
        xPlt = torch.tensor(varScaleVec) * (1.1) **c
        ax.errorbar(xPlt, errType[e][covType]['median'],
                    yerr=yerr, fmt='o-', label=covTypeCap[c], color=typeColor[covType])
        # Plot empirical error
        #ax.plot(varScaleVec, errType_E[e][covType]['median'], 'o--')
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


