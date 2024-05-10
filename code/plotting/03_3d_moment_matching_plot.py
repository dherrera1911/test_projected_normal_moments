##################
#
# Simulate data coming from a projected normal.
# Fit a projected normal to the data through moment
# matching and compare the results to the true
# parameters of the projected normal.
#
##################

import torch
import numpy as np
import qr_library as qr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
sys.path.append('./code/')
from analysis_functions import *
from plotting_functions import *
import time
import copy

saveFig = True
plotDir = './plots/3d_fit_performance/'
os.makedirs(plotDir, exist_ok=True)

# set seed
np.random.seed(1911)
# Parameters of simulation
nDim = 3

##############
# 1) PLOT FIT RESULTS FOR INDIVIDUAL EXAMPLES
##############

# Parameters of simulation
varScaleVec = [0.0625, 0.125, 0.25, 0.5, 1.0]
covTypeVec = ['uncorrelated', 'correlated', 'symmetric']
nSamples = 10**6
dtype = torch.float32

# Training parameters
lossType = 'mse'
nIter = 400
lr = 0.2
optimizerType = 'NAdam'
decayIter = 10
lrGamma = 0.9
nCycles = 1
cycleMult = 0.25

for c in range(len(covTypeVec)):
    covType = covTypeVec[c]
    for v in range(len(varScaleVec)):
        varScale = varScaleVec[v]
        # Get parameters
        mu, covariance = sample_parameters(nDim, covType=covType)
        covariance = covariance * varScale
        # Initialize the projected normal with the true parameters
        prnorm = qr.ProjNorm(nDim=nDim, muInit=mu, covInit=covariance,
                             requires_grad=False, dtype=dtype)
        # Get the Taylor approximation moments
        meanT, covT = prnorm.get_moments()
        # Get empirical moment estimates
        meanE, covE = prnorm.empirical_moments(nSamples=nSamples)
        # Fit the projected normal to the empirical moments
        muInit = meanE / meanE.norm()
        covInit = torch.eye(nDim, dtype=dtype) * 0.1
        prnormFit = qr.ProjNorm(nDim=nDim, muInit=muInit, covInit=covInit, dtype=dtype)
        start = time.time()
        loss = prnormFit.moment_match(muObs=meanE, covObs=covE, nIter=nIter, lr=lr,
                             lrGamma=lrGamma, decayIter=20, lossType=lossType,
                             nCycles=nCycles, cycleMult=cycleMult)
        end = time.time()
        print(f'Fitting took {end - start:.2f} seconds')
        # Get the fitted moments without gradient
        with torch.no_grad():
            meanFT, covFT = prnormFit.get_moments() # Taylor approximation
            meanFE, covFE = prnormFit.empirical_moments(nSamples=nSamples) # Empirical
        # Get the fitted parameters
        muF = prnormFit.mu.detach()
        covarianceF = prnormFit.cov.detach()
        # Plot ellipses of projected normal moments
        plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
        fig, ax = plt.subplots(2, 2, figsize=(5, 5))
        # Set up variables to plot
        muList = [meanT, meanFT, meanE, meanFE]
        covList = [covT, covFT, covE, covFE]
        colorList = ['orangered', 'royalblue', 'orangered', 'royalblue']
        styleList = ['-', '-', '--', '--']
        nameList = ['Taylor', 'Fit-Taylor', 'Empirical', 'Fit-Empirical']
        # Plot ellipses
        plot_3d_approximation_ellipses(axes=ax, muList=muList, covList=covList,
                                       colorList=colorList, styleList=styleList,
                                       nameList=nameList, limits=[-1.5, 1.5])
        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2,
                   bbox_to_anchor=(0.5, 1.11))
        # Print error value in ax[0,1]
        muErr = (meanT - meanE).norm() / meanE.norm() * 100
        covErr = (covT - covE).norm() / covE.norm() * 100
        muErr_F = (meanFT - meanE).norm() / meanE.norm() * 100
        covErr_F = (covFT - covE).norm() / covE.norm() * 100
        muErr_FE = (meanFE - meanE).norm() / meanE.norm() * 100
        covErr_FE = (covFE - covE).norm() / covE.norm() * 100
        ax[0,1].text(0.5, 0.5, r'As % of ||$\mu_{E}$|| and ||$\Sigma_{E}$||' f'\n'\
                               r'||$\mu_{T}$ - $\mu_{E}$||' f'= {muErr:.2f}%\n'\
                               r'||$\mu_{FT}$ - $\mu_{E}$||' f'= {muErr_F:.2f}%\n'\
                               r'||$\mu_{FE}$ - $\mu_{E}$||' f'= {muErr_FE:.2f}%\n'\
                               r'||$\Sigma_{T}$ - $\Sigma_{E}$||' f' = {covErr:.2f}%\n'\
                               r'||$\Sigma_{FT}$ - $\Sigma_{E}$||' f' = {covErr_F:.2f}%\n'
                               r'||$\Sigma_{FE}$ - $\Sigma_{E}$||' f' = {covErr_FE:.2f}%\n',
                      horizontalalignment='center', verticalalignment='center')
        plt.tight_layout()
        if saveFig:
            plt.savefig(plotDir + f'1_prnorm_{covType}_'\
                f'var_{int(varScale*100)}.png', bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        # Plot ellipses of fitted parameters
        plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
        fig, ax = plt.subplots(2, 2, figsize=(5, 5))
        # Set up variables to plot
        muList = [mu, muF]
        covList = [covariance, covarianceF]
        colorList = ['orangered', 'royalblue']
        styleList = ['-', '-']
        nameList = ['True', 'Fit']
        plot_3d_approximation_ellipses(axes=ax, muList=muList, covList=covList,
                                       colorList=colorList, styleList=styleList,
                                       nameList=nameList, limits=[-2, 2])
        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2,
                   bbox_to_anchor=(0.5, 1.11))
        # Print error value in ax[0,1]
        muErr = (mu - muF).norm() / mu.norm() * 100
        covErr = (covariance - covarianceF).norm() / covariance.norm() * 100
        ax[0,1].text(0.5, 0.5, r'As % of ||$\mu_{E}$|| and ||$\Sigma_{E}$||'f'\n'\
                               r'||$\mu$ - $\mu_{F}$||' f'= {muErr:.2f}%\n'\
                               r'||$\Sigma$ - $\Sigma_{F}$||' f' = {covErr:.2f}%\n',
                      horizontalalignment='center', verticalalignment='center')
        plt.tight_layout()
        if saveFig:
            plt.savefig(plotDir + f'2_params_{covType}_'\
                f'var_{int(varScale*100)}.png', bbox_inches='tight')
            plt.close()
        else:
            plt.show()


##############
# 2) PLOT DISTRIBUTION OF ERRORS IN ESTIMATED PARAMETERS
##############

# Parameters of simulation
varScaleVec = [0.0625, 0.125, 0.25, 0.5, 1.0]
covTypeVec = ['uncorrelated', 'correlated', 'symmetric']
nSamples = 100000

# Fitting parameters
nIter = 200
lossFun = 'mse'
nReps = 100
corrMagnitude = 0.5
dtype = torch.float32
lr = 0.2
optimizerType = 'NAdam'
decayIter = 5
lrGamma = 0.97
nCycles = 2
cycleMult = 0.25

# Initialize dictionaries to store results
parErrMu = {} # (param) True mu vs fitted mu
parErrCov = {} # (param) True cov vs fitted cov
fitErrMu = {} # (output) True Empirical mu vs fitted Taylor
fitErrCov = {} # (output) True Empirical cov vs fitted Taylor
approxErrMu = {} # (output) True Empirical mu vs true Taylor
approxErrCov = {} # (output) True Empirical cov vs true Taylor
outErrMu = {} # (output) True Empirical mu vs fitted empirical
outErrCov = {} # (output) True Empirical cov vs fitted empirical
# Statistics of errors
parErrMu_S = {} # (param) True mu vs fitted mu
parErrCov_S = {} # (param) True cov vs fitted cov
fitErrMu_S = {} # (output) True Empirical mu vs fitted Taylor
fitErrCov_S = {} # (output) True Empirical cov vs fitted Taylor
approxErrMu_S = {} # (output) True Empirical mu vs true Taylor
approxErrCov_S = {} # (output) True Empirical cov vs true Taylor
outErrMu_S = {} # (output) True Empirical mu vs fitted empirical
outErrCov_S = {} # (output) True Empirical cov vs fitted empirical

for c in range(len(covTypeVec)):
    covType = covTypeVec[c]
    zerosArr = torch.zeros(len(varScaleVec), nReps)
    dictTemplate = {'absolute': zerosArr.clone(), 'relative': zerosArr.clone()}
    dictTemplate2 = {'absolute': [], 'relative': []}
    parErrMu[covType] = copy.deepcopy(dictTemplate)
    parErrCov[covType] = copy.deepcopy(dictTemplate)
    fitErrMu[covType] = copy.deepcopy(dictTemplate)
    fitErrCov[covType] = copy.deepcopy(dictTemplate)
    approxErrMu[covType] = copy.deepcopy(dictTemplate)
    approxErrCov[covType] = copy.deepcopy(dictTemplate)
    outErrMu[covType] = copy.deepcopy(dictTemplate)
    outErrCov[covType] = copy.deepcopy(dictTemplate)
    # Statistics
    parErrMu_S[covType] = copy.deepcopy(dictTemplate2)
    parErrCov_S[covType] = copy.deepcopy(dictTemplate2)
    fitErrMu_S[covType] = copy.deepcopy(dictTemplate2)
    fitErrCov_S[covType] = copy.deepcopy(dictTemplate2)
    approxErrMu_S[covType] = copy.deepcopy(dictTemplate2)
    approxErrCov_S[covType] = copy.deepcopy(dictTemplate2)
    outErrMu_S[covType] = copy.deepcopy(dictTemplate2)
    outErrCov_S[covType] = copy.deepcopy(dictTemplate2)
    for v in range(len(varScaleVec)):
        varScale = varScaleVec[v]
        for r in range(nReps):
            # Get parameters
            mu, covariance = sample_parameters(nDim, covType=covType,
                                               corrMagnitude=corrMagnitude)
            covariance = covariance * varScale
            # Initialize the projected normal with the true parameters
            prnorm = qr.ProjNorm(nDim=nDim, muInit=mu, covInit=covariance,
                                 requires_grad=False, dtype=dtype)
            # Get the Taylor approximation moments
            meanT, covT = prnorm.get_moments()
            # Get empirical moment estimates
            meanE, covE = prnorm.empirical_moments(nSamples=nSamples)
            # Fit the projected normal to the empirical moments
            muInit = meanE / meanE.norm()
            covInit = torch.eye(nDim, dtype=dtype) * 0.1
            prnormFit = qr.ProjNorm(nDim=nDim, muInit=muInit, covInit=covInit, dtype=dtype)
            loss = prnormFit.moment_match(muObs=meanE, covObs=covE, nIter=nIter, lr=lr,
                                 lrGamma=lrGamma, decayIter=20, lossType=lossType,
                                 nCycles=nCycles, cycleMult=cycleMult)
            # Get the fitted moments without gradient
            with torch.no_grad():
                meanFT, covFT = prnormFit.get_moments()
                meanFE, covFE = prnormFit.empirical_moments(nSamples=nSamples) # Empirical
            # Get the fitted parameters
            muF = prnormFit.mu.detach()
            covarianceF = prnormFit.cov.detach()

            # Store errors
            parErrMu[covType]['absolute'][v,r] = (mu - muF).norm()
            parErrMu[covType]['relative'][v,r] = parErrMu[covType]['absolute'][v,r] / mu.norm()
            parErrCov[covType]['absolute'][v,r] = (covariance - covarianceF).norm()
            parErrCov[covType]['relative'][v,r] = parErrCov[covType]['absolute'][v,r] / covariance.norm()
            fitErrMu[covType]['absolute'][v,r] = (meanE - meanFT).norm()
            fitErrMu[covType]['relative'][v,r] = fitErrMu[covType]['absolute'][v,r] / meanE.norm()
            fitErrCov[covType]['absolute'][v,r] = (covE - covFT).norm()
            fitErrCov[covType]['relative'][v,r] = fitErrCov[covType]['absolute'][v,r] / covE.norm()
            approxErrMu[covType]['absolute'][v,r] = (meanE - meanT).norm()
            approxErrMu[covType]['relative'][v,r] = approxErrMu[covType]['absolute'][v,r] / meanE.norm()
            approxErrCov[covType]['absolute'][v,r] = (covE - covT).norm()
            approxErrCov[covType]['relative'][v,r] = approxErrCov[covType]['absolute'][v,r] / covE.norm()
            outErrMu[covType]['absolute'][v,r] = (meanE - meanFE).norm()
            outErrMu[covType]['relative'][v,r] = outErrMu[covType]['absolute'][v,r] / meanE.norm()
            outErrCov[covType]['absolute'][v,r] = (covE - covFE).norm()
            outErrCov[covType]['relative'][v,r] = outErrCov[covType]['absolute'][v,r] / covE.norm()
            # Store statistics
            parErrMu_S[covType]['absolute'] = error_stats(parErrMu[covType]['absolute'])
            parErrMu_S[covType]['relative'] = error_stats(parErrMu[covType]['relative'])
            parErrCov_S[covType]['absolute'] = error_stats(parErrCov[covType]['absolute'])
            parErrCov_S[covType]['relative'] = error_stats(parErrCov[covType]['relative'])
            fitErrMu_S[covType]['absolute'] = error_stats(fitErrMu[covType]['absolute'])
            fitErrMu_S[covType]['relative'] = error_stats(fitErrMu[covType]['relative'])
            fitErrCov_S[covType]['absolute'] = error_stats(fitErrCov[covType]['absolute'])
            fitErrCov_S[covType]['relative'] = error_stats(fitErrCov[covType]['relative'])
            approxErrMu_S[covType]['absolute'] = error_stats(approxErrMu[covType]['absolute'])
            approxErrMu_S[covType]['relative'] = error_stats(approxErrMu[covType]['relative'])
            approxErrCov_S[covType]['absolute'] = error_stats(approxErrCov[covType]['absolute'])
            approxErrCov_S[covType]['relative'] = error_stats(approxErrCov[covType]['relative'])
            outErrMu_S[covType]['absolute'] = error_stats(outErrMu[covType]['absolute'])
            outErrMu_S[covType]['relative'] = error_stats(outErrMu[covType]['relative'])
            outErrCov_S[covType]['absolute'] = error_stats(outErrCov[covType]['absolute'])
            outErrCov_S[covType]['relative'] = error_stats(outErrCov[covType]['relative'])


# Plot the mean error as a function of the variance scale
plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})

errType = [parErrMu_S, parErrCov_S,
           fitErrMu_S, fitErrCov_S,
           approxErrMu_S, approxErrCov_S,
           outErrMu_S, outErrCov_S]

errLabel = [r'||$\mu - \mu_F$||', r'||$\Sigma - \Sigma_F$||',
           r'||$\mu_E - \mu_{FT}$||', r'||$\Sigma_E - \Sigma_{FT}$||',
           r'||$\mu_E - \mu_{T}$||', r'||$\Sigma_E - \Sigma_{T}$||',
           r'||$\mu_E - \mu_{FE}$||', r'||$\Sigma_E - \Sigma_{FE}$||']

errName = ['parameter_error_mean', 'parameter_error_covariance',
            'loss_mean', 'loss_covariance',
            'approximation_error_mean', 'approximation_error_covariance',
            'output_error_mean', 'output_error_covariance']

typeColor = {'uncorrelated': 'turquoise', 'correlated': 'orange', 'symmetric': 'blueviolet'}
rel = ['absolute', 'relative']

varScaleVec = np.array(varScaleVec)
for r in range(2):
    normType = rel[r]
    for e in range(len(errType)):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        for c in range(len(covTypeVec)):
            covType = covTypeVec[c]
            errDict = errType[e][covType][normType]
            meanErr = errDict['median']
            # Plot median error with bars showing quartiles
            yerr = torch.stack((errDict['q1'], errDict['q3']))
            if normType == 'relative':
                meanErr = meanErr * 100
                yerr = yerr * 100
            xPlt = varScaleVec * (1 + 0.1 * c)
            ax.errorbar(xPlt, meanErr, yerr=yerr, fmt='o-',
                        label=covType, color=typeColor[covType])
        # Plot empirical error
        ax.set_xlabel('Variance scale')
        if normType == 'absolute':
            ax.set_ylabel(errLabel[e])
        else:
            ax.set_ylabel(errLabel[e] + ' (%)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        # Put legend on top left
        ax.legend(loc='upper left')
        plt.tight_layout()
        if saveFig:
            plt.savefig(plotDir + f'3_{errName[e]}_vs_var_scale_{normType}.png',
                        bbox_inches='tight')
            plt.close()
        else:
            plt.show()


##############
# 3) PLOT RELATION OF FITTING ERROR TO APPROXIMATION ERROR
##############

colors = plt.cm.viridis(np.linspace(0, 1, len(varScaleVec)))

param = ['mean', 'covariance']

# 3.1 Plot parameter error vs approximation error
plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
approxErrList = [approxErrMu, approxErrCov]
parErrList = [parErrMu, parErrCov]

for p in range(len(param)):
    for c in range(len(covTypeVec)):
        covType = covTypeVec[c]
        approxErr = approxErrList[p]
        parErr = parErrList[p]
        for v in range(len(varScaleVec)):
            varLabel = f'{varScaleVec[v]:.2f}'
            color = colors[v]
            plt.plot(approxErr[covType]['absolute'][v,:],
                     parErr[covType]['absolute'][v,:], 'o', color=color,
                     label=varLabel, markersize=5, alpha=0.7)
        # Add color legend outside the plot to the right
        plt.legend(title='Variance scale', bbox_to_anchor=(1.05, 1), loc='upper left')
        # Make log scales
        plt.xscale('log')
        plt.yscale('log')
        if p == 0:
            plt.xlabel(r'Approx error (||$\mu_E - \mu_{T}$||)')
            plt.ylabel(r'Parameter error (||$\mu - \mu_F$||)')
        else:
            plt.xlabel(r'Approx error (||$\Sigma_E - \Sigma_{T}$||)')
            plt.ylabel(r'Parameter error (||$\Sigma - \Sigma_F$||)')
        plt.tight_layout()
        if saveFig:
            plt.savefig(plotDir + f'4_{param[p]}_fit_vs_approx_error_{covType}.png',
                        bbox_inches='tight')
            plt.close()
        else:
            plt.show()


# 3.2 Plot parameter error vs fitting error
plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
fitErrList = [fitErrMu, fitErrCov]
parErrList = [parErrMu, parErrCov]

for p in range(len(param)):
    for c in range(len(covTypeVec)):
        covType = covTypeVec[c]
        fitErr = fitErrList[p]
        parErr = parErrList[p]
        for v in range(len(varScaleVec)):
            varLabel = f'{varScaleVec[v]:.2f}'
            color = colors[v]
            plt.plot(fitErr[covType]['absolute'][v,:],
                     parErr[covType]['absolute'][v,:], 'o', color=color,
                     label=varLabel, markersize=5, alpha=0.7)
        # Add color legend outside the plot to the right
        plt.legend(title='Variance scale', bbox_to_anchor=(1.05, 1), loc='upper left')
        # Make log scales
        plt.xscale('log')
        plt.yscale('log')
        if p == 0:
            plt.xlabel(r'Fit error (||$\mu_E - \mu_{FT}$||)')
            plt.ylabel(r'Parameter error (||$\mu - \mu_F$||)')
        else:
            plt.xlabel(r'Fit error (||$\Sigma_E - \Sigma_{FT}$||)')
            plt.ylabel(r'Parameter error (||$\Sigma - \Sigma_F$||)')
        plt.tight_layout()
        if saveFig:
            plt.savefig(plotDir + f'4_{param[p]}_fit_vs_fit_error_{covType}.png',
                        bbox_inches='tight')
            plt.close()
        else:
            plt.show()


# 3.3 Plot parameter error vs fitting error
plt.rcParams.update({'font.size': 12, 'font.family': 'Nimbus Sans'})
outErrList = [outErrMu, outErrCov]
parErrList = [parErrMu, parErrCov]

for p in range(len(param)):
    for c in range(len(covTypeVec)):
        covType = covTypeVec[c]
        outErr = outErrList[p]
        parErr = parErrList[p]
        for v in range(len(varScaleVec)):
            varLabel = f'{varScaleVec[v]:.2f}'
            color = colors[v]
            plt.plot(parErr[covType]['absolute'][v,:],
                     outErr[covType]['absolute'][v,:], 'o', color=color,
                     label=varLabel, markersize=5, alpha=0.7)
        # Add color legend outside the plot to the right
        plt.legend(title='Variance scale', bbox_to_anchor=(1.05, 1), loc='upper left')
        # Make log scales
        plt.xscale('log')
        plt.yscale('log')
        if p == 0:
            plt.xlabel(r'Parameter error (||$\mu - \mu_F$||)')
            plt.ylabel(r'Output error (||$\mu_E - \mu_{FE}$||)')
        else:
            plt.xlabel(r'Parameter error (||$\Sigma - \Sigma_F$||)')
            plt.ylabel(r'Output error (||$\Sigma_E - \Sigma_{FE}$||)')
        plt.tight_layout()
        if saveFig:
            plt.savefig(plotDir + f'4_{param[p]}_output_vs_param_error_{covType}.png',
                        bbox_inches='tight')
            plt.close()
        else:
            plt.show()

