##################
#
# For the 3D case, plot the training dynamics and compare to
# model initialized from true parameters
#
##################

import torch
import numpy as np
import projected_normal as pn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import os
import sys
sys.path.append('./code/')
from analysis_functions import *
from plotting_functions import *

saveFig = True
plotDir = './plots/04_nd_mm_training/'
os.makedirs(plotDir, exist_ok=True)


# Load numpy data
nDim = 25
lossType = 'mse'
resDir = f'./results/controls/04_nd_mm_training/'
gammaTrue = np.load(resDir + f'gammaTrue_correlated_n_{nDim}_{lossType}.npy')
gammaFit = np.load(resDir + f'gammaFit_correlated_n_{nDim}_{lossType}.npy')
gammaOracle = np.load(resDir + f'gammaOracle_correlated_n_{nDim}_{lossType}.npy')
psiTrue = np.load(resDir + f'psiTrue_correlated_n_{nDim}_{lossType}.npy')
psiFit = np.load(resDir + f'psiFit_correlated_n_{nDim}_{lossType}.npy')
psiOracle = np.load(resDir + f'psiOracle_correlated_n_{nDim}_{lossType}.npy')
muTrue = np.load(resDir + f'muTrue_correlated_n_{nDim}_{lossType}.npy')
muFit = np.load(resDir + f'muFit_correlated_n_{nDim}_{lossType}.npy')
muOracle = np.load(resDir + f'muOracle_correlated_n_{nDim}_{lossType}.npy')
covTrue = np.load(resDir + f'covTrue_correlated_n_{nDim}_{lossType}.npy')
covFit = np.load(resDir + f'covFit_correlated_n_{nDim}_{lossType}.npy')
covOracle = np.load(resDir + f'covOracle_correlated_n_{nDim}_{lossType}.npy')
lossFit = np.load(resDir + f'lossFit_correlated_n_{nDim}_{lossType}.npy')
lossOracle = np.load(resDir + f'lossOracle_correlated_n_{nDim}_{lossType}.npy')

# Compute errors
gammaLoss = np.linalg.norm(gammaFit - gammaTrue[:, np.newaxis, :], axis=0)
psiLoss = np.linalg.norm(psiFit - psiTrue[:, :, np.newaxis, :], axis=(0,1))
gammaLossOr = np.linalg.norm(gammaOracle - gammaTrue[:, np.newaxis, :], axis=0)
psiLossOr = np.linalg.norm(psiOracle - psiTrue[:, :, np.newaxis, :], axis=(0,1))
muError = np.linalg.norm(muFit - muTrue[:, np.newaxis, :], axis=0)
covError = np.linalg.norm(covFit - covTrue[:, :, np.newaxis, :], axis=(0,1))
muErrorOr = np.linalg.norm(muOracle - muTrue[:, np.newaxis, :], axis=0)
covErrorOr = np.linalg.norm(covOracle - covTrue[:, :, np.newaxis, :], axis=(0,1))

# Relative errors
gammaLossRel = np.einsum('ir,r->ir', gammaLoss, 100/np.linalg.norm(gammaTrue, axis=0))
psiLossRel = np.einsum('ir,r->ir', psiLoss, 100/np.linalg.norm(psiTrue, axis=(0,1)))
gammaLossOrRel = np.einsum('ir,r->ir', gammaLossOr, 100/np.linalg.norm(gammaTrue, axis=0))
psiLossOrRel = np.einsum('ir,r->ir', psiLossOr, 100/np.linalg.norm(psiTrue, axis=(0,1)))
muErrorRel = np.einsum('ir,r->ir', muError, 100/np.linalg.norm(muTrue, axis=0))
covErrorRel = np.einsum('ir,r->ir', covError, 100/np.linalg.norm(covTrue, axis=(0,1)))
muErrorOrRel = np.einsum('ir,r->ir', muErrorOr, 100/np.linalg.norm(muTrue, axis=0))
covErrorOrRel = np.einsum('ir,r->ir', covErrorOr, 100/np.linalg.norm(covTrue, axis=(0,1)))


############################
# 1) PLOT ORACLE MODEL TRAINING DYNAMICS
############################

# Plot oracle loss through iterations
plt.figure(figsize=(12, 4))
lossList = [lossOracle, gammaLossOr, psiLossOr]
labels = ['Total loss', 'Gamma loss', 'Psi loss']
for i in range(len(lossList)):
    plt.subplot(1, 3, i+1)
    plt.plot(lossList[i])
    plt.ylabel(labels[i])
    plt.xlabel('Iteration')
    plt.yscale('log')
plt.savefig(plotDir + f'01_1_oracle_loss.png',
            bbox_inches='tight')
plt.close()

# Plot oracle error through iterations
plt.figure(figsize=(12, 4))
lossList = [muError + covError, muErrorOr, covErrorOr]
labels = ['Total error', 'Mu error', 'Cov error']
for i in range(len(lossList)):
    plt.subplot(1, 3, i+1)
    plt.plot(lossList[i])
    plt.ylabel(labels[i])
    plt.xlabel('Iteration')
    plt.yscale('log')
plt.savefig(plotDir + f'01_2_oracle_error.png',
            bbox_inches='tight')
plt.close()

# Plot oracle error vs loss
plt.figure(figsize=(12, 4))
lossList = [lossOracle, gammaLossOr, psiLossOr]
errorList = [muError+covError, muErrorOr, covErrorOr]
labels = ['Total loss', 'Gamma loss', 'Psi loss']
nLines = 5
for i in range(len(lossList)):
    plt.subplot(1, len(lossList), i+1)
    plt.plot(lossList[i][:,:nLines], errorList[i][:,:nLines])
    plt.ylabel('Error')
    plt.xlabel(labels[i])
    plt.yscale('log')
    plt.xscale('log')
plt.savefig(plotDir + f'01_3_oracle_error_vs_loss.png',
            bbox_inches='tight')
plt.close()


############################
# 2) PLOT FINAL TRAINING ERROR VS ORACLE ERROR
############################

# Extract final errors
relError = True
if not relError:
    muErrorFinal = muError[-1,:]
    covErrorFinal = covError[-1,:]
    totErrorFinal = muErrorFinal + covErrorFinal
    muErrorFinalOr = muErrorOr[-1,:]
    covErrorFinalOr = covErrorOr[-1,:]
    totErrorFinalOr = muErrorFinalOr + covErrorFinalOr
else:
    muErrorFinal = muErrorRel[-1,:]
    covErrorFinal = covErrorRel[-1,:]
    totErrorFinal = (muErrorFinal + covErrorFinal)/2
    muErrorFinalOr = muErrorOrRel[-1,:]
    covErrorFinalOr = covErrorOrRel[-1,:]
    totErrorFinalOr = (muErrorFinalOr + covErrorFinalOr)/2

# Plot training vs oracle error
plt.figure(figsize=(16, 4))
errorList = [totErrorFinal, muErrorFinal, covErrorFinal]
errorListOr = [totErrorFinalOr, muErrorFinalOr, covErrorFinalOr]
labels = ['Total error', 'Mu error', 'Cov error']
logScale = True
for i in range(len(errorList)):
    plt.subplot(1, 3, i+1)
    plt.scatter(errorList[i], errorListOr[i])
    plt.ylabel('Oracle error')
    plt.xlabel('Error')
    plt.title(labels[i])
    minVal = min(np.min(errorList[i]), np.min(errorListOr[i]))
    maxVal = max(np.max(errorList[i]), np.max(errorListOr[i]))
    if logScale:
        plt.yscale('log')
        plt.xscale('log')
    # Plot identity line
    plt.plot([minVal, maxVal], [minVal, maxVal], 'k--')

# Prevent overlap
plt.tight_layout()
plt.savefig(plotDir + f'02_1_error_vs_oracle_error.png',
            bbox_inches='tight')
plt.close()


# Plot training vs oracle error
errorList = [muErrorRel, covErrorRel]
errorListOr = [muErrorOrRel, covErrorOrRel]
labels = ['Mu error', 'Cov error']
fileId = ['mu', 'cov']
logScale = True
for i in range(len(errorList)):
    plt.figure(figsize=(12, 12))
    for n in range(9):
        plt.subplot(3, 3, n+1)
        plt.plot(errorList[i][:,n], color='black', label='Fit')
        plt.plot(errorListOr[i][:,n], color='red', label='Oracle')
        plt.ylabel(labels[i])
        plt.xlabel('Iteration')
        plt.legend()
        if logScale:
            plt.yscale('log')
        plt.ylim([0.1, 300])
    # Prevent overlap
    plt.tight_layout()
    plt.savefig(plotDir + f'02_2_error_relative_iteration_{fileId[i]}.png',
                bbox_inches='tight')
    plt.close()


############################
# 3) PLOT RELATIVE LOSS
############################

# Plot training vs oracle error
errorList = [gammaLoss, covErrorRel]
errorListOr = [muErrorOrRel, covErrorOrRel]
labels = ['Mu error', 'Cov error']
fileId = ['mu', 'cov']
logScale = True
for i in range(len(errorList)):
    plt.figure(figsize=(12, 12))
    for n in range(9):
        plt.subplot(3, 3, n+1)
        plt.plot(errorList[i][:,n], color='black', label='Fit')
        plt.plot(errorListOr[i][:,n], color='red', label='Oracle')
        plt.ylabel(labels[i])
        plt.xlabel('Iteration')
        plt.legend()
        if logScale:
            plt.yscale('log')
        plt.ylim([0.1, 300])
    # Prevent overlap
    plt.tight_layout()
    plt.savefig(plotDir + f'02_2_error_relative_iteration_{fileId[i]}.png',
                bbox_inches='tight')
    plt.close()

