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
sys.path.append('../')
from analysis_functions import *
from plotting_functions import *

saveFig = True
plotDir = '../../plots/control/02_3d_mm_training/'
os.makedirs(plotDir, exist_ok=True)


# Load numpy data
dataDir = '../../results/controls/02_3d_mm_training/'
covType = 'uncorrelated'
lossType = 'mse'
gammaTrue = np.load(dataDir + f'gammaTrue_{covType}_{lossType}.npy')
gammaFit = np.load(dataDir + f'gammaFit_{covType}_{lossType}.npy')
gammaOracle = np.load(dataDir + f'gammaOracle_{covType}_{lossType}.npy')
psiTrue = np.load(dataDir + f'psiTrue_{covType}_{lossType}.npy')
psiFit = np.load(dataDir + f'psiFit_{covType}_{lossType}.npy')
psiOracle = np.load(dataDir + f'psiOracle_{covType}_{lossType}.npy')
muTrue = np.load(dataDir + f'muTrue_{covType}_{lossType}.npy')
muFit = np.load(dataDir + f'muFit_{covType}_{lossType}.npy')
muOracle = np.load(dataDir + f'muOracle_{covType}_{lossType}.npy')
covTrue = np.load(dataDir + f'covTrue_{covType}_{lossType}.npy')
covFit = np.load(dataDir + f'covFit_{covType}_{lossType}.npy')
covOracle = np.load(dataDir + f'covOracle_{covType}_{lossType}.npy')
lossFit = np.load(dataDir + f'lossFit_{covType}_{lossType}.npy')
lossOracle = np.load(dataDir + f'lossOracle_{covType}_{lossType}.npy')

# Compute errors
gammaLoss = np.linalg.norm(gammaFit - gammaTrue[:, np.newaxis, :], axis=0)
psiLoss = np.linalg.norm(psiFit - psiTrue[:, :, np.newaxis, :], axis=(0,1))
gammaLossOr = np.linalg.norm(gammaOracle - gammaTrue[:, np.newaxis, :], axis=0)
psiLossOr = np.linalg.norm(psiOracle - psiTrue[:, :, np.newaxis, :], axis=(0,1))
muError = np.linalg.norm(muFit - muTrue[:, np.newaxis, :], axis=0)
covError = np.linalg.norm(covFit - covTrue[:, :, np.newaxis, :], axis=(0,1))
muErrorOr = np.linalg.norm(muOracle - muTrue[:, np.newaxis, :], axis=0)
covErrorOr = np.linalg.norm(covOracle - covTrue[:, :, np.newaxis, :], axis=(0,1))


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
lossList = [muErrorOr, covErrorOr]
labels = ['Mu error', 'Cov loss']
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
lossList = [gammaLossOr, psiLossOr]
errorList = [muErrorOr, covErrorOr]
labels = ['Gamma loss', 'Psi loss']
nLines = 5
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.plot(lossList[i][:,:nLines], errorList[i][:,:nLines])
    plt.ylabel(labels[i])
    plt.xlabel('Error')
    plt.yscale('log')
    plt.xscale('log')
plt.savefig(plotDir + f'01_3_oracle_error_vs_loss.png',
            bbox_inches='tight')
plt.close()


############################
# 2) PLOT FINAL TRAINING ERROR VS ORACLE ERROR
############################

# Extract final errors
muErrorFinal = muError[-1,:]
covErrorFinal = covError[-1,:]
totErrorFinal = muErrorFinal + covErrorFinal
muErrorFinalOr = muErrorOr[-1,:]
covErrorFinalOr = covErrorOr[-1,:]
totErrorFinalOr = muErrorFinalOr + covErrorFinalOr

# Plot training vs oracle error
plt.figure(figsize=(12, 4))
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

plt.savefig(plotDir + f'02_1_error_vs_oracle_error.png',
            bbox_inches='tight')
plt.close()


