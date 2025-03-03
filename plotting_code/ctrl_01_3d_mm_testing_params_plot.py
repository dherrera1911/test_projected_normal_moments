##################
#
# For the 3D case, plot the training dynamics and compare to
# model initialized from true parameters
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
plotDir = '../../plots/control/01_3d_mm_testing_params/'
os.makedirs(plotDir, exist_ok=True)

lossTypeVec = ['mse', 'norm']
covWeightVec = [1, 2.5]

# Load numpy data
dataDir = '../../results/controls/02_3d_mm_training_test_params/'
covType = 'correlated'
gammaTrue = np.load(dataDir + f'gammaTrue_{covType}.npy')
gammaFit = np.load(dataDir + f'gammaFit_{covType}.npy')
gammaOracle = np.load(dataDir + f'gammaOracle_{covType}.npy')
psiTrue = np.load(dataDir + f'psiTrue_{covType}.npy')
psiFit = np.load(dataDir + f'psiFit_{covType}.npy')
psiOracle = np.load(dataDir + f'psiOracle_{covType}.npy')
muTrue = np.load(dataDir + f'muTrue_{covType}.npy')
muFit = np.load(dataDir + f'muFit_{covType}.npy')
muOracle = np.load(dataDir + f'muOracle_{covType}.npy')
covTrue = np.load(dataDir + f'covTrue_{covType}.npy')
covFit = np.load(dataDir + f'covFit_{covType}.npy')
covOracle = np.load(dataDir + f'covOracle_{covType}.npy')
lossFit = np.load(dataDir + f'lossFit_{covType}.npy')
lossOracle = np.load(dataDir + f'lossOracle_{covType}.npy')

# Compute errors
gammaLoss = la.norm(gammaFit - np.expand_dims(gammaTrue, axis=(0,1,3)), axis=2)
psiLoss = la.norm(psiFit - np.expand_dims(psiTrue, axis=(0,1,4)), axis=(2,3))
muError = la.norm(muFit - np.expand_dims(muTrue, axis=(0,1,3)), axis=2)
covError = la.norm(covFit - np.expand_dims(covTrue, axis=(0,1,4)), axis=(2,3))
# Make errors relative
gammaLossR = gammaLoss / np.expand_dims(la.norm(gammaTrue, axis=0), axis=(0,1,2)) * 100
psiLossR = psiLoss / np.expand_dims(la.norm(psiTrue, axis=(0,1)), axis=(0,1,2)) * 100
muErrorR = muError / np.expand_dims(la.norm(muTrue, axis=0), axis=(0,1,2)) * 100
covErrorR = covError / np.expand_dims(la.norm(covTrue, axis=(0,1)), axis=(0,1,2)) * 100
# Compute oracle errors
gammaLossOr = la.norm(gammaOracle - np.expand_dims(gammaTrue, axis=1), axis=0)
psiLossOr = la.norm(psiOracle - np.expand_dims(psiTrue, axis=2), axis=(0,1))
muErrorOr = la.norm(muOracle - np.expand_dims(muTrue, axis=1), axis=0)
covErrorOr = la.norm(covOracle - np.expand_dims(covTrue, axis=2), axis=(0,1))
# Make errors relative
gammaLossOrR = gammaLossOr / la.norm(gammaTrue, axis=0) * 100
psiLossOrR = psiLossOr / la.norm(psiTrue, axis=(0,1)) * 100
muErrorOrR = muErrorOr / la.norm(muTrue, axis=0) * 100
covErrorOrR = covErrorOr / la.norm(covTrue, axis=(0,1)) * 100


############################
# 1) PLOT THE FINAL ERRORS FOR THE DIFFERENT TRAINING PARAMETERS
############################

# Extract final errors
#muErrorFinal = muError[:,:,-1,:]
#covErrorFinal = covError[:,:,-1,:]
#totErrorFinal = muErrorFinal + covErrorFinal
muErrorFinal = muErrorR[:,:,-1,:]
covErrorFinal = covErrorR[:,:,-1,:]
totErrorFinal = (muErrorFinal + covErrorFinal) / 2

# Extract final oracle errors
muErrorFinalOr = muErrorOrR[-1,:]
covErrorFinalOr = covErrorOrR[-1,:]
totErrorFinalOr = (muErrorFinalOr + covErrorFinalOr) / 2

# Put in list to iterate over
errorList = [totErrorFinal, muErrorFinal, covErrorFinal]
errorListOr = [totErrorFinalOr, muErrorFinalOr, covErrorFinalOr]

# Plot the two loss function errors
plt.figure(figsize=(12, 4))
labels = ['Total error', 'Mu error', 'Cov error']
logScale = True
for i in range(len(errorList)):
    plt.subplot(1, 3, i+1)
    plt.scatter(errorList[i][0,0,:], errorList[i][1,0,:], label='MSE', c='b')
    plt.scatter(errorList[i][0,0,:], errorListOr[i], label='Oracle', c='r')
    #plt.ylabel(f'Error {lossTypeVec[0]}')
    plt.xlabel(f'Error {lossTypeVec[0]}')
    plt.ylabel(f'Error')
    plt.title(labels[i])
    minVal = np.min(errorList[i][:,0,:])
    maxVal = np.max(errorList[i][:,0,:])
    if logScale:
        plt.yscale('log')
        plt.xscale('log')
    # Plot identity line
    plt.plot([minVal, maxVal], [minVal, maxVal], 'k--')
    plt.legend()

# Prevent overlap
plt.tight_layout()
plt.savefig(plotDir + f'01_final_error_loss_functions.png',
            bbox_inches='tight')
plt.close()


# Plot the upweighting of the covariance
plt.figure(figsize=(12, 4))
labels = ['Total error', 'Mu error', 'Cov error']
logScale = True
l = 0
for i in range(len(errorList)):
    plt.subplot(1, 3, i+1)
    plt.scatter(errorList[i][l,0,:], errorList[i][l,1,:], label='Upweight cov', c='b')
    plt.scatter(errorList[i][l,0,:], errorListOr[i], label='Oracle', c='r')
    #plt.ylabel(f'Error {lossTypeVec[0]}')
    plt.ylabel(f'Error')
    plt.xlabel(f'Error {lossTypeVec[1]}')
    plt.title(labels[i])
    minVal = np.min(errorList[i][:,0,:])
    maxVal = np.max(errorList[i][:,0,:])
    if logScale:
        plt.yscale('log')
        plt.xscale('log')
    # Plot identity line
    plt.plot([minVal, maxVal], [minVal, maxVal], 'k--')
    plt.legend()

# Prevent overlap
plt.tight_layout()
plt.savefig(plotDir + f'01_final_error_cov_weight_functions.png',
            bbox_inches='tight')
plt.close()


############################
# 2) PLOT THE FINAL LOSS FOR THE DIFFERENT TRAINING PARAMETERS
############################

# Extract final errors
#gammaLossFinal = gammaLoss[:,:,-1,:]
#psiLossFinal = psiLoss[:,:,-1,:]
#totErrorFinal = gammaLossFinal + psiLossFinal
gammaLossFinal = gammaLossR[:,:,-1,:]
psiLossFinal = psiLossR[:,:,-1,:]
totErrorFinal = (gammaLossFinal + psiLossFinal) / 2

# Extract final oracle errors
gammaLossFinalOr = gammaLossOrR[-1,:]
psiLossFinalOr = psiLossOrR[-1,:]
totErrorFinalOr = (gammaLossFinalOr + psiLossFinalOr) / 2

# Put in list to iterate over
errorList = [totErrorFinal, gammaLossFinal, psiLossFinal]
errorListOr = [totErrorFinalOr, gammaLossFinalOr, psiLossFinalOr]

# Plot the two loss function errors
plt.figure(figsize=(12, 4))
labels = ['Total error', 'Mu error', 'Cov error']
logScale = True
for i in range(len(errorList)):
    plt.subplot(1, 3, i+1)
    plt.scatter(errorList[i][0,0,:], errorList[i][1,0,:], label='Norm', c='b')
    plt.scatter(errorList[i][0,0,:], errorListOr[i], label='Oracle', c='r')
    #plt.ylabel(f'Error {lossTypeVec[0]}')
    plt.xlabel(f'Error {lossTypeVec[0]}')
    plt.ylabel(f'Error')
    plt.title(labels[i])
    minVal = np.min(errorList[i][:,0,:])
    maxVal = np.max(errorList[i][:,0,:])
    if logScale:
        plt.yscale('log')
        plt.xscale('log')
    # Plot identity line
    plt.plot([minVal, maxVal], [minVal, maxVal], 'k--')
    plt.legend()

# Prevent overlap
plt.tight_layout()
plt.savefig(plotDir + f'02_final_loss_loss_functions.png',
            bbox_inches='tight')
plt.close()


# Plot the upweighting of the covariance
plt.figure(figsize=(12, 4))
labels = ['Total error', 'Mu error', 'Cov error']
logScale = True
l = 0
for i in range(len(errorList)):
    plt.subplot(1, 3, i+1)
    plt.scatter(errorList[i][l,0,:], errorList[i][l,1,:], label='Upweight cov', c='b')
    plt.scatter(errorList[i][l,0,:], errorListOr[i], label='Oracle', c='r')
    #plt.ylabel(f'Error {lossTypeVec[0]}')
    plt.ylabel(f'Error')
    plt.xlabel(f'Error unweighted')
    plt.title(labels[i])
    minVal = np.min(errorList[i][:,0,:])
    maxVal = np.max(errorList[i][:,0,:])
    if logScale:
        plt.yscale('log')
        plt.xscale('log')
    # Plot identity line
    plt.plot([minVal, maxVal], [minVal, maxVal], 'k--')
    plt.legend()

# Prevent overlap
plt.tight_layout()
plt.savefig(plotDir + f'02_final_loss_cov_weight_functions.png',
            bbox_inches='tight')
plt.close()


