##################
#
# For the case of a sphere embedded in 3D space,
# fit a projected normal to the empirical moments
# and compare the results to the true parameters.
#
##################

import torch
import numpy as np
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
resultsDir = '../../results/03_3d_moment_matching/'
os.makedirs(resultsDir, exist_ok=True)

# set seed
np.random.seed(1911)
# Parameters of simulation
nDim = 3

##############
# PLOT DISTRIBUTION OF ERRORS IN ESTIMATED PARAMETERS
##############

# Parameters of simulation
varScaleVec = [0.0625, 0.125, 0.25, 0.5, 1.0]
covTypeVec = ['uncorrelated', 'correlated', 'symmetric']
nSamples = 1000000

# Fitting parameters
nIter = 300
lossType = 'mse'
nReps = 5
dtype = torch.float32
lr = 0.2
optimizerType = 'NAdam'
decayIter = 10
lrGamma = 0.9
nCycles = 2
cycleMult = 0.2

start = time.time()
for c in range(len(covTypeVec)):
    # Arrays to save results
    covType = covTypeVec[c]
    gammaTrue = torch.zeros(len(varScaleVec), nDim, nReps)
    gammaApprox = torch.zeros(len(varScaleVec), nDim, nReps)
    gammaTaylor = torch.zeros(len(varScaleVec), nDim, nReps)
    psiTrue = torch.zeros(len(varScaleVec), nDim, nDim, nReps)
    psiApprox = torch.zeros(len(varScaleVec), nDim, nDim, nReps)
    psiTaylor = torch.zeros(len(varScaleVec), nDim, nDim, nReps)
    muTrue = torch.zeros(len(varScaleVec), nDim, nReps)
    muFit = torch.zeros(len(varScaleVec), nDim, nReps)
    covTrue = torch.zeros(len(varScaleVec), nDim, nDim, nReps)
    covFit = torch.zeros(len(varScaleVec), nDim, nDim, nReps)
    lossArray = torch.zeros(len(varScaleVec), nIter*nCycles, nReps)
    for v in range(len(varScaleVec)):
        varScale = varScaleVec[v]
        for r in range(nReps):
            # Get parameters
            muTrue[v,:,r], covTrue[v,:,:,r] = sample_parameters(nDim, covType=covType)
            covTrue[v,:,:,r] = covTrue[v,:,:,r] * varScale
            # Initialize the projected normal with the true parameters
            prnorm = pn.ProjNorm(nDim=nDim, muInit=muTrue[v,:,r], covInit=covTrue[v,:,:,r],
                                 requires_grad=False, dtype=dtype)
            # Get empirical moment estimates
            gammaTrue[v,:,r], psiTrue[v,:,:,r] = prnorm.empirical_moments(nSamples=nSamples)
            # Fit the projected normal to the empirical moments
            muInit = gammaTrue[v,:,r] / torch.norm(gammaTrue[v,:,r])
            covInit = torch.eye(nDim, dtype=dtype) * 0.1
            prnormFit = pn.ProjNorm(nDim=nDim, muInit=muInit, covInit=covInit, dtype=dtype)
            loss = prnormFit.moment_match(muObs=gammaTrue[v,:,r],
                                          covObs=psiTrue[v,:,:,r], nIter=nIter, lr=lr,
                                          lrGamma=lrGamma, decayIter=20, lossType=lossType,
                                          nCycles=nCycles, cycleMult=cycleMult,
                                          optimizerType=optimizerType)
            # Get the fitted moments without gradient
            with torch.no_grad():
                gammaTaylor[v,:,r], psiTaylor[v,:,:,r] = prnormFit.get_moments()
                gammaApprox[v,:,r], psiApprox[v,:,:,r] = prnormFit.empirical_moments(nSamples=nSamples) # Empirical
            # Get the fitted parameters
            muFit[v,:,r] = prnormFit.mu.detach()
            covFit[v,:,:,r] = prnormFit.cov.detach()
            # Save the loss
            lossArray[v,:,r] = loss.detach()
    np.save(resultsDir + f'gammaTrue_{covType}.npy', gammaTrue.numpy())
    np.save(resultsDir + f'gammaApprox_{covType}.npy', gammaApprox.numpy())
    np.save(resultsDir + f'gammaTaylor_{covType}.npy', gammaTaylor.numpy())
    np.save(resultsDir + f'psiTrue_{covType}.npy', psiTrue.numpy())
    np.save(resultsDir + f'psiApprox_{covType}.npy', psiApprox.numpy())
    np.save(resultsDir + f'psiTaylor_{covType}.npy', psiTaylor.numpy())
    np.save(resultsDir + f'muTrue_{covType}.npy', muTrue.numpy())
    np.save(resultsDir + f'muFit_{covType}.npy', muFit.numpy())
    np.save(resultsDir + f'covTrue_{covType}.npy', covTrue.numpy())
    np.save(resultsDir + f'covFit_{covType}.npy', covFit.numpy())
    np.save(resultsDir + f'lossArray_{covType}.npy', lossArray.numpy())

print(f'Time taken: {time.time() - start:.2f} seconds')

