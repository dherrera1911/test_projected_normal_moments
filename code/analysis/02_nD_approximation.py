##################
#
# For the case of the sphere embedded in nD, obtain the empirical
# moments of the projected normal, and the Taylor approximation
# to the moments. Save the results.
#
# Additionally, a second empirical estimate of the moments can be
# computed to estimate the error in the empirical moments for
# a given number of samples.
#
##################

import torch
import numpy as np
import projected_normal as pn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import os
import time
import sys
sys.path.append('../')
from analysis_functions import *
from plotting_functions import *

saveFig = True
resultsDir = '../../results/02_nD_approximation/'
os.makedirs(resultsDir, exist_ok=True)
getEmpiricalError = False # If true, compute and save the empirical error, but takes longer

# set seed
np.random.seed(1911)

##############
# GET APPROXIMATION ERRORS
##############

# Parameters of simulation
nDimList = [3, 5, 10, 25, 50, 100]
varScaleVec = [0.0625, 0.25, 1.0, 4.0]
covTypeVec = ['uncorrelated', 'correlated', 'symmetric']
nSamples = 10**6
nReps = 200

start = time.time()
for c in range(len(covTypeVec)):
    for n in range(len(nDimList)):
        nDim = nDimList[n]
        covType = covTypeVec[c]
        gammaTrue = torch.zeros(len(varScaleVec), nDim, nReps)
        gammaTaylor = torch.zeros(len(varScaleVec), nDim, nReps)
        psiTrue = torch.zeros(len(varScaleVec), nDim, nDim, nReps)
        psiTaylor = torch.zeros(len(varScaleVec), nDim, nDim, nReps)
        mu = torch.zeros(len(varScaleVec), nDim, nReps)
        cov = torch.zeros(len(varScaleVec), nDim, nDim, nReps)
        if getEmpiricalError:
            gammaTrue2 = torch.zeros(len(varScaleVec), nDim, nReps)
            psiTrue2 = torch.zeros(len(varScaleVec), nDim, nDim, nReps)

        for v in range(len(varScaleVec)):
            varScale = varScaleVec[v] / torch.tensor(nDim/3.0)
            for r in range(nReps):
                # Get parameters
                mu[v,:,r], cov[v,:,:,r] = sample_parameters(nDim, covType=covType)
                cov[v,:,:,r] = cov[v,:,:,r] * varScale
                # Initialize the projected normal
                prnorm = pn.ProjNorm(nDim=nDim, muInit=mu[v,:,r],
                                     covInit=cov[v,:,:,r], requires_grad=False)
                # Get empirical moment estimates
                gammaTrue[v,:,r], psiTrue[v,:,:,r] = prnorm.empirical_moments(nSamples=nSamples)
                # Get the Taylor approximation moments
                gammaTaylor[v,:,r], psiTaylor[v,:,:,r] = prnorm.get_moments()
                if getEmpiricalError:
                    # Compute second empirical estimate
                    gammaTrue2[v,:,r], psiTrue2[v,:,r] = prnorm.empirical_moments(nSamples=nSamples)

        # Save the error samples
        np.save(resultsDir + f'gammaTrue_{covType}_n_{nDim}.npy', gammaTrue.numpy())
        np.save(resultsDir + f'gammaTaylor_{covType}_n_{nDim}.npy', gammaTaylor.numpy())
        np.save(resultsDir + f'psiTrue_{covType}_n_{nDim}.npy', psiTrue.numpy())
        np.save(resultsDir + f'psiTaylor_{covType}_n_{nDim}.npy', psiTaylor.numpy())
        np.save(resultsDir + f'mu_{covType}_n_{nDim}.npy', mu.numpy())
        np.save(resultsDir + f'cov_{covType}_n_{nDim}.npy', cov.numpy())
        if getEmpiricalError:
            np.save(resultsDir + f'gammaTrue2_{covType}_n_{nDim}.npy', gammaTrue2.numpy())
            np.save(resultsDir + f'psiTrue2_{covType}_n_{nDim}.npy', psiTrue2.numpy())

print(f'Time taken: {time.time() - start:.2f} seconds')

