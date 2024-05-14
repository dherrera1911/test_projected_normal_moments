##################
#
# For the case of the sphere embedded in n-dimensional
# space, test the approximation
# of the moments of the projected normal by comparing to
# moments obtained from sampling.
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
resultsDir = '../../results/02_nD_approximation/'
os.makedirs(resultsDir, exist_ok=True)

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
nReps = 500

muErrType = {}
covErrType = {}
muErrRelType = {}
covErrRelType = {}
#muErrType_E = {}
#covErrType_E = {}
#muErrRelType_E = {}
#covErrRelType_E = {}

start = time.time()
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
                mu, covariance = sample_parameters(nDim, covType=covType)
                covariance = covariance * varScale
                # Initialize the projected normal
                prnorm = pn.ProjNorm(nDim=nDim, muInit=mu, covInit=covariance, requires_grad=False)
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
    np.save(plotDir + f'mu_error_{covType}.npy', muErr.numpy())
    np.save(plotDir + f'cov_error_{covType}.npy', covErr.numpy())
    np.save(plotDir + f'mu_error_rel_{covType}.npy', muErrRel.numpy())
    np.save(plotDir + f'cov_error_rel_{covType}.npy', covErrRel.numpy())

print(f'Time taken: {time.time() - start:.2f} seconds')
