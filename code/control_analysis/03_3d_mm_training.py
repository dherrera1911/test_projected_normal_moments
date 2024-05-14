##################
#
# Simulate data coming from a projected normal.
# Fit a projected normal to the data through moment
# matching and test how stable the optimization is.
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
resultsDir = '../../results/controls/03_3d_mm_training/'
os.makedirs(resultsDir, exist_ok=True)

# set seed
np.random.seed(1911)
# Parameters of simulation
nDim = 3

# CREATE A NEW CLASS WHERE THE FITTING SAVES AND RETURNS
# THE MOMENTS AT EACH ITERATION
class ProjNormFit(pn.ProjNorm):
    def fit(self, muObs, covObs, nIter=100, lr=0.5, lrGamma=0.75,
            nCycles=1, decayIter=20, lossType='norm', optimizerType='SGD'):
        # Initialize loss function
        if lossType == 'norm':
            lossFunc = pn.loss_norm
        elif lossType == 'mse':
            lossFunc = pn.loss_mse
        else:
            raise ValueError('Loss function not recognized.')
        # Initialize the loss list
        lossList = []
        muList = []
        covList = []
        gammaList = []
        psiList = []
        for c in range(nCycles):
            lrCycle = lr * 0.25**c # Decrease the initial learning rate
            # Initialize the optimizer
            if optimizerType == 'SGD':
                optimizer = torch.optim.SGD(self.parameters(), lr=lrCycle)
            elif optimizerType == 'Adam':
                optimizer = torch.optim.Adam(self.parameters(), lr=lrCycle)
            elif optimizerType == 'NAdam':
                optimizer = torch.optim.NAdam(self.parameters(), lr=lrCycle)
            elif optimizerType == 'LBFGS':
                optimizer = torch.optim.LBFGS(self.parameters(), lr=lrCycle)
            else:
                raise ValueError('Optimizer not recognized.')
            # Initialize the scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decayIter,
                                                        gamma=lrGamma)
            # Iterate over the number of iterations
            for i in range(nIter):
                # Zero the gradients
                optimizer.zero_grad()
                muOut, covOut = self.get_moments()
                loss = lossFunc(muOut, covOut, muObs, covObs)
                # Compute the gradients
                loss.backward()
                # Optimize the parameters
                optimizer.step()
                # Append the loss to the list
                lossList.append(loss.item())
                # Step the scheduler
                scheduler.step()
                with torch.no_grad():
                    muList.append(self.mu.detach().clone())
                    covList.append(self.cov.detach().clone())
                    gamma, psi = self.get_moments()
                    gammaList.append(gamma.detach().clone())
                    psiList.append(psi.detach().clone())
        lossVec = torch.tensor(lossList)
        # Concatenate the lists
        muList = torch.stack(muList)
        covList = torch.stack(covList)
        gammaList = torch.stack(gammaList)
        psiList = torch.stack(psiList)
        return lossVec, muList, covList, gammaList, psiList

##############
# FIT THROUGH MOMENT MATCHING WITH DIFFERENT INITIALIZATIONS
##############

# Parameters of simulation
varScale = 0.5
nSamples = 1000000
covType = 'correlated'
nReps = 30

# Parameters of fitting
nIter = 300
lossType = 'mse'
dtype = torch.float32
lr = 0.2
optimizerType = 'NAdam'
decayIter = 10
lrGamma = 0.9
nCycles = 2
cycleMult = 0.2


gammaTrueArr = torch.zeros(nDim, nReps)
gammaFitArr = torch.zeros(nDim, nIter*nCycles, nReps)
gammaOracleArr = torch.zeros(nDim, nIter*nCycles, nReps)
psiTrueArr = torch.zeros(nDim, nDim, nReps)
psiFitArr = torch.zeros(nDim, nDim, nIter*nCycles, nReps)
psiOracleArr = torch.zeros(nDim, nDim, nIter*nCycles, nReps)
muTrueArr = torch.zeros(nDim, nReps)
muFitArr = torch.zeros(nDim, nIter*nCycles, nReps)
muOracleArr = torch.zeros(nDim, nIter*nCycles, nReps)
covTrueArr = torch.zeros(nDim, nDim, nReps)
covFitArr = torch.zeros(nDim, nDim, nIter*nCycles, nReps)
covOracleArr = torch.zeros(nDim, nDim, nIter*nCycles, nReps)
lossFitArr = torch.zeros(nIter*nCycles, nReps)
lossOracleArr = torch.zeros(nIter*nCycles, nReps)

start = time.time()
for r in range(nReps):
    # Get parameters
    mu, cov = sample_parameters(nDim, covType=covType)
    cov = cov * varScale
    # Initialize the projected normal with the true parameters
    prnorm = pn.ProjNorm(nDim=nDim, muInit=mu, covInit=cov,
                         requires_grad=False, dtype=dtype)
    # Get empirical moment estimates
    gammaE, psiE = prnorm.empirical_moments(nSamples=nSamples)

    ##### Get the reference fit
    prnormFit = ProjNormFit(nDim=nDim, muInit=mu, covInit=cov, dtype=dtype)
    lossOrac, muOrac, covOrac, gammaOrac, psiOrac = \
        prnormFit.fit(muObs=gammaE, covObs=psiE, nIter=nIter, lr=lr*0.1,
                      lrGamma=lrGamma, nCycles=nCycles, decayIter=decayIter,
                      lossType=lossType, optimizerType=optimizerType)

    #### Get regular fit
    muInit = gammaE / gammaE.norm()
    covInit = torch.eye(nDim, dtype=dtype) * 0.1
    prnormFit = ProjNormFit(nDim=nDim, muInit=muInit, covInit=covInit, dtype=dtype)
    loss, muFit, covFit, gammaFit, psiFit = \
      prnormFit.fit(muObs=gammaE, covObs=psiE, nIter=nIter, lr=lr, lrGamma=lrGamma,
                    nCycles=nCycles, decayIter=decayIter, lossType=lossType,
                    optimizerType=optimizerType)

    # Save the results
    gammaTrueArr[:, r] = gammaE
    gammaFitArr[:, :, r] = gammaFit.t()
    gammaOracleArr[:, :, r] = gammaOrac.t()
    psiTrueArr[:, :, r] = psiE
    psiFitArr[:, :, :, r] = psiFit.permute(1, 2, 0)
    psiOracleArr[:, :, :, r] = psiOrac.permute(1, 2, 0)
    muTrueArr[:, r] = mu
    muFitArr[:, :, r] = muFit.t()
    muOracleArr[:, :, r] = muOrac.t()
    covTrueArr[:, :, r] = cov
    covFitArr[:, :, :, r] = covFit.permute(1, 2, 0)
    covOracleArr[:, :, :, r] = covOrac.permute(1, 2, 0)
    lossFitArr[:, r] = loss
    lossOracleArr[:, r] = lossOrac


np.save(resultsDir + f'gammaTrue_{covType}Arr.npy', gammaTrueArr.numpy())
np.save(resultsDir + f'gammaFit_{covType}Arr.npy', gammaFitArr.numpy())
np.save(resultsDir + f'gammaOracle_{covType}Arr.npy', gammaOracleArr.numpy())
np.save(resultsDir + f'psiTrue_{covType}Arr.npy', psiTrueArr.numpy())
np.save(resultsDir + f'psiFit_{covType}Arr.npy', psiFitArr.numpy())
np.save(resultsDir + f'psiOracle_{covType}Arr.npy', psiOracleArr.numpy())
np.save(resultsDir + f'muTrue_{covType}Arr.npy', muTrueArr.numpy())
np.save(resultsDir + f'muFit_{covType}Arr.npy', muFitArr.numpy())
np.save(resultsDir + f'muOracle_{covType}Arr.npy', muOracleArr.numpy())
np.save(resultsDir + f'covTrue_{covType}Arr.npy', covTrueArr.numpy())
np.save(resultsDir + f'covFit_{covType}Arr.npy', covFitArr.numpy())
np.save(resultsDir + f'covOracle_{covType}Arr.npy', covOracleArr.numpy())
np.save(resultsDir + f'lossFitArray_{covType}Arr.npy', lossFitArr.numpy())
np.save(resultsDir + f'lossOracleArray_{covType}Arr.npy', lossOracleArr.numpy())

print(f'Time taken: {time.time() - start:.2f} seconds')

