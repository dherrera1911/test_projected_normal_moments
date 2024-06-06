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
resultsDir = '../../results/controls/01_3d_mm_training_test_params/'
os.makedirs(resultsDir, exist_ok=True)

# set seed
np.random.seed(1911)
# Parameters of simulation
nDim = 3

# CREATE A NEW CLASS WHERE THE FITTING SAVES AND RETURNS
# THE MOMENTS AT EACH ITERATION
class ProjNormFit(pn.ProjNorm):
    def fit(self, muObs, covObs, nIter=100, lr=0.5, lrGamma=0.75,
            nCycles=1, decayIter=20, lossType='norm', cycleMult=0.25,
            covWeight=1, optimizerType='SGD'):
        # Initialize loss function
        if lossType == 'norm':
            lossFunc = pn.loss_norm
        elif lossType == 'mse':
            lossFunc = pn.loss_mse
        else:
            raise ValueError('Loss function not recognized.')
        # Initialize the loss list
        with torch.no_grad():
            muList = [self.mu.detach().clone()]
            covList = [self.cov.detach().clone()]
            gamma, psi = self.get_moments()
            gammaList = [gamma.detach().clone()]
            psiList = [psi.detach().clone()]
            loss = lossFunc(gamma, psi, muObs, covObs)
            lossList = [loss.item()]
        for c in range(nCycles):
            lrCycle = lr * cycleMult**c # Decrease the initial learning rate
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
                loss = lossFunc(muOut, covOut*covWeight, muObs, covObs*covWeight)
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
varScale = 0.25
nSamples = 100000
covType = 'correlated'
nReps = 20

# Parameters of fitting
nIter = 300
lossTypeVec = ['mse', 'norm']
dtype = torch.float32
lr = 0.3
optimizerType = 'NAdam'
decayIter = 10
lrGamma = 0.9
nCycles = 4
covWeightVec = [1, 2.5]

# Initialize lists for storing results
d1 = len(lossTypeVec)
d2 = len(covWeightVec)
gammaTrueArr = torch.zeros(nDim, nReps)
gammaFitArr = torch.zeros(d1, d2, nDim, nIter*nCycles+1, nReps)
gammaOracleArr = torch.zeros(nDim, nIter*nCycles+1, nReps)
psiTrueArr = torch.zeros(nDim, nDim, nReps)
psiFitArr = torch.zeros(d1, d2, nDim, nDim, nIter*nCycles+1, nReps)
psiOracleArr = torch.zeros(nDim, nDim, nIter*nCycles+1, nReps)
muTrueArr = torch.zeros(nDim, nReps)
muFitArr = torch.zeros(d1, d2, nDim, nIter*nCycles+1, nReps)
muOracleArr = torch.zeros(nDim, nIter*nCycles+1, nReps)
covTrueArr = torch.zeros(nDim, nDim, nReps)
covFitArr = torch.zeros(d1, d2, nDim, nDim, nIter*nCycles+1, nReps)
covOracleArr = torch.zeros(nDim, nDim, nIter*nCycles+1, nReps)
lossFitArr = torch.zeros(d1, d2, nIter*nCycles+1, nReps)
lossOracleArr = torch.zeros(nIter*nCycles+1, nReps)

start = time.time()
for r in range(nReps):
    print(f'Iteration {r+1}/{nReps}')
    # Get parameters
    mu, cov = sample_parameters(nDim, covType=covType)
    cov = cov * varScale
    # Initialize the projected normal with the true parameters
    prnorm = pn.ProjNorm(nDim=nDim, muInit=mu, covInit=cov,
                         requires_grad=False, dtype=dtype)
    # Get empirical moment estimates
    gammaE, psiE = prnorm.empirical_moments(nSamples=nSamples)
    muTrueArr[:, r] = mu
    covTrueArr[:, :, r] = cov
    gammaTrueArr[:, r] = gammaE
    psiTrueArr[:, :, r] = psiE

    ##### Get the reference fit
    prnormFit = ProjNormFit(nDim=nDim, muInit=mu, covInit=cov, dtype=dtype)
    lossOrac, muOrac, covOrac, gammaOrac, psiOrac = \
        prnormFit.fit(muObs=gammaE, covObs=psiE, nIter=nIter, lr=lr*0.1,
                      lrGamma=lrGamma, nCycles=nCycles, decayIter=decayIter,
                      lossType=lossTypeVec[0], covWeight=covWeightVec[0],
                      optimizerType=optimizerType)
    gammaOracleArr[:, :, r] = gammaOrac.t()
    muOracleArr[:, :, r] = muOrac.t()
    psiOracleArr[:, :, :, r] = psiOrac.permute(1, 2, 0)
    covOracleArr[:, :, :, r] = covOrac.permute(1, 2, 0)
    lossOracleArr[:, r] = lossOrac

    # Get the actual fits
    for i in range(d1):
        for j in range(d2):
            ##### Get the fit
            muInit = gammaE / gammaE.norm()
            covInit = torch.eye(nDim, dtype=dtype) * 0.1
            prnormFit = ProjNormFit(nDim=nDim, muInit=muInit, covInit=covInit, dtype=dtype)
            loss, muFit, covFit, gammaFit, psiFit = \
                prnormFit.fit(muObs=gammaE, covObs=psiE, nIter=nIter, lr=lr,
                              lrGamma=lrGamma, nCycles=nCycles, decayIter=decayIter,
                              lossType=lossTypeVec[i], covWeight=covWeightVec[j],
                              optimizerType=optimizerType)

            # Save the results
            gammaFitArr[i, j, :, :, r] = gammaFit.t()
            psiFitArr[i, j, :, :, :, r] = psiFit.permute(1, 2, 0)
            muFitArr[i, j, :, :, r] = muFit.t()
            covFitArr[i, j, :, :, :, r] = covFit.permute(1, 2, 0)
            lossFitArr[i, j, :, r] = loss

np.save(resultsDir + f'gammaTrue_{covType}.npy', gammaTrueArr.numpy())
np.save(resultsDir + f'gammaFit_{covType}.npy', gammaFitArr.numpy())
np.save(resultsDir + f'gammaOracle_{covType}.npy', gammaOracleArr.numpy())
np.save(resultsDir + f'psiTrue_{covType}.npy', psiTrueArr.numpy())
np.save(resultsDir + f'psiFit_{covType}.npy', psiFitArr.numpy())
np.save(resultsDir + f'psiOracle_{covType}.npy', psiOracleArr.numpy())
np.save(resultsDir + f'muTrue_{covType}.npy', muTrueArr.numpy())
np.save(resultsDir + f'muFit_{covType}.npy', muFitArr.numpy())
np.save(resultsDir + f'muOracle_{covType}.npy', muOracleArr.numpy())
np.save(resultsDir + f'covTrue_{covType}.npy', covTrueArr.numpy())
np.save(resultsDir + f'covFit_{covType}.npy', covFitArr.numpy())
np.save(resultsDir + f'covOracle_{covType}.npy', covOracleArr.numpy())
np.save(resultsDir + f'lossFit_{covType}.npy', lossFitArr.numpy())
np.save(resultsDir + f'lossOracle_{covType}.npy', lossOracleArr.numpy())

print(f'Time taken: {time.time() - start:.2f} seconds')
