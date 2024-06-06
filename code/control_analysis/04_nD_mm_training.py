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
resultsDir = '../../results/controls/04_nd_mm_training/'
os.makedirs(resultsDir, exist_ok=True)

# set seed
np.random.seed(1911)
# Parameters of simulation
nDim = 25

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
        with torch.no_grad():
            muList = [self.mu.detach().clone()]
            covList = [self.cov.detach().clone()]
            gamma, psi = self.get_moments()
            gammaList = [gamma.detach().clone()]
            psiList = [psi.detach().clone()]
            loss = lossFunc(gamma, psi, muObs, covObs)
            lossList = [loss.item()]
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
                gamma, psi = self.get_moments()
                loss = lossFunc(gamma, psi*5, muObs, covObs*5)
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
nReps = 30

# Parameters of fitting
nIter = 300
lossType = 'mse'
dtype = torch.float32
lr = 0.2
optimizerType = 'NAdam'
decayIter = 10
lrGamma = 0.9
nCycles = 6

# Initialize lists for storing results
gammaTrueArr = torch.zeros(nDim, nReps)
gammaFitArr = torch.zeros(nDim, nIter*nCycles+1, nReps)
gammaOracleArr = torch.zeros(nDim, nIter*nCycles+1, nReps)
psiTrueArr = torch.zeros(nDim, nDim, nReps)
psiFitArr = torch.zeros(nDim, nDim, nIter*nCycles+1, nReps)
psiOracleArr = torch.zeros(nDim, nDim, nIter*nCycles+1, nReps)
muTrueArr = torch.zeros(nDim, nReps)
muFitArr = torch.zeros(nDim, nIter*nCycles+1, nReps)
muOracleArr = torch.zeros(nDim, nIter*nCycles+1, nReps)
covTrueArr = torch.zeros(nDim, nDim, nReps)
covFitArr = torch.zeros(nDim, nDim, nIter*nCycles+1, nReps)
covOracleArr = torch.zeros(nDim, nDim, nIter*nCycles+1, nReps)
lossFitArr = torch.zeros(nIter*nCycles+1, nReps)
lossOracleArr = torch.zeros(nIter*nCycles+1, nReps)

start = time.time()
for r in range(nReps):
    # Get parameters
    mu, cov = sample_parameters(nDim, covType=covType)
    varScaleAdjusted = varScale / torch.tensor(nDim/3.0)
    cov = cov * varScaleAdjusted
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


np.save(resultsDir + f'gammaTrue_{covType}_n_{nDim}_{lossType}.npy', gammaTrueArr.numpy())
np.save(resultsDir + f'gammaFit_{covType}_n_{nDim}_{lossType}.npy', gammaFitArr.numpy())
np.save(resultsDir + f'gammaOracle_{covType}_n_{nDim}_{lossType}.npy', gammaOracleArr.numpy())
np.save(resultsDir + f'psiTrue_{covType}_n_{nDim}_{lossType}.npy', psiTrueArr.numpy())
np.save(resultsDir + f'psiFit_{covType}_n_{nDim}_{lossType}.npy', psiFitArr.numpy())
np.save(resultsDir + f'psiOracle_{covType}_n_{nDim}_{lossType}.npy', psiOracleArr.numpy())
np.save(resultsDir + f'muTrue_{covType}_n_{nDim}_{lossType}.npy', muTrueArr.numpy())
np.save(resultsDir + f'muFit_{covType}_n_{nDim}_{lossType}.npy', muFitArr.numpy())
np.save(resultsDir + f'muOracle_{covType}_n_{nDim}_{lossType}.npy', muOracleArr.numpy())
np.save(resultsDir + f'covTrue_{covType}_n_{nDim}_{lossType}.npy', covTrueArr.numpy())
np.save(resultsDir + f'covFit_{covType}_n_{nDim}_{lossType}.npy', covFitArr.numpy())
np.save(resultsDir + f'covOracle_{covType}_n_{nDim}_{lossType}.npy', covOracleArr.numpy())
np.save(resultsDir + f'lossFit_{covType}_n_{nDim}_{lossType}.npy', lossFitArr.numpy())
np.save(resultsDir + f'lossOracle_{covType}_n_{nDim}_{lossType}.npy', lossOracleArr.numpy())

print(f'Time taken: {time.time() - start:.2f} seconds')

