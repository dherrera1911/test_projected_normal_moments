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
import os
import time
import argparse
import yaml
import sys
sys.path.append('../')
from analysis_functions import *
from plotting_functions import *

# Uncomment block corresponding to how this is being run
#### TO RUN FROM THE COMMAND LINE
# Set up command-line argument parsing
parser = argparse.ArgumentParser(description='Run analysis with specified configuration file.')
parser.add_argument('config_path', type=str, help='Path to the configuration YAML file.')
args = parser.parse_args()
# Load the YAML file
with open(args.config_path, 'r') as file:
    config = yaml.safe_load(file)
###

### TO RUN INTERACTIVE
#fileName = 'par_mm_3d.yaml'
#config = yaml.safe_load(open(fileName, 'r'))
###

# Simulation parameters
varScaleVec = config['SimulationParameters']['varScaleVec']
covTypeVec = config['SimulationParameters']['covTypeVec']
nSamples = config['SimulationParameters']['nSamples']
nReps = config['SimulationParameters']['nReps']
nDimVec = config['SimulationParameters']['nDimVec']

# Fitting parameters
nIter = config['FittingParameters']['nIter']
lossType = config['FittingParameters']['lossType']
lr = config['FittingParameters']['lr']
optimizerType = config['FittingParameters']['optimizerType']
decayIter = config['FittingParameters']['decayIter']
lrGamma = config['FittingParameters']['lrGamma']
nCycles = config['FittingParameters']['nCycles']
cycleMult = config['FittingParameters']['cycleMult']
covWeight = config['FittingParameters']['covWeight']

# Results saving directory
resultsDir = config['SavingDir']['resultsDir']
saveFig = True
os.makedirs(resultsDir, exist_ok=True)

# set seed
np.random.seed(1911)
# Set the data type
dtype = torch.float32

##############
# PLOT DISTRIBUTION OF ERRORS IN ESTIMATED PARAMETERS
##############

start = time.time()
for c in range(len(covTypeVec)):
    print('Covariance type:', covTypeVec[c])
    for n in range(len(nDimVec)):
        nDim = nDimVec[n]
        print('Dimension:', nDim)
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
            print('Variance scale:', varScaleVec[v])
            varScale = varScaleVec[v] / torch.tensor(nDim/3.0)
            for r in range(nReps):
                print('Rep:', r)
                # Get parameters
                muTrue[v,:,r], covTrue[v,:,:,r] = sample_parameters(nDim, covType=covType)
                covTrue[v,:,:,r] = covTrue[v,:,:,r] * varScale
                # Initialize the projected normal with the true parameters
                prnorm = pn.ProjNorm(nDim=nDim, muInit=muTrue[v,:,r],
                                     covInit=covTrue[v,:,:,r], requires_grad=False,
                                     dtype=dtype)
                # Get empirical moment estimates
                gammaTrue[v,:,r], psiTrue[v,:,:,r] = prnorm.empirical_moments(nSamples=nSamples)
                # Fit the projected normal to the empirical moments
                muInit = gammaTrue[v,:,r] / torch.norm(gammaTrue[v,:,r])
                covInit = torch.eye(nDim, dtype=dtype) * 0.2
                prnormFit = pn.ProjNorm(nDim=nDim, muInit=muInit, covInit=covInit, dtype=dtype)
                loss = prnormFit.moment_match(muObs=gammaTrue[v,:,r],
                                              covObs=psiTrue[v,:,:,r], nIter=nIter, lr=lr,
                                              lrGamma=lrGamma, decayIter=decayIter,
                                              lossType=lossType, nCycles=nCycles,
                                              cycleMult=cycleMult, covWeight=covWeight,
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
        np.save(resultsDir + f'gammaTrue_{covType}_n{nDim}.npy', gammaTrue.numpy())
        np.save(resultsDir + f'gammaApprox_{covType}_n{nDim}.npy', gammaApprox.numpy())
        np.save(resultsDir + f'gammaTaylor_{covType}_n{nDim}.npy', gammaTaylor.numpy())
        np.save(resultsDir + f'psiTrue_{covType}_n{nDim}.npy', psiTrue.numpy())
        np.save(resultsDir + f'psiApprox_{covType}_n{nDim}.npy', psiApprox.numpy())
        np.save(resultsDir + f'psiTaylor_{covType}_n{nDim}.npy', psiTaylor.numpy())
        np.save(resultsDir + f'muTrue_{covType}_n{nDim}.npy', muTrue.numpy())
        np.save(resultsDir + f'muFit_{covType}_n{nDim}.npy', muFit.numpy())
        np.save(resultsDir + f'covTrue_{covType}_n{nDim}.npy', covTrue.numpy())
        np.save(resultsDir + f'covFit_{covType}_n{nDim}.npy', covFit.numpy())
        np.save(resultsDir + f'lossArray_{covType}_n{nDim}.npy', lossArray.numpy())

print(f'Time taken: {time.time() - start:.2f} seconds')

