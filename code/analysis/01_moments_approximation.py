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
import os
import time
import argparse
import yaml
import sys

sys.path.append("../")
from analysis_functions import *
from plotting_functions import *

# Uncomment block corresponding to how this is being run
#### TO RUN FROM THE COMMAND LINE
# Set up command-line argument parsing
parser = argparse.ArgumentParser(
    description="Run analysis with specified configuration file."
)
parser.add_argument(
    "config_path", type=str, help="Path to the configuration YAML file."
)
args = parser.parse_args()
# Load the YAML file
with open(args.config_path, "r") as file:
    config = yaml.safe_load(file)
###

### TO RUN INTERACTIVE
# fileName = 'par_approx_3d.yaml'
# config = yaml.safe_load(open(fileName, 'r'))
###

# Simulation parameters
varScaleVec = config["SimulationParameters"]["varScaleVec"]
covTypeVec = config["SimulationParameters"]["covTypeVec"]
nSamples = config["SimulationParameters"]["nSamples"]
nReps = config["SimulationParameters"]["nReps"]
nDimVec = config["SimulationParameters"]["nDimVec"]

# Create saving directory
resultsDir = config["SavingDir"]["resultsDir"]
os.makedirs(resultsDir, exist_ok=True)
getEmpiricalError = (
    False  # If true, compute and save the empirical error, but takes longer
)

# set seed
np.random.seed(1911)

##############
# GET APPROXIMATION ERRORS
##############

start = time.time()
for c, covType in enumerate(covTypeVec):
    # loop over covariance types
    for n, nDim in enumerate(nDimVec):
        # loop over dimensions

        # Initialize the tensors to store the moments
        gammaTrue = torch.zeros(len(varScaleVec), nDim, nReps)
        gammaTaylor = torch.zeros(len(varScaleVec), nDim, nReps)
        psiTrue = torch.zeros(len(varScaleVec), nDim, nDim, nReps)
        psiTaylor = torch.zeros(len(varScaleVec), nDim, nDim, nReps)
        mu = torch.zeros(len(varScaleVec), nDim, nReps)
        cov = torch.zeros(len(varScaleVec), nDim, nDim, nReps)
        if getEmpiricalError:
            gammaTrue2 = torch.zeros(len(varScaleVec), nDim, nReps)
            psiTrue2 = torch.zeros(len(varScaleVec), nDim, nDim, nReps)

        for v, varScaleUn in enumerate(varScaleVec):
            # loop over variance scales
            varScale = varScaleUn / torch.tensor(nDim / 3.0)

            for r in range(nReps):
                progressStr = f"covType = {covType}, nDim = {nDim}, varScale = {varScale}, rep = {r}"
                print(progressStr)
                # Get parameters
                mu[v, :, r], cov[v, :, :, r] = sample_parameters(nDim, covType=covType)
                cov[v, :, :, r] = cov[v, :, :, r] * varScale
                # Initialize the projected normal
                prnorm = pn.ProjNorm(
                    nDim=nDim,
                    muInit=mu[v, :, r],
                    covInit=cov[v, :, :, r],
                    requires_grad=False,
                )
                # Get empirical moment estimates
                gammaTrue[v, :, r], psiTrue[v, :, :, r] = prnorm.empirical_moments(
                    nSamples=nSamples
                )
                # Get the Taylor approximation moments
                gammaTaylor[v, :, r], psiTaylor[v, :, :, r] = prnorm.get_moments()
                if getEmpiricalError:
                    # Compute second empirical estimate
                    gammaTrue2[v, :, r], psiTrue2[v, :, r] = prnorm.empirical_moments(
                        nSamples=nSamples
                    )

        # Save the moments
        np.save(resultsDir + f"gammaTrue_{covType}_n_{nDim}.npy", gammaTrue.numpy())
        np.save(resultsDir + f"gammaTaylor_{covType}_n_{nDim}.npy", gammaTaylor.numpy())
        np.save(resultsDir + f"psiTrue_{covType}_n_{nDim}.npy", psiTrue.numpy())
        np.save(resultsDir + f"psiTaylor_{covType}_n_{nDim}.npy", psiTaylor.numpy())
        np.save(resultsDir + f"mu_{covType}_n_{nDim}.npy", mu.numpy())
        np.save(resultsDir + f"cov_{covType}_n_{nDim}.npy", cov.numpy())
        if getEmpiricalError:
            np.save(
                resultsDir + f"gammaTrue2_{covType}_n_{nDim}.npy", gammaTrue2.numpy()
            )
            np.save(resultsDir + f"psiTrue2_{covType}_n_{nDim}.npy", psiTrue2.numpy())

print(f"Time taken: {time.time() - start:.2f} seconds")
