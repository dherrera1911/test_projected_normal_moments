import torch
import numpy as np
import geotorch
import scipy


def sample_parameters(nDim, covType='uncorrelated', corrMagnitude=1):
    # Sample a random mean in the unit sphere
    mu = torch.randn(nDim) # Random mean
    mu = mu / torch.norm(mu) # Project to the unit sphere
    if covType == 'isotropic':
        covariance = torch.eye(nDim)
    elif covType == 'uncorrelated':
        var = torch.rand(nDim) * 0.45 + 0.05
        covariance = torch.diag(var)
    elif covType == 'correlated':
        #eig = np.exp(np.linspace(start=-0.5, stop=0.5, num=nDim))
        #eig = np.exp(np.linspace(start=-0.5*np.sqrt((nDim/3)),
        #                         stop=0.5*np.sqrt((nDim/3)), num=nDim))
        eig = np.exp(np.linspace(start=-0.5*np.sqrt(nDim),
                                 stop=0.5*np.sqrt(nDim), num=nDim))
        correlation = make_random_correlation(eig) * corrMagnitude + \
            torch.eye(nDim) * (1 - corrMagnitude)
        var = torch.rand(nDim) * 0.45 + 0.05
        covariance = corr_2_cov(corr=correlation, variances=var)
    elif covType == 'symmetric':
        # Generate random orthogonal matrix
        soGroup = geotorch.SO(size=(nDim,nDim))
        orthogonal = soGroup.sample()
        # Generate random diagonal matrix with 
        eig = torch.rand(nDim) * 0.45 + 0.05
        # Generate covariance matrix
        covariance = torch.einsum('ij,j,jk->ik', orthogonal, eig, orthogonal.t())
        # Take as mu the first eigenvector
        mu = orthogonal[:,0]
    return mu, covariance


def corr_2_cov(corr, variances):
    """ Convert a correlation matrix to a covariance matrix.
    ----------------
    Arguments:
    ----------------
      - corr: Correlation matrix. Shape (n x n).
      - variances: Vector of variances. Shape (n).
    ----------------
    Outputs:
    ----------------
      - covariance: Covariance matrix. Shape (n x n).
    """
    D = torch.diag(torch.sqrt(variances))
    covariance = torch.einsum('ij,jk,kl->il', D, corr, D)
    return covariance


def make_random_correlation(eig):
    """ Create a random covariance matrix with the given variances, and
    whose correlation matrix has the given eigenvalues.
    ----------------
    Arguments:
    ----------------
      - variances: Vector of variances. Shape (n).
      - eig: Vector of eigenvalues of correlation matrix. Shape (n).
    ----------------
    Outputs:
    ----------------
      - covariance: Random covariance matrix. Shape (n x n).
    """
    # Make sum of eigenvalues equal to n
    eig = np.array(eig)
    eig = eig / eig.sum() * len(eig)
    # Create random correlation matrix
    randCorr = scipy.stats.random_correlation(eig).rvs()
    randCorr = torch.as_tensor(randCorr, dtype=torch.float32)
    return randCorr


# Get statistics of errors across samples
def error_stats(err):
    return {'mean': err.mean(dim=-1), 'median': err.median(dim=-1).values,
            'std': err.std(dim=-1), 'q1': err.quantile(0.25, dim=-1),
            'q3': err.quantile(0.75, dim=-1)}


