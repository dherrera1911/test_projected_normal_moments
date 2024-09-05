import torch
import numpy as np
import geotorch
import scipy


def sample_parameters(n_dim, covariance_type='uncorrelated', correlation_magnitude=1):
    # Sample a random mean in the unit sphere
    mu = torch.randn(n_dim) # Random mean
    mu = mu / torch.norm(mu) # Project to the unit sphere
    if covariance_type == 'isotropic':
        covariance = torch.eye(n_dim)
    elif covariance_type == 'uncorrelated':
        var = torch.rand(n_dim) * 0.45 + 0.05
        covariance = torch.diag(var)
    elif covariance_type == 'correlated':
        #eig = np.exp(np.linspace(start=-0.5, stop=0.5, num=n_dim))
        #eig = np.exp(np.linspace(start=-0.5*np.sqrt((n_dim/3)),
        #                         stop=0.5*np.sqrt((n_dim/3)), num=n_dim))
        eig = np.exp(np.linspace(start=-0.5*np.sqrt(n_dim),
                                 stop=0.5*np.sqrt(n_dim), num=n_dim))
        correlation = make_random_correlation(eig) * correlation_magnitude + \
            torch.eye(n_dim) * (1 - correlation_magnitude)
        var = torch.rand(n_dim) * 0.45 + 0.05
        covariance = corr_2_cov(corr=correlation, variances=var)
    elif covariance_type == 'symmetric':
        # Generate random orthogonal matrix
        soGroup = geotorch.SO(size=(n_dim,n_dim))
        orthogonal = soGroup.sample()
        # Generate random diagonal matrix with 
        eig = torch.rand(n_dim) * 0.45 + 0.05
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
    err = torch.as_tensor(err)
    return {'mean': err.mean(dim=-1), 'median': err.median(dim=-1).values,
            'std': err.std(dim=-1), 'q1': err.quantile(0.25, dim=-1),
            'q3': err.quantile(0.75, dim=-1)}

def list_2d(n1, n2):
    return [[None for _ in range(n1)] for _ in range(n2)]

