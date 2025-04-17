"""Class implementing ellipsoid matrix B with constraints."""
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize


def make_basis(n_dim, n_basis):
    """Create a set of orthonormal basis vectors using cosine and sine functions.
    The basis vectors are normalized to have unit norm.

    Parameters
    ----------
    n_dim : int
        The dimension of the basis vectors.

    n_basis : int
        Number of basis functions. Must be even.

    Returns
    -------
    torch.Tensor, shape (n_basis, n_dim)
        The orthonormal basis vectors.
    """
    # Check that m is even
    if n_basis % 2 != 0:
        raise ValueError("Number of basis vectors must be even")

    # Initialize the basis vectors
    x = torch.linspace(0, 1, n_dim+1) * 2 * torch.pi
    x = x[:-1].unsqueeze(0)
    x = x.repeat(n_basis // 2, 1)
    x = x * torch.arange(1, n_basis // 2 + 1).unsqueeze(1)

    # Compute the cosine and sine basis functions
    cos = torch.cos(x)
    sin = torch.sin(x)
    cos = cos / torch.norm(cos, dim=1, keepdim=True)
    sin = sin / torch.norm(sin, dim=1, keepdim=True)
    # Concatenate the cosine and sine basis functions
    basis = torch.cat((cos, sin), dim=0)
    return basis


class ParametrizedVector(nn.Module):
    """Class implementing a set of vectors parametrized by a basis
    of orthonormal cosine and sine functions."""

    def __init__(self, n_dim, n_basis):
        """Initialize the class.

        Parameters
        ----------
          n_dim : int
              The dimension of the parametrized vector.

          n_basis : int
              Number of basis functions. Must be even.
        """
        super().__init__()
        self.n_dim = n_dim
        self.n_basis = n_basis
        basis = make_basis(n_dim, n_basis)
        self.register_buffer("basis", basis)

    def forward(self, coeffs):
        """Get the parametrized vectors

        Parameters
        ----------
        coeffs : torch.Tensor, shape (n_vecs, n_basis)
            Coefficients of the basis functions for the parametrized vectors.

        Returns
        -------
        torch.Tensor, shape (n_vecs, n_dim)
            Vectors in subspace spanned by the basis functions.
        """
        coeffs_norm = coeffs / torch.norm(coeffs, dim=1, keepdim=True)
        output = torch.einsum("ki,ij->kj", coeffs_norm, self.basis)
        return output


    def right_inverse(self, vecs):
        """Get the right inverse of the parametrized vectors.

        Parameters
        ----------
        vecs : torch.Tensor, shape (n_vecs, n_dim)
            Vectors in subspace spanned by the basis functions.

        Returns
        -------
        torch.Tensor, shape (n_vecs, n_basis)
            Coefficients of the basis functions for the parametrized vectors.
        """
        coeffs = torch.einsum("kj,ij->ki", vecs, self.basis)
        coeffs = coeffs / torch.norm(coeffs, dim=1, keepdim=True)
        return coeffs

