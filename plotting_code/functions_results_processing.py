import torch

# Extract the results
def list_2_tensor(tensor_list):
    return torch.stack([torch.stack(tensor_list[i]) for i in range(len(tensor_list))])


def list_2_tensor_results(results_dict):
    # Apply list_2_tensor for each key in the dictionary, except for the 'sigma' key
    for key in results_dict.keys():
        if key != 'sigma':
            results_dict[key] = list_2_tensor(results_dict[key])
    return results_dict


def error_rel(x, y):
    """
    Get the relative error between two tensors as the
    percentage of the mean of their norms.
    """
    if x.dim()==2:
        error = 2 * torch.abs(x - y) / torch.abs(x + y)
    if x.dim()==3:
        error = 2 * torch.norm(x - y, dim=2) \
            / (torch.norm(x, dim=2) + torch.norm(y, dim=-1))
    elif x.dim()==4:
        error = 2 * torch.norm(x - y, dim=(-1,-2)) \
            / (torch.norm(x, dim=(-1,-2)) + torch.norm(y, dim=(-1,-2)))
    return error * 100


def error_stats(error_tensor):
    error_stats_dict = {
      'median': error_tensor.median(dim=-1).values,
      'q1': error_tensor.quantile(0.25, dim=-1),
      'q3': error_tensor.quantile(0.75, dim=-1)
    }
    return error_stats_dict


def remove_mean_component(means, covariances):
    """
    Project the covariances onto the subspace orthogonal
    to the means.
    """
    coefs = torch.einsum(
      '...i,...ij,...j->...', means, covariances, means
    )

    orthogonal_covariances = covariances \
        - torch.einsum('...i,...,...j->...ij', means, coefs, means)

    return orthogonal_covariances

