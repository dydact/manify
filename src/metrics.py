"""Implementation of different measurement metrics"""
import torch
import networkx as nx

from torchtyping import TensorType


def distortion_loss(
    estimated_distances: TensorType["n_points", "n_points"], true_distances: TensorType["n_points", "n_points"]
) -> float:
    """
    Compute the distortion loss between estimated SQUARED distances and true SQUARED distances.
    Args:
        estimated_distances (n_points, n_points): A tensor of estimated pairwise distances.
        true_distances (n_points, n_points).: A tensor of true pairwise distances. 
                            
    Returns:
        float: A float indicating the distortion loss, calculated as the sum of the squared relative 
               errors between the estimated and true squared distances.   
    """
    n = true_distances.shape[0]
    idx = torch.triu_indices(n, n, offset=1)

    pdist_true = true_distances[idx[0], idx[1]]
    pdist_est = estimated_distances[idx[0], idx[1]]

    return torch.sum(torch.abs((pdist_est / pdist_true) ** 2 - 1))


def d_avg(
    estimated_distances: TensorType["n_points", "n_points"], true_distances: TensorType["n_points", "n_points"]
) -> float:
    """Average distance error D_av
    Args:
        estimated_distances (n_points, n_points): A tensor of estimated pairwise distances.
        true_distances (n_points, n_points).: A tensor of true pairwise distances. 
                            
    Returns:
        float: A float indicating the average distance error D_avg, calculated as the 
        mean relative error across all pairwise distances.
    """
    n = true_distances.shape[0]
    idx = torch.triu_indices(n, n, offset=1)

    pdist_true = true_distances[idx[0], idx[1]]
    pdist_est = estimated_distances[idx[0], idx[1]]

    # Note that D_avg uses nonsquared distances:
    return torch.mean(torch.abs(pdist_est - pdist_true) / pdist_true)


def mean_average_precision(x_embed: TensorType["n_points", "n_dim"], graph: nx.Graph) -> float:
    """Mean averae precision (mAP) from the Gu et al paper."""
    raise NotImplementedError


def dist_component_by_manifold(pm, x_embed):
    """
    Compute the variance in pairwise distances explained by each manifold component.

    Args:
        pm: The product manifold. 
        x_embed (n_points, n_dim): A tensor of embeddings. 

    Returns:
        List[float]: A list of proportions, where each value represents the fraction 
                     of total distance variance explained by the corresponding 
                     manifold component.
    """
    sq_dists_by_manifold = [M.pdist2(x_embed[:, pm.man2dim[i]]) for i, M in enumerate(pm.P)]
    total_sq_dist = pm.pdist2(x_embed)

    return [
        torch.sum(D.triu(diagonal=1) / torch.sum(total_sq_dist.triu(diagonal=1))).item() for D in sq_dists_by_manifold
    ]

