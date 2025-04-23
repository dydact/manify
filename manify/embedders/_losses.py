"""Implementation of metrics and loss functions for evaluating embedding quality.

This module provides various functions to measure the quality of embeddings
in Riemannian manifolds, including distortion metrics, average distance error,
and other evaluation measures for both graph and general embedding tasks.
"""

from __future__ import annotations

from typing import List

import networkx as nx
import torch
from jaxtyping import Float

from ..manifolds import ProductManifold


def distortion_loss(
    D_est: Float[torch.Tensor, "n_points n_points"],
    D_true: Float[torch.Tensor, "n_points n_points"],
    pairwise: bool = False,
) -> Float[torch.Tensor, ""]:
    r"""Computes the distortion loss between estimated and true squared distances.

    The distortion loss measures how well the pairwise distances in the embedding space match the true distances. It is
    calculated as

    $$\sum_{i,j} \left(\left(\frac{D_{\text{est}}(i,j)}{D_{\text{true}}(i,j)}\right)^2 - 1\right).$$

    Args:
        D_est: Tensor of estimated pairwise squared distances.
        D_true: Tensor of true pairwise squared distances.
        pairwise: Whether to consider only unique pairs (upper triangular part of the matrices).

    Returns:
        loss: Scalar tensor representing the distortion loss.

    Note:
        This is similar to the `square_loss` in HazyResearch hyperbolics repository:
        https://github.com/HazyResearch/hyperbolics/blob/master/pytorch/hyperbolic_models.py#L178
    """

    # Turn into flat vectors of pairwise distances. For pairwise distances, we only consider the upper triangle.
    if pairwise:
        n = D_true.shape[0]
        idx = torch.triu_indices(n, n, offset=1)
        D_true = D_true[idx[0], idx[1]]
        D_est = D_est[idx[0], idx[1]]
    else:
        D_true = D_true.flatten()
        D_est = D_est.flatten()

    # Mask out any infinite or nan values
    mask = torch.isfinite(D_true) & ~torch.isnan(D_true)
    D_true = D_true[mask]
    D_est = D_est[mask]

    return torch.sum(torch.abs((D_est / D_true) ** 2 - 1))


def d_avg(
    D_est: Float[torch.Tensor, "n_points n_points"],
    D_true: Float[torch.Tensor, "n_points n_points"],
    pairwise: bool = False,
) -> Float[torch.Tensor, ""]:
    r"""Computes the average relative distance error (D_avg).

    The average distance error is the mean relative error between the estimated and true distances:

    $$
        D_{\text{avg}} = \frac{1}{N} \sum_{i,j} \frac{
            |D_{\text{est}}(i,j) - D_{\text{true}}(i,j)|
        }{
            D_{\text{true}}(i,j)
        },
    $$

    where $N$ is the number of distances being considered. This metric provides a normalized measure of how accurately
    the embedding preserves the original distances.

    Args:
        D_est: Tensor of estimated pairwise distances.
        D_true: Tensor of true pairwise distances.
        pairwise: Whether to consider only unique pairs (upper triangular part of the matrices).

    Returns:
        d_avg: Scalar tensor representing the average relative distance error.
    """

    if pairwise:
        n = D_true.shape[0]
        idx = torch.triu_indices(n, n, offset=1)
        D_true = D_true[idx[0], idx[1]]
        D_est = D_est[idx[0], idx[1]]
    else:
        D_true = D_true.flatten()
        D_est = D_est.flatten()

    # Mask out any infinite or nan values
    mask = torch.isfinite(D_true) & ~torch.isnan(D_true)
    D_true = D_true[mask]
    D_est = D_est[mask]

    # Note that D_avg uses nonsquared distances:
    return torch.mean(torch.abs(D_est - D_true) / D_true)


def mean_average_precision(x_embed: Float[torch.Tensor, "n_points n_dim"], graph: nx.Graph) -> Float[torch.Tensor, ""]:
    r"""Computes the mean average precision (mAP) for graph embedding evaluation.

    This metric is used to evaluate how well an embedding preserves the neighborhood structure of a graph, as described
    in Gu et al. (2019): "Learning Mixed-Curvature Representations in Product Spaces".

    Args:
        x_embed: Tensor containing the embeddings of the graph nodes.
        graph: NetworkX graph representing the original graph structure.

    Returns:
        mAP: Mean average precision score.

    Note:
        This function is currently not implemented.
    """
    raise NotImplementedError


def dist_component_by_manifold(pm: ProductManifold, x_embed: Float[torch.Tensor, "n_points n_dim"]) -> List[float]:
    r"""Computes the proportion of variance in pairwise distances explained by each manifold component.

    The contribution is calculated as the ratio of the sum of squared distances in each component to the total squared
    distance:

    $$\text{contribution}_k = \frac{\sum_{i<j} D^2_k(x_i, x_j)}{\sum_{i<j} D^2_{\text{total}}(x_i, x_j)}$$

    where $D^2_k$ is the squared distance in the $k$-th manifold component.

    Args:
        pm: The product manifold containing multiple component manifolds.
        x_embed: Tensor of embeddings in the product manifold.

    Returns:
        contributions: List of proportions, where each value represents the fraction of total distance variance
        explained by the corresponding manifold component.
    """
    sq_dists_by_manifold = [M.pdist2(x_embed[:, pm.man2dim[i]]) for i, M in enumerate(pm.P)]
    total_sq_dist = pm.pdist2(x_embed)

    return [
        torch.sum(D.triu(diagonal=1) / torch.sum(total_sq_dist.triu(diagonal=1))).item() for D in sq_dists_by_manifold
    ]
