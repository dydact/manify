r"""Methods for estimating sectional curvature of graphs.

This module provides functions to estimate the sectional curvature of graphs based on the approach from:
Gu et al. "Learning mixed-curvature representations in product spaces." ICLR 2019.

Sectional curvature estimation uses a discrete triangle comparison theorem to measure local curvature
at nodes in a graph. The approach compares triangle geometry to determine whether regions are
positively curved (spherical-like), negatively curved (hyperbolic-like), or flat (Euclidean-like).

This module provides two implementations:

1. Sampling-based approximation for efficient curvature estimation
2. Full computation over all valid triangles for precise analysis
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx
import torch

if TYPE_CHECKING:
    from jaxtyping import Float, Int


def _discrete_curvature_estimator(D: Float[torch.Tensor, "n_points n_points"], a: int, b: int, c: int, m: int) -> float:
    r"""Computes discrete curvature estimate for triangle using comparison theorem.

    Based on the triangle comparison theorem from Toponogov's theorem, this function computes:

    $$\xi_G(a,b,c) = d_G(a,m)^2 + \frac{d_G(b,c)^2}{4} - \frac{d_G(a,b)^2 + d_G(a,c)^2}{2}$$

    where $m$ is the midpoint of edge $bc$. This quantity is:
    - Positive for positively curved (spherical-like) regions
    - Zero for flat (Euclidean) regions
    - Negative for negatively curved (hyperbolic-like) regions

    Args:
        D: Pairwise distance matrix.
        a: Reference point index.
        b: First triangle vertex index.
        c: Second triangle vertex index.
        m: Midpoint reference index.

    Returns:
        Discrete curvature estimate for the triangle.
    """
    if a == m:
        return 0.0

    # Normalize by distance to reference point for scale invariance
    xi = D[a, m] ** 2 + (D[b, c] ** 2) / 4.0 - (D[a, b] ** 2 + D[a, c] ** 2) / 2.0
    xi = xi / (2 * D[a, m])

    return float(xi)


def sampled_sectional_curvature(
    D: Float[torch.Tensor, "n_points n_points"], n_samples: int = 1000, relative: bool = True
) -> tuple[Float[torch.Tensor, "n_samples"], Int[torch.Tensor, "n_samples 4"]]:
    r"""Estimates sectional curvature by sampling random triangles and reference points.

    For large graphs, this approximates sectional curvature by randomly sampling triangles and reference
    points. For each sampled configuration, computes the discrete curvature estimator:

    $$\xi_G(a,b,c;m) = \frac{1}{2d_G(a,m)} \left( d_G(a,m)^2 + \frac{d_G(b,c)^2}{4} - \frac{d_G(a,b)^2 + d_G(a,c)^2}{2} \right)$$

    Args:
        D: Pairwise distance matrix.
        n_samples: Number of triangle configurations to sample.
        relative: Whether to normalize by the maximum distance.

    Returns:
        curvatures: Sectional curvature estimates for each sampled configuration.
        indices: Indices of sampled points [a, b, c, m] for each configuration.
    """
    n = D.shape[0]

    # Sample random configurations (reference point a, triangle vertices b,c, midpoint m)
    indices = torch.randint(0, n, (n_samples, 4))

    a, b, c, m = indices.T

    # Filter out degenerate cases where a == m
    valid_mask = a != m

    # Compute discrete curvature estimator
    xi = torch.zeros(n_samples, dtype=torch.float32)

    if valid_mask.any():
        valid_a, valid_b, valid_c, valid_m = a[valid_mask], b[valid_mask], c[valid_mask], m[valid_mask]

        # Compute the triangle comparison quantity
        xi_valid = (
            D[valid_a, valid_m] ** 2
            + (D[valid_b, valid_c] ** 2) / 4.0
            - (D[valid_a, valid_b] ** 2 + D[valid_a, valid_c] ** 2) / 2.0
        )

        # Normalize by distance to reference point
        xi_valid = xi_valid / (2 * D[valid_a, valid_m])

        xi[valid_mask] = xi_valid

    # Apply relative normalization if requested
    if relative:
        xi = xi / torch.max(D)

    return xi, indices


def vectorized_sectional_curvature(
    D: Float[torch.Tensor, "n_points n_points"], relative: bool = True, full: bool = False, k_neighbors: int = 5
) -> Float[torch.Tensor, "n_points"] | float:
    r"""Computes sectional curvature estimates for all nodes in a graph or manifold.

    For each node $m$, computes the sectional curvature by averaging over all
    valid triangles involving pairs of neighbors. The curvature at node $m$ is:

    $$\kappa_m = \frac{1}{|\mathcal{T}_m|} \sum_{(b,c) \in \mathcal{T}_m} \frac{1}{|V|-1} \sum_{a \neq m} \xi_G(a,b,c;m)$$

    where $\mathcal{T}_m$ is the set of neighbor pairs of node $m$.

    Args:
        D: Pairwise distance matrix.
        relative: Whether to normalize by the maximum distance.
        full: Whether to return per-node curvatures or the global average.
        k_neighbors: Number of nearest neighbors to use for continuous distance matrices.
                    If distances contain exact 1.0 values (graph case), uses those instead.

    Returns:
        curvatures: Either per-node curvature estimates (if full=True) or global average (if full=False).
    """
    n = D.shape[0]
    node_curvatures = torch.zeros(n, dtype=torch.float32)

    # Check if this is a graph distance matrix (contains exact 1.0 distances)
    has_unit_distances = torch.any(D == 1.0)

    # For each node, compute curvature based on neighbor triangles
    for m in range(n):
        if has_unit_distances:
            # Graph case: find neighbors at distance 1
            neighbors = torch.where(D[m] == 1)[0]
        else:
            # Continuous distance case: use k-nearest neighbors (excluding self)
            distances_from_m = D[m].clone()
            distances_from_m[m] = float('inf')  # Exclude self
            _, neighbor_indices = torch.topk(distances_from_m, k=min(k_neighbors, n-1), largest=False)
            neighbors = neighbor_indices

        if len(neighbors) < 2:
            continue  # Need at least 2 neighbors to form triangles

        curvature_estimates = []

        # Consider all pairs of neighbors
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                b, c = neighbors[i], neighbors[j]

                # Average over all reference points
                triangle_curvatures = []
                for a in range(n):
                    if a != m:
                        xi = _discrete_curvature_estimator(D, a, b.item(), c.item(), m)
                        triangle_curvatures.append(xi)

                if triangle_curvatures:
                    curvature_estimates.append(torch.tensor(triangle_curvatures).mean())

        if curvature_estimates:
            node_curvatures[m] = torch.stack(curvature_estimates).mean()

    # Apply relative normalization if requested
    if relative:
        node_curvatures = node_curvatures / torch.max(D)

    return node_curvatures if full else node_curvatures.mean().item()


def sectional_curvature(
    input_data: nx.Graph | Float[torch.Tensor, "n_points n_points"],
    method: str = "sampled",
    **kwargs: Any
) -> Float[torch.Tensor, "n_points"] | float:
    r"""Estimates the sectional curvature of a graph or from a distance matrix.

    This function implements the graph sectional curvature estimation described in Gu et al. 2019.
    Uses a discrete triangle comparison theorem to estimate local curvature at each node.

    The discrete curvature estimator compares triangle geometry to theoretical predictions:
    - Positive values indicate spherical-like (positively curved) regions
    - Zero values indicate Euclidean (flat) regions
    - Negative values indicate hyperbolic-like (negatively curved) regions

    Args:
        input_data: Either a NetworkX graph or a pairwise distance matrix as a torch.Tensor.
        method: Estimation method. Options:
            - "sampled": Random sampling approach, returns array of curvature samples
            - "per_node": Per-node curvature computation, returns curvature for each node
            - "global": Global average curvature, returns single scalar value
            - "full": DEPRECATED - use "per_node" instead
        **kwargs: Additional arguments passed to the estimation function.
            For "sampled": n_samples, relative
            For "per_node"/"global": relative

    Returns:
        curvature_estimates: Sectional curvature estimates.
            - "sampled": torch.Tensor of shape (n_samples,)
            - "per_node": torch.Tensor of shape (n_points,)
            - "global": float scalar
    """
    # Handle different input types
    if isinstance(input_data, nx.Graph):
        # Compute shortest path distance matrix from graph
        D = torch.tensor(nx.floyd_warshall_numpy(input_data), dtype=torch.float32)
    elif isinstance(input_data, torch.Tensor):
        # Use distance matrix directly
        D = input_data.float()
    else:
        raise TypeError(f"input_data must be a NetworkX graph or torch.Tensor, got {type(input_data)}")

    if method == "sampled":
        curvatures, _ = sampled_sectional_curvature(D, **kwargs)
        return curvatures
    elif method == "per_node":
        return vectorized_sectional_curvature(D, full=True, **kwargs)
    elif method == "global":
        return vectorized_sectional_curvature(D, full=False, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'sampled', 'per_node', 'global'")
