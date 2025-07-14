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

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from typing import Any

    from jaxtyping import Float, Int


def _discrete_curvature_estimator(D: Float[torch.Tensor, "n_points n_points"], a: int, b: int, c: int, m: int) -> float:
    r"""Computes discrete curvature estimate for triangle using comparison theorem.

    Based on the triangle comparison theorem from Toponogov's theorem, this function computes:

    $$\xi_G(a,b,c;m) = \frac{1}{2d_G(a,m)} \left( d_G(a,m)^2 + \frac{d_G(b,c)^2}{4} - \frac{d_G(a,b)^2 + d_G(a,c)^2}{2} \right)$$

    where $m$ is a reference node for the triangle formed by nodes $b$ and $c$. This quantity is:
    - Positive for positively curved (spherical-like) regions
    - Zero for flat (Euclidean) regions
    - Negative for negatively curved (hyperbolic-like) regions

    Args:
        D: Pairwise distance matrix.
        a: Reference point index.
        b: First triangle vertex index.
        c: Second triangle vertex index.
        m: Reference node index.

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
    A: Float[torch.Tensor, "n_points n_points"],
    D: Float[torch.Tensor, "n_points n_points"],
    relative: bool = True,
    full: bool = False,
) -> Float[torch.Tensor, "n_points"] | float:
    r"""Computes sectional curvature estimates for all nodes in a graph.

    For each node $m$, computes the sectional curvature by averaging over all
    valid triangles involving pairs of neighbors. The curvature at node $m$ is:

    $$\kappa_m = \frac{1}{|\mathcal{T}_m|} \sum_{(b,c) \in \mathcal{T}_m} \frac{1}{|V|-1} \sum_{a \neq m} \xi_G(a,b,c;m)$$

    where $\mathcal{T}_m$ is the set of neighbor pairs of node $m$.

    Args:
        A: Adjacency matrix indicating graph connections.
        D: Pairwise shortest path distance matrix.
        relative: Whether to normalize by the maximum distance.
        full: Whether to return per-node curvatures or the global average.

    Returns:
        curvatures: Either per-node curvature estimates (if full=True) or global average (if full=False).
    """
    n = A.shape[0]
    node_curvatures = torch.zeros(n, dtype=torch.float32)

    # For each node, compute curvature based on neighbor triangles
    for m in range(n):
        # Find neighbors using adjacency matrix
        neighbors = torch.where(A[m] == 1)[0]

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
    adjacency_matrix: Float[torch.Tensor, "n_points n_points"],
    distance_matrix: Float[torch.Tensor, "n_points n_points"],
    method: str = "sampled",
    **kwargs: Any,
) -> Float[torch.Tensor, "n_points"] | float:
    r"""Estimates the sectional curvature of a graph from adjacency and distance matrices.

    This function implements the graph sectional curvature estimation described in Gu et al. 2019.
    Uses a discrete triangle comparison theorem to estimate local curvature at each node.

    The discrete curvature estimator compares triangle geometry to theoretical predictions:
    - Positive values indicate spherical-like (positively curved) regions
    - Zero values indicate Euclidean (flat) regions
    - Negative values indicate hyperbolic-like (negatively curved) regions

    Args:
        adjacency_matrix: Binary adjacency matrix indicating graph connections.
        distance_matrix: Pairwise shortest path distance matrix.
        method: Estimation method. Options:
            - "sampled": Random sampling approach, returns array of curvature samples
            - "per_node": Per-node curvature computation, returns curvature for each node
            - "global": Global average curvature, returns single scalar value
        **kwargs: Additional arguments passed to the estimation function.
            For "sampled": n_samples, relative
            For "per_node"/"global": relative

    Returns:
        curvature_estimates: Sectional curvature estimates.
            - "sampled": torch.Tensor of shape (n_samples,)
            - "per_node": torch.Tensor of shape (n_points,)
            - "global": float scalar
    """
    # Validate input matrices
    if not isinstance(adjacency_matrix, torch.Tensor) or not isinstance(distance_matrix, torch.Tensor):
        raise TypeError("Both adjacency_matrix and distance_matrix must be torch.Tensors")

    if adjacency_matrix.shape != distance_matrix.shape:
        raise ValueError("Adjacency matrix and distance matrix must have the same shape")

    A = adjacency_matrix.float()
    D = distance_matrix.float()

    if method == "sampled":
        curvatures, _ = sampled_sectional_curvature(D, **kwargs)
        return curvatures
    elif method == "per_node":
        return vectorized_sectional_curvature(A, D, full=True, **kwargs)
    elif method == "global":
        return vectorized_sectional_curvature(A, D, full=False, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'sampled', 'per_node', 'global'")
