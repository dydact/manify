r"""Sectional curvature estimation for graphs.

This module implements the graph sectional curvature estimation from:
Gu et al. "Learning mixed-curvature representations in product spaces." ICLR 2019.

Estimates local curvature at nodes using a discrete triangle comparison theorem.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from jaxtyping import Float


def sectional_curvature(
    adjacency_matrix: Float[torch.Tensor, "n_points n_points"],
    distance_matrix: Float[torch.Tensor, "n_points n_points"],
    samples: int | None = None,
    relative: bool = True,
) -> Float[torch.Tensor, "n_points"] | Float[torch.Tensor, "samples"]:
    r"""Estimates sectional curvature of a graph.

    Uses discrete triangle comparison theorem to estimate local curvature.
    Positive values indicate spherical-like regions, negative values indicate
    hyperbolic-like regions, zero indicates flat regions.

    Args:
        adjacency_matrix: Binary adjacency matrix indicating graph connections.
        distance_matrix: Pairwise shortest path distance matrix.
        samples: Number of triangle configurations to sample. If None, computes per-node curvatures.
        relative: Whether to normalize by maximum distance.

    Returns:
        Sectional curvature estimates:
        - When samples is not None: torch.Tensor of shape (samples,)
        - When samples is None: torch.Tensor of shape (n_points,) with per-node curvatures

    Note:
        For global statistics, call .mean() or other aggregation functions on the result.
    """
    if not isinstance(adjacency_matrix, torch.Tensor) or not isinstance(distance_matrix, torch.Tensor):
        raise TypeError("Both adjacency_matrix and distance_matrix must be torch.Tensors")

    if adjacency_matrix.shape != distance_matrix.shape:
        raise ValueError("Adjacency matrix and distance matrix must have the same shape")

    A = adjacency_matrix.float()
    D = distance_matrix.float()

    if samples is not None:
        return _sample_curvatures(D, samples, relative)
    else:
        return _compute_node_curvatures(A, D, relative)


def _sample_curvatures(
    D: Float[torch.Tensor, "n_points n_points"], 
    n_samples: int, 
    relative: bool
) -> Float[torch.Tensor, "n_samples"]:
    """Sample random triangle configurations and compute curvature estimates."""
    n = D.shape[0]
    
    # Sample random configurations: reference point a, triangle vertices b,c, midpoint m
    indices = torch.randint(0, n, (n_samples, 4))
    a, b, c, m = indices.T

    # Filter out degenerate cases where a == m
    valid_mask = a != m
    curvatures = torch.zeros(n_samples, dtype=torch.float32)

    if valid_mask.any():
        valid_a, valid_b, valid_c, valid_m = a[valid_mask], b[valid_mask], c[valid_mask], m[valid_mask]

        # Compute discrete curvature estimator: ξ(a,b,c;m) = (d²(a,m) + d²(b,c)/4 - (d²(a,b) + d²(a,c))/2) / (2*d(a,m))
        curvature_values = (
            D[valid_a, valid_m] ** 2
            + (D[valid_b, valid_c] ** 2) / 4.0
            - (D[valid_a, valid_b] ** 2 + D[valid_a, valid_c] ** 2) / 2.0
        ) / (2 * D[valid_a, valid_m])

        curvatures[valid_mask] = curvature_values

    if relative:
        curvatures = curvatures / torch.max(D)

    return curvatures


def _compute_node_curvatures(
    A: Float[torch.Tensor, "n_points n_points"],
    D: Float[torch.Tensor, "n_points n_points"],
    relative: bool,
) -> Float[torch.Tensor, "n_points"]:
    """Compute curvature for each node by averaging over neighbor triangles."""
    n = A.shape[0]
    node_curvatures = torch.zeros(n, dtype=torch.float32)

    for m in range(n):
        neighbors = torch.where(A[m] == 1)[0]
        
        if len(neighbors) < 2:
            continue  # Need at least 2 neighbors to form triangles

        triangle_curvatures = []
        
        # Consider all pairs of neighbors
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                b, c = neighbors[i], neighbors[j]
                
                # Average curvature over all reference points
                reference_curvatures = []
                for a in range(n):
                    if a != m:
                        # Compute ξ(a,b,c;m)
                        curvature = (
                            D[a, m] ** 2 + (D[b, c] ** 2) / 4.0 - (D[a, b] ** 2 + D[a, c] ** 2) / 2.0
                        ) / (2 * D[a, m])
                        reference_curvatures.append(curvature)

                if reference_curvatures:
                    triangle_curvatures.append(torch.tensor(reference_curvatures).mean())

        if triangle_curvatures:
            node_curvatures[m] = torch.stack(triangle_curvatures).mean()

    if relative:
        node_curvatures = node_curvatures / torch.max(D)

    return node_curvatures