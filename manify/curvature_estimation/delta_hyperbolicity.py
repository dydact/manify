r"""δ-hyperbolicity computation for metric spaces.

The δ-hyperbolicity measures how close a metric space is to a tree.
Smaller values indicate the space is more hyperbolic (tree-like).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from jaxtyping import Float


def delta_hyperbolicity(
    distance_matrix: Float[torch.Tensor, "n_points n_points"], 
    samples: int | None = None,
    reference_idx: int = 0,
    relative: bool = True,
) -> Float[torch.Tensor, "n_points n_points n_points"] | Float[torch.Tensor, "samples"]:
    r"""Computes δ-hyperbolicity from a distance matrix.

    For each triplet of points (x,y,z) and reference point w, computes:
    δ(x,y,z) = min((x,y)_w, (y,z)_w) - (x,z)_w

    where (a,b)_w = ½(d(w,a) + d(w,b) - d(a,b)) is the Gromov product.

    Args:
        distance_matrix: Pairwise distance matrix.
        samples: Number of triplets to sample. If None, computes full δ tensor over all triplets.
        reference_idx: Index of the reference point w.
        relative: Whether to normalize by maximum distance.

    Returns:
        δ-hyperbolicity estimates:
        - When samples is not None: torch.Tensor of shape (samples,)
        - When samples is None: torch.Tensor of shape (n_points, n_points, n_points)

    Note:
        For global statistics, call .max() or other aggregation functions on the result.
    """
    if not isinstance(distance_matrix, torch.Tensor):
        raise TypeError(f"distance_matrix must be a torch.Tensor, got {type(distance_matrix)}")

    D = distance_matrix.float()

    if samples is not None:
        return _sample_delta_values(D, samples, reference_idx, relative)
    else:
        return _compute_full_delta_tensor(D, reference_idx, relative)


def _sample_delta_values(
    D: Float[torch.Tensor, "n_points n_points"], 
    n_samples: int, 
    reference_idx: int, 
    relative: bool
) -> Float[torch.Tensor, "n_samples"]:
    """Sample random triplets and compute δ values."""
    n = D.shape[0]
    
    # Sample random triplets
    indices = torch.randint(0, n, (n_samples, 3))
    x, y, z = indices.T
    w = reference_idx

    # Compute Gromov products: (a,b)_w = ½(d(w,a) + d(w,b) - d(a,b))
    xy_w = 0.5 * (D[w, x] + D[w, y] - D[x, y])
    xz_w = 0.5 * (D[w, x] + D[w, z] - D[x, z])
    yz_w = 0.5 * (D[w, y] + D[w, z] - D[y, z])

    # δ(x,y,z) = min((x,y)_w, (y,z)_w) - (x,z)_w
    deltas = torch.minimum(xy_w, yz_w) - xz_w

    if relative:
        deltas = 2 * deltas / torch.max(D)

    return deltas


def _compute_full_delta_tensor(
    D: Float[torch.Tensor, "n_points n_points"], 
    reference_idx: int, 
    relative: bool
) -> Float[torch.Tensor, "n_points n_points n_points"]:
    """Compute full δ tensor over all triplets."""
    n = D.shape[0]
    w = reference_idx

    # Vectorized computation of all Gromov products
    row = D[w, :].unsqueeze(0)  # (1, n)
    col = D[:, w].unsqueeze(1)  # (n, 1)
    gromov_products = 0.5 * (row + col - D)  # (n, n)

    # Expand to (n, n, n) for all triplet combinations
    xy_w = gromov_products.unsqueeze(2).expand(-1, -1, n)  # (x,y)_w for all z
    xz_w = gromov_products.unsqueeze(1).expand(-1, n, -1)  # (x,z)_w for all y
    yz_w = gromov_products.unsqueeze(0).expand(n, -1, -1)  # (y,z)_w for all x

    # δ(x,y,z) = min((x,y)_w, (y,z)_w) - (x,z)_w
    deltas = torch.minimum(xy_w, yz_w) - xz_w

    if relative:
        max_dist = torch.max(D)
        if max_dist > 0:
            deltas = 2 * deltas / max_dist

    return deltas