r"""Methods for computing $\delta$-hyperbolicity of a metric space.

The $\delta$-hyperbolicity measures how close a metric space is to a tree. The value $\delta \geq 0$ is a global
property; smaller values indicate the space is more hyperbolic.

This module provides two implementations:

1. Full computation of $\delta$-hyperbolicity over all possible point triplets
2. Sampling-based approximation for large metric spaces
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from typing import Any

    from jaxtyping import Float, Int


def sampled_delta_hyperbolicity(
    D: Float[torch.Tensor, "n_points n_points"], n_samples: int = 1000, reference_idx: int = 0, relative: bool = True
) -> tuple[Float[torch.Tensor, "n_samples"], Int[torch.Tensor, "n_samples 3"]]:
    r"""Computes $\delta$-hyperbolicity by sampling random point triplets.

    For large metric spaces, this approximates $\delta$-hyperbolicity by randomly sampling triplets. For each triplet
    $(x,y,z)$ and reference point $w$, this function computes:

    $$\delta(x,y,z) = \min((x,y)_w, (y,z)_w) - (x,z)_w$$

    where $(a,b)_w = \frac{1}{2}(d(w,a) + d(w,b) - d(a,b))$ is the Gromov product.

    Args:
        D: Pairwise distance matrix.
        n_samples: Number of triplets to sample.
        reference_idx: Index of the reference point w.
        relative: Whether to normalize by the maximum distance.

    Returns:
        deltas: $\delta$-hyperbolicity values for each sampled triplet.
        indices: Indices of the sampled triplets.
    """
    n = D.shape[0]
    # Sample n_samples triplets of points randomly
    indices = torch.randint(0, n, (n_samples, 3))

    # Get gromov products
    # (j,k)_i = .5 (d(i,j) + d(i,k) - d(j,k))

    x, y, z = indices.T
    w = reference_idx  # set reference point

    xy_w = 0.5 * (D[w, x] + D[w, y] - D[x, y])
    xz_w = 0.5 * (D[w, x] + D[w, z] - D[x, z])
    yz_w = 0.5 * (D[w, y] + D[w, z] - D[y, z])

    # delta(x,y,z) = min((x,y)_w,(y-z)_w) - (x,z)_w
    deltas = torch.minimum(xy_w, yz_w) - xz_w

    deltas = 2 * deltas / torch.max(D) if relative else deltas

    return deltas, indices


def vectorized_delta_hyperbolicity(
    D: Float[torch.Tensor, "n_points n_points"], reference_idx: int = 0, relative: bool = True, full: bool = False
) -> Float[torch.Tensor, "n_points n_points n_points"] | float:
    r"""Computes the exact delta-hyperbolicity of a metric space over all point triplets.

    For a metric space with distance matrix $\mathbf{D}$, computes the $\delta$-hyperbolicity by:

    $$\delta = \max_{x,y,z} \min((x,y)_w, (y,z)_w) - (x,z)_w$$

    where $(a,b)_w = \frac{1}{2}(d(w,a) + d(w,b) - d(a,b))$ is the Gromov product.

    This is equivalent to the 4-point definition but more efficient to compute.

    Args:
        D: Pairwise distance matrix.
        reference_idx: Index of the reference point $w$.
        relative: Whether to normalize by the maximum distance.
        full: Whether to return the full $\delta$ tensor or just the maximum value.

    Returns:
        delta: Either the maximum $\delta$ value (if full=False) or the full $\delta$ tensor
            over all triplets.
    """
    n = D.shape[0]
    w = reference_idx

    row = D[w, :].unsqueeze(0)  # (1,N)
    col = D[:, w].unsqueeze(1)  # (N,1)
    XY_w = 0.5 * (row + col - D)

    XY_w_xy = XY_w.unsqueeze(2).expand(-1, -1, n)  # (n,n,n)
    XY_w_yz = XY_w.unsqueeze(0).expand(n, -1, -1)  # (n,n,n)
    XY_w_xz = XY_w.unsqueeze(1).expand(-1, n, -1)  # (n,n,n)

    out = torch.minimum(XY_w_xy, XY_w_yz)

    if full:
        delta = out - XY_w_xz
        if relative:
            max_dist = torch.max(D)
            delta = 2 * delta / max_dist if max_dist > 0 else torch.zeros_like(delta)

    else:
        delta = (out - XY_w_xz).max().item()
        if relative:
            max_dist = torch.max(D).item()
            delta = 2 * delta / max_dist if max_dist > 0 else 0.0

    return delta


def delta_hyperbolicity(
    distance_matrix: Float[torch.Tensor, "n_points n_points"], method: str = "global", **kwargs: Any
) -> Float[torch.Tensor, "n_points"] | float:
    r"""Computes the δ-hyperbolicity from a distance matrix.

    This function implements δ-hyperbolicity computation, which measures how close a metric
    space is to a tree. The value δ ≥ 0 is a global property; smaller values indicate
    the space is more hyperbolic (tree-like).

    For each triplet of points (x,y,z) and reference point w, computes:
    δ(x,y,z) = min((x,y)_w, (y,z)_w) - (x,z)_w

    where (a,b)_w = ½(d(w,a) + d(w,b) - d(a,b)) is the Gromov product.

    Args:
        distance_matrix: Pairwise distance matrix as a torch.Tensor.
        method: Computation method. Options:
            - "sampled": Random sampling approach, returns array of δ values for sampled triplets
            - "global": Global maximum δ value over all triplets, returns single scalar
            - "full": Full δ tensor over all triplets, returns tensor of shape (n,n,n)
        **kwargs: Additional arguments passed to the computation function.
            For "sampled": n_samples, reference_idx, relative
            For "global"/"full": reference_idx, relative

    Returns:
        delta_values: δ-hyperbolicity estimates.
            - "sampled": torch.Tensor of shape (n_samples,)
            - "global": float scalar (maximum δ value)
            - "full": torch.Tensor of shape (n_points, n_points, n_points)
    """
    # Validate input
    if not isinstance(distance_matrix, torch.Tensor):
        raise TypeError(f"distance_matrix must be a torch.Tensor, got {type(distance_matrix)}")

    D = distance_matrix.float()

    if method == "sampled":
        deltas, _ = sampled_delta_hyperbolicity(D, **kwargs)
        return deltas
    elif method == "global":
        return vectorized_delta_hyperbolicity(D, full=False, **kwargs)
    elif method == "full":
        return vectorized_delta_hyperbolicity(D, full=True, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'sampled', 'global', 'full'")
