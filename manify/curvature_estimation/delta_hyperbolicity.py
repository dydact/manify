r"""Methods for computing $\delta$-hyperbolicity of a metric space.

The $\delta$-hyperbolicity measures how close a metric space is to a tree. The value $\delta \geq 0$ is a global
property; smaller values indicate the space is more hyperbolic.

This module provides two implementations:

1. Full computation of $\delta$-hyperbolicity over all possible point triplets
2. Sampling-based approximation for large metric spaces
"""

from __future__ import annotations

import torch
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
        n_samples: Number of triplets to sample. Defaults to 1000.
        reference_idx: Index of the reference point w. Defaults to 0.
        relative: Whether to normalize by the maximum distance. Defaults to True.

    Returns:
        deltas: $\delta$-hyperbolicity of each sampled triplet.
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
) -> Float[torch.Tensor, "n_points n_points n_points"]:
    r"""Computes the exact delta-hyperbolicity of a metric space over all point triplets.

    For a metric space with distance matrix $\mathbf{D}$, computes the $\delta$-hyperbolicity by:

    $\delta = \max_{x,y,z} \min((x,y)_w, (y,z)_w) - (x,z)_w$

    where $(a,b)_w = \frac{1}{2}(d(w,a) + d(w,b) - d(a,b))$ is the Gromov product.

    This is equivalent to the 4-point definition but more efficient to compute.

    Args:
        D: Pairwise distance matrix.
        reference_idx: Index of the reference point $w$. Defaults to 0.
        relative: Whether to normalize by the maximum distance. Defaults to True.
        full: Whether to return the full $\delta$ tensor or just the maximum value. Defaults to False.

    Returns:
        delta: Either the maximum $\delta$ value (if full=False) or the full $\delta$ tensor.
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

    if not full:
        delta = (out - XY_w_xz).max().item()
    else:
        delta = out - XY_w_xz

    delta = 2 * delta / torch.max(D) if relative else delta

    return delta
