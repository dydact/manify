"""Compute delta-hyperbolicity of a metric space."""

from __future__ import annotations

from typing import Tuple

import torch
from jaxtyping import Float


def sampled_delta_hyperbolicity(
    D: Float[torch.Tensor, "n_points n_points"], n_samples: int = 1000, reference_idx: int = 0, relative: bool = True
) -> Tuple[Float[torch.Tensor, "n_samples,"], Float[torch.Tensor, "n_samples 3"]]:
    """Sampled delta-hyperbolicity computation with optional relative scaling.

    Args:
        D: Distance matrix of the metric space.
        n_samples: Number of samples to draw.
        reference_idx: Index of the reference point.
        relative: Whether to return the relative delta-hyperbolicity.

    Returns:
        rel_deltas: Relative delta-hyperbolicity values.
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


def delta_hyperbolicity(
    D: Float[torch.Tensor, "n_points n_points"], reference_idx: int = 0, relative: bool = True, full: bool = False
) -> Float[torch.Tensor, "n_points n_points n_points"]:
    """
    Compute the delta-hyperbolicity of a metric space.

    Args:
        D: Distance matrix of the metric space.
        relative: Whether to return the relative delta-hyperbolicity.
        full: Whether to return the full delta tensor or just the maximum delta.

    Returns:
        delta: Delta-hyperbolicity of the metric space.
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
