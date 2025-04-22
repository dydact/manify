"""Compute delta-hyperbolicity of a metric space."""

from __future__ import annotations

import torch
from jaxtyping import Float
from typing import Tuple


def sampled_delta_hyperbolicity(
    D: Float[torch.Tensor, "n_points n_points"], n_samples: int = 1000, reference_idx: int = 0
) -> Tuple[Float[torch.Tensor, "n_samples,"], Float[torch.Tensor, "n_samples 3"]]:
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
    diam = torch.max(D)
    rel_deltas = 2 * deltas / diam

    return rel_deltas, indices


def iterative_delta_hyperbolicity(
    D: Float[torch.Tensor, "n_points n_points"], reference_idx: int = 0
) -> Float[torch.Tensor, "n_points n_points n_points"]:
    """delta(x,y,z) = min((x,y)_w,(y-z)_w) - (x,z)_w"""
    n = D.shape[0]
    w = reference_idx
    gromov_products = torch.zeros((n, n))
    deltas = torch.zeros((n, n, n))

    # Get Gromov Products
    for x in range(n):
        for y in range(n):
            gromov_products[x, y] = gromov_product(w, x, y, D)

    # Get Deltas
    for x in range(n):
        for y in range(n):
            for z in range(n):
                xz_w = gromov_products[x, z]
                xy_w = gromov_products[x, y]
                yz_w = gromov_products[y, z]
                deltas[x, y, z] = torch.minimum(xy_w, yz_w) - xz_w

    diam = torch.max(D)
    rel_deltas = 2 * deltas / diam

    return rel_deltas, gromov_products


def gromov_product(i: int, j: int, k: int, D: Float[torch.Tensor, "n_points n_points"]) -> float:
    """(j,k)_i = 0.5 (d(i,j) + d(i,k) - d(j,k))"""
    return float(0.5 * (D[i, j] + D[i, k] - D[j, k]))


def delta_hyperbolicity(
    D: Float[torch.Tensor, "n_points n_points"], reference_idx: int = 0, relative: bool = True, full: bool = False
) -> Float[torch.Tensor, ""]:
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

    if relative:
        diam = torch.max(D).item()
        delta = 2 * delta / diam

    return delta
