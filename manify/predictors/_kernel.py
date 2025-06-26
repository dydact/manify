"""Kernel matrix calculation for product manifolds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from jaxtyping import Float

from ..manifolds import Manifold, ProductManifold


def compute_kernel_and_norm_manifold(
    manifold: Manifold,
    X_source: Float[torch.Tensor, "n_points_source n_dim"],
    X_target: Float[torch.Tensor, "n_points_target n_dim"] | None,
) -> tuple[Float[torch.Tensor, "n_points_source n_points_target"], Float[torch.Tensor, ""]]:
    """Computes the kernel matrix for a single manifold.

    Args:
        manifold: The manifold for the computation.
        X_source: Tensor of source points.
        X_target: Tensor of target points.

    Returns:
        kernel_matrix: The kernel matrix between source and target points.
        norm_constant: Scalar normalization constant for the kernel.
    """
    if X_target is None:
        X_target = X_source

    ip = manifold.inner(X_source, X_target)
    ip *= manifold.scale
    if manifold.type == "E":
        # K_E is just inner products
        K = ip
        norm = torch.tensor(1.0)
    elif manifold.type == "S":
        # K_S is asin(C_S * inner products)
        # C_S is the curvature (see p.5 of Tabaghi paper)
        C_S = manifold.curvature
        K = torch.asin(torch.clamp(ip * C_S, -1, 1)) * C_S**0.5
        norm = torch.tensor(C_S**0.5)
        # norm is sqrt(C_S) (see p.16 of Tabaghi paper)
    elif manifold.type == "H":
        # K_H is asinh(R^-2 * Lorentz inner products) * sqrt(-C_H)
        C_H = abs(manifold.curvature)
        R = -1 * manifold.scale
        # R = (X_source @ X_target.T).sqrt().max()
        # K = torch.asinh(torch.clamp(ip / R**2, -1, 1)) * C_H**0.5
        K = torch.asinh(ip / R**2) * C_H**0.5
        # norm = torch.tensor(C_H)
        # norm is sqrt(-C_H)
        # norm = torch.asinh(-(R**2) * C_H)
        norm = torch.tensor(C_H) ** 0.5
    else:
        raise ValueError("Invalid manifold type!")

    return K, norm


def product_kernel(
    pm: ProductManifold,
    X_source: Float[torch.Tensor, "n_points_source n_dim"],
    X_target: Float[torch.Tensor, "n_points_target n_dim"] | None,
) -> tuple[list[Float[torch.Tensor, "n_points_source n_points_target"]], list[Float[torch.Tensor, ""]]]:
    """Computes the kernel matrix for a product manifold.

    Args:
        pm: The product manifold for the computation.
        X_source: Tensor of source points.
        X_target: Tensor of target points.

    Returns:
        kernel_matrices: List of kernel matrices for each component manifold.
        norm_constants: List of normalization constants for each kernel.
    """
    # If X_target is None, set it to X_source
    if X_target is None:
        X_target = X_source

    # Compute the kernel matrix and norm for each manifold
    Ks = []
    norms = []
    for M, x_source, x_target in zip(pm.P, pm.factorize(X_source), pm.factorize(X_target)):
        K_m, norm_m = compute_kernel_and_norm_manifold(M, x_source, x_target)
        Ks.append(K_m)
        norms.append(norm_m)

    return Ks, norms
