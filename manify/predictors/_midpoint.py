"""Compute the angular midpoints between two angular coordinates in different geometric spaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from jaxtyping import Float

from ..manifolds import Manifold


def hyperbolic_midpoint(
    u: Float[torch.Tensor, ""], v: Float[torch.Tensor, ""], assert_hyperbolic: bool = False
) -> Float[torch.Tensor, ""]:
    """Compute the hyperbolic midpoint between two angular coordinates u and v.

    Args:
        u: The first angular coordinate.
        v: The second angular coordinate.
        assert_hyperbolic: If True, verifies that the midpoint satisfies the hyperbolic distance property.

    Returns:
        midpoint: The computed hyperbolic midpoint between u and v.
    """
    w = torch.sin(2.0 * u - 2.0 * v) / (torch.sin(u + v) * torch.sin(v - u))
    coef = -1.0 if u + v < torch.pi else 1.0
    sol = (-w + coef * torch.sqrt(w**2 - 4.0)) / 2.0
    m = torch.arctan2(torch.tensor(1.0), sol) % torch.pi
    if assert_hyperbolic:
        assert is_hyperbolic_midpoint(u, v, m)
    return m


def is_hyperbolic_midpoint(u: Float[torch.Tensor, ""], v: Float[torch.Tensor, ""], m: Float[torch.Tensor, ""]) -> bool:
    r"""Verify if $\mathbf{m}$ is the true hyperbolic midpoint between $\mathbf{u}$ and $\mathbf{v}$.

    Args:
        u: The first angular coordinate.
        v: The second angular coordinate.
        m: The candidate midpoint to verify.

    Returns:
        is_midpoint: True if m is the true hyperbolic midpoint between u and v, otherwise False.
    """

    def _a(x: Float[torch.Tensor, ""]) -> Float[torch.Tensor, ""]:
        """Compute the alpha coefficient to reach the hyperbolic manifold."""
        return torch.sqrt(-1.0 / torch.cos(x))

    def _d(x: Float[torch.Tensor, ""], y: Float[torch.Tensor, ""]) -> Float[torch.Tensor, ""]:
        """Compute the hyperbolic distance function (angular)."""
        return _a(x) * _a(y) * torch.cos(x - y)

    return torch.isclose(_d(u, m), _d(m, v))  # type: ignore


def spherical_midpoint(u: Float[torch.Tensor, ""], v: Float[torch.Tensor, ""]) -> Float[torch.Tensor, ""]:
    """Compute the spherical midpoint between two angular coordinates u and v.

    Args:
        u: The first angular coordinate.
        v: The second angular coordinate.

    Returns:
        midpoint: The computed spherical midpoint between u and v.
    """
    return (u + v) / 2.0


def euclidean_midpoint(u: Float[torch.Tensor, ""], v: Float[torch.Tensor, ""]) -> Float[torch.Tensor, ""]:
    """Compute the euclidean midpoint between two angular coordinates u and v.

    Args:
        u: The first angular coordinate.
        v: The second angular coordinate.

    Returns:
        midpoint: The computed euclidean midpoint between u and v.
    """
    return torch.arctan2(torch.tensor(2.0), (1.0 / torch.tan(u) + 1.0 / torch.tan(v)))


def midpoint(
    u: Float[torch.Tensor, ""], v: Float[torch.Tensor, ""], manifold: Manifold, special_first: bool = False
) -> Float[torch.Tensor, ""]:
    """Compute the midpoint between two angular coordinates given the manifold type.

    This function automatically selects the appropriate midpoint calculation depending
    on the manifold type. It supports hyperbolic, Euclidean, and spherical geometries.

    Args:
        u: The first angular coordinate.
        v: The second angular coordinate.
        manifold: An object representing the manifold type.
        special_first: If True, uses the manifold-specific midpoint calculations given the manifold type of hyperbolic
            or euclidean.

    Returns:
        midpoint: The computed midpoint between u and v, based on the selected geometry.
    """
    if torch.isclose(u, v):
        return u

    elif manifold.type == "H" and special_first:
        return hyperbolic_midpoint(u, v)

    elif manifold.type == "E" and special_first:
        return euclidean_midpoint(u, v)

    # Spherical midpoint handles all spherical angles
    # *AND* any angles that don't involve figuring out where you hit the manifold
    else:
        return spherical_midpoint(u, v)
