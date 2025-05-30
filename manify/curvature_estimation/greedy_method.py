"""Greedy method for estimating mixed-curvature product manifold signatures.

This module implements the greedy signature selection approach described in Tabaghi, Pan, Chien, Peng & Milenkovic.
"Linear Classifiers in Product Space Forms" (2021).
"""

from __future__ import annotations

import torch
from jaxtyping import Float

from ..manifolds import ProductManifold


def greedy_signature_selection(
    pm: ProductManifold,
    dists: Float[torch.Tensor, "n_points n_points"],
    candidate_components: tuple[tuple[float, int], ...] = ((-1.0, 2), (0.0, 2), (1.0, 2)),
    max_components: int = 3,
) -> None:
    r"""Greedily estimates an optimal product manifold signature.

    This implements the greedy signature selection algorithm that incrementally builds a product manifold
    by selecting components that best preserve distances. At each step, it chooses the manifold component
    that maximizes distortion reduction.

    Args:
        pm: Initial product manifold to use as starting point.
        dists: Pairwise distance matrix to approximate.
        candidate_components: Candidate (curvature, dimension) pairs to consider.
            Defaults to ((-1, 2), (0.0, 2), (1, 2)).
        max_components: Maximum number of components to include. Defaults to 3.

    Returns:
        optimal_pm: Optimized product manifold with the selected signature.

    Note:
        This function is not yet implemented.
    """
    raise NotImplementedError
