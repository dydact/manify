"""Greedy method for estimating mixed-curvature product manifold signatures.

This module implements the greedy signature selection approach described in Tabaghi, Pan, Chien, Peng & Milenkovic.
"Linear Classifiers in Product Space Forms" (2021).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Any

from ..manifolds import ProductManifold
from ._pipelines import distortion_pipeline


def greedy_signature_selection(
    candidate_components: Iterable[tuple[float, int]] = ((-1.0, 2), (0.0, 2), (1.0, 2)),
    max_components: int = 3,
    pipeline: Callable[..., float] = distortion_pipeline,
    **kwargs: dict[str, Any],
) -> tuple[ProductManifold, list[float]]:
    r"""Greedily estimates an optimal product manifold signature.

    This implements the greedy signature selection algorithm that incrementally builds a product manifold
    by selecting components that best preserve distances. At each step, it chooses the manifold component
    that maximizes distortion reduction.

    Args:
        candidate_components: Candidate (curvature, dimension) pairs to consider.
        max_components: Maximum number of components to include.
        pipeline: Function that takes a ProductManifold, plus additional arguments, and returns a loss value.
        **kwargs: Additional keyword arguments to pass to the pipeline function.

    Returns:
        optimal_pm: Optimized product manifold with the selected signature.
    """
    # Initialize variables
    signature: list[tuple[float, int]] = []
    loss_history: list[float] = []
    current_loss = float("inf")
    candidate_components_list = list(candidate_components)  # For type safe iteration

    # Greedy loop
    for _ in range(max_components):
        best_loss, best_idx = current_loss, -1

        # Try each candidate
        for idx, comp in enumerate(candidate_components_list):
            pm = ProductManifold(signature=signature + [comp])
            loss = pipeline(pm, **kwargs)
            if loss < best_loss:
                best_loss, best_idx = loss, idx

        # If no improvement, stop
        if best_idx < 0:
            break

        # Otherwise accept that component
        signature.append(candidate_components_list[best_idx])
        current_loss = best_loss
        loss_history.append(current_loss)

    # Return final manifold
    optimal_pm = ProductManifold(signature=signature)
    return optimal_pm, loss_history
