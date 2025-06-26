"""Preprocessing utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from jaxtyping import Float


def knn_graph(x: Float[torch.Tensor, "n_points n_dim"], k: int) -> Float[torch.Tensor, "n_points n_points"]:
    """Compute the k-nearest neighbor graph from ambient coordinates.

    Args:
        x: Points in ambient space.
        k: Number of nearest neighbors.

    Returns:
        adjacency_matrix: k-nearest neighbor adjacency matrix.

    Note:
        This function is not yet implemented.
    """
    raise NotImplementedError
