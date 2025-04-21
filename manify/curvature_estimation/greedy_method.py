"""Greedy selection of signatures, as described in Tabaghi et al. at https://arxiv.org/pdf/2102.10204"""

from __future__ import annotations

from typing import Tuple

import torch

from ..manifolds import ProductManifold


def greedy_curvature_method(
    pm: ProductManifold,
    dists: torch.Tensor,
    candidate_components: Tuple[Tuple[float, int], ...] = (
        (-1.0, 2),
        (0.0, 2),
        (1.0, 2),
    ),
    max_components: int = 3,
):
    """The greedy curvature estimation method from Tabaghi et al. at https://arxiv.org/pdf/2102.10204"""
    raise NotImplementedError
