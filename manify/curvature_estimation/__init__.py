"""Methods for estimating curvature in metric spaces and graphs.

This module provides tools to estimate various curvature properties of metric spaces and graphs:

* `delta_hyperbolicity`: Computes the Gromov delta-hyperbolicity, which measures how close a metric space is to a tree.
* `greedy_method`: Implements the greedy signature selection method from Tabaghi et al.
* `sectional_curvature`: Estimates the sectional curvature of a graph from its distance matrix.
"""

from manify.curvature_estimation.delta_hyperbolicity import delta_hyperbolicity
from manify.curvature_estimation.greedy_method import greedy_signature_selection
from manify.curvature_estimation.sectional_curvature import sectional_curvature

__all__ = [
    "greedy_signature_selection",
    "sectional_curvature",
    "delta_hyperbolicity",
]
