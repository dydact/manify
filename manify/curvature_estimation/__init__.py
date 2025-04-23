"""Methods for estimating curvature in metric spaces and graphs.

This module provides tools to estimate various curvature properties of metric spaces and graphs:

* `delta_hyperbolicity`: Computes the Gromov delta-hyperbolicity, which measures how close a metric space is to a tree.
* `greedy_method`: Implements the greedy signature selection method from Tabaghi et al.
* `sectional_curvature`: Estimates the sectional curvature of a graph from its distance matrix.
"""

import manify.curvature_estimation.delta_hyperbolicity
import manify.curvature_estimation.greedy_method
import manify.curvature_estimation.sectional_curvature
