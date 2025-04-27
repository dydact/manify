"""Manify: A Python Library for Learning Non-Euclidean Representations."""

from manify.curvature_estimation import (
    sampled_delta_hyperbolicity,
    delta_hyperbolicity,
    sectional_curvature,
    greedy_signature_selection,
)
from manify.embedders import CoordinateLearning, ProductSpaceVAE, SiameseNetwork
from manify.manifolds import Manifold, ProductManifold
from manify.predictors import ProductSpaceDT, ProductSpaceRF, KappaGCN, ProductSpacePerceptron, ProductSpaceSVM

# import manify.utils

# Define version and other package metadata
__version__ = "0.0.2"
__author__ = "Philippe Chlenski"
__email__ = "pac@cs.columbia.edu"
__license__ = "MIT"

# Export modules
__all__ = [
    # manify.manifolds
    "Manifold",
    "ProductManifold",
    # manify.embedders
    "CoordinateLearning",
    "ProductSpaceVAE",
    "SiameseNetwork",
    # manify.predictors
    "ProductSpaceDT",
    "ProductSpaceRF",
    "KappaGCN",
    "ProductSpacePerceptron",
    "ProductSpaceSVM",
    # manify.curvature_estimation
    "delta_hyperbolicity",
    "sampled_delta_hyperbolicity",
    "sectional_curvature",
    "greedy_signature_selection",
    # no utils
]
