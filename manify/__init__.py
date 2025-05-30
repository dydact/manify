"""Manify: A Python Library for Learning Non-Euclidean Representations."""

import os

if os.getenv("BEARTYPE_ENABLE", "false").lower() == "true":
    from jaxtyping import install_import_hook

    install_import_hook("manify", "beartype.beartype")

from manify.curvature_estimation import greedy_signature_selection, sampled_delta_hyperbolicity, sectional_curvature
from manify.embedders import CoordinateLearning, ProductSpaceVAE, SiameseNetwork
from manify.manifolds import Manifold, ProductManifold
from manify.predictors import KappaGCN, ProductSpaceDT, ProductSpacePerceptron, ProductSpaceRF, ProductSpaceSVM

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
