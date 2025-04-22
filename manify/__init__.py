"""Manify: A Python Library for Learning Non-Euclidean Representations."""

import manify.curvature_estimation
import manify.embedders
import manify.manifolds
import manify.predictors

# Define version
__version__ = "0.0.2"

# Dynamically check for utils dependencies
try:
    import importlib.util

    _utils_deps = ["networkx", "pandas", "matplotlib", "scipy", "anndata", "basemap"]
    _missing_deps = [dep for dep in _utils_deps if importlib.util.find_spec(dep) is None]

    if not _missing_deps:
        import manify.utils

        HAS_UTILS = True
    else:
        HAS_UTILS = False
except ImportError:
    HAS_UTILS = False

# Expose flag for users to check
__all__ = ["curvature_estimation", "embedders", "manifolds", "predictors"]
if HAS_UTILS:
    __all__.append("utils")
