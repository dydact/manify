"""Base embedder class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
from sklearn.base import BaseEstimator, TransformerMixin

from ..manifolds import ProductManifold


class BaseEmbedder(BaseEstimator, TransformerMixin, ABC):
    """Base class for everything in manify.embedders.

    This is an abstract class that that defines a common interface for all embedding methods. We assume only that a
    ProductManifold object is given. We try to follow the scikit-learn API's fit/transform paradigm as closely as
    possible, while accommodating the nuances of product manifold geometry and Pytorch/Geoopt.

    Attributes:
        pm: ProductManifold object associated with the embedder.
        random_state: Random state for reproducibility.
        device: Device for tensor computations. If not provided, defaults to pm.device.
        is_fitted_: Boolean flag indicating if the embedder has been fitted.
    """

    def __init__(self, pm: ProductManifold, random_state: Optional[int] = None, device: Optional[str] = None) -> None:
        self.pm = pm
        self.random_state = random_state
        self.device = pm.device if device is None else device
        self.is_fitted_ = False

    @abstractmethod
    def fit(
        self,
        X: Optional[Float[torch.Tensor, "n_points n_features"]] = None,
        D: Optional[Float[torch.Tensor, "n_points n_points"]] = None,
    ) -> "BaseEmbedder":
        """Abstract method to fit an embedder. Requires at least one of (features, distances).

        Args:
            X: Features to embed. Used by Mixed-curvature VAE and Siamese Network classes.
            D: Distacnes to embed. Used by coordinate learning and Siamese Network classes.

        Returns:
            self: Fitted embedder instance.
        """
        pass

    @abstractmethod
    def transform(
        self, X: Optional[Float[torch.Tensor, "n_points n_features"]]
    ) -> Float[torch.Tensor, "n_points embedding_dim"]:
        """Apply embedding to new data. Not defined for coordinate learning.

        Args:
            X: New features to embed using the trained embedder.

        Returns:
            X_embedded: Embedded representation of the input features.
        """
        pass

    def fit_transform(
        self,
        X: Optional[Float[torch.Tensor, "n_points n_features"]] = None,
        D: Optional[Float[torch.Tensor, "n_points n_points"]] = None,
    ) -> Float[torch.Tensor, "n_points embedding_dim"]:
        """Fit the embedder and transform the data in one step.

        Args:
            X: Features to embed.
            D: Distances to embed.

        Returns:
            X_embedded: Embedded representation of the input features.
        """
        return self.fit(X, D).transform(X)
