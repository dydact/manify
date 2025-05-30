"""Base embedder class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from beartype.typing import Any
from jaxtyping import Float
from sklearn.base import BaseEstimator, TransformerMixin

from ..manifolds import ProductManifold


class BaseEmbedder(BaseEstimator, TransformerMixin, ABC):
    """Base class for everything in `manify.embedders`.

    This is an abstract class that that defines a common interface for all embedding methods. We assume only that a
    ProductManifold object is given. We try to follow the scikit-learn API's fit/transform paradigm as closely as
    possible, while accommodating the nuances of product manifold geometry and Pytorch/Geoopt.

    Attributes:
        pm: ProductManifold object associated with the embedder.
        random_state: Random state for reproducibility.
        device: Device for tensor computations. If not provided, defaults to pm.device.
        loss_history_: History of loss values during training.
        is_fitted_: Boolean flag indicating if the embedder has been fitted.
    """

    def __init__(self, pm: ProductManifold, random_state: int | None = None, device: str | None = None) -> None:
        self.pm = pm
        self.random_state = random_state
        self.device = pm.device if device is None else device
        self.loss_history_: dict[str, list[float]] = {}
        self.is_fitted_: bool = False

    @abstractmethod
    def fit(
        self,
        X: Float[torch.Tensor, "n_points n_features"] | None = None,
        D: Float[torch.Tensor, "n_points n_points"] | None = None,
        lr: float = 1e-2,
        burn_in_lr: float = 1e-3,
        curvature_lr: float = 0.0,  # Off by default
        burn_in_iterations: int = 2_000,
        training_iterations: int = 18_000,
        loss_window_size: int = 100,
        logging_interval: int = 10,
    ) -> "BaseEmbedder":
        """Abstract method to fit an embedder. Requires at least one of (features, distances).

        Args:
            X: Features to embed. Used by Mixed-curvature VAE and Siamese Network classes.
            D: Distances to embed. Used by coordinate learning and Siamese Network classes.
            lr: Learning rate for the main training phase.
            burn_in_lr: Learning rate for the burn-in phase.
            curvature_lr: Learning rate for optimizing manifold scale factors. Off (no learning) by default.
            burn_in_iterations: Number of iterations for the burn-in phase.
            training_iterations: Number of iterations for the main training phase.
            loss_window_size: Window size for computing moving average loss.
            logging_interval: Interval for logging training progress.

        Returns:
            self: Fitted embedder instance.
        """
        pass

    @abstractmethod
    def transform(
        self, X: Float[torch.Tensor, "n_points n_features"] | None
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
        X: Float[torch.Tensor, "n_points n_features"] | None = None,
        D: Float[torch.Tensor, "n_points n_points"] | None = None,
        **fit_kwargs: Any,
    ) -> Float[torch.Tensor, "n_points embedding_dim"]:
        """Fit the embedder and transform the data in one step.

        Args:
            X: Features to embed.
            D: Distances to embed.
            **fit_kwargs: Additional keyword arguments for fitting.

        Returns:
            X_embedded: Embedded representation of the input features.
        """
        return self.fit(X=X, D=D, **fit_kwargs).transform(X=X)
