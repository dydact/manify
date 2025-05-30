"""Base predictor class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
from jaxtyping import Float
from sklearn.base import BaseEstimator

from ..manifolds import ProductManifold


class BasePredictor(BaseEstimator, ABC):
    """Base class for everything in `manify.predictors`.

    This is an abstract class that defines a common interface for all mixed-curvature predictors. We assume only that a
    ProductManifold object is given. We try to follow the scikit-learn API's fit/predict_proba/predict paradigm as
    closely as possible, while accommodating the nuances of product manifold geometry and Pytorch/Geoopt.

    Attributes:
        pm: ProductManifold object associated with the predictor.
        task: Task type, either "classification" or "regression".
        random_state: Random state for reproducibility.
        device: Device for tensor computations. If not provided, defaults to pm.device.
        loss_history_: History of loss values during training.
        is_fitted_: Boolean flag indicating if the predictor has been fitted.
    """

    def __init__(
        self,
        pm: ProductManifold,
        task: Literal["classification", "regression"],
        random_state: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        self.pm = pm
        self.task = task
        self.random_state = random_state
        self.device = pm.device if device is None else device
        self.loss_history_: Dict[str, List[float]] = {}
        self.is_fitted_: bool = False

    @abstractmethod
    def fit(
        self, X: Float[torch.Tensor, "n_points n_features"], y: Float[torch.Tensor, "n_points n_classes"]
    ) -> "BasePredictor":
        """Abstract method to fit a predictor. Requires features and labels.

        Args:
            X: Features to fit.
            y: Labels for the features.

        Returns:
            self: Fitted predictor instance.
        """
        pass

    @abstractmethod
    def predict_proba(
        self, X: Optional[Float[torch.Tensor, "n_points n_features"]]
    ) -> Float[torch.Tensor, "n_points n_classes"]:
        """Compute the predicted probabilities for the given features.

        Args:
            X: New inputs for which to make predictions.

        Returns:
            X_proba: Predicted probabilities for the input features.
        """
        pass

    def predict(
        self, X: Optional[Float[torch.Tensor, "n_points n_features"]]
    ) -> Float[torch.Tensor, "n_points n_classes"]:
        """Compute the predicted classes for the given features.

        Args:
            X: New inputs for which to make predictions.

        Returns:
            X_proba: Predicted probabilities for the input features.
        """
        if self.task == "regression":
            return self.predict_proba(X=X)
        return self.predict_proba(X=X).argmax(dim=-1)
