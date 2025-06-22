"""Base predictor class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

if TYPE_CHECKING:
    from beartype.typing import Literal
    from jaxtyping import Float

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
        task: Literal["classification", "regression", "link_prediction"],
        random_state: int | None = None,
        device: str | None = None,
    ) -> None:
        self.pm = pm
        self.task = task
        self.random_state = random_state
        self.device = pm.device if device is None else device
        self.loss_history_: dict[str, list[float]] = {}
        self.is_fitted_: bool = False

        # Initialize appropriate base class depending on task
        if task == "classification":
            ClassifierMixin.__init__(self)
        elif task == "regression":
            RegressorMixin.__init__(self)
        elif task == "link_prediction":
            # For link prediction, we also use ClassifierMixin, as we think of it as binary classificaiton.
            ClassifierMixin.__init__(self)
        else:
            raise ValueError(f"Unknown task type: {task}")

    def _store_classes(
        self, y: Float[torch.Tensor, "n_points n_classes"] | Float[torch.Tensor, "n_points"]
    ) -> Float[torch.Tensor, "n_points"]:
        """Store unique classes and return relabeled y for classification tasks."""
        if self.task == "classification":
            self.classes_, y_relabeled = y.unique(return_inverse=True)
            return y_relabeled
        else:
            return y

    def _get_class_predictions(self, class_indices: torch.Tensor) -> torch.Tensor:
        """Convert class indices back to original labels."""
        if hasattr(self, "classes_") and self.task == "classification":
            return self.classes_[class_indices]
        return class_indices

    @abstractmethod
    def fit(
        self,
        X: Float[torch.Tensor, "n_points n_features"],
        y: Float[torch.Tensor, "n_points n_classes"] | Float[torch.Tensor, "n_points"],
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
    def predict_proba(self, X: Float[torch.Tensor, "n_points n_features"]) -> Float[torch.Tensor, "n_points n_classes"]:
        """Compute the predicted probabilities for the given features.

        Args:
            X: New inputs for which to make predictions.

        Returns:
            X_proba: Predicted probabilities for the input features.
        """
        pass

    def predict(self, X: Float[torch.Tensor, "n_points n_features"], **kwargs: dict) -> Float[torch.Tensor, "n_points"]:
        """Compute the predicted classes for the given features.

        Args:
            X: New inputs for which to make predictions.
            **kwargs: Additional keyword arguments that get passed to `self.predict_proba()`.

        Returns:
            X_proba: Predicted probabilities for the input features.
        """
        if self.task == "regression":
            return self.predict_proba(X=X, **kwargs).squeeze(-1)
        elif self.task == "link_prediction":
            logits = self.predict_proba(X=X, **kwargs)
            return (logits > 0.5).float()  # Threshold at 0.5
        else:  # classification
            class_indices = self.predict_proba(X=X, **kwargs).argmax(dim=-1)
            return self._get_class_predictions(class_indices)

    def score(
        self,
        X: Float[torch.Tensor, "n_points n_features"],
        y: Float[torch.Tensor, "n_points n_classes"] | Float[torch.Tensor, "n_points"],
        **kwargs: dict,
    ) -> float:
        """Return the mean accuracy/R² score.

        Args:
            X: Input features.
            y: Target labels.
            **kwargs: Additional keyword arguments that get passed to `self.predict_proba()`.

        Returns:
            score: Mean accuracy (classification) or R² score (regression).
        """
        predictions = self.predict(X, **kwargs)

        if self.task == "classification":
            out = ((predictions == y).float() * sample_weight).mean().item()
        elif self.task == "regression":
            out = (((predictions - y) ** 2 * sample_weight).mean()).item()
        else:  # link_prediction
            out = ((predictions == y).float() * sample_weight).mean().item()

        return float(out)
