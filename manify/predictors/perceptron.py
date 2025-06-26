"""Product space perceptron implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from jaxtyping import Float, Int

from ..manifolds import ProductManifold
from ._base import BasePredictor
from ._kernel import product_kernel


class ProductSpacePerceptron(BasePredictor):
    """A product-space perceptron model for multiclass classification in the product manifold space.

    Args:
        pm: ProductManifold object for the product space.
        max_epochs: Maximum number of training epochs.
        patience: Number of consecutive epochs without improvement to consider convergence.
        weights: Per-manifold weights for kernel combination.
        task: Task type (defaults to "classification").
        random_state: Random seed for reproducibility.
        device: Device for tensor computations.

    Attributes:
        pm: ProductManifold object associated with the predictor.
        max_epochs: Maximum number of training epochs.
        patience: Number of consecutive epochs without improvement to consider convergence.
        weights: Per-manifold weights for kernel combination.
        alpha: Dictionary storing perceptron coefficients for each class.
        X_train_: Training data points.
        y_train_: Training labels.
        is_fitted_: Boolean flag indicating if the predictor has been fitted.
    """

    def __init__(
        self,
        pm: ProductManifold,
        max_epochs: int = 1_000,
        patience: int = 5,
        weights: Float[torch.Tensor, "n_manifolds"] | None = None,
        task: str = "classification",
        random_state: int | None = None,
        device: str | None = None,
    ):
        # Initialize base class
        super().__init__(pm=pm, task=task, random_state=random_state, device=device)
        self.pm = pm  # ProductManifold instance
        self.max_epochs = max_epochs
        self.patience = patience  # Number of consecutive epochs without improvement to consider convergence
        if weights is None:
            self.weights = torch.ones(len(pm.P), dtype=torch.float32)
        else:
            assert len(weights) == len(pm.P), "Number of weights must match the number of manifolds."
            self.weights = weights

    def fit(
        self, X: Float[torch.Tensor, "n_samples n_manifolds"], y: Int[torch.Tensor, "n_samples"]
    ) -> ProductSpacePerceptron:
        """Trains the perceptron model using the provided data and labels.

        Args:
            X: Training data tensor.
            y: Class labels for the training data.

        Returns:
            self: Fitted perceptron model.
        """
        # Identify unique classes for multiclass classification
        self._store_classes(y)
        n_samples = X.shape[0]

        # Precompute kernel matrix
        Ks, _ = product_kernel(self.pm, X, None)
        K = torch.ones((n_samples, n_samples), dtype=X.dtype, device=X.device)
        for K_m, w in zip(Ks, self.weights, strict=False):
            K += w * K_m

        # Store training data and labels for prediction
        self.X_train_ = X
        self.y_train_ = y

        # Initialize dictionary to store alpha coefficients for each class
        self.alpha = {}

        # For patience checking
        best_epoch, least_errors = 0, n_samples + 1

        for class_label in self.classes_:
            class_label_item = class_label.item()

            # One-vs-rest labels
            y_binary = torch.where(y == class_label_item, 1, -1)  # Shape: (n_samples,)

            # Initialize alpha coefficients for this class
            alpha = torch.zeros(n_samples, dtype=X.dtype, device=X.device)

            for epoch in range(self.max_epochs):
                # Compute decision function: f = K @ (alpha * y_binary)
                f = K @ (alpha * y_binary)  # Shape: (n_samples,)

                # Compute predictions
                predictions = torch.sign(f)

                # Find misclassified samples
                misclassified = predictions != y_binary

                # If no misclassifications, break early
                if not misclassified.any():
                    break

                # Test patience
                n_errors = misclassified.sum().item()
                if n_errors < least_errors:
                    best_epoch, least_errors = epoch, n_errors
                if epoch - best_epoch >= self.patience:
                    break

                # Update alpha coefficients for misclassified samples
                alpha[misclassified] += 1

            # Store the alpha coefficients for the current class
            self.alpha[class_label_item] = alpha

        self.is_fitted_ = True
        return self

    def predict_proba(
        self,
        X: Float[torch.Tensor, "n_points n_features"],  # type: ignore[override]
    ) -> Float[torch.Tensor, "n_points n_classes"]:
        """Predicts the decision values for each class.

        Args:
            X: Test data tensor.

        Returns:
            decision_values: Decision values for each test sample and each class.
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        decision_values = torch.zeros((n_samples, n_classes), dtype=X.dtype, device=X.device)

        # Compute kernel matrix between training data and test data
        Ks, _ = product_kernel(self.pm, self.X_train_, X)
        K_test = torch.ones((self.X_train_.shape[0], n_samples), dtype=X.dtype, device=X.device)
        for K_m, w in zip(Ks, self.weights, strict=False):
            K_test += w * K_m
        # K_test = self.X_train_ @ X.T

        for idx, class_label in enumerate(self.classes_):
            class_label_item = class_label.item()
            alpha = self.alpha[class_label_item]  # Shape: (n_samples_train,)
            y_binary = torch.where(self.y_train_ == class_label_item, 1, -1)  # Shape: (n_samples_train,)

            # Compute decision function for test samples
            f = (alpha * y_binary) @ K_test  # Shape: (n_samples_test,)
            decision_values[:, idx] = f

        return decision_values
