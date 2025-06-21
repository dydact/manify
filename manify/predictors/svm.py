"""Implementation for Support Vector Machine in Product Manifolds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cvxpy
import numpy as np
import torch
from sklearn.base import BaseEstimator

if TYPE_CHECKING:
    from beartype.typing import Literal
    from jaxtyping import Float, Int

from ..manifolds import ProductManifold
from ._base import BasePredictor
from ._kernel import product_kernel


class ProductSpaceSVM(BaseEstimator):
    """Product Space SVM class."""

    def __init__(
        self,
        pm: ProductManifold,
        weights: Float[torch.Tensor, "n_manifolds"] | None = None,
        h_constraints: bool = True,
        e_constraints: bool = True,
        s_constraints: bool = True,
        task: Literal["classification", "regression"] = "classification",
        epsilon: float = 1e-5,
        random_state: int | None = None,
        device: str | None = None,
    ):
        # Initialize base class
        super().__init__(pm=pm, task=task, random_state=random_state, device=device)

        self.pm = pm
        self.h_constraints = h_constraints
        self.s_constraints = s_constraints
        self.e_constraints = e_constraints
        self.eps = epsilon
        self.task = task
        if weights is None:
            self.weights = torch.ones(len(pm.P), dtype=torch.float32)
        else:
            assert len(weights) == len(pm.P), "Number of weights must match the number of manifolds."
            self.weights = weights

    def fit(self, X: Float[torch.Tensor, "n_samples n_manifolds"], y: Int[torch.Tensor, "n_samples"]) -> None:
        """Trains the SVM model using the provided data and labels.

        Args:
            X: The training data of shape (n_samples, n_features).
            y: The class labels for the training data of shape (n_samples,).
        """
        # Identify unique classes for multiclass classification
        self.classes_ = torch.unique(y).tolist()
        n_samples = X.shape[0]

        # Precompute kernel matrix
        Ks, norms = product_kernel(self.pm, X, None)
        K = torch.ones((n_samples, n_samples), dtype=X.dtype, device=X.device)
        for K_m, w in zip(Ks, self.weights):
            K += w * K_m

        # Make numpy arrays
        X = X.detach().cpu().numpy()
        # Y = torch.diagflat(y).detach().cpu().numpy()
        K = K.detach().cpu().numpy()

        # Initialize dicts for SVM parameters
        self.beta = {}
        self.zeta = {}
        self.epsilon = {}

        for class_label in self.classes_:
            # Get labels
            # y_binary = torch.where(y == class_label, 1, -1)  # Shape: (n_samples,)
            y_binary = torch.where(y == class_label, 1, 0)  # Shape: (n_samples,)
            Y = torch.diagflat(y_binary).detach().cpu().numpy()

            # Make variables
            zeta = cvxpy.Variable(X.shape[0])
            beta = cvxpy.Variable(X.shape[0])
            epsilon = cvxpy.Variable(1)

            # Get constraints
            # TODO: try putting this outside of this loop?
            constraints = [
                epsilon >= 0,
                zeta >= 0,
                Y @ (K @ beta + cvxpy.sum(beta)) >= epsilon - zeta,
                # Y @ (K @ beta) >= epsilon - zeta,  # Replaced with sum-less form
            ]
            for M, K_component, norm in zip(self.pm.P, Ks, norms):
                K_component = K_component.detach().cpu().numpy()
                norm = norm.item()
                if M.type == "E" and self.e_constraints:
                    alpha_E = 1.0  # TODO: make this flexible
                    constraints.append(cvxpy.quad_form(beta, K_component) <= alpha_E**2)
                elif M.type == "S" and self.s_constraints:
                    constraints.append(cvxpy.quad_form(beta, K_component) <= np.pi / 2)
                elif M.type == "H" and self.h_constraints:
                    K_component_pos = K_component.clip(0, None)
                    K_component_neg = K_component.clip(None, 0)
                    constraints.append(cvxpy.quad_form(beta, K_component_neg) <= self.eps)
                    constraints.append(cvxpy.quad_form(beta, K_component_pos) <= self.eps + norm)

            # CVXPY solver
            cvxpy.Problem(
                objective=cvxpy.Minimize(-epsilon + cvxpy.sum(zeta)),
                constraints=constraints,
            ).solve()

            self.zeta[class_label] = zeta.value
            self.beta[class_label] = beta.value
            self.epsilon[class_label] = epsilon.value

            # Need to store X for prediction
            self.X_train_ = torch.tensor(X, dtype=torch.float32)

    def predict_proba(
        self, X: Float[torch.Tensor, "n_samples n_manifolds"]
    ) -> Float[torch.Tensor, "n_samples n_classes"]:
        """Predicts the probability of each class for the given test data.

        Args:
            X: The test data of shape (n_samples, n_features).

        Returns:
            probs: The probability of each class for the given test data of shape (n_samples, n_classes).
        """
        # Ensure X is a torch tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        # Ensure X is on the same device as the training data
        device = self.X_train_.device
        X = X.to(device)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # Compute the kernel between training data and test data
        Ks_test, _ = product_kernel(self.pm, self.X_train_, X)
        K_test = torch.ones((self.X_train_.shape[0], n_samples), dtype=X.dtype, device=X.device)
        for K_m, w in zip(Ks_test, self.weights):
            K_test += w * K_m

        # Convert to NumPy array
        K_test = K_test.detach().cpu().numpy()

        # Initialize array to store decision function values
        decision_function = np.zeros((n_samples, n_classes))
        for idx, class_label in enumerate(self.classes_):
            beta = self.beta[class_label]  # Shape: (n_train_samples,)
            # Compute decision function: f(x) = K_test^T @ beta + sum(beta)
            f = K_test.T @ beta + np.sum(beta)
            decision_function[:, idx] = f

        # Convert decision function values to probabilities using softmax
        exp_scores = np.exp(decision_function - np.max(decision_function, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs
