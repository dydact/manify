"""Implementation for direct coordinate optimization in Riemannian manifolds.

This module provides functions for learning optimal embeddings in product manifolds by directly optimizing the
coordinates using Riemannian optimization. This approach is particularly useful for embedding graphs using metric
learning to maintain pairwise distances in the target space. The optimization is performed using Riemannian gradient
descent with support for non-transductive training, in which gradients from the test set to the training set are masked
out.
"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING

import geoopt
import numpy as np
import torch

if TYPE_CHECKING:
    from beartype.typing import Any
    from jaxtyping import Float, Int

from ..manifolds import ProductManifold
from ._base import BaseEmbedder
from ._losses import d_avg, distortion_loss

# TQDM: notebook or regular
if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class CoordinateLearning(BaseEmbedder):
    """Coordinate learning method class.

    This embedder implements the approach described in Gu et al., "Learning Mixed-Curvature Representations in Product
    Spaces". It directly optimizes point coordinates to preserve a given distance matrix, using Riemannian optimization
    techniques.

    Trains point coordinates in a product manifold to match target distances.

    This class optimizes the coordinates of points in a product manifold to match a given distance matrix. The
    optimization is performed in two phases:

    1. Burn-in phase: Initial optimization with a smaller learning rate to find a good starting configuration.
    2. Training phase: Fine-tuning of the coordinates with a larger learning rate, and optionally optimizing the scale
        factors (curvatures) of the manifold components.

    The optimization uses Riemannian Adam optimizer to respect the manifold structure during gradient updates. The loss
    is computed based on the distortion between the pairwise distances in the embedding and the target distances.

    For non-transductive settings, the class supports split between training and testing points, optimizing different
    combinations of distances (train-train, test-test, train-test).

    Attributes:
        pm: Product manifold defining the target embedding space.
        embeddings_: Optimized point coordinates after fitting.
        loss_history_: Training loss history.
        is_fitted_: Boolean flag indicating if the embedder has been fitted.

    Args:
        pm: ProductManifold object defining the target embedding space.
        random_state: Optional random state for reproducibility.
        device: Optional device for tensor computations.
    """

    def __init__(self, pm: ProductManifold, random_state: int | None = None, device: str | None = None) -> None:
        super().__init__(pm=pm, random_state=random_state, device=device)

    def fit(  # type: ignore[override]
        self,
        X: None,
        D: Float[torch.Tensor, "n_points n_points"],
        test_indices: Int[torch.Tensor, "n_test"] = torch.tensor([]),
        lr: float = 1e-2,
        burn_in_lr: float = 1e-3,
        curvature_lr: float = 0.0,  # Off by default
        burn_in_iterations: int = 2_000,
        training_iterations: int = 18_000,
        loss_window_size: int = 100,
        logging_interval: int = 10,
    ) -> "CoordinateLearning":
        """Fit the Coordinate Learning Embedder. Sets attributes `embeddings_`, `loss_history_`, and `is_fitted_`.

        Args:
            X: Ignored.
            D: Tensor representing the target pairwise distance matrix between points.
            test_indices: Tensor containing indices of test points for transductive learning.
                Defaults to an empty tensor (all points are used for training).
            lr: Learning rate for the main training phase.
            burn_in_lr: Learning rate for the burn-in phase.
            curvature_lr: Learning rate for optimizing manifold scale factors. Off (no learning) by default.
            burn_in_iterations: Number of iterations for the burn-in phase.
            training_iterations: Number of iterations for the main training phase.
            loss_window_size: Window size for computing moving average loss.
            logging_interval: Interval for logging training progress.

        Returns:
            self: Fitted embedder instance.

        Raises:
            ValueError: If the distance matrix D is None or if X is provided.
            Warning: If X is provided, it will be ignored during fitting.
        """
        # Input validation
        if D is None:
            raise ValueError("Distance matrix D is needed for coordinate learning")
        if X is not None:
            warnings.warn(
                "Input X has been given. This will be ignored during fitting. If you have provided a distance matrix,"
                "please run embedder.fit(None, D) instead."
            )

        # Set random seed if provided
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Move everything to the device; initialize random embeddings
        n = D.shape[0]
        covs = [torch.stack([torch.eye(M.dim) / self.pm.dim] * n).to(self.device) for M in self.pm.P]
        means = torch.vstack([self.pm.mu0] * n).to(self.device)
        X_embed, _ = self.pm.sample(z_mean=means, sigma_factorized=covs)
        D = D.to(self.device)

        # Get train and test indices set up
        use_test = len(test_indices) > 0
        test = torch.tensor([i in test_indices for i in range(len(D))]).to(self.device)
        train = ~test

        # Initialize optimizer
        X_embed = geoopt.ManifoldParameter(X_embed, manifold=self.pm.manifold)
        ropt = geoopt.optim.RiemannianAdam(
            [{"params": [X_embed], "lr": burn_in_lr}, {"params": self.pm.parameters(), "lr": 0}]
        )

        # Init TQDM
        my_tqdm = tqdm(total=burn_in_iterations + training_iterations, leave=False)

        # Outer training loop - mostly setting optimizer learning rates up here
        losses: dict[str, list[float]] = {"train_train": [], "test_test": [], "train_test": [], "total": []}

        # Actual training loop
        for i in range(burn_in_iterations + training_iterations):
            if i == burn_in_iterations:
                # Optimize curvature by changing lr
                ropt.param_groups[0]["lr"] = lr
                ropt.param_groups[1]["lr"] = curvature_lr

            # Zero grad
            ropt.zero_grad()

            # 1. Train-train loss
            X_t = X_embed[train]
            D_tt = self.pm.pdist(X_t)
            L_tt = distortion_loss(D_tt, D[train][:, train], pairwise=True)
            L_tt.backward(retain_graph=True)
            losses["train_train"].append(L_tt.item())

            if use_test:
                # 2. Test-test loss
                X_q = X_embed[test]
                D_qq = self.pm.pdist(X_q)
                L_qq = distortion_loss(D_qq, D[test][:, test], pairwise=True)
                L_qq.backward(retain_graph=True)
                losses["test_test"].append(L_qq.item())

                # 3. Train-test loss
                X_t_detached = X_embed[train].detach()
                D_tq = self.pm.dist(X_t_detached, X_q)  # Note 'dist' not 'pdist', as we're comparing different sets
                L_tq = distortion_loss(D_tq, D[train][:, test], pairwise=False)
                L_tq.backward()
                losses["train_test"].append(L_tq.item())
            else:
                L_qq = 0
                L_tq = 0

            # Step
            ropt.step()
            L = L_tt + L_qq + L_tq
            losses["total"].append(L.item())

            # TQDM management
            my_tqdm.update(1)
            my_tqdm.set_description(f"Loss: {L.item():.3e}")

            # Logging
            if i % logging_interval == 0:
                d = {f"r{i}": f"{logscale.item():.3f}" for i, logscale in enumerate(self.pm.parameters())}
                d["D_avg"] = f"{d_avg(D_tt, D[train][:, train], pairwise=True):.4f}"
                d["L_avg"] = f"{np.mean(losses['total'][-loss_window_size:]):.3e}"
                my_tqdm.set_postfix(d)

            # Early stopping for errors
            if torch.isnan(L):
                raise ValueError("Loss is NaN")

        # Final maintenance: update attributes
        self.embeddings_ = X_embed.data.detach()
        self.loss_history_ = losses
        self.is_fitted_ = True

        return self

    def transform(self, X: None = None) -> Float[torch.Tensor, "n_points embedding_dim"]:
        """Transform data using learned embedding. This is not meaningful for new data during coordinate learning.

        Args:
            X: Ignored.

        Returns:
            embeddings: Learned embeddings.

        Raises:
            ValueError: If the embedder has not been fitted yet.
            Warning: If X is provided, as it will be ignored.
        """
        if not self.is_fitted_:
            raise ValueError("The embedder has not been fitted yet.")

        if X is not None:
            warnings.warn("Coordinate learning can only return trained embeddings. X will be ignored.")

        return self.embeddings_

    def fit_transform(  # type: ignore[override]
        self, X: None, D: Float[torch.Tensor, "n_points n_points"], **fit_kwargs: Any
    ) -> Float[torch.Tensor, "n_points embedding_dim"]:
        """Transform data using learned embedding based on the provided distance matrix D.

        This method overrides the base class method `BaseEmbedder.fit_transform()` to not use the input data X.

        Args:
            X: Ignored.
            D: Distance matrix for the points.

        Returns:
            embeddings: Learned embeddings.
        """
        return self.fit(X=None, D=D, **fit_kwargs).transform(X=None)
