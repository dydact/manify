"""Implementation for direct coordinate optimization in Riemannian manifolds.

This module provides functions for learning optimal embeddings in product manifolds by directly optimizing the
coordinates using Riemannian optimization. This approach is particularly useful for embedding graphs using metric learning
to maintain pairwise distances in the target space. The optimization is performed using Riemannian gradient descent
with support for non-transductive training, in which gradients from the test set to the training set are masked out.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Tuple

import geoopt
import numpy as np
import torch
from jaxtyping import Float, Int

from ..manifolds import ProductManifold
from ._losses import d_avg, distortion_loss

# TQDM: notebook or regular
if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def train_coords(
    pm: ProductManifold,
    dists: Float[torch.Tensor, "n_points n_points"],
    test_indices: Int[torch.Tensor, "n_test,"] = torch.tensor([]),
    device: str = "cpu",
    burn_in_learning_rate: float = 1e-3,
    burn_in_iterations: int = 2_000,
    learning_rate: float = 1e-2,
    scale_factor_learning_rate: float = 0.0,  # Off by default
    training_iterations: int = 18_000,
    loss_window_size: int = 100,
    logging_interval: int = 10,
) -> Tuple[Float[torch.Tensor, "n_points n_dim"], Dict[str, List[float]]]:
    r"""Trains point coordinates in a product manifold to match target distances.

    This function optimizes the coordinates of points in a product manifold to match a given distance matrix. The
    optimization is performed in two phases:

    1. Burn-in phase: Initial optimization with a smaller learning rate to find a good starting configuration.
    2. Training phase: Fine-tuning of the coordinates with a larger learning rate, and optionally optimizing the scale
        factors (curvatures) of the manifold components.

    The optimization uses Riemannian Adam optimizer to respect the manifold structure during gradient updates. The loss
    is computed based on the distortion between the pairwise distances in the embedding and the target distances.

    For non-transductive settings, the function supports split between training and testing points, optimizing different
    combinations of distances (train-train, test-test, train-test).

    Args:
        pm: ProductManifold object specifying the target manifold structure.
        dists: Tensor representing the target pairwise distance matrix between points.
        test_indices: Tensor containing indices of test points for transductive learning.
            Defaults to an empty tensor (all points are used for training).
        device: Device for tensor computations.
        burn_in_learning_rate: Learning rate for the burn-in phase.
        burn_in_iterations: Number of iterations for the burn-in phase.
        learning_rate: Learning rate for the main training phase.
        scale_factor_learning_rate: Learning rate for optimizing manifold scale factors. Off (no learning) by default.
        training_iterations: Number of iterations for the main training phase.
        loss_window_size: Window size for computing moving average loss.
        logging_interval: Interval for logging training progress.

    Returns:
        embeddings: Tensor of shape (n_points, n_dim) with optimized coordinates in the manifold.
        losses: Dictionary containing loss histories for different components:

            * 'train_train': Loss between training points
            * 'test_test': Loss between test points (if test_indices is provided)
            * 'train_test': Loss between training and test points (if test_indices is provided)
            * 'total': Sum of all loss components
    """
    # Move everything to the device
    n = dists.shape[0]
    covs = [torch.stack([torch.eye(M.dim) / pm.dim] * n).to(device) for M in pm.P]
    means = torch.stack([pm.mu0] * n).to(device)
    X, _ = pm.sample(z_mean=means, sigma_factorized=covs)
    dists = dists.to(device)

    # Get train and test indices set up
    use_test = len(test_indices) > 0
    test = torch.tensor([i in test_indices for i in range(len(dists))]).to(device)
    train = ~test

    # Initialize optimizer
    X = geoopt.ManifoldParameter(X, manifold=pm.manifold)
    ropt = geoopt.optim.RiemannianAdam(
        [
            {"params": [X], "lr": burn_in_learning_rate},
            {"params": pm.parameters(), "lr": 0},
        ]
    )

    # Init TQDM
    my_tqdm = tqdm(total=burn_in_iterations + training_iterations, leave=False)

    # Outer training loop - mostly setting optimizer learning rates up here
    losses: Dict[str, List[float]] = {"train_train": [], "test_test": [], "train_test": [], "total": []}

    # Actual training loop
    for i in range(burn_in_iterations + training_iterations):
        if i == burn_in_iterations:
            # Optimize curvature by changing lr
            # opt.lr = scale_factor_learning_rate
            # ropt.lr = learning_rate
            ropt.param_groups[0]["lr"] = learning_rate
            ropt.param_groups[1]["lr"] = scale_factor_learning_rate

        # Zero grad
        ropt.zero_grad()
        # opt.zero_grad()

        # 1. Train-train loss
        X_t = X[train]
        D_tt = pm.pdist(X_t)
        L_tt = distortion_loss(D_tt, dists[train][:, train], pairwise=True)
        L_tt.backward(retain_graph=True)
        losses["train_train"].append(L_tt.item())

        if use_test:
            # 2. Test-test loss
            X_q = X[test]
            D_qq = pm.pdist(X_q)
            L_qq = distortion_loss(D_qq, dists[test][:, test], pairwise=True)
            L_qq.backward(retain_graph=True)
            losses["test_test"].append(L_qq.item())

            # 3. Train-test loss
            X_t_detached = X[train].detach()
            D_tq = pm.dist(X_t_detached, X_q)  # Note 'dist' not 'pdist', as we're comparing different sets
            L_tq = distortion_loss(D_tq, dists[train][:, test], pairwise=False)
            L_tq.backward()
            losses["train_test"].append(L_tq.item())
        else:
            L_qq = 0
            L_tq = 0

        # Step
        # opt.step()
        ropt.step()
        L = L_tt + L_qq + L_tq
        losses["total"].append(L.item())

        # TQDM management
        my_tqdm.update(1)
        my_tqdm.set_description(f"Loss: {L.item():.3e}")

        # Logging
        if i % logging_interval == 0:
            d = {f"r{i}": f"{x._log_scale.item():.3f}" for i, x in enumerate(pm.manifold.manifolds)}
            d["D_avg"] = f"{d_avg(D_tt, dists[train][:, train], pairwise=True):.4f}"
            d["L_avg"] = f"{np.mean(losses['total'][-loss_window_size:]):.3e}"
            my_tqdm.set_postfix(d)

        # Early stopping for errors
        if torch.isnan(L):
            raise ValueError("Loss is NaN")

    return X.data.detach(), losses
