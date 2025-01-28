"""Implementation for coordinate training and optimization"""
from typing import Optional, Tuple, List, Dict
from torchtyping import TensorType

import sys
import torch
import numpy as np
import geoopt
from .metrics import distortion_loss, d_avg
from .manifolds import ProductManifold

# TQDM: notebook or regular
if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def train_coords(
    pm: ProductManifold,
    dists: TensorType["n_points", "n_points"],
    test_indices: Optional[TensorType["n_test"]] = torch.tensor([]),
    device: str = "cpu",
    burn_in_learning_rate: float = 1e-3,
    burn_in_iterations: int = 2_000,
    learning_rate: float = 1e-2,
    scale_factor_learning_rate: float = 0.0,  # Off by default
    training_iterations: int = 18_000,
    loss_window_size: int = 100,
    logging_interval: int = 10,
) -> Tuple[TensorType["n_points", "n_dim"], Dict[str, List[float]]]:
    """
    Coordinate training and optimization

    Args:
        pm: ProductManifold object that encapsulates the manifold and its signature.
        dists: (n_points, n_points) Tensor representing the pairwise distance matrix between points.
        test_indices: (n_test) Tensor representing the indices of the test points.
        device: Device for training (default: "cpu").
        burn_in_learning_rate: Learning rate during the burn-in phase (default: 1e-3).
        burn_in_iterations: Number of iterations during the burn-in phase (default: 2,000).
        learning_rate: Learning rate during the training phase (default: 1e-2).
        scale_factor_learning_rate: Learning rate for scale factor optimization (default: 0.0).
        training_iterations: Number of iterations for the training phase (default: 18,000).
        loss_window_size: Window size for computing the moving average of the loss (default: 100).
        logging_interval: Interval for logging the training progress (default: 10).

    Returns:
        pm.x_embed: Tensor of the final learned coordinates in the manifold.
        losses: List of loss values at each iteration during training.
    """
    # Move everything to the device
    X = pm.initialize_embeddings(n_points=len(dists)).to(device)
    dists = dists.to(device)

    # Get train and test indices set up
    use_test = len(test_indices) > 0
    test = torch.tensor([i in test_indices for i in range(len(dists))]).to(device)
    train = ~test

    # Initialize optimizer
    X = geoopt.ManifoldParameter(X, manifold=pm.manifold)
    ropt = geoopt.optim.RiemannianAdam(params=[X], lr=burn_in_learning_rate)
    opt = torch.optim.Adam(params=[x._log_scale for x in pm.manifold.manifolds], lr=0)

    # Init TQDM
    my_tqdm = tqdm(total=burn_in_iterations + training_iterations, leave=False)

    # Outer training loop - mostly setting optimizer learning rates up here
    losses = {"train_train": [], "test_test": [], "train_test": [], "total": []}
    for lr, n_iters in ((burn_in_learning_rate, burn_in_iterations), (learning_rate, training_iterations)):
        # Actual training loop
        for i in range(n_iters):
            if i == burn_in_iterations:
                # Optimize curvature by changing lr
                opt.lr = scale_factor_learning_rate
                ropt.lr = learning_rate
            
            # Zero grad
            ropt.zero_grad()
            opt.zero_grad()

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
            opt.step()
            ropt.step()
            L = L_tt + L_qq + L_tq
            losses["total"].append(L.item())

            # TQDM management
            my_tqdm.update(1)
            my_tqdm.set_description(f"Loss: {L.item():.3e}")

            # Logging
            if i % logging_interval == 0:
                d = {f"r{i}": f"{x._log_scale.item():.3f}" for i, x in enumerate(pm.manifold.manifolds)}
                # d_avg_this_iter = d_avg(dist_est, dists)
                # d_avg_this_iter = (
                #     d_avg(D_tt, dists[train][:, train], pairwise=True) * len(train) * (len(train) - 1) / 2  # triu
                #     + d_avg(D_qq, dists[test][:, test], pairwise=True) * len(test) * (len(test) - 1)  # triu
                #     + d_avg(D_tq, dists[train][:, test], pairwise=False) * len(train) * len(test)  # full
                # ) / (
                #     len(pm.x_embed) * (len(pm.x_embed) - 1) / 2
                # )  # triu
                # d["D_avg"] = f"{d_avg_this_iter:.4f}"
                d["D_avg"] = f"{d_avg(D_tt, dists[train][:, train], pairwise=True):.4f}"
                d["L_avg"] = f"{np.mean(losses['total'][-loss_window_size:]):.3e}"
                my_tqdm.set_postfix(d)

    return X, losses
