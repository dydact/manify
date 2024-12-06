import sys

# TQDM: notebook or regular
if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import torch
import numpy as np
import geoopt

from torchtyping import TensorType

from .metrics import distortion_loss, d_avg
from .manifolds import ProductManifold

"""Implementation for coordinate training and optimization"""
def train_coords(
    pm: ProductManifold,
    dists: TensorType["n_points", "n_points"],
    device: str = "cpu",
    burn_in_learning_rate: float = 1e-3,
    burn_in_iterations: int = 2_000,
    learning_rate: float = 1e-2,
    scale_factor_learning_rate: float = 0.0,  # Off by default
    training_iterations: int = 18_000,
    loss_window_size: int = 100,
    logging_interval: int = 10,
):
    """
    Coordinate training and optimization

    Args:
        pm: productManifold object that encapsulates the manifold and its signature.
        dists: (n_points, n_points) tensor representing the pairwise distance matrix between points.
        device: device for training (default: "cpu").
        burn_in_learning_rate: learning rate during the burn-in phase (default: 1e-3).
        burn_in_iterations: number of iterations during the burn-in phase (default: 2,000).
        learning_rate: learning rate during the training phase (default: 1e-2).
        scale_factor_learning_rate: learning rate for scale factor optimization (default: 0.0).
        training_iterations: number of iterations for the training phase (default: 18,000).
        loss_window_size: window size for computing the moving average of the loss (default: 100).
        logging_interval: interval for logging the training progress (default: 10).

    Returns:
        pm.x_embed: tensor of the final learned coordinates in the manifold.
        losses: list of loss values at each iteration during training.
    """    
    # Move everything to the device
    pm.x_embed = pm.initialize_embeddings(n_points=len(dists)).to(device)
    dists = dists.to(device)

    # Initialize optimizer
    pm.x_embed = geoopt.ManifoldParameter(pm.x_embed, manifold=pm.manifold)
    pm.opt = geoopt.optim.RiemannianAdam(
        [
            {"params": [pm.x_embed], "lr": burn_in_learning_rate},
            {"params": [x._log_scale for x in pm.manifold.manifolds], "lr": 0},
        ]
    )

    # Init TQDM
    my_tqdm = tqdm(total=burn_in_iterations + training_iterations, leave=False)

    # Outer training loop - mostly setting optimizer learning rates up here
    losses = []
    for lr, n_iters in ((burn_in_learning_rate, burn_in_iterations), (learning_rate, training_iterations)):
        # Set the learning rate
        pm.opt.param_groups[0]["lr"] = lr
        if lr == learning_rate:
            pm.opt.param_groups[1]["lr"] = scale_factor_learning_rate

        # Actual training loop
        for i in range(n_iters):
            pm.opt.zero_grad()
            dist_est = pm.pdist(pm.x_embed)
            L = distortion_loss(dist_est, dists)
            L.backward()
            losses.append(L.item())
            pm.opt.step()

            # TQDM management
            my_tqdm.update(1)
            my_tqdm.set_description(f"Loss: {L.item():.3e}")

            # Logging
            if i % logging_interval == 0:
                d = {f"r{i}": f"{x._log_scale.item():.3f}" for i, x in enumerate(pm.manifold.manifolds)}
                d_avg_this_iter = d_avg(dist_est, dists)
                d["D_avg"] = f"{d_avg_this_iter:.4f}"
                d["L_avg"] = f"{np.mean(losses[-loss_window_size:]):.3e}"
                my_tqdm.set_postfix(d)

    return pm.x_embed, losses
