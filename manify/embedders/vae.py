"""Product space variational autoencoder implementation"""

from __future__ import annotations

import sys
from typing import List, Tuple

import torch
from jaxtyping import Float

from ..manifolds import ProductManifold

# TQDM: notebook or regular
if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class ProductSpaceVAE(torch.nn.Module):
    """
    Variational Autoencoder (VAE) for data in a mixed-curvature product manifold space.
    This VAE model leverages a product manifold structure for latent representations, enabling
    flexible encodings in spaces with different curvature properties (e.g., hyperbolic, Euclidean, spherical).
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        pm: ProductManifold,
        beta: float = 1.0,
        reconstruction_loss: str = "mse",
        device: str = "cpu",
        n_samples: int = 16,
    ):
        super(ProductSpaceVAE, self).__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.pm = pm.to(device)
        self.beta = beta
        self.device = device
        self.n_samples = n_samples

        if reconstruction_loss == "mse":
            self.reconstruction_loss = torch.nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unknown reconstruction loss: {reconstruction_loss}")

    def encode(self, x: Float[torch.Tensor, "batch_size n_features"]) -> Tuple[
        Float[torch.Tensor, "batch_size n_latent"],
        Float[torch.Tensor, "batch_size n_latent"],
    ]:
        """Must return z_mean, z_logvar"""
        z_mean_tangent, z_logvar = self.encoder(x)
        z_mean_ambient = z_mean_tangent @ self.pm.projection_matrix  # Adds zeros in the right places
        z_mean = self.pm.expmap(u=z_mean_ambient, base=None)
        return z_mean, z_logvar

    def decode(self, z: Float[torch.Tensor, "batch_size n_latent"]) -> Float[torch.Tensor, "batch_size n_features"]:
        """Decoding in product space VAE"""
        return self.decoder(z)

    def forward(self, x: Float[torch.Tensor, "batch_size n_features"]) -> Tuple[
        Float[torch.Tensor, "batch_size n_features"],
        Float[torch.Tensor, "batch_size n_latent"],
        List[Float[torch.Tensor, "n_latent n_latent"]],
    ]:
        """
        Performs the forward pass of the VAE.

        Encodes the input, samples latent variables, and decodes to reconstruct the input.

        Args:
            x (torch.Tensor): Input data of shape (batch_size, n_features).

        Returns:
            tuple: Reconstructed data, latent means, and latent variances.
        """
        z_means, z_logvars = self.encode(x)
        sigma_factorized = self.pm.factorize(z_logvars, intrinsic=True)
        sigmas = [torch.diag_embed(torch.exp(z_logvar) + 1e-8) for z_logvar in sigma_factorized]
        z, _ = self.pm.sample(z_means, sigmas)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z_means, sigmas

    def kl_divergence(
        self,
        z_mean: Float[torch.Tensor, "batch_size n_latent"],
        sigma_factorized: List[Float[torch.Tensor, "n_latent n_latent"]],
    ) -> Float[torch.Tensor, "batch_size,"]:
        """
        Computes the KL divergence between posterior and prior distributions.

        Args:
            z_mean (torch.Tensor): Latent means of shape (batch_size, n_latent).
            sigma_factorized (list of torch.Tensor): Factorized covariance matrices for each latent dimension.

        Returns:
            torch.Tensor: KL divergence values for each data point in the batch.
        """
        # Get KL divergence as the average of log q(z|x) - log p(z)
        # See http://joschu.net/blog/kl-approx.html for more info
        means = torch.repeat_interleave(z_mean, self.n_samples, dim=0)
        sigmas_factorized_interleaved = [
            torch.repeat_interleave(sigma, self.n_samples, dim=0) for sigma in sigma_factorized
        ]
        z_samples, _ = self.pm.sample(means, sigmas_factorized_interleaved)
        log_qz = self.pm.log_likelihood(z_samples, means, sigmas_factorized_interleaved)
        log_pz = self.pm.log_likelihood(z_samples)
        return (log_qz - log_pz).view(-1, self.n_samples).mean(dim=1)

    def elbo(
        self, x: Float[torch.Tensor, "batch_size n_features"]
    ) -> Tuple[Float[torch.Tensor, ""], Float[torch.Tensor, ""], Float[torch.Tensor, ""]]:
        """
        Computes the Evidence Lower Bound (ELBO).

        Args:
            x (torch.Tensor): Input data of shape (batch_size, n_features).

        Returns:
            tuple: Mean ELBO, mean log-likelihood, and mean KL divergence across the batch.
        """
        x_reconstructed, z_means, sigma_factorized = self(x)
        kld = self.kl_divergence(z_means, sigma_factorized)
        ll = -self.reconstruction_loss(x_reconstructed.view(x.shape[0], -1), x.view(x.shape[0], -1)).sum(dim=1)
        return (ll - self.beta * kld).mean(), ll.mean(), kld.mean()

    def _grads_ok(self):
        out = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient in {name}")
                    out = False
                if torch.isinf(param.grad).any():
                    print(f"Inf gradient in {name}")
                    out = False
        return out

    def fit(
        self,
        X_train: Float[torch.Tensor, "n_points n_features"],
        burn_in_epochs: int = 100,
        epochs: int = 1900,
        batch_size: int = 32,
        seed: int = None,
        lr: float = 1e-3,
        curvature_lr: float = 1e-4,
        clip_grad: bool = True,
    ) -> List[float]:
        """Fits the VAE model to the training data.

        Args:
            X_train (torch.Tensor): Training data of shape (n_points, n_features).
            burn_in_epochs (int): Number of burn-in epochs.
            epochs (int): Number of training epochs.
            batch_size (int): Size of each training batch.
            seed (int, optional): Random seed for reproducibility.
            lr (float): Learning rate for the optimizer.
            curvature_lr (float): Learning rate for the curvature parameters.
            clip_grad (bool): Whether to clip gradients.

        Returns:
            List[float]: A list of loss values recorded during training.
        """
        if seed is not None:
            torch.manual_seed(seed)

        my_tqdm = tqdm(total=(burn_in_epochs + epochs) * len(X_train))
        opt = torch.optim.Adam(
            [{"params": self.parameters(), "lr": lr * 0.1}, {"params": self.pm.parameters(), "lr": curvature_lr}]
        )
        losses = []
        for epoch in range(burn_in_epochs + epochs):
            if epoch == burn_in_epochs:
                opt.param_groups[0]["lr"] = lr
                opt.param_groups[1]["lr"] = curvature_lr

            for i in range(0, len(X_train), batch_size):
                opt.zero_grad()
                X_batch = X_train[i : i + batch_size]
                elbo, ll, kl = self.elbo(X_batch)
                loss = -elbo
                losses.append(loss.item())
                loss.backward()

                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Invalid loss detected at epoch {epoch}, batch {i}")
                    continue
                elif self._grads_ok():
                    opt.step()

                my_tqdm.update(batch_size)
                my_tqdm.set_description(f"Epoch {epoch + 1}/{burn_in_epochs + epochs}, Loss: {loss.item():.4f}")
                my_tqdm.set_postfix(loss=loss.item(), epoch=epoch + 1)

        return losses
