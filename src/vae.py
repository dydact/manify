"""Product space variational autoencoder implementation"""
from typing import List
from torchtyping import TensorType as TT

import torch

from .manifolds import ProductManifold

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
        n_samples=16,
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

    def encode(self, x: TT["batch_size", "n_features"]) -> (TT["batch_size", "n_latent"], TT["batch_size", "n_latent"]):
        """Must return z_mean, z_logvar"""
        return self.encoder(x)

    def decode(self, z: TT["batch_size", "n_latent"]) -> TT["batch_size", "n_features"]:
        """Decoding in product space VAE"""
        return self.decoder(z)

    def forward(self, x: TT["batch_size", "n_features"]) -> TT["batch_size", "n_features"]:
        """
        Performs the forward pass of the VAE.

        Encodes the input, samples latent variables, and decodes to reconstruct the input.

        Args:
            x (torch.Tensor): Input data of shape (batch_size, n_features).

        Returns:
            tuple: Reconstructed data, latent means, and latent variances.
        """
        z_means, z_logvars = self.encode(x)
        # sigma = torch.diag_embed(torch.exp(z_logvars) + 1e-8)
        # z = self.pm.sample(z_means, sigma)
        # sigma_factorized = [torch.diag_embed(torch.exp(z_logvar) + 1e-8) for z_logvar in self.pm.factorize(z_logvars)]
        # sigma_factorized = [
        #     torch.diag_embed(torch.exp(z_logvars)[pm.intrinsic2man[i]] + 1e-8) for i in
        # ]
        sigma_factorized = self.pm.factorize(z_logvars, intrinsic=True)
        sigmas = [torch.diag_embed(torch.exp(z_logvar) + 1e-8) for z_logvar in sigma_factorized]
        z = self.pm.sample(z_means, sigmas)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z_means, sigmas

    def kl_divergence(
        self,
        z_mean: TT["batch_size", "n_latent"],
        # sigma: TT["n_latent", "n_latent"],
        sigma_factorized: List[TT["batch_size", "n_latent", "n_latent"]],
    ) -> TT["batch_size"]:
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
        # sigmas = torch.repeat_interleave(sigma, self.n_samples, dim=0)
        sigmas_factorized_interleaved = [
            torch.repeat_interleave(sigma, self.n_samples, dim=0) for sigma in sigma_factorized
        ]
        # z_samples = self.product_manifold.sample(means, sigmas)
        z_samples = self.pm.sample(means, sigmas_factorized_interleaved)
        # log_qz = self.product_manifold.log_likelihood(z_samples, means, sigmas)
        log_qz = self.pm.log_likelihood(z_samples, means, sigmas_factorized_interleaved)
        log_pz = self.pm.log_likelihood(z_samples)
        return (log_qz - log_pz).view(-1, self.n_samples).mean(dim=1)

    def elbo(self, x: TT["batch_size", "n_features"]) -> TT["batch_size"]:
        """
        Computes the Evidence Lower Bound (ELBO).

        Args:
            x (torch.Tensor): Input data of shape (batch_size, n_features).

        Returns:
            tuple: Mean ELBO, mean log-likelihood, and mean KL divergence across the batch.
        """
        # x_reconstructed, z_means, sigmas = self(x)
        # kld = self.kl_divergence(z_means, sigmas)
        x_reconstructed, z_means, sigma_factorized = self(x)
        kld = self.kl_divergence(z_means, sigma_factorized)
        ll = -self.reconstruction_loss(x_reconstructed.view(x.shape[0], -1), x.view(x.shape[0], -1)).sum(dim=1)
        return (ll - self.beta * kld).mean(), ll.mean(), kld.mean()

