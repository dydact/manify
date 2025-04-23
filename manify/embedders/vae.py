"""Variational autoencoder implementation for product manifold spaces.

This module provides a variational autoencoder (VAE) implementation specifically designed for learning representations
in mixed-curvature product spaces. The implementation handles the complexities of sampling, KL divergence calculation,
and reparameterization in curved spaces, supporting combinations of hyperbolic, Euclidean, and spherical geometries
within a single latent space.

For more information, see Skopek et al (2020): Mixed Curvature Variational Autoencoders
"""

from __future__ import annotations

import sys
from typing import List, Optional, Tuple

import torch
from jaxtyping import Float

from ..manifolds import ProductManifold

# TQDM: notebook or regular
if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class ProductSpaceVAE(torch.nn.Module):
    r"""Product Space Variational Autoencoder.

    The probabilistic model is defined as:

    - Prior: $p(z) = \mathcal{WN}(z; \mu_0, I)$ (wrapped normal distribution centered at manifold origin)
    - Likelihood: $p_\theta(x|z) = \mathcal{N}(x; f_\theta(z), \sigma^2 I)$ or other reconstruction distribution
    - Posterior approximation: $q_\phi(z|x) = \mathcal{WN}(z; \mu_\phi(x), \Sigma_\phi(x))$

    where $\mathcal{WN}$ is a wrapped normal distribution on the manifold.

    The model is trained by maximizing the evidence lower bound (ELBO):

    $\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot D_{KL}(q_\phi(z|x) || p(z))$

    Attributes:
        encoder: Neural network that outputs mean and log-variance parameters.
        decoder: Neural network that reconstructs inputs from latent embeddings.
        pm: Product manifold defining the structure of the latent space.
        beta: Weight for the KL divergence term in the ELBO. Defaults to 1.0.
        device: Device for tensor computations. Defaults to "cpu".
        n_samples: Number of samples for Monte Carlo estimation of KL divergence.
        reconstruction_loss: Type of reconstruction loss to use.

    Args:
        encoder: Neural network module that produces mean (first half of output) and log-variance (second half of
            output) of the posterior distribution. The output dimension should match twice the intrinsic dimension of
            the product manifold.
        decoder: Neural network module that maps latent representations back to the input space.
        pm: Product manifold defining the structure of the latent space.
        beta: Weight of the KL divergence term in the ELBO loss. Values < 1 give a $\beta$-VAE with a looser constraint
            on the latent space. Defaults to 1.0.
        reconstruction_loss: Type of reconstruction loss to use. Currently only "mse" (mean squared error) is supported.
            Defaults to "mse".
        device: Device for tensor computations. Defaults to "cpu".
        n_samples: Number of Monte Carlo samples to use when estimating the KL divergence. Defaults to 16.

    Raises:
        ValueError: If an unsupported reconstruction_loss is specified.
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

    def encode(
        self, x: Float[torch.Tensor, "batch_size n_features"]
    ) -> Tuple[Float[torch.Tensor, "batch_size n_latent"], Float[torch.Tensor, "batch_size n_latent"]]:
        r"""Encodes input data to obtain latent means and log-variances in the manifold.

        This method processes input data through the encoder network to obtain parameters of the approximate posterior
        distribution $q(z|x)$ in the product manifold space. For non-Euclidean components, the method:

        1. Gets tangent space vectors and log-variances from the encoder,
        2. Projects tangent vectors to the ambient space by adding zeros in the right places, and
        3. Maps the ambient space vectors to the manifold using the exponential map

        Args:
            x: Input data tensor.

        Returns:
            z_mean: Mean of the posterior distribution in the manifold space.
            z_logvar: Log-variance of the posterior distribution, used for constructing the covariance matrices.
        """
        z_mean_tangent, z_logvar = self.encoder(x)
        z_mean_ambient = z_mean_tangent @ self.pm.projection_matrix  # Adds zeros in the right places
        z_mean = self.pm.expmap(u=z_mean_ambient, base=None)
        return z_mean, z_logvar

    def decode(self, z: Float[torch.Tensor, "batch_size n_latent"]) -> Float[torch.Tensor, "batch_size n_features"]:
        """Decodes latent points from the manifold space back to the input space.

        Takes points from the product manifold latent space and passes them through
        the decoder network to reconstruct the original input data.

        Args:
            z: Latent points in the product manifold, with shape (batch_size, n_latent).

        Returns:
            reconstructed: Tensor containing the reconstructed input data,
                with shape (batch_size, n_features).
        """
        return self.decoder(z)

    def forward(self, x: Float[torch.Tensor, "batch_size n_features"]) -> Tuple[
        Float[torch.Tensor, "batch_size n_features"],
        Float[torch.Tensor, "batch_size n_latent"],
        List[Float[torch.Tensor, "n_latent n_latent"]],
    ]:
        r"""Performs the forward pass of the VAE in product manifold space.

        This method implements the complete VAE forward pass:

        1. Encode the input to get posterior parameters (`z_means`, `z_logvars`)
        2. Factorize the log-variances for each manifold component
        3. Convert log-variances to covariance matrices (adding a small epsilon for numerical stability)
        4. Sample points from the posterior distributions in the product manifold
        5. Decode the sampled points to get reconstructions

        Args:
            x: Input data tensor.

        Returns:
            x_reconstructed: Reconstructed data tensor with the same shape as the input.
            z_means: Means of the posterior distributions in the manifold space.
            sigmas: List of covariance matrices for each manifold component.
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
        r"""Computes the KL divergence between posterior and prior distributions in the manifold.

        For distributions in Riemannian manifolds, computing the KL divergence analytically
        is often intractable. This method uses Monte Carlo sampling to approximate the KL divergence:

        $$D_{KL}(q(z|x) || p(z)) \approx \frac{1}{N} \sum_{i=1}^{N} [\log q(z_i|x) - \log p(z_i)]$$

        where $z_i$ are samples from $q(z|x)$.

        This implementation follows the approach described in:
        http://joschu.net/blog/kl-approx.html

        Args:
            z_mean: Means of the posterior distributions in the manifold.
            sigma_factorized: List of covariance matrices for each manifold component.

        Returns:
            kl_divergence: KL divergence values for each data point in the batch.
        """
        # Get KL divergence as the average of log q(z|x) - log p(z)
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
        r"""Computes the Evidence Lower Bound (ELBO) for the VAE objective.

        The ELBO is the standard objective function for variational autoencoders, consisting of a reconstruction term
        (log-likelihood) and a regularization term (KL divergence):

        $$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot D_{KL}(q_\phi(z|x) || p(z)),$$

        where:

        - $\theta$ are the decoder parameters
        - $\phi$ are the encoder parameters
        - $\beta$ is a weight for the KL term (setting $\beta < 1$ creates a $\beta$-VAE)

        Args:
            x: Input data tensor.

        Returns:
            elbo: Mean ELBO value across the batch (higher is better).
            log_likelihood: Mean reconstruction log-likelihood across the batch.
            kl_divergence: Mean KL divergence across the batch.
        """
        x_reconstructed, z_means, sigma_factorized = self(x)
        kld = self.kl_divergence(z_means, sigma_factorized)
        ll = -self.reconstruction_loss(x_reconstructed.view(x.shape[0], -1), x.view(x.shape[0], -1)).sum(dim=1)
        return (ll - self.beta * kld).mean(), ll.mean(), kld.mean()

    def _grads_ok(self) -> bool:
        """Checks if all gradients are valid (no NaN or Inf values).

        This is a helper method used during training to ensure numerical stability. It checks each parameter's gradient
        for NaN or Inf values and reports any issues.

        Returns:
            valid: True if all gradients are valid, False otherwise.
        """
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
        seed: Optional[int] = None,
        lr: float = 1e-3,
        curvature_lr: float = 1e-4,
        clip_grad: bool = True,
    ) -> List[float]:
        """Trains the VAE model on the provided data.

        The training process consists of two phases:

        1. Burn-in phase: Initial training with a lower learning rate for stability
        2. Main training phase: Training with the full learning rate and optional curvature optimization

        Training uses Adam optimizer with gradient clipping to prevent exploding gradients. During training, the model
        maximizes the Evidence Lower Bound (ELBO).

        Args:
            X_train: Training data tensor.
            burn_in_epochs: Number of initial training epochs with reduced learning rate. Defaults to 100.
            epochs: Number of main training epochs. Defaults to 1900.
            batch_size: Number of samples per mini-batch. Defaults to 32.
            seed: Random seed for reproducibility. Defaults to None.
            lr: Learning rate for network parameters. Defaults to 1e-3.
            curvature_lr: Learning rate for manifold curvature parameters. Defaults to 1e-4.
            clip_grad: Whether to apply gradient clipping. Defaults to True.

        Returns:
            losses: List of loss values recorded during training.
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
