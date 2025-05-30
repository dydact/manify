'''
The Riemannian Fuzzy K-Means algorithm is a clustering algorithm that operates on Riemannian manifolds. 
Compared to a straightforward extension of K-Means or Fuzzy K-Means to Riemannian manifolds, 
it offers significant acceleration while achieving lower loss. For more details, 
please refer to the paper: https://openreview.net/forum?id=9VmOgMN4Ie

If you find this work useful, please cite the paper as follows:


@article{Yuan2025,
  title={Riemannian Fuzzy K-Means},
  author={Anonymous},
  journal={OpenReview},
  year={2025},
  url={https://openreview.net/forum?id=9VmOgMN4Ie}
}

If you have questions about the code, feel free to contact: yuanjinghuiiii@gmail.com.
'''

import torch
from geoopt import ManifoldParameter
from geoopt.optim import RiemannianAdam
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from ..optimizers.radan import RiemannianAdan
from ..manifolds import Manifold, ProductManifold



class RiemannianFuzzyKMeans(BaseEstimator, ClusterMixin):
    """
    Riemannian Fuzzy K-Means.

    param:
    ----------
    n_clusters : int
        The number of clusters to form.
    manifold : Manifold or ProductManifold
        An initialized manifold object (from manifolds.py) on which clustering will be performed.
    m : float, default=2.0
        Fuzzifier parameter. Controls the softness of the partition.
    lr : float, default=0.1
        Learning rate for the optimizer.
    max_iter : int, default=100
        Maximum number of iterations for the optimization.
    tol : float, default=1e-4
        Tolerance for convergence. If the change in loss is less than tol, iteration stops.
    optimizer : {'adan','adam'}, default='adan'
        The optimizer to use for updating cluster centers.
    random_state : int or None, default=None
        Seed for random number generation for reproducibility.
    verbose : bool, default=False
        Whether to print loss information during iterations.
    """
    def __init__(self, n_clusters, manifold, m=2.0, lr=0.1, max_iter=100,
                 tol=1e-4, optimizer='adan',
                 random_state=None, verbose=False):
        self.n_clusters = n_clusters
        self.manifold = manifold 
        self.m = m
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        if optimizer not in ('adan','adam'):
            raise ValueError("optimizer must be 'adan' or 'adam'")
        self.optimizer = optimizer
        self.random_state = random_state
        self.verbose = verbose

    def _init_centers(self, X):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Input data X's second dimension should match the manifold's ambient dimension
        if X.shape[1] != self.manifold.ambient_dim:
            raise ValueError(
                f"Input data X's dimension ({X.shape[1]}) does not match "
                f"the manifold's ambient dimension ({self.manifold.ambient_dim})."
            )

        # Generate initial centers using the manifold's sample method
        # We want n_clusters points, each sampled around the manifold's origin (mu0)
        # The .sample() method in manifolds.py handles z_mean and sigma/sigma_factorized
        # defaulting to mu0 and identity covariances if z_mean or sigma are not fully specified
        # or are set to None in a way that triggers this default.

        # For sampling initial centers, we want n_clusters distinct points.
        # The .sample() method typically takes a z_mean of shape (num_points_to_sample, ambient_dim).
        # If we provide self.manifold.mu0 repeated n_clusters times,
        # it samples n_clusters points, each around mu0.
        means_for_sampling_centers = self.manifold.mu0.repeat(self.n_clusters, 1)
        
        if isinstance(self.manifold, ProductManifold):
            # sigma_factorized should be a list of [n_clusters, M.dim, M.dim] tensors
            # Setting to None will use default identity covariances in .sample()
            centers, _ = self.manifold.sample(
                z_mean=means_for_sampling_centers,
                sigma_factorized=None
            )
        elif isinstance(self.manifold, Manifold):
            # sigma should be a [n_clusters, self.manifold.dim, self.manifold.dim] tensor
            # Setting to None will use default identity covariance in .sample()
            centers, _ = self.manifold.sample(
                z_mean=means_for_sampling_centers,
                sigma=None
            )
        else:
            # Fallback: Randomly select points from X if the manifold type isn't directly supported for sampling
            # This is a common k-means initialization strategy.
            # Ensure X is on the correct device first.
            X_device = X.to(self.manifold.device) # Ensure X is on the manifold's device
            indices = np.random.choice(X_device.shape[0], self.n_clusters, replace=False)
            centers = X_device[indices]
            # Ensure centers are detached if they came from X which might require grad
            centers = centers.detach()


        # IMPORTANT: Use self.manifold.manifold for ManifoldParameter,
        # as self.manifold is our wrapper and self.manifold.manifold is the geoopt object.
        self.mu_ = ManifoldParameter(centers.clone().detach(), manifold=self.manifold.manifold) # Ensure centers are detached
        self.mu_.requires_grad_(True)

        if self.optimizer == 'adan':
            self.opt_ = RiemannianAdan([self.mu_], lr=self.lr, betas=[0.7, 0.999, 0.999])
        else:
            self.opt_ = RiemannianAdam([self.mu_], lr=self.lr, betas=[0.99, 0.999])

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).type(torch.get_default_dtype())
        elif not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.get_default_dtype())

        # Ensure X is on the same device as the manifold
        X = X.to(self.manifold.device)

        if X.shape[1] != self.manifold.ambient_dim:
             raise ValueError(
                f"Input data X's dimension ({X.shape[1]}) in fit() does not match "
                f"the manifold's ambient dimension ({self.manifold.ambient_dim})."
            )

        self._init_centers(X)
        m, tol = self.m, self.tol
        losses = []
        for i in range(self.max_iter):
            self.opt_.zero_grad()
            # self.manifold.dist is implemented in manifolds.py and handles broadcasting
            d = self.manifold.dist(X, self.mu_) # X is (N,D), mu_ is (K,D) -> d is (N,K)
            # Original RFK: d = self.manifold.dist(X.unsqueeze(1), self.mu_.unsqueeze(0))
            # The .dist in manifolds.py uses X[:, None] and Y[None, :], so direct call should work if mu_ is (K,D)

            S = torch.sum(d.pow(-2 / (m - 1)) + 1e-8, dim=1)  # Add epsilon for stability
            loss = torch.sum(S.pow(1 - m))
            loss.backward()
            losses.append(loss.item())
            self.opt_.step()
            if self.verbose:
                print(f"RFK iter {i + 1}, loss={loss.item():.4f}")
            if i > 0 and abs(losses[-1] - losses[-2]) < tol:
                break
        # save the result
        self.losses_ = np.array(losses)
        with torch.no_grad(): # Ensure no gradients are computed for final calculations
            dfin = self.manifold.dist(X, self.mu_) # Re-calculate dist to final centers
            inv = dfin.pow(-2 / (m - 1)) + 1e-8 # Add epsilon
            u_final = inv / (inv.sum(dim=1, keepdim=True) + 1e-8) # Add epsilon
        self.u_ = u_final.detach().cpu().numpy()
        self.labels_ = np.argmax(self.u_, axis=1)
        self.cluster_centers_ = self.mu_.data.clone().detach().cpu().numpy()
        return self

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).type(torch.get_default_dtype())
        elif not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.get_default_dtype())

        # Ensure X is on the same device as the manifold
        X = X.to(self.manifold.device)

        if X.shape[1] != self.manifold.ambient_dim:
            raise ValueError(
                f"Input data X's dimension ({X.shape[1]}) in predict() does not match "
                f"the manifold's ambient dimension ({self.manifold.ambient_dim})."
            )

        if not hasattr(self, 'mu_') or self.mu_ is None:
            raise RuntimeError("The RFK model has not been fitted yet. Call 'fit' before 'predict'.")

        with torch.no_grad():
            dmat = self.manifold.dist(X, self.mu_) # X is (N,D), mu_ is (K,D) -> dmat is (N,K)
            inv = dmat.pow(-2 / (self.m - 1)) + 1e-8 # Add epsilon
            u = inv / (inv.sum(dim=1, keepdim=True) + 1e-8) # Add epsilon
            labels = torch.argmax(u, dim=1).cpu().numpy()
        return labels