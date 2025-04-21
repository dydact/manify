import torch
import pytest
from manify.manifolds import Manifold, ProductManifold
import geoopt
import math


def test_manifold_methods():
    for curv, dim in [(-1, 2), (0, 2), (1, 2), (-1, 64), (0, 64), (1, 64)]:
        M = Manifold(curvature=curv, dim=dim)

        # Does device switching work?
        M.to("cpu")

        # Do attributes work correctly?
        if curv < 0:
            assert M.type == "H" and isinstance(M.manifold.base, geoopt.Lorentz)
        elif curv == 0:
            assert M.type == "E" and isinstance(M.manifold.base, geoopt.Euclidean)
        else:
            assert M.type == "S" and isinstance(M.manifold.base, geoopt.Sphere)

        # get some vectors via gaussian mixture
        cov = torch.eye(M.dim) / M.dim / 100
        means = torch.vstack([M.mu0] * 10)
        covs = torch.stack([cov] * 10)
        X1, _ = M.sample(z_mean=means, sigma=covs)
        X2, _ = M.sample(z_mean=means[:5], sigma=covs[:5])

        # Verify points are on manifold
        assert M.manifold.check_point(X1), "X1 is not on the manifold"
        assert M.manifold.check_point(X2), "X2 is not on the manifold"

        # Inner products
        ip_11 = M.inner(X1, X1)
        assert ip_11.shape == (10, 10), "Inner product shape mismatch for X1"
        ip_12 = M.inner(X1, X2)
        assert ip_12.shape == (10, 5), "Inner product shape mismatch for X1 and X2"
        if curv == 0:
            assert torch.allclose(ip_11, X1 @ X1.T, atol=1e-5), "Euclidean inner products do not match for X1"
            assert torch.allclose(ip_12, X1 @ X2.T, atol=1e-5), "Euclidean inner products do not match for X1 and X2"

        # Dists
        dists_11 = M.dist(X1, X1)
        assert dists_11.shape == (10, 10), "Distance shape mismatch for X1"
        dists_12 = M.dist(X1, X2)
        assert dists_12.shape == (10, 5), "Distance shape mismatch for X1 and X2"
        if curv == 0:
            assert torch.allclose(
                dists_12, torch.linalg.norm(X1[:, None] - X2[None, :], dim=-1)
            ), "Euclidean distances do not match for X1 and X2"
            assert torch.allclose(
                dists_11, torch.linalg.norm(X1[:, None] - X1[None, :], dim=-1)
            ), "Euclidean distances do not match for X1"
        assert (dists_11.triu(1) >= 0).all(), "Distances for X1 should be non-negative"
        assert (dists_12.triu(1) >= 0).all(), "Distances for X2 should be non-negative"
        assert torch.allclose(dists_11.triu(1), M.pdist(X1).triu(1), atol=1e-5), "dist and pdist diverge for X1"

        # Square dists
        sqdists_11 = M.dist2(X1, X1)
        assert sqdists_11.shape == (10, 10), "Squared distance shape mismatch for X1"
        sqdists_12 = M.dist2(X1, X2)
        assert sqdists_12.shape == (10, 5), "Squared distance shape mismatch for X1 and X2"
        if curv == 0:
            assert torch.allclose(
                sqdists_12, torch.linalg.norm(X1[:, None] - X2[None, :], dim=-1) ** 2
            ), "Euclidean squared distances do not match for X1 and X2"
            assert torch.allclose(
                sqdists_11, torch.linalg.norm(X1[:, None] - X1[None, :], dim=-1) ** 2
            ), "Euclidean squared distances do not match for X1"
        assert (sqdists_11.triu(1) >= 0).all(), "Squared distances for X1 should be non-negative"
        assert (sqdists_12.triu(1) >= 0).all(), "Squared distances for X1 and X2 should be non-negative"
        assert torch.allclose(
            sqdists_11.triu(1), M.pdist2(X1).triu(1), atol=1e-5
        ), "sqdists_11 and pdist2 diverge for X1"

        # Log-likelihood
        lls = M.log_likelihood(X1)
        if curv == 0:
            # Evaluate as ll of gaussian with mean 0, variance 1:
            assert torch.allclose(
                lls,
                -0.5 * (torch.sum(X1**2, dim=-1) + X1.size(-1) * math.log(2 * math.pi)),
                atol=1e-5,
            ), "Log-likelihood mismatch for Gaussian"
        assert (lls <= 0).all(), "Log-likelihood should be non-positive"

        # Logmap and expmap
        logmap_x1 = M.logmap(X1)
        assert M.manifold.check_vector(logmap_x1), "Logmap point should be in the tangent plane"
        expmap_x1 = M.expmap(logmap_x1)
        assert M.manifold.check_point(expmap_x1), "Expmap point should be on the manifold"
        assert torch.allclose(expmap_x1, X1, atol=1e-5), "Expmap does not return the original points"

        # Stereographic conversions
        M_stereo, X1_stereo, X2_stereo = M.stereographic(X1, X2)
        assert M_stereo.is_stereographic
        X_inv_stereo, X1_inv_stereo, X2_inv_stereo = M_stereo.inverse_stereographic(X1_stereo, X2_stereo)
        assert not X_inv_stereo.is_stereographic
        assert torch.allclose(X1_inv_stereo, X1, atol=1e-5), "Inverse stereographic conversion mismatch for X1"
        assert torch.allclose(X2_inv_stereo, X2, atol=1e-5), "Inverse stereographic conversion mismatch for X2"

        # Apply
        @M.apply
        def apply_function(x):
            return torch.nn.functional.relu(x)

        result = apply_function(X1)
        assert result.shape == X1.shape, "Result shape mismatch for apply_function"
        assert M.manifold.check_point(result)
