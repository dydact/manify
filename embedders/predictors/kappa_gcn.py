"""
Kappa GCN implementation
"""
import torch
import geoopt
from geoopt.manifolds.stereographic.math import arsin_k


# A kappa-GCN layer
class KappaGCNLayer(torch.nn.Module):
    """
    Implementation for the Kappa GCN layer

    Parameters
    ----------
    in_features: Number of input features
    out_features: Number of output features
    manifold: Manifold object for the Kappa GCN
    nonlinearity: Function for nonlinear activation.
    """

    def __init__(self, in_features, out_features, manifold, nonlinearity=torch.relu):
        super().__init__()

        # Parameters are Euclidean, straightforardly
        self.W = torch.nn.Parameter(torch.rand(in_features, out_features))
        # self.b = torch.nn.Parameter(torch.rand(out_features))

        # Noninearity must be applied via wrapper
        if nonlinearity is None:
            self.sigma = lambda x: x
        else:

            @manifold.apply
            def sigma(x):
                return nonlinearity(x)

            self.sigma = sigma

        # Also store manifold
        self.manifold = manifold

    def left_multiply(self, A, X):
        """
        Implementation for Kappa left matrix multiplication for message passing in product space

        Args:
            A: Adjacency matrix of the graph
            X: Embedding matrix of the graph.

        Returns:
            out: result of the Kappa left matrix multiplication.
        """
        out = torch.zeros_like(X)
        for i, (A_i, X_i) in enumerate(zip(A, X)):
            m_i = self.manifold.manifold.weighted_midpoint(xs=X_i, weights=A_i)
            out[i] = self.manifold.manifold.mobius_scalar_mul(r=A_i.sum(), x=m_i)
        return out

    def forward(self, X, A_hat):
        """
        Forward pass for the Kappa GCN layer.

        Args:
            X: Embedding matrix
            A_hat: Normalized adjacency matrix

        Returns:
            AXW: Transformed node features after message passing and nonlinear activation.
        """
        # 1. right-multiply X by W - mobius_matvec broadcasts correctly (verified)
        XW = self.manifold.manifold.mobius_matvec(m=self.W, x=X)

        # 2. left-multiply (X @ W) by A_hat - we need our own implementation for this
        AXW = self.left_multiply(A_hat, XW)

        # 3. Apply nonlinearity - note that sigma is wrapped with our manifold.apply decorator
        AXW = self.sigma(AXW)

        return AXW


class KappaGCN(torch.nn.Module):
    """
    Implementation for the Kappa GCN

    Parameters
    ----------
    in_features: Number of input features
    out_features: Number of output features
    manifold: Manifold object for the Kappa GCN
    nonlinearity: Function for nonlinear activation.
    """

    def __init__(self, in_features, out_features, manifold, nonlinearity=torch.relu):
        super().__init__()
        self.layer = KappaGCNLayer(in_features, in_features, manifold, nonlinearity)
        self.manifold = manifold

        # Final layer params
        self.W_logits = torch.nn.Parameter(torch.rand(in_features, out_features))
        self.p_ks = geoopt.ManifoldParameter(
            manifold.expmap(torch.rand(out_features, in_features), base=None), manifold=manifold.manifold
        )

    def forward(self, X, A_hat, softmax=False):
        """
        Forward pass for the Kappa GCN.

        Args:
            X: Embedding matrix
            A_hat: Normalized adjacency matrix
            softmax: boolean for whether to use softmax function

        Returns:
            logits_agg: output of Kappa GCN network
        """
        H0 = X
        H1 = self.layer(H0, A_hat)
        logits = self.get_logits(self.manifold, H1, self.W_logits, self.p_ks)
        logits_agg = A_hat @ logits
        if softmax:
            return torch.softmax(logits_agg, dim=1)
        else:
            return logits_agg

    def get_logits(self, X, W, b, return_inner_products=False):
        """
        Computes logits given the manifold.

        Credit to the Curve Your Attention paper for an implementation we referenced:
        https://openreview.net/forum?id=AN5uo4ByWH

        Args:
            X: Input points tensor of shape (n, d), where n is the number of points and d is the dimensionality.
            W: Weight tensor of shape (d, k), where k is the number of classes.
            b: Bias tensor of shape (k,).
            return_inner_products: If True, returns the inner products between the weight vectors and the input points.

        Returns:
            Logits: tensor of shape (n, k).
        """

        # For convenience, get curvature and manifold
        M = self.manifold
        kappa = torch.tensor(M.curvature, dtype=X.dtype, device=X.device)

        # Change shapes
        b = b[None, :]  # (1, k)
        X = X[:, None]  # (n, 1, d)

        # Need transposes because column vectors live in the tangent space
        # W = M.manifold.transp0(b, W.T).T  # (d, k)

        # 1. Get z_k = -p_k \oplus_\kappa x (vectorized)
        # This works for the Euclidean case too
        z_ks = M.manifold.mobius_add(-b, X)  # (n, k, d)
        # z_ks = M.manifold.projx(z_ks)  # (n, k, d)

        # 2. Get norms for relevant terms
        z_k_norms = torch.norm(z_ks, dim=-1).clamp_min(1e-10)  # (n, k)
        a_k_norms = torch.norm(W, dim=0).clamp_min(1e-10)  # (k,)

        # 3. Get the distance to the hyperplane
        za = torch.einsum("nkd,dk->nk", z_ks, W)  # (n, k)

        # 4. Get the logits
        if abs(kappa) < 1e-4:
            # Euclidean case: it's just a dot product
            logits = 4 * za
        else:
            # Non-Euclidean case: need to do the arsinh
            dist = 2 * za / ((1 + kappa * z_k_norms**2) * a_k_norms)
            dist = geoopt.manifolds.stereographic.math.arsin_k(dist, kappa * abs(kappa))

            # Get the coefficients
            lambda_pks = M.manifold.lambda_x(b)  # (k,)
            coeffs = lambda_pks * a_k_norms  # / abs(kappa) ** 0.5
            logits = coeffs * dist

        if return_inner_products:
            return logits, za
        else:
            return logits
