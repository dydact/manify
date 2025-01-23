"""
Kappa GCN implementation
"""
import torch
import geoopt
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
    def get_logits(self, man, X, W, b):
        """
        Computes logits given the manifold.
    
        Args:
            man: The manifold object, containing the curvature and relevant operations.
            X: Input points tensor of shape (n, d), where n is the number of points and d is the dimensionality.
            W: Weight tensor of shape (d, k), where k is the number of classes.
            b: Bias tensor of shape (k,).
    
        Returns:
            Logits: tensor of shape (n, k).
        """
        k = W.shape[1]
        n = X.shape[0]
        kappa = man.curvature
        logits = torch.zeros(n, k)
        for class_idx in range(k):
            # 0. Get parameters
            a_k = W[:, class_idx]
            p_k = b[class_idx]
    
            # 1. Let's get z_k = -p_k \oplus_\kappa x
            z_k = man.manifold.mobius_add(-p_k, X)
    
            # 2. Get norms for relevant terms
            z_k_norm = torch.norm(z_k)
            a_k_norm = torch.norm(a_k)
    
            # 3. Get the argument for sin_kappa^-1
            num = 2 * abs(kappa) ** 0.5 * man.manifold.inner(z_k, a_k)
            denom = (1 + kappa * z_k_norm**2) * a_k_norm
            arg = num / denom
    
            # 4. Get the right-hand side
            sin_kappa_inv = torch.asin if kappa >= 0 else torch.asinh
            rhs = sin_kappa_inv(arg)
    
            # 5. Get the left-hand side
            lambda_pk = man.manifold.lambda_x(p_k)
            lhs = lambda_pk * a_k_norm / abs(kappa) ** 0.5
    
            # 6. Get the logits
            logits[:, class_idx] = (lhs * rhs).flatten()
    
        return logits
