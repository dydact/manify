"""Neural network layers for product manifolds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geoopt
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from beartype.typing import Callable
    from jaxtyping import Float

from ...manifolds import Manifold, ProductManifold


class KappaGCNLayer(torch.nn.Module):
    """Implementation for the Kappa GCN layer.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        manifold: Manifold object for the Kappa GCN
        nonlinearity: Function for nonlinear activation.

    Attributes:
        W: Weight matrix parameter.
        sigma: Nonlinear activation function applied via the manifold.
        manifold: The manifold object for geometric operations.
    """

    def __init__(
        self, in_features: int, out_features: int, manifold: Manifold, nonlinearity: Callable | None = torch.relu
    ):
        super().__init__()

        # Parameters are Euclidean, straightforardly
        # self.W = torch.rand(in_features, out_features)
        self.W = torch.nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        # self.b = torch.nn.Parameter(torch.rand(out_features))

        # Noninearity must be applied via the manifold
        if nonlinearity is None:
            self.sigma = lambda x: x
        else:
            self.sigma = lambda x: manifold.expmap(nonlinearity(manifold.logmap(x)))

        # Also store manifold
        self.manifold = manifold

    def _left_multiply(
        self, A: Float[torch.Tensor, "n_nodes n_nodes"], X: Float[torch.Tensor, "n_nodes dim"], M: Manifold
    ) -> Float[torch.Tensor, "n_nodes dim"]:
        r"""$\kappa$-left matrix multiply two matrices $\mathbf{A}$ and $\mathbf{X}$.

        $$\mathbf{A} \boxtimes_\kappa \mathbf{X}$$

        Args:
            A: Adjacency matrix of the graph
            X: Embedding matrix of the graph.
            M: Manifold object for the Kappa GCN - need to specify in case we're going by component

        Returns:
            out: result of the Kappa left matrix multiplication.
        """
        # Vectorized version:
        return M.manifold.weighted_midpoint(
            xs=X.unsqueeze(0),  # (1, N, D)
            weights=A,  # (N, N)
            reducedim=[1],  # Sum over the N points dimension (dim 1)
            dim=-1,  # Compute conformal factors along the points dimension
            keepdim=False,  # Squeeze the batch dimension out
            lincomb=True,  # Scale by sum of weights (A.sum(dim=1))
            posweight=False,
        )

    def forward(
        self, X: Float[torch.Tensor, "n_nodes dim"], A_hat: Float[torch.Tensor, "n_nodes n_nodes"] | None = None
    ) -> Float[torch.Tensor, "n_nodes dim"]:
        """Forward pass for the Kappa GCN layer.

        Args:
            X: Embedding matrix
            A_hat: Normalized adjacency matrix

        Returns:
            AXW: Transformed node features after message passing and nonlinear activation.
        """
        # 1. right-multiply X by W - mobius_matvec broadcasts correctly (verified)
        XW = self.manifold.manifold.mobius_matvec(m=self.W, x=X)

        # 2. left-multiply (X @ W) by A_hat - we need our own implementation for this
        if A_hat is None:
            AXW = XW
        elif isinstance(self.manifold, ProductManifold):
            XWs = self.manifold.factorize(XW)
            AXW = torch.hstack([self._left_multiply(A_hat, XW, M) for XW, M in zip(XWs, self.manifold.P)])
        else:
            AXW = self._left_multiply(A_hat, XW, self.manifold)

        # 3. Apply nonlinearity - note that sigma is wrapped with our manifold.apply decorator
        AXW = self.sigma(AXW)

        return AXW


class KappaSequential(nn.Module):
    """Sequential container for κ-layers that properly handles adjacency matrices.

    Similar to nn.Sequential but passes the adjacency matrix through each layer.
    All layers should accept (X, A_hat) and return X.

    Args:
        *layers: Variable number of layers to be added to the sequence.
        Each layer should be a subclass of nn.Module that implements a forward method accepting (X, A_hat).
    """

    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self, X: Float[torch.Tensor, "n_nodes dim"], A_hat: Float[torch.Tensor, "n_nodes n_nodes"] | None = None
    ) -> Float[torch.Tensor, "n_nodes out_dim"]:
        """Forward pass through all layers.

        Args:
            X: Input features
            A_hat: Adjacency matrix passed to each layer

        Returns:
            Output after passing through all layers
        """
        for layer in self.layers:
            X = layer(X, A_hat)
        return X

    def append(self, layer: nn.Module) -> None:
        """Add a layer to the end of the sequence."""
        self.layers.append(layer)

    def __len__(self) -> int:
        """Return the number of layers in the sequence.

        Returns:
            int: Number of layers in the KappaSequential.
        """
        return len(self.layers)

    def __getitem__(self, idx: int) -> nn.Module:
        """Get a layer by index.

        Args:
            idx: Index of the layer to retrieve

        Returns:
            nn.Module: The layer at the specified index.
        """
        return self.layers[idx]


class StereographicLogits(nn.Module):
    """Stereographic logits layer for classification and regression on product manifolds.

    Computes signed distances from hyperplanes in the product manifold space.
    Can optionally apply softmax for classification tasks.

    Args:
        out_features: Number of output classes (dimensionality of output space)
        manifold: Manifold or ProductManifold object defining the geometry
        apply_softmax: Whether to apply softmax to the output logits (default: False)
    """

    def __init__(self, out_features: int, manifold: Manifold | ProductManifold, apply_softmax: bool = False):
        super().__init__()

        self.out_features = out_features
        self.manifold = manifold
        self.apply_softmax = apply_softmax

        # Weight matrix (Euclidean parameters)
        self.W = nn.Parameter(torch.randn(manifold.dim, out_features) * 0.01)

        # Bias points on the manifold
        self.p_ks = geoopt.ManifoldParameter(torch.zeros(out_features, manifold.dim), manifold=manifold.manifold)

    def _get_logits_single_manifold(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        W: Float[torch.Tensor, "dim n_classes"],
        b: Float[torch.Tensor, "n_classes dim"],
        M: Manifold,
        return_inner_products: bool = False,
    ) -> (
        tuple[Float[torch.Tensor, "n_nodes n_classes"], Float[torch.Tensor, "n_nodes n_classes"]]
        | Float[torch.Tensor, "n_nodes n_classes"]
    ):
        """Compute logits for a single manifold."""
        kappa = torch.tensor(M.curvature, dtype=X.dtype, device=X.device)

        # Change shapes
        b = b[None, :]  # (1, k)
        X = X[:, None]  # (n, 1, d)

        # 1. Get z_k = -p_k ⊕_κ x (vectorized)
        z_ks = M.manifold.mobius_add(-b, X)  # (n, k, d)

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
            coeffs = lambda_pks * a_k_norms
            logits = coeffs * dist

        if return_inner_products:
            return logits, za
        else:
            return logits

    def _get_logits_product_manifold(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        W: Float[torch.Tensor, "dims n_classes"],
        b: Float[torch.Tensor, "n_classes dim"],
        M: ProductManifold,
    ) -> Float[torch.Tensor, "n_nodes n_classes"]:
        """Helper function for get_logits."""
        # For convenience, get curvature and manifold
        # kappas = [man.curvature for manifold in M.P]
        Xs = M.factorize(X)
        bs = M.factorize(b)
        Ws = [w.T for w in M.factorize(W.T)]
        res = [
            self._get_logits_single_manifold(X_man, W_man, b_man, man, return_inner_products=True)
            for X_man, W_man, b_man, man in zip(Xs, Ws, bs, M.P)
        ]

        # Each result is (n, k) logits and (n, k) inner products
        logits, inner_products = zip(*res)

        # Final logits: l2 norm of logits * sign of inner product
        stacked_logits = torch.stack(logits, dim=2)  # (n, k, m)
        stack_products = torch.stack(inner_products, dim=2)  # (n, k, m)

        # Reduce
        logits = torch.norm(stacked_logits, dim=2)  # (n, k)
        signs = torch.sign(stack_products.sum(dim=2))  # (n, k)

        return logits * signs

    def forward(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        A_hat: Float[torch.Tensor, "n_nodes n_nodes"] | None = None,
        aggregate_logits: bool = False,
    ) -> Float[torch.Tensor, "n_nodes n_classes"]:
        """Forward pass through stereographic logits.

        Args:
            X: Input features
            A_hat: Optional adjacency matrix for logit aggregation
            aggregate_logits: Whether to aggregate logits using adjacency matrix

        Returns:
            Logits (or probabilities if apply_softmax=True)
        """
        # Compute logits based on manifold type
        if isinstance(self.manifold, ProductManifold):
            logits = self._get_logits_product_manifold(X, self.W, self.p_ks, self.manifold)
        else:
            logits = self._get_logits_single_manifold(X, self.W, self.p_ks, self.manifold, return_inner_products=False)

        # Optional aggregation via adjacency matrix
        if A_hat is not None and aggregate_logits:
            logits = A_hat @ logits

        # Optional softmax for classification
        if self.apply_softmax:
            logits = torch.softmax(logits, dim=-1)

        return logits


class FermiDiracDecoder(nn.Module):
    """Fermi-Dirac decoder for link prediction tasks.

    Computes pairwise distances and applies Fermi-Dirac transformation
    to predict edge probabilities.

    Args:
        manifold: Manifold or ProductManifold object defining the geometry
        learnable_params: If True, temperature and bias are learnable parameters. If False, they are fixed to 1.0 and
            0.0, respectively.
    """

    def __init__(self, manifold: Manifold | ProductManifold, learnable_params: bool = True):
        super().__init__()

        self.manifold = manifold

        if learnable_params:
            self.temperature = nn.Parameter(torch.tensor(1.0))
            self.bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("temperature", torch.tensor(1.0))
            self.register_buffer("bias", torch.tensor(0.0))

    def forward(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        A_hat: Float[torch.Tensor, "n_nodes n_nodes"] | None = None,
    ) -> Float[torch.Tensor, "n_nodes n_nodes"]:
        """Forward pass through Fermi-Dirac decoder.

        Args:
            X: Node embeddings
            A_hat: Ignored (for compatibility)
            return_pairwise: If True, return full pairwise matrix. If False, return flattened upper triangle.

        Returns:
            Edge probabilities (logits, apply sigmoid if needed)
        """
        # Compute pairwise distances
        pairwise_dist = self.manifold.pdist2(X)

        # Apply Fermi-Dirac transformation
        logits = -(pairwise_dist - self.bias) / self.temperature

        return logits


class StereographicLayerNorm(nn.Module):
    """Stereographic Layer Normalization."""

    def __init__(self, manifold: Manifold | ProductManifold, num_heads: int):
        raise NotImplementedError

    def forward(self, X: Float[torch.Tensor, "n_nodes dim"]) -> Float[torch.Tensor, "n_nodes dim"]:
        """Apply layer normalization on the stereographic manifold."""
        raise NotImplementedError


class StereographicAttention(nn.Module):
    """Stereographic Attention Layer."""

    def __init__(self, manifold: Manifold | ProductManifold, num_heads: int):
        raise NotImplementedError

    def forward(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        mask: Float[torch.Tensor, "n_nodes n_nodes"] | None = None,
    ) -> Float[torch.Tensor, "n_nodes dim"]:
        """Forward pass for the stereographic attention layer."""
        raise NotImplementedError


class StereographicTransformer(nn.Module):
    """Stereographic Transformer Block."""

    def __init__(self, manifold: Manifold | ProductManifold, num_blocks: int, num_heads: int):
        raise NotImplementedError

    def forward(
        self,
        X: Float[torch.Tensor, "n_nodes dim"],
        mask: Float[torch.Tensor, "n_nodes n_nodes"] | None = None,
    ) -> Float[torch.Tensor, "n_nodes dim"]:
        """Forward pass through the stereographic transformer block."""
        raise NotImplementedError
