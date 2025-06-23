"""Preprocessing datasets for link prediction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from beartype.typing import Any
    from jaxtyping import Float, Int

from ..manifolds import ProductManifold


def make_link_prediction_dataset(
    X_embed: Float[torch.Tensor, "batch n_dim"],
    pm: ProductManifold,
    adj: Float[torch.Tensor, "batch batch"],
    add_dists: bool = True,
) -> tuple[Float[torch.Tensor, "batch**2 n_dim*2"], Float[torch.Tensor, "batch**2"], ProductManifold]:
    r"""Preprocess a graph link prediction task into a binary classification problem on a new product manifold.

    This function constructs a dataset for link prediction by creating pairwise embeddings from the input node
    embeddings, optionally appending pairwise distances, and returning labels from an adjacency matrix. It also updates
    the manifold signature correspondingly.

    Args:
        X_embed: Node embeddings.
        pm : The manifold on which the embeddings lie.
        adj: A binary adjacency matrix indicating edges between nodes.
        add_dists: If True, appends pairwise distances to the feature vectors. Default is True.

    Returns:
        X: Node-pair embeddings in $\mathcal{M} \times \mathcal{M}$
        y: Edge labels derived from the adjacency matrix.
        new_pm: A new instance of `ProductManifold` with an updated signature reflecting the feature space
            $\mathcal{M} \times \mathcal{M}$.

    """
    # Stack embeddings
    X = torch.stack([torch.cat([X_i, X_j]) for X_i in X_embed for X_j in X_embed])

    # Add distances
    if add_dists:
        dists = pm.pdist(X_embed)
        X = torch.cat([X, dists.flatten().unsqueeze(1)], dim=1)

    y = adj.flatten()

    # Binarize y
    y = (y > 0).long()

    # Make a new signature
    new_sig = pm.signature + pm.signature
    if add_dists:
        new_sig.append((0.0, 1))
    new_pm = ProductManifold(signature=new_sig)

    return X, y, new_pm


def split_link_prediction_dataset(
    X: Float[torch.Tensor, "n_pairs n_dims"],
    y: Int[torch.Tensor, "n_pairs"],
    test_size: float = 0.2,
    downsample: int | None = None,
    **kwargs: Any,
) -> tuple[
    Float[torch.Tensor, "n_pairs n_dims"],
    Float[torch.Tensor, "n_pairs n_dims"],
    Int[torch.Tensor, "n_pairs"],
    Int[torch.Tensor, "n_pairs"],
    Int[torch.Tensor, "n_pairs"],
    Int[torch.Tensor, "n_pairs"],
]:
    """Split a link prediction dataset into train and test sets in a stratified (non-leaky) manner.

    Args:
        X: Node-pair embeddings.
        y: Edge labels derived from the adjacency matrix.
        test_size: Proportion of the dataset to include in the test split.
        downsample: Optional number of positive and negative samples to retain.
        **kwargs: Additional keyword arguments for train_test_split.

    Returns:
        X_train: Training node-pair embeddings.
        X_test: Testing node-pair embeddings.
        y_train: Training edge labels.
        y_test: Testing edge labels.
        idx_train: Indices of training nodes.
        idx_test: Indices of testing nodes.
    """


def split_link_prediction_dataset(
    X: Float[torch.Tensor, "n_pairs n_dims"],
    y: Int[torch.Tensor, "n_pairs"],
    test_size: float = 0.2,
    downsample: int | None = None,
    random_state: int | None = None,
    **kwargs: Any,
) -> tuple[
    Float[torch.Tensor, "... n_dims"],
    Float[torch.Tensor, "... n_dims"],
    Int[torch.Tensor, "..."],
    Int[torch.Tensor, "..."],
    Int[torch.Tensor, "..."],
    Int[torch.Tensor, "..."],
]:
    """Split a link prediction dataset into train and test sets.

    Args:
        X: Node-pair embeddings of shape (n_nodes^2, n_dims).
        y: Edge labels of shape (n_nodes^2,).
        test_size: Proportion of nodes to include in test set.
        downsample: If provided, downsample to this many pos/neg pairs each.
        random_state: Random seed for reproducibility.
        **kwargs: Additional arguments for train_test_split.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, idx_train, idx_test).
    """
    if random_state is not None:
        torch.manual_seed(random_state)

    n_pairs, n_dims = X.shape
    n_nodes = int(n_pairs**0.5)
    assert n_nodes**2 == n_pairs, f"Expected {n_nodes}^2 = {n_nodes**2} pairs, got {n_pairs}"

    # Downsample if requested (before split to maintain structure)
    if downsample is not None:
        pos_mask = y == 1
        neg_mask = y == 0

        pos_indices = torch.where(pos_mask)[0]
        neg_indices = torch.where(neg_mask)[0]

        # Sample up to 'downsample' examples from each class
        n_pos = min(len(pos_indices), downsample)
        n_neg = min(len(neg_indices), downsample)

        sampled_pos = pos_indices[torch.randperm(len(pos_indices))[:n_pos]]
        sampled_neg = neg_indices[torch.randperm(len(neg_indices))[:n_neg]]

        # Create a mask for selected pairs
        mask = torch.zeros(n_pairs, dtype=torch.bool)
        mask[sampled_pos] = True
        mask[sampled_neg] = True

        # Zero out unselected pairs
        X_filtered = X.clone()
        y_filtered = y.clone()
        X_filtered[~mask] = 0
        y_filtered[~mask] = 0
    else:
        X_filtered = X
        y_filtered = y

    # Reshape to adjacency format
    X_adj = X_filtered.view(n_nodes, n_nodes, n_dims)
    y_adj = y_filtered.view(n_nodes, n_nodes)

    # Split nodes into train/test
    node_indices = torch.arange(n_nodes)
    idx_train, idx_test = train_test_split(node_indices, test_size=test_size, random_state=random_state, **kwargs)

    # Extract train and test subgraphs and flatten
    X_train = X_adj[idx_train][:, idx_train].reshape(-1, n_dims)
    y_train = y_adj[idx_train][:, idx_train].reshape(-1)

    X_test = X_adj[idx_test][:, idx_test].reshape(-1, n_dims)
    y_test = y_adj[idx_test][:, idx_test].reshape(-1)

    return X_train, X_test, y_train, y_test, idx_train, idx_test
