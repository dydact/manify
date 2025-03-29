"""Preprocessing with link prediction"""

from typing import Tuple, Optional
import torch
from jaxtyping import Float, Int
from sklearn.model_selection import train_test_split
from ..manifolds import ProductManifold


def make_link_prediction_dataset(
    X_embed: Float[torch.Tensor, "batch n_dim"],
    pm: ProductManifold,
    adj: Float[torch.Tensor, "batch batch"],
    add_dists: bool = True,
) -> Tuple[Float[torch.Tensor, "batch**2 n_dim*2"], Float[torch.Tensor, "batch**2"], ProductManifold]:
    """
    Generate a dataset for link prediction tasks with product manifold

    This function constructs a dataset for link prediction by creating pairwise
    embeddings from the input node embeddings, optionally appending pairwise
    distances, and returning labels from an adjacency matrix. It also updates the
    manifold signature correspondingly.

    Args:
        X_embed (batch, n_dim): A tensor of node embeddings.
        pm : The manifold on which the embeddings lie.
        adj (batch, batch): A binary adjacency matrix indicating edges between nodes.
        add_dists: If True, appends pairwise distances to the feature vectors. Default is True.

    Returns:
        Tuple[Float[torch.Tensor, "batch**2 2*n_dim"], Float[torch.Tensor, "batch**2"], ProductManifold]:
            - `X` (batch**2, 2*n_dim): A tensor of pairwise embeddings
            - `y` (batch**2,).: A tensor of labels derived from the adjacency matrix with.
            - `new_pm`: A new ProductManifold instance with an updated signature reflecting the feature space.

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
        new_sig.append((0, 1))
    new_pm = ProductManifold(signature=new_sig)

    return X, y, new_pm


def split_dataset(
    X: Float[torch.Tensor, "n_pairs n_dims"],
    y: Int[torch.Tensor, "n_pairs"],
    test_size: float = 0.2,
    downsample: Optional[int] = None,
    **kwargs,
) -> Tuple[
    Float[torch.Tensor, "n_pairs n_dims"],
    Float[torch.Tensor, "n_pairs n_dims"],
    Int[torch.Tensor, "n_pairs"],
    Int[torch.Tensor, "n_pairs"],
]:
    """Split a link prediction dataset into train and test sets"""
    n_pairs, n_dims = X.shape
    n_nodes = int(n_pairs**0.5)

    # Reshape
    X_reshaped = X.view(n_nodes, n_nodes, -1)
    y_reshaped = y.view(n_nodes, n_nodes)

    # Optionally, stratified downsampling according to y
    if downsample is not None:
        n_pos = y_reshaped.sum(dim=1)
        n_neg = y_reshaped.shape[1] - n_pos
        n_pos = torch.min(n_pos, torch.tensor(downsample))
        n_neg = torch.min(n_neg, torch.tensor(downsample))
        pos_idx = torch.where(y_reshaped == 1)[0]
        neg_idx = torch.where(y_reshaped == 0)[0]
        pos_idx = pos_idx[torch.randperm(len(pos_idx))[:n_pos]]
        neg_idx = neg_idx[torch.randperm(len(neg_idx))[:n_neg]]
        idx = torch.cat([pos_idx, neg_idx])
    else:
        idx = torch.arange(n_nodes)

    # Take 20% Of the nodes as test nodes
    idx_train, idx_test = train_test_split(idx, test_size=test_size, **kwargs)

    # Return test and train sets
    X_train = X_reshaped[idx_train][:, idx_train].reshape(-1, n_dims)
    y_train = y_reshaped[idx_train][:, idx_train].reshape(-1)

    X_test = X_reshaped[idx_test][:, idx_test].reshape(-1, n_dims)
    y_test = y_reshaped[idx_test][:, idx_test].reshape(-1)

    return X_train, X_test, y_train, y_test
