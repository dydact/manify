"""Preprocessing with link prediction"""

from typing import Tuple
import torch
from torchtyping import TensorType as TT
from .manifolds import ProductManifold

def make_link_prediction_dataset(
    X_embed: TT["batch", "n_dim"], pm: ProductManifold, adj: TT["batch", "batch"], add_dists: bool = True
) -> Tuple[TT["batch**2", "2*n_dim"], TT["batch**2"], ProductManifold]:
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
        Tuple[Tensor, Tensor, ProductManifold]: 
            - `X` (batch**2, 2*n_dim): A tensor of pairwise embeddings 
            - `y` (batch**2,).: A tensor of labels derived from the adjacency matrix with.
            - `new_pm`: A new ProductManifold instance with an updated signature reflecting the feature space.

    """
    # Stack embeddings
    # emb = []
    # for X_i in X_embed:
    #     for X_j in X_embed:
    #         joint_embed = torch.cat([X_embed[i], X_embed[j]])
    #         emb.append(joint_embed)
    # embed = [[torch.cat([X_i, X_j]) for X_j in X_embed] for X_i in X_embed]

    # X = torch.stack(emb)
    X = torch.stack([torch.cat([X_i, X_j]) for X_i in X_embed for X_j in X_embed])

    # Add distances
    if add_dists:
        dists = pm.pdist(X_embed)
        X = torch.cat([X, dists.flatten().unsqueeze(1)], dim=1)

    # y = torch.tensor(adj.flatten())
    y = adj.flatten()

    # Make a new signature
    new_sig = pm.signature + pm.signature
    if add_dists:
        new_sig.append((0, 1))
    new_pm = ProductManifold(signature=new_sig)

    return X, y, new_pm
