from jaxtyping import Float
import torch


def knn_graph(x: Float[torch.Tensor, "n_points n_dim"], k: int) -> Float[torch.Tensor, "n_points n_points"]:
    """Compute the k-nearest neighbor graph from ambient coordinates."""
    raise NotImplementedError
