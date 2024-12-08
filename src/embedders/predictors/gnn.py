import torch
from torch_geometric.nn import GCNConv
from torch import nn
from .mlp import MLP

"""Implementation for Graph Neural Network and its utility functions"""
# Create edges for a subset of nodes
def get_subset_edges(dist_matrix, node_indices):
    """
    Create edges for a subset of nodes based on a distance threshold.

    Args:
        dist_matrix (torch.Tensor): A square distance matrix.
        node_indices (torch.Tensor): Indices of nodes to include in the subset.

    Returns:
        torch.Tensor: Edge indices (2xE).
    """
    # Get submatrix of distances
    sub_dist = dist_matrix[node_indices][:, node_indices]

    # Create edges based on threshold
    threshold = sub_dist.mean()
    edges = (sub_dist < threshold).nonzero().t()

    return edges


def get_dense_edges(dist_matrix, node_indices):
    """
    Generate dense (all-to-all) edges with corresponding edge weights.

    Args:
        dist_matrix (torch.Tensor): A square distance matrix.
        node_indices (torch.Tensor): Indices of nodes to include in the subset.

    Returns:
        tuple:
            - torch.Tensor: Edge indices (2xE).
            - torch.Tensor: Edge weights corresponding to the distances.
    """
    # Get submatrix of distances
    sub_dist = dist_matrix[node_indices][:, node_indices]

    # Create dense edges (all-to-all connections)
    n = len(node_indices)
    rows = torch.arange(n, device=dist_matrix.device).repeat_interleave(n)
    cols = torch.arange(n, device=dist_matrix.device).repeat(n)
    edge_index = torch.stack([rows, cols])

    # Get corresponding distances as edge weights
    edge_weights = sub_dist.flatten()

    # Convert distances to weights (you can modify this function)
    edge_weights = torch.exp(-edge_weights)  # Gaussian kernel
    # Alternative weightings:
    # edge_weights = 1 / (edge_weights + 1e-6)  # Inverse distance
    # edge_weights = torch.softmax(-edge_weights, dim=0)  # Softmax of negative distances

    return edge_index, edge_weights


def get_nonzero(adj_matrix, node_indices):
    """
    Retrieve edges corresponding to non-zero values in an adjacency matrix for a subset of nodes.

    Args:
        adj_matrix (torch.Tensor): A square adjacency matrix.
        node_indices (torch.Tensor): Indices of nodes to include in the subset.

    Returns:
        tuple:
            - torch.Tensor: Edge indices (2xE).
            - torch.Tensor: Edge weights corresponding to the non-zero values.
    """
    # Get submatrix of distances
    sub_adj = adj_matrix[node_indices][:, node_indices]

    # Create edges based on threshold
    edges = sub_adj.nonzero().t()

    # Get weights
    edge_weights = sub_adj[sub_adj.nonzero(as_tuple=True)]

    return edges, edge_weights


def get_all(adj_matrix, node_indices):
    """
    Generate all possible edges for a subset of nodes with corresponding weights.

    Args:
        adj_matrix (torch.Tensor): A square adjacency matrix.
        node_indices (torch.Tensor): Indices of nodes to include in the subset.

    Returns:
        tuple:
            - torch.Tensor: Edge indices (2xE).
            - torch.Tensor: Edge weights for all edges.
    """
    n = len(node_indices)
    rows = torch.arange(n, device=adj_matrix.device).repeat_interleave(n)
    cols = torch.arange(n, device=adj_matrix.device).repeat(n)
    edge_index = torch.stack([rows, cols])

    # Get corresponding distances as edge weights
    edge_weights = adj_matrix[node_indices][:, node_indices].flatten()

    return edge_index, edge_weights


class GNN(nn.Module):
    def __init__(
        self,
        pm,
        input_dim=0,
        hidden_dims=None,
        output_dim=0,
        tangent=True,
        task="classification",
        activation=nn.ReLU,
        edge_func=get_dense_edges,
    ):
        super().__init__()
        self.pm = pm
        self.task = task
        self.tangent = tangent
        self.edge_func = edge_func

        # Build architecture
        if hidden_dims is None:
            hidden_dims = []

        dimensions = [input_dim] + hidden_dims + [output_dim]
        layers = []

        # Create layers
        for i in range(len(dimensions) - 1):
            # bias = True if tangent or i > 0 else False  # First layer for non-tangent = no bias
            bias = not tangent and i == 0
            layers.append(GCNConv(dimensions[i], dimensions[i + 1], bias=bias))

            # Add activation function after all layers except the last one
            if i < len(dimensions) - 2:
                layers.append(activation())

        # Register layers as a ModuleList
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Perform a forward pass through the GNN.
    
        Args:
            x (torch.Tensor): Input feature matrix (NxD).
            edge_index (torch.Tensor): Edge indices (2xE).
            edge_weight (torch.Tensor, optional): Edge weights (E).
    
        Returns:
            torch.Tensor: Output features after passing through the network.
        """
        for layer in self.layers:
            if isinstance(layer, GCNConv):
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x)
        return x

    def fit(self, X, y, adj, train_idx, epochs=200, lr=1e-2):
        """
        Train the GNN on the provided data.
    
        Args:
            X (torch.Tensor): Feature matrix.
            y (torch.Tensor): Labels for training nodes.
            adj (torch.Tensor): Adjacency or distance matrix.
            train_idx (torch.Tensor): Indices of nodes for training.
            epochs: Number of training epochs (default=200).
            lr: Learning rate (default=1e-2).
        """
        # Get edges for training set
        train_edges, train_weights = self.edge_func(adj, train_idx)
        X_train = X[train_idx]
        y_train = y[train_idx]

        # This part is the same as MLP.fit()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        if self.task == "classification":
            loss_fn = nn.CrossEntropyLoss()
            y_train = y_train.long()
        elif self.task == "regression":
            loss_fn = nn.MSELoss()
            y_train = y_train.float()
        elif self.task == "link_prediction":
            loss_fn = nn.BCEWithLogitsLoss()
            y_train = y_train.float()

        self.train()
        for i in range(epochs):
            opt.zero_grad()
            y_pred = self(X_train, train_edges, train_weights)
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            opt.step()

    def predict(self, X, adj, test_idx):
        """
        Make predictions using the trained GNN.
    
        Args:
            X (torch.Tensor): Feature matrix (NxD).
            adj (torch.Tensor): Adjacency or distance matrix (NxN).
            test_idx (torch.Tensor): Indices of nodes for testing.
    
        Returns:
            torch.Tensor: Predicted labels or outputs.
        """
        # Get edges for test set
        self.eval()
        # dist_matrix = self.pm.pdist(X).detach()
        test_edges, test_weights = self.edge_func(adj, test_idx)
        X_test = X[test_idx]
        y_pred = self(X_test, test_edges, test_weights)
        # return y_pred.argmax(1).detach()
        if self.task == "classification":
            return y_pred.argmax(1).detach()
        else:
            return y_pred.detach()
