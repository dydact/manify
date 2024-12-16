"""Implementation for multilayered perception(MLP)"""
import torch
import torch.nn as nn

class MLP(nn.Module):
    """The MLP class and its functions"""
    def __init__(
        self, pm, input_dim=0, hidden_dims=None, output_dim=0, tangent=True, task="classification", activation=nn.ReLU
    ):
        super().__init__()
        self.pm = pm  # Not used in the basic implementation
        self.task = task

        # Build architecture
        if hidden_dims is None:
            hidden_dims = []

        dimensions = [input_dim] + hidden_dims + [output_dim]
        layers = []

        # Create layers
        for i in range(len(dimensions) - 1):
            # bias = True if tangent or i > 0 else False  # First layer for non-tangent = no bias
            bias = not tangent and i == 0
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1], bias=bias))

            # Add activation function after all layers except the last one
            if i < len(dimensions) - 2:
                layers.append(activation())

        # Register layers as a ModuleList
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Feature in tensor

        Returns:
            x: Result after the forward pass
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def fit(self, X, y, epochs=1_000, lr=1e-2):
        """
        Train the model.
        
        Args:
            X: feature tensor for training the model
            y: labels tensor the training the model
            epochs: number of epochs in training, default to 1000
            lr: learning rate of training, default to 1e-2
        """
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        if self.task == "classification":
            loss_fn = nn.CrossEntropyLoss()
            y = y.long()
        else:
            loss_fn = nn.MSELoss()
            y = y.float()

        self.train()
        for i in range(epochs):
            opt.zero_grad()
            y_pred = self(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            opt.step()

    def predict(self, X):
        """
        Make predictions.

        Args:
            X: feature in tensors for predictions
            
        Returns: 
            self(X).detach(): Result of the prediction
        """
        self.eval()
        with torch.no_grad():
            if self.task == "classification":
                return self(X).argmax(1).detach()
            else:
                return self(X).detach()
