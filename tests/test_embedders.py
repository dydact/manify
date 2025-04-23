import torch
from sklearn.model_selection import train_test_split
from manify.embedders.coordinate_learning import train_coords
from manify.embedders.vae import ProductSpaceVAE
from manify.utils.dataloaders import load_hf
from manify.manifolds import ProductManifold


def test_train_coords():
    # Load karate club dataset
    _, dists, adj, _ = load_hf("karate_club")
    pm = ProductManifold(signature=[(-1, 2), (0, 2), (1, 2)])

    # Run without train_test_split
    X, losses = train_coords(pm=pm, dists=dists, burn_in_iterations=10, training_iterations=90)
    assert pm.manifold.check_point(X), "Output points should be on the manifold"
    assert losses["train_train"]
    assert not losses["train_test"]
    assert not losses["test_test"]
    assert losses["total"]
    assert not torch.isnan(torch.tensor(losses["total"])).any(), "Losses tensor contains NaN values"
    assert not torch.isinf(torch.tensor(losses["total"])).any(), "Losses tensor contains infinite values"
    assert len(losses["total"]) == 100, "Losses tensor should have the same number of elements as training iterations"

    # Run with train_test_split
    train_idx, test_idx = train_test_split(torch.arange(len(X)))
    X2, losses = train_coords(pm=pm, dists=dists, test_indices=test_idx, burn_in_iterations=10, training_iterations=90)
    assert pm.manifold.check_point(X2), "Output points should be on the manifold"
    assert losses["train_train"]
    assert losses["train_test"]
    assert losses["test_test"]
    assert losses["total"]
    assert not torch.isnan(torch.tensor(losses["total"])).any(), "Losses tensor contains NaN values"
    assert not torch.isinf(torch.tensor(losses["total"])).any(), "Losses tensor contains infinite values"
    assert len(losses["total"]) == 100, "Losses tensor should have the same number of elements as training iterations"


def test_vae():
    class Encoder(torch.nn.Module):
        def __init__(self, pm):
            super().__init__()
            self.pm = pm
            self.fc1 = torch.nn.Linear(784, 400)
            self.fc2_z_mean = torch.nn.Linear(400, pm.dim)
            self.fc2_z_logvar = torch.nn.Linear(400, pm.dim)

        def forward(self, x):
            h1 = torch.relu(self.fc1(x))
            z_mean_tangent = self.fc2_z_mean(h1)
            z_logvar = self.fc2_z_logvar(h1)
            return z_mean_tangent, z_logvar

    class Decoder(torch.nn.Module):
        def __init__(self, pm):
            super().__init__()
            self.pm = pm
            self.fc1 = torch.nn.Linear(pm.ambient_dim, 400)
            self.fc2 = torch.nn.Linear(400, 784)

        def forward(self, z):
            h1 = torch.relu(self.fc1(z))
            x = torch.sigmoid(self.fc2(h1))
            return x

    pm = ProductManifold(signature=[(-1, 2), (0, 2), (1, 2)])
    X, _, _, _ = load_hf("mnist")
    X = X.reshape(X.shape[0], -1)
    vae = ProductSpaceVAE(encoder=Encoder(pm=pm), decoder=Decoder(pm=pm), pm=pm)
    vae.fit(X_train=X[:64], burn_in_epochs=10, epochs=90, batch_size=16, seed=42)
