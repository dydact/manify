import torch
from sklearn.model_selection import train_test_split
from manify.embedders.coordinate_learning import CoordinateLearning
from manify.embedders.siamese import SiameseNetwork
from manify.embedders.vae import ProductSpaceVAE
from manify.utils.dataloaders import load_hf
from manify.manifolds import ProductManifold


def test_train_coords():
    # Load karate club dataset
    _, D, _, _ = load_hf("karate_club")
    pm = ProductManifold(signature=[(-1.0, 2), (0.0, 2), (1.0, 2)])

    # Init embedder
    embedder = CoordinateLearning(pm=pm, random_state=42)

    # Run without train_test_split
    X = embedder.fit_transform(X=None, D=D, burn_in_iterations=10, training_iterations=90)
    losses = embedder.loss_history_
    assert pm.manifold.check_point(X), "Output points should be on the manifold"
    assert losses["train_train"]
    assert not losses["train_test"]
    assert not losses["test_test"]
    assert losses["total"]
    assert not torch.isnan(torch.tensor(losses["total"])).any(), "Losses tensor contains NaN values"
    assert not torch.isinf(torch.tensor(losses["total"])).any(), "Losses tensor contains infinite values"
    assert len(losses["total"]) == 100, "Losses tensor should have the same number of elements as training iterations"
    assert losses["total"][-1] < losses["total"][0], "Loss should go down"

    # Run with train_test_split
    _, test_idx = train_test_split(torch.arange(len(X)))
    X2 = embedder.fit_transform(X=None, D=D, test_indices=test_idx, burn_in_iterations=10, training_iterations=90)
    losses2 = embedder.loss_history_
    assert pm.manifold.check_point(X2), "Output points should be on the manifold"
    assert losses2["train_train"]
    assert losses2["train_test"]
    assert losses2["test_test"]
    assert losses2["total"]
    assert not torch.isnan(torch.tensor(losses2["total"])).any(), "Losses tensor contains NaN values"
    assert not torch.isinf(torch.tensor(losses2["total"])).any(), "Losses tensor contains infinite values"
    assert len(losses2["total"]) == 100, "Losses tensor should have the same number of elements as training iterations"
    assert losses2["total"][-1] < losses2["total"][0], "Loss should go down"


def test_vae():
    # Load MNIST dataset
    X, _, _, _ = load_hf("mnist")
    X = X.reshape(X.shape[0], -1)
    pm = ProductManifold(signature=[(-1.0, 2), (0.0, 2), (1.0, 2)])

    # Init embedder
    encoder = torch.nn.Sequential(torch.nn.Linear(784, 128), torch.nn.ReLU(), torch.nn.Linear(128, 2 * pm.dim))
    decoder = torch.nn.Sequential(torch.nn.Linear(pm.ambient_dim, 128), torch.nn.Linear(128, 784), torch.nn.Sigmoid())
    vae = ProductSpaceVAE(pm=pm, encoder=encoder, decoder=decoder, random_state=42)
    vae.fit(X=X[:64], burn_in_iterations=10, training_iterations=90, batch_size=16, lr=1e-4)

    # Get outputs
    X_embedded = vae.transform(X[:128])
    assert X_embedded.shape == (
        128,
        pm.ambient_dim,
    ), f"Embedded output shape should match expected dimensions: Got {X_embedded.shape}, expected {(128, pm.ambient_dim)}"


def test_siamese():
    # Load qiita dataset
    X, D, _, _ = load_hf("qiita")
    X = X[:128]
    D = D[:128, :128] / D.max()
    pm = ProductManifold(signature=[(-1.0, 2), (0.0, 2), (1.0, 2)])

    # Init embedder
    encoder = torch.nn.Sequential(torch.nn.Linear(152, 64), torch.nn.ReLU(), torch.nn.Linear(64, pm.dim))
    decoder = torch.nn.Sequential(torch.nn.Linear(pm.ambient_dim, 64), torch.nn.Linear(64, 152))
    snn = SiameseNetwork(pm=pm, encoder=encoder, decoder=decoder, random_state=42)
    snn.fit(X=X, D=D, burn_in_iterations=1, training_iterations=9, batch_size=16, lr=1e-4)

    # Get outputs
    X_embedded = snn.transform(X)
    assert X_embedded.shape == (
        128,
        pm.ambient_dim,
    ), f"Embedded output shape should match expected dimensions: Got {X_embedded.shape}, expected {(128, pm.ambient_dim)}"
