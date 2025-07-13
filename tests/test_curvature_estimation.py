import torch

from manify.curvature_estimation.delta_hyperbolicity import delta_hyperbolicity
from manify.curvature_estimation.sectional_curvature import sectional_curvature
from manify.manifolds import ProductManifold


def test_delta_hyperbolicity():
    torch.manual_seed(42)
    pm = ProductManifold(signature=[(-1.0, 2)])
    X, _ = pm.sample(z_mean=torch.stack([pm.mu0] * 10))
    dists = pm.pdist(X)

    # Test sampled method
    sampled_deltas = delta_hyperbolicity(dists, method="sampled", n_samples=10)
    assert sampled_deltas.shape == (10,)
    assert (sampled_deltas <= 1).all()
    assert (sampled_deltas >= -2).all()

    # Test global method
    global_delta = delta_hyperbolicity(dists, method="global")
    assert isinstance(global_delta, float)
    assert -2 <= global_delta <= 1

    # Test full method
    full_deltas = delta_hyperbolicity(dists, method="full")
    assert full_deltas.shape == (10, 10, 10)
    assert (full_deltas <= 1).all()
    assert (full_deltas >= -2).all()


def test_sectional_curvature():
    torch.manual_seed(42)
    n = 8
    # Create simple adjacency matrix (ring graph)
    A = torch.zeros(n, n)
    for i in range(n):
        A[i, (i + 1) % n] = 1
        A[(i + 1) % n, i] = 1
    
    # Create distance matrix (shortest path distances)
    D = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            D[i, j] = min(abs(i - j), n - abs(i - j))

    # Test sampled method
    sampled_curvatures = sectional_curvature(A, D, method="sampled", n_samples=10)
    assert sampled_curvatures.shape == (10,)

    # Test per_node method
    node_curvatures = sectional_curvature(A, D, method="per_node")
    assert node_curvatures.shape == (n,)

    # Test global method
    global_curvature = sectional_curvature(A, D, method="global")
    assert isinstance(global_curvature, float)

