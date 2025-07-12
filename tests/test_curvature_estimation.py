import torch

from manify.curvature_estimation.delta_hyperbolicity import sampled_delta_hyperbolicity, vectorized_delta_hyperbolicity, delta_hyperbolicity
from manify.manifolds import ProductManifold


def iterative_delta_hyperbolicity(D, reference_idx=0, relative=True):
    """delta(x,y,z) = min((x,y)_w,(y-z)_w) - (x,z)_w"""
    n = D.shape[0]
    w = reference_idx
    gromov_products = torch.zeros((n, n))
    deltas = torch.zeros((n, n, n))

    # Get Gromov Products
    for x in range(n):
        for y in range(n):
            gromov_products[x, y] = gromov_product(w, x, y, D)

    # Get Deltas
    for x in range(n):
        for y in range(n):
            for z in range(n):
                xz_w = gromov_products[x, z]
                xy_w = gromov_products[x, y]
                yz_w = gromov_products[y, z]
                deltas[x, y, z] = torch.minimum(xy_w, yz_w) - xz_w

    deltas = 2 * deltas / torch.max(D) if relative else deltas

    return deltas, gromov_products


def gromov_product(i, j, k, D):
    """(j,k)_i = 0.5 (d(i,j) + d(i,k) - d(j,k))"""
    return float(0.5 * (D[i, j] + D[i, k] - D[j, k]))


def test_delta_hyperbolicity():
    torch.manual_seed(42)
    pm = ProductManifold(signature=[(-1.0, 2)])
    X, _ = pm.sample(z_mean=torch.vstack([pm.mu0] * 10))
    dists = pm.pdist(X)
    dists_max = dists.max()

    # Iterative deltas
    iterative_deltas, gromov_products = iterative_delta_hyperbolicity(dists, relative=True)
    assert (gromov_products >= 0).all()
    assert (gromov_products <= dists_max).all()
    assert (iterative_deltas <= 1).all(), "Deltas should be in the range [-2, 1]"
    assert (iterative_deltas >= -2).all(), "Deltas should be in the range [-2, 1]"
    assert iterative_deltas.shape == (10, 10, 10)

    # Vectorized deltas
    vectorized_deltas = vectorized_delta_hyperbolicity(dists, full=True, relative=True)
    assert (vectorized_deltas <= 1).all(), "Deltas should be in the range [-2, 1]"
    assert (vectorized_deltas >= -2).all(), "Deltas should be in the range [-2, 1]"
    assert vectorized_deltas.shape == (10, 10, 10)
    assert torch.allclose(vectorized_deltas, iterative_deltas, atol=1e-5), (
        "Vectorized deltas should be close to iterative deltas."
    )

    # Sampled deltas
    sampled_deltas, indices = sampled_delta_hyperbolicity(dists, n_samples=10, relative=True)
    assert (sampled_deltas <= 1).all(), "Sampled deltas should be in the range [-2, 1]"
    assert (sampled_deltas >= -2).all(), "Sampled deltas should be in the range [-2, 1]"
    assert sampled_deltas.shape == (10,), "There should be 10 sampled deltas"
    assert torch.allclose(sampled_deltas, vectorized_deltas[indices[:, 0], indices[:, 1], indices[:, 2]], atol=1e-5), (
        "Sampled deltas should be close to vectorized deltas."
    )

    # Test centralized delta_hyperbolicity function
    # Test global method
    global_delta = delta_hyperbolicity(dists, method="global", relative=True)
    assert isinstance(global_delta, float), "Global method should return a float"
    assert global_delta == vectorized_delta_hyperbolicity(dists, full=False, relative=True), "Global method should match vectorized_delta_hyperbolicity"
    
    # Test full method
    full_delta = delta_hyperbolicity(dists, method="full", relative=True)
    assert torch.allclose(full_delta, vectorized_deltas, atol=1e-5), "Full method should match vectorized result"
    
    # Test sampled method
    sampled_delta_centralized = delta_hyperbolicity(dists, method="sampled", n_samples=10, relative=True)
    assert sampled_delta_centralized.shape == (10,), "Sampled method should return correct shape"
