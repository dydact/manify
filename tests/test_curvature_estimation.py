import torch

from manify.curvature_estimation._pipelines import (
    distortion_pipeline,
    predictor_pipeline,
)
from manify.curvature_estimation.delta_hyperbolicity import delta_hyperbolicity
from manify.curvature_estimation.greedy_method import greedy_signature_selection
from manify.curvature_estimation.sectional_curvature import sectional_curvature
from manify.manifolds import ProductManifold
from manify.utils.dataloaders import load_hf


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
    X = pm.sample(10)
    dists = pm.pdist(X)

    # Iterative deltas
    iterative_deltas, gromov_products = iterative_delta_hyperbolicity(dists, relative=True)
    assert (gromov_products >= 0).all()
    assert (gromov_products <= dists.max()).all()
    assert (iterative_deltas <= 1).all(), "Deltas should be in the range [-2, 1]"
    assert (iterative_deltas >= -2).all(), "Deltas should be in the range [-2, 1]"
    assert iterative_deltas.shape == (10, 10, 10)

    # Test sampled method
    sampled_deltas = delta_hyperbolicity(dists, samples=10)
    assert sampled_deltas.shape == (10,)
    assert (sampled_deltas <= 1).all()
    assert (sampled_deltas >= -2).all()

    # Test global method (user calls .max() themselves)
    full_deltas = delta_hyperbolicity(dists)
    global_delta = full_deltas.max().item()
    assert isinstance(global_delta, float)
    assert -2 <= global_delta <= 1

    # Test full method
    assert full_deltas.shape == (10, 10, 10)
    assert (full_deltas <= 1).all()
    assert (full_deltas >= -2).all()

    assert torch.allclose(full_deltas, iterative_deltas, atol=1e-5), (
        "Vectorized deltas should be close to iterative deltas."
    )
    assert torch.isclose(full_deltas.max(), torch.tensor(global_delta), atol=1e-5), (
        "Maximum of vectorized deltas should match global delta."
    )


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
    sampled_curvatures = sectional_curvature(A, D, samples=10)
    assert sampled_curvatures.shape == (10,)

    # Test per_node method
    node_curvatures = sectional_curvature(A, D)
    assert node_curvatures.shape == (n,)

    # Test global method (user calls .mean() themselves)
    global_curvature = node_curvatures.mean().item()
    assert isinstance(global_curvature, float)


def test_greedy_method():
    # Get a very small subset of the polblogs dataset
    _, D, _, y = load_hf("polblogs")
    D = D[:128, :128]
    y = y[:128]
    D = D / D.max()

    max_components = 3
    embedder_init_kwargs = {"random_state": 42}
    embedder_fit_kwargs = {"burn_in_iterations": 10, "training_iterations": 90, "lr": 1e-2}

    # Try distortion pipeline
    optimal_pm, loss_history = greedy_signature_selection(
        pipeline=distortion_pipeline,
        dists=D,
        embedder_init_kwargs=embedder_init_kwargs,
        embedder_fit_kwargs=embedder_fit_kwargs,
        verbose=True,  # Ensure this hits the print statements
    )
    # assert set(optimal_pm.signature) == set(pm.signature), "Optimal signature should match the initial signature"
    assert len(optimal_pm.signature) == len(loss_history)
    assert len(optimal_pm.signature) <= max_components
    assert len(optimal_pm.signature) > 0, "Optimal signature should not be empty"
    assert len(loss_history) > 0, "Loss history should not be empty"
    if len(loss_history) > 1:
        assert loss_history[-1] < loss_history[0], "Loss should decrease over iterations"

    # Try classifier pipeline
    optimal_pm, loss_history = greedy_signature_selection(
        pipeline=predictor_pipeline,
        labels=y,
        dists=D,
        embedder_init_kwargs=embedder_init_kwargs,
        embedder_fit_kwargs=embedder_fit_kwargs,
    )
    # assert set(optimal_pm.signature) == set(pm.signature), "Optimal signature should match the initial signature"
    assert len(optimal_pm.signature) == len(loss_history)
    assert len(optimal_pm.signature) <= max_components
    assert len(optimal_pm.signature) > 0, "Optimal signature should not be empty"
    assert len(loss_history) > 0, "Loss history should not be empty"
    if len(loss_history) > 1:
        assert loss_history[-1] < loss_history[0], "Loss should decrease over iterations"
