import torch

from manify.curvature_estimation._pipelines import (
    distortion_pipeline,
    predictor_pipeline,
)
from manify.curvature_estimation.delta_hyperbolicity import delta_hyperbolicity
from manify.curvature_estimation.sectional_curvature import sectional_curvature
from manify.curvature_estimation.greedy_method import greedy_signature_selection
from manify.manifolds import ProductManifold
from manify.utils.dataloaders import load_hf


def test_delta_hyperbolicity():
    torch.manual_seed(42)
    pm = ProductManifold(signature=[(-1.0, 2)])
    X = pm.sample(z_mean=torch.stack([pm.mu0] * 10))
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
