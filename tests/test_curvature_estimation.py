import torch

from manify.curvature_estimation._pipelines import classifier_pipeline, distortion_pipeline
from manify.curvature_estimation.delta_hyperbolicity import sampled_delta_hyperbolicity, vectorized_delta_hyperbolicity
from manify.curvature_estimation.greedy_method import greedy_signature_selection
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


def test_greedy_method():
    # Get a very small subset of the polblogs dataset
    _, D, _, y = load_hf("polblogs")
    D = D[:128, :128] / D.max()
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
        pipeline=classifier_pipeline,
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
