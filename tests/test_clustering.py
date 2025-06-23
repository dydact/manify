from manify.manifolds import ProductManifold
from manify.clustering import RiemannianFuzzyKMeans


def test_riemannian_fuzzy_k_means():
    pm = ProductManifold(signature=[(-1.0, 4), (-1.0, 2), (0.0, 2), (1.0, 2), (1.0, 4)])
    X, _ = pm.gaussian_mixture(num_points=100)

    for optimizer in ["adam", "adan"]:
        kmeans = RiemannianFuzzyKMeans(manifold=pm, n_clusters=5)
        kmeans.fit(X)
        preds = kmeans.predict(X)
        assert preds.shape == (100,), f"Predictions should have shape (100,) (optimizer: {optimizer})"

        # Also do a single manifold
        kmeans = RiemannianFuzzyKMeans(manifold=pm.P[0], n_clusters=5, optimizer=optimizer)
        X0 = pm.factorize(X)[0]
        kmeans.fit(X0)
        preds = kmeans.predict(X0)
        assert preds.shape == (100,), f"Predictions should have shape (100,) (optimizer: {optimizer})"
