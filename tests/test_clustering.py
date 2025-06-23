from manify.manifolds import ProductManifold
from manify.clustering import RiemannianFuzzyKMeans


def test_riemannian_fuzzy_k_means():
    pm = ProductManifold(signature=[(-1.0, 4), (-1.0, 2), (0.0, 2), (1.0, 2), (1.0, 4)])
    X, y = pm.gaussian_mixture(num_points=100)
    kmeans = RiemannianFuzzyKMeans(manifold=pm, n_clusters=5)
    kmeans.fit(X)
    preds = kmeans.predict(X)
    assert preds.shape == (100,), "Predictions should have shape (100,)"
