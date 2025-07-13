from manify.embedders._losses import dist_component_by_manifold
from manify.manifolds import ProductManifold


def test_dist_components_by_manifold():
    pm = ProductManifold(signature=[(-1.0, 2), (0.0, 2), (1.0, 2)])
    X, _ = pm.gaussian_mixture(num_points=100, num_classes=2, seed=42)
    dists = dist_component_by_manifold(pm, X)

    assert len(dists) == 3, "Should have 3 distance components for 3 manifolds"
