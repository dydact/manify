from manify.manifolds import ProductManifold
from manify.utils.benchmarks import benchmark
from manify.utils.dataloaders import load_hf
from manify.utils.visualization import S2_to_polar, hyperboloid_to_poincare, spherical_to_polar


def test_benchmark():
    print("Testing benchmark")
    pm = ProductManifold(signature=[(-1.0, 4), (-1.0, 2), (0.0, 2), (1.0, 2), (1.0, 4)])
    X, y = pm.gaussian_mixture(num_points=100)

    models = [
        "sklearn_dt",
        "sklearn_rf",
        "product_dt",
        "product_rf",
        "tangent_dt",
        "tangent_rf",
        "knn",
        "ps_perceptron",
        "svm",
        "ps_svm",
        "kappa_mlp",
        "tangent_mlp",
        "ambient_mlp",
        "tangent_gcn",
        "ambient_gcn",
        "kappa_gcn",
        "ambient_mlr",
        "tangent_mlr",
        "kappa_mlr",
    ]

    metrics = ["accuracy", "f1-micro", "f1-macro", "time"]

    out = benchmark(X, y, pm, task="classification", epochs=10, models=models)
    target_keys = set(f"{model}_{metric}" for model in models for metric in metrics)
    assert set(out.keys()) == target_keys, "Output keys do not match"
    assert all(out[key] >= 0 for key in target_keys), "All scores should be non-negative"


def test_dataloaders():
    print("Testing dataloaders")
    # I apologize for the handcoded nature of this, but this encodes my expectations for how each of the datasets
    # behaves. I added this to the docstring as well.
    for dataset_name, features_expected, dists_expected, labels_expected, adjacency_expected in [
        ("cities", False, True, False, False),
        ("cs_phds", False, True, True, True),
        ("polblogs", False, True, True, True),
        ("polbooks", False, True, True, True),
        ("cora", False, True, True, True),
        ("citeseer", False, True, True, True),
        ("karate_club", False, True, False, True),
        ("lesmis", False, True, False, True),
        ("adjnoun", False, True, False, True),
        ("football", False, True, False, True),
        ("dolphins", False, True, False, True),
        ("blood_cells", True, False, True, False),
        ("lymphoma", True, False, True, False),
        ("cifar_100", True, False, True, False),
        ("mnist", True, False, True, False),
        ("temperature", True, False, True, False),
        ("landmasses", True, False, True, False),
        ("neuron_33", True, False, True, False),
        ("neuron_46", True, False, True, False),
        ("traffic", True, False, True, False),
        ("qiita", True, True, False, False),
    ]:
        print(f"  Testing {dataset_name}")
        features, dists, adjacency, labels = load_hf(dataset_name)

        assert features_expected or dists_expected, "Must have features or distances"

        if features_expected:
            assert features is not None, f"Features should not be None for {dataset_name}"
            n = features.shape[0]
        else:
            assert features is None, f"Features should be None for {dataset_name}"

        if dists_expected:
            assert dists is not None, f"Distances should not be None for {dataset_name}"
            n = dists.shape[0]
        else:
            assert dists is None, f"Distances should be None for {dataset_name}"

        if adjacency_expected:
            assert adjacency is not None, f"Adjacency should not be None for {dataset_name}"
            assert adjacency.shape[0] == adjacency.shape[1] == n, "All adjacency matrix dimensions should be n"
        else:
            assert adjacency is None, "Adjacency should be None for {dataset_name}"

        if labels_expected:
            assert labels is not None, f"Labels should not be None for {dataset_name}"
            assert labels.shape[0] == n, "Number of labels should be n"
        else:
            assert labels is None, f"Labels should be None for {dataset_name}"

        print("Done testing dataloaders")


def test_visualization():
    print("Testing visualization functions")

    # 2-D (special case)
    pm = ProductManifold(signature=[(-1.0, 2), (1.0, 2)])
    X, _ = pm.gaussian_mixture(num_points=100, num_classes=2, seed=42)

    X_H, X_S = pm.factorize(X)
    assert X_H.shape == (100, 3), "Hyperbolic factor should have 3 dimensions"
    assert X_S.shape == (100, 3), "Spherical factor should have 3 dimensions"

    X_H_poincare = hyperboloid_to_poincare(X_H)
    X_S_polar = spherical_to_polar(X_S)
    X_S2_polar = S2_to_polar(X_S)
    assert X_H_poincare.shape == (100, 2), "Poincare coordinates should have 2 dimensions"
    assert X_S_polar.shape == (100, 2), "Polar coordinates should have  2 dimensions"
    assert X_S2_polar.shape == (100, 2), "S^2 polar coordinates should have 2 dimensions"

    # Higher dimensions are basically all the same
    pm = ProductManifold(signature=[(-1.0, 4), (1.0, 4)])
    X, _ = pm.gaussian_mixture(num_points=100, num_classes=2, seed=42)
    X_H, X_S = pm.factorize(X)
    assert X_H.shape == (100, 5), "Hyperbolic factor should have 5 dimensions"
    assert X_S.shape == (100, 5), "Spherical factor should have 5 dimensions"

    X_H_poincare = hyperboloid_to_poincare(X_H)
    X_S_polar = spherical_to_polar(X_S)
    assert X_H_poincare.shape == (100, 4), "Poincare coordinates should have 4 dimensions"
    assert X_S_polar.shape == (100, 4), "Polar coordinates should have 4 dimensions"
