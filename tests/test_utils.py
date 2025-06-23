from manify.utils.dataloaders import load_hf
from manify.utils.benchmarks import benchmark
from manify.utils.visualization import hyperboloid_to_poincare, spherical_to_polar, S2_to_polar
from manify.manifolds import ProductManifold


def test_benchmark():
    print("Testing benchmark")
    pm = ProductManifold(signature=[(-1.0, 4), (-1.0, 2), (0.0, 2), (1.0, 2), (1.0, 4)])
    X, y = pm.gaussian_mixture(num_points=100)
    out = benchmark(X, y, pm, task="classification", epochs=10)
    target_keys = set(
        [
            "sklearn_dt_accuracy",
            "sklearn_dt_f1-micro",
            "sklearn_dt_f1-macro",
            "sklearn_dt_time",
            "sklearn_rf_accuracy",
            "sklearn_rf_f1-micro",
            "sklearn_rf_f1-macro",
            "sklearn_rf_time",
            "product_dt_accuracy",
            "product_dt_f1-micro",
            "product_dt_f1-macro",
            "product_dt_time",
            "product_rf_accuracy",
            "product_rf_f1-micro",
            "product_rf_f1-macro",
            "product_rf_time",
            "tangent_dt_accuracy",
            "tangent_dt_f1-micro",
            "tangent_dt_f1-macro",
            "tangent_dt_time",
            "tangent_rf_accuracy",
            "tangent_rf_f1-micro",
            "tangent_rf_f1-macro",
            "tangent_rf_time",
            "knn_accuracy",
            "knn_f1-micro",
            "knn_f1-macro",
            "knn_time",
            "ps_perceptron_accuracy",
            "ps_perceptron_f1-micro",
            "ps_perceptron_f1-macro",
            "ps_perceptron_time",
            "ps_svm_accuracy",
            "ps_svm_f1-micro",
            "ps_svm_f1-macro",
            "ps_svm_time",
            "ambient_mlp_accuracy",
            "ambient_mlp_f1-micro",
            "ambient_mlp_f1-macro",
            "ambient_mlp_time",
            "ambient_gcn_accuracy",
            "ambient_gcn_f1-micro",
            "ambient_gcn_f1-macro",
            "ambient_gcn_time",
            "tangent_gcn_accuracy",
            "tangent_gcn_f1-micro",
            "tangent_gcn_f1-macro",
            "tangent_gcn_time",
            "kappa_gcn_accuracy",
            "kappa_gcn_f1-micro",
            "kappa_gcn_f1-macro",
            "kappa_gcn_time",
            "kappa_mlr_accuracy",
            "kappa_mlr_f1-micro",
            "kappa_mlr_f1-macro",
            "kappa_mlr_time",
            "tangent_mlr_accuracy",
            "tangent_mlr_f1-micro",
            "tangent_mlr_f1-macro",
            "tangent_mlr_time",
            "ambient_mlr_accuracy",
            "ambient_mlr_f1-micro",
            "ambient_mlr_f1-macro",
            "ambient_mlr_time",
        ]
    )
    assert out.keys() == target_keys, "Output keys do not match"
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
    X, y = pm.gaussian_mixture(num_points=100, num_classes=2, seed=42)

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
    pm = ProductManifold(signature=[(-1.0, 10), (1.0, 10)])
    X, y = pm.gaussian_mixture(num_points=100, num_classes=2, seed=42)
    X_H, X_S = pm.factorize(X)
    assert X_H.shape == (100, 11), "Hyperbolic factor should have 11 dimensions"
    assert X_S.shape == (100, 11), "Spherical factor should have 11 dimensions"

    X_H_poincare = hyperboloid_to_poincare(X_H)
    X_S_polar = spherical_to_polar(X_S)
    assert X_H_poincare.shape == (100, 10), "Poincare coordinates should have 10 dimensions"
    assert X_S_polar.shape == (100, 10), "Polar coordinates should have 10 dimensions"
