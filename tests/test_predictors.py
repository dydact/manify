import torch

from sklearn.model_selection import train_test_split

from manify.predictors.decision_tree import ProductSpaceDT, ProductSpaceRF
from manify.predictors.kappa_gcn import KappaGCN, get_A_hat
from manify.predictors.perceptron import ProductSpacePerceptron
from manify.predictors.svm import ProductSpaceSVM
from manify.manifolds import ProductManifold


def _test_base_classifier(model, X_train, X_test, y_train, y_test, task="classification"):
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    assert preds.shape[0] == X_test.shape[0], "Predictions should match the number of test samples"
    assert preds.ndim == 1, "Predictions should be a 1D array"

    if task == "classification":
        probs = model.predict_proba(X_test)
        assert probs.shape == (X_test.shape[0], 2), "Probabilities should match the number of test samples and classes"
        assert probs.ndim == 2, "Probabilities should be a 2D array"
        assert (
            torch.argmax(probs, dim=1) == preds
        ).all(), "Predictions should match the class with the highest probability"

        # accuracy = model.score(X_test, y_test)
        accuracy = (preds == y_test).float().mean()
        # assert torch.isclose(accuracy, (preds == y_test).float().mean(), atol=1e-5), "Accuracy calculation mismatch"
        assert accuracy >= 0.5, f"Model {model.__class__.__name__} did not achieve sufficient accuracy"
    elif task == "regression":
        pass  # No further tests are really possible for regression


def _test_kappa_gcn_model(model, X_train, X_test, y_train, y_test, pm, task="classification"):
    X_train_kernel = torch.exp(-pm.pdist2(X_train))
    X_test_kernel = torch.exp(-pm.pdist2(X_test))
    A_train = get_A_hat(X_train_kernel)
    A_test = get_A_hat(X_test_kernel)
    model.fit(X_train, y_train, A=A_train, use_tqdm=False, epochs=100)

    preds = model.predict(X_test, A=A_test)
    assert preds.shape[0] == X_test.shape[0], "Predictions should match the number of test samples"
    assert preds.ndim == 1, "Predictions should be a 1D array"

    if task == "classification":
        probs = model.predict_proba(X_test, A=A_test)
        assert probs.shape == (X_test.shape[0], 2), "Probabilities should match the number of test samples and classes"
        assert (
            torch.argmax(probs, dim=1) == preds
        ).all(), "Predictions should match the class with the highest probability"
        assert (
            torch.argmax(probs, dim=1).shape[0] == preds.shape[0]
        ), "The number of predicted classes should match the number of predictions"

        accuracy = (preds == y_test).float().mean()
        assert accuracy >= 0.5, f"Model {model.__class__.__name__} did not achieve sufficient accuracy"
    elif task == "regression":
        pass


def test_all_classifiers():
    print("Testing basic classifier functionality")
    pm = ProductManifold(signature=[(-1.0, 2), (0.0, 2), (1.0, 2)])
    X, y = pm.gaussian_mixture(num_points=100, num_classes=2, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Init models
    for model_class in [
        ProductSpaceDT,
        ProductSpaceRF,
        ProductSpacePerceptron,
        # ProductSpaceSVM,
    ]:
        model = model_class(pm=pm)
        _test_base_classifier(model, X_train, X_test, y_train, y_test)

    # Kappa-GCN needs its own thing
    pm_stereo, X_train_stereo, X_test_stereo = pm.stereographic(X_train, X_test)
    kappa_gcn = KappaGCN(pm=pm_stereo, output_dim=2, hidden_dims=[pm.dim, pm.dim])
    _test_kappa_gcn_model(kappa_gcn, X_train_stereo, X_test_stereo, y_train, y_test, pm=pm_stereo)

    print("All classifiers tested successfully.")


def test_all_regressors():
    print("Testing basic regressor functionality")
    pm = ProductManifold(signature=[(-1.0, 2), (0.0, 2), (1.0, 2)])
    X, y = pm.gaussian_mixture(num_points=100, num_classes=2, seed=42, task="regression")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Init models
    for model_class in [
        ProductSpaceDT,
        ProductSpaceRF,
        # ProductSpacePerceptron,
        # ProductSpaceSVM,
    ]:
        model = model_class(pm=pm, task="regression")
        _test_base_classifier(model, X_train, X_test, y_train, y_test, task="regression")

    # Kappa-GCN needs its own thing
    pm_stereo, X_train_stereo, X_test_stereo = pm.stereographic(X_train, X_test)
    kappa_gcn = KappaGCN(pm=pm_stereo, output_dim=1, hidden_dims=[pm.dim, pm.dim], task="regression")
    _test_kappa_gcn_model(kappa_gcn, X_train_stereo, X_test_stereo, y_train, y_test, pm=pm_stereo, task="regression")

    print("All regressors tested successfully.")
