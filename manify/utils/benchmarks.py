"""Implementation for benchmarking different product space machine learning methods"""

from __future__ import annotations

import time
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
from jaxtyping import Float, Real
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..manifolds import ProductManifold
from ..predictors.decision_tree import ProductSpaceDT, ProductSpaceRF
from ..predictors.kappa_gcn import KappaGCN, get_A_hat
from ..predictors.perceptron import ProductSpacePerceptron
from ..predictors.svm import ProductSpaceSVM

MODELTYPE = Literal[
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
    "single_manifold_rf",
]
SCORETYPE = Literal["accuracy", "f1-micro", "f1-macro", "mse", "percent_rmse", "time"]
TASKTYPE = Literal["classification", "regression", "link_prediction"]


def _score(
    _X: Float[torch.Tensor, "n_samples n_dims"],
    _y: Real[torch.Tensor, "n_samples,"],
    model: BaseEstimator,
    y_pred_override: Optional[Real[torch.Tensor, "n_samples,"]] = None,
    use_torch: bool = False,
    score: Optional[List[SCORETYPE]] = None,
) -> Dict[SCORETYPE, float]:
    if score is None:
        score = ["accuracy", "f1-micro"]
    y_pred = y_pred_override if y_pred_override is not None else model.predict(_X)
    if use_torch:
        y_pred = y_pred.detach().cpu().numpy()
    scoring_funcs = {
        "accuracy": accuracy_score,
        "f1-micro": lambda y, p: f1_score(y, p, average="micro"),
        "f1-macro": lambda y, p: f1_score(y, p, average="macro"),
        "mse": mean_squared_error,
        "rmse": root_mean_squared_error,
        "percent_rmse": lambda y, p: (root_mean_squared_error(y, p, multioutput="raw_values") / np.abs(y)).mean(),
    }
    return {s: scoring_funcs[s](_y, y_pred) if s in scoring_funcs else np.nan for s in score}


def benchmark(
    X: Float[torch.Tensor, "batch dim"],
    y: Real[torch.Tensor, "batch,"],
    pm: ProductManifold,
    device: Literal["cpu", "cuda", "mps"] = "cpu",
    score: Optional[List[SCORETYPE]] = None,
    models: Optional[List[MODELTYPE]] = None,
    max_depth: int = 5,
    n_estimators: int = 12,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    task: TASKTYPE = "classification",
    seed: Optional[int] = None,
    use_special_dims: bool = False,
    n_features: Literal["d", "d_choose_2"] = "d_choose_2",
    X_train: Optional[Float[torch.Tensor, "n_samples n_manifolds"]] = None,
    X_test: Optional[Float[torch.Tensor, "n_samples n_manifolds"]] = None,
    y_train: Optional[Real[torch.Tensor, "n_samples,"]] = None,
    y_test: Optional[Real[torch.Tensor, "n_samples,"]] = None,
    batch_size: Optional[int] = None,
    adj: Optional[Float[torch.Tensor, "n_nodes n_nodes"]] = None,
    A_train: Optional[Float[torch.Tensor, "n_samples n_samples"]] = None,
    A_test: Optional[Float[torch.Tensor, "n_samples n_samples"]] = None,
    hidden_dims: Optional[List[int]] = None,
    epochs: int = 4_000,
    lr: float = 1e-4,
    kappa_gcn_layers: int = 1,
    lp_train_idx: Optional[Float[torch.Tensor, "n_samples,"]] = None,
    lp_test_idx: Optional[Float[torch.Tensor, "n_samples,"]] = None,
) -> Dict[str, float]:
    """
    Benchmarks various machine learning models on a dataset using a product manifold structure.

    Args:
        X (batch, dim): Input tensor of features
        y (batch,): Input tensor of labels.
        pm: The defined product manifold for benchmarks.
        split: Data splitting strategy ('train_test' or 'cross_val').
        device: Device for computation ('cpu', 'cuda', 'mps').
        score: Scoring metric for model evaluation ('accuracy', 'f1-micro', etc.).
        models: List of model names to evaluate. Options include:
            * "sklearn_dt": Decision tree from scikit-learn.
            * "sklearn_rf": Random forest from scikit-learn.
            * "product_dt": Product space decision tree.
            * "product_rf": Product space random forest.
            * "tangent_dt": Decision tree on tangent space.
            * "tangent_rf": Random forest on tangent space.
            * "knn": k-nearest neighbors.
            * "ps_perceptron": Product space perceptron.
        max_depth: Maximum depth of tree-based models in integer.
        n_estimators: Integer number of estimators for random forest models.
        min_samples_split: Minimum number of samples required to split an internal node.
        min_samples_leaf: Minimum number of samples in a leaf node.
        task: Task type ('classification' or 'regression').
        seed: Random seed for reproducibility.
        use_special_dims: Boolean for whether to use special manifold dimensions.
        n_features: Feature dimensionality type ('d' or 'd_choose_2').
        X_train, X_test, y_train, y_test: Training and testing datasets, X: feature, y: label.
        batch_size: Batch size for certain models.

    Returns:
        Dict[str, float]: Dictionary of model names and their corresponding evaluation scores.

    """
    if score is None:
        score = ["accuracy", "f1-micro", "f1-macro"]
    if models is None:
        models = [
            "sklearn_dt",
            "sklearn_rf",
            "product_dt",
            "product_rf",
            "tangent_dt",
            "tangent_rf",
            "knn",
            "ps_perceptron",
            # "svm",
            # "ps_svm",
            # "tangent_mlp",
            "ambient_mlp",
            "tangent_gcn",
            "ambient_gcn",
            "kappa_gcn",
            "ambient_mlr",
            "tangent_mlr",
            "kappa_mlr",
            "single_manifold_rf",
        ]
    if hidden_dims is None:
        hidden_dims = [32, 32]

    # Input validation on (task, score) pairing
    if task in ["classification", "link_prediction"]:
        assert all(s in ["accuracy", "f1-micro", "f1-macro", "time"] for s in score)
    elif task == "regression":
        assert all(s in ["mse", "rmse", "percent_rmse", "time"] for s in score)

    # Input validation on (task, score) pairing
    if task in ["classification", "link_prediction"]:
        assert all(s in ["accuracy", "f1-micro", "f1-macro", "time"] for s in score)
    elif task == "regression":
        assert all(s in ["mse", "rmse", "percent_rmse", "time"] for s in score)
    else:
        raise ValueError(f"Unknown task: {task}")

    # Make sure we're on the right device
    pm = pm.to(device)

    # Split data
    if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
        # Coerce to tensor as needed
        if not torch.is_tensor(X_train):
            X_train = torch.tensor(X_train)
        if not torch.is_tensor(X_test):
            X_test = torch.tensor(X_test)
        if not torch.is_tensor(y_train):
            y_train = torch.tensor(y_train)
        if not torch.is_tensor(y_test):
            y_test = torch.tensor(y_test)

        # Move to device
        X_train = X_train.to(device)
        X_test = X_test.to(device)
        y_train = y_train.to(device)
        y_test = y_test.to(device)

        # Get X and y
        X = torch.cat([X_train, X_test])
        y = torch.cat([y_train, y_test])
        train_idx = np.arange(len(X_train))
        test_idx = np.arange(len(X_train), len(X))

    else:
        # Coerce to tensor as needed
        if not torch.is_tensor(X):
            X = torch.tensor(X)
        if not torch.is_tensor(y):
            y = torch.tensor(y)

        X = X.to(device)
        y = y.to(device)

        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(X, y, np.arange(len(X)), test_size=0.2)

    # Make sure classification labels are formatted correctly
    if task in ["classification", "link_prediction"]:
        y = torch.unique(y, return_inverse=True)[1]
        y_train = y[train_idx]
        y_test = y[test_idx]

    # Make sure everything is detached
    X, X_train, X_test = X.detach(), X_train.detach(), X_test.detach()
    y, y_train, y_test = y.detach(), y_train.detach(), y_test.detach()

    # Get pdists
    pdists = pm.pdist(X).detach()

    # Get tangent plane
    X_train_tangent = pm.logmap(X_train).detach()
    X_test_tangent = pm.logmap(X_test).detach()

    # Get numpy versions
    X_train_np, X_test_np = X_train.detach().cpu().numpy(), X_test.detach().cpu().numpy()
    y_train_np, y_test_np = y_train.detach().cpu().numpy(), y_test.detach().cpu().numpy()
    X_train_tangent_np, X_test_tangent_np = X_train_tangent.cpu().numpy(), X_test_tangent.cpu().numpy()

    # Get stereographic version
    pm_stereo, X_train_stereo, X_test_stereo = pm.stereographic(X_train, X_test)
    assert isinstance(X_train_stereo, torch.Tensor)
    X_train_stereo = X_train_stereo.detach()
    assert isinstance(X_test_stereo, torch.Tensor)
    X_test_stereo = X_test_stereo.detach()

    # Also euclidean """PM"""
    pm_euc = ProductManifold(signature=[(0, X.shape[1])], device=device, stereographic=True)

    # Get A_hat
    if adj is not None:
        A_hat = get_A_hat(adj).detach()
    else:
        dists = pdists**2
        dists_train = dists[train_idx][:, train_idx]
        dists /= dists_train[torch.isfinite(dists_train)].max()
        A_hat = get_A_hat(dists).detach()
    A_hat = A_hat.to(device)

    if A_train is None and A_test is None:
        A_train = A_hat[train_idx][:, train_idx].detach()
        A_test = A_hat[test_idx][:, test_idx].detach()
    else:
        assert A_train is not None
        assert A_test is not None
        A_train = A_train.to(device).detach()
        A_test = A_test.to(device).detach()

    # Aggregate arguments
    tree_kwargs = {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "min_samples_split": min_samples_split}
    prod_kwargs = {"use_special_dims": use_special_dims, "n_features": n_features, "batch_size": batch_size}
    rf_kwargs = {"n_estimators": n_estimators, "n_jobs": -1, "random_state": seed}
    nn_outdim = 1 if task == "regression" else len(torch.unique(y))
    nn_kwargs = {"task": task, "output_dim": nn_outdim, "hidden_dims": hidden_dims}
    nn_train_kwargs = {"epochs": epochs, "lr": lr, "lp_indices": lp_train_idx}

    # Define your models
    if task in ["classification", "link_prediction"]:
        dt_class = DecisionTreeClassifier
        rf_class = RandomForestClassifier
        knn_class = KNeighborsClassifier
        svm_class = SVC

    else:  # task == "regression"
        dt_class = DecisionTreeRegressor
        rf_class = RandomForestRegressor
        knn_class = KNeighborsRegressor
        svm_class = SVR

    # Get link prediction indices: random 1000 edges
    if task == "link_prediction":
        # Want shape to be (1000, 2) in range 0, n_nodes:
        assert adj is not None
        n_nodes = adj.shape[0]
        n_edges = 1000
        lp_indices = torch.randint(0, n_nodes, (n_edges, 2))

    # Evaluate sklearn
    accs: Dict[MODELTYPE, Dict[SCORETYPE, float]] = {}
    if "sklearn_dt" in models:
        dt = dt_class(**tree_kwargs)
        t1 = time.time()
        dt.fit(X_train_np, y_train_np)
        t2 = time.time()
        accs["sklearn_dt"] = _score(X_test_np, y_test_np, dt, use_torch=False, score=score)
        accs["sklearn_dt"]["time"] = t2 - t1

    if "sklearn_rf" in models:
        rf = rf_class(**tree_kwargs, **rf_kwargs)
        t1 = time.time()
        rf.fit(X_train_np, y_train_np)
        t2 = time.time()
        accs["sklearn_rf"] = _score(X_test_np, y_test_np, rf, use_torch=False, score=score)
        accs["sklearn_rf"]["time"] = t2 - t1

    if "product_dt" in models:
        psdt = ProductSpaceDT(pm=pm, task=task, **tree_kwargs, **prod_kwargs)  # type: ignore
        t1 = time.time()
        psdt.fit(X_train, y_train)
        t2 = time.time()
        accs["product_dt"] = _score(X_test, y_test_np, psdt, use_torch=True, score=score)
        accs["product_dt"]["time"] = t2 - t1

    if "product_rf" in models:
        psrf = ProductSpaceRF(pm=pm, task=task, **tree_kwargs, **rf_kwargs, **prod_kwargs)  # type: ignore
        t1 = time.time()
        psrf.fit(X_train, y_train)
        t2 = time.time()
        accs["product_rf"] = _score(X_test, y_test_np, psrf, use_torch=True, score=score)
        accs["product_rf"]["time"] = t2 - t1

    # if "single_manifold_rf" in models:
    #     smrf = SingleManifoldEnsembleRF(pm=pm, task=task, n_estimators=n_estimators)
    #     t1 = time.time()
    #     smrf.fit(X_train, y_train)
    #     t2 = time.time()
    #     accs["single_manifold_rf"] = _score(X_test, y_test_np, smrf, torch=True, score=score)
    #     accs["single_manifold_rf"]["time"] = t2 - t1

    if "tangent_dt" in models:
        tdt = dt_class(**tree_kwargs)
        t1 = time.time()
        tdt.fit(X_train_tangent_np, y_train_np)
        t2 = time.time()
        accs["tangent_dt"] = _score(X_test_tangent_np, y_test_np, tdt, use_torch=False, score=score)
        accs["tangent_dt"]["time"] = t2 - t1

    if "tangent_rf" in models:
        trf = rf_class(**tree_kwargs, **rf_kwargs)
        t1 = time.time()
        trf.fit(X_train_tangent_np, y_train_np)
        t2 = time.time()
        accs["tangent_rf"] = _score(X_test_tangent_np, y_test_np, trf, use_torch=False, score=score)
        accs["tangent_rf"]["time"] = t2 - t1

    if "knn" in models:
        # Get dists - max imputation is a workaround for some nan values we occasionally get
        t1 = time.time()
        train_dists = pm.pdist(X_train)
        train_dists = torch.nan_to_num(train_dists, nan=train_dists[~train_dists.isnan()].max().item())
        train_test_dists = pm.dist(X_test, X_train)
        train_test_dists = torch.nan_to_num(
            train_test_dists,
            nan=train_test_dists[~train_test_dists.isnan()].max().item(),
        )

        # Convert to numpy
        train_dists = train_dists.detach().cpu().numpy()
        train_test_dists = train_test_dists.detach().cpu().numpy()

        # Train classifier on distances
        knn = knn_class(metric="precomputed")
        t2 = time.time()
        knn.fit(train_dists, y_train_np)
        t3 = time.time()
        accs["knn"] = _score(train_test_dists, y_test_np, knn, use_torch=False, score=score)
        accs["knn"]["time"] = t3 - t1

    # if "perceptron" in models:
    #     loss = "perceptron" if task == "classification" else "squared_error"
    #     ptron = perceptron_class(
    #         loss=loss,
    #         learning_rate="constant",
    #         fit_intercept=False,
    #         eta0=1.0,
    #         max_iter=10_000,
    #     )  # fit_intercept must be false for ambient coordinates
    #     t1 = time.time()
    #     ptron.fit(X_train_np, y_train_np)
    #     t2 = time.time()
    #     accs["perceptron"] = _score(X_test_np, y_test_np, ptron, torch=False, score=score)
    #     accs["perceptron"]["time"] = t2 - t1

    if "ps_perceptron" in models:
        if task == "classification":
            ps_per = ProductSpacePerceptron(pm=pm)
            t1 = time.time()
            ps_per.fit(X_train, y_train)
            t2 = time.time()
            accs["ps_perceptron"] = _score(X_test, y_test_np, ps_per, use_torch=True, score=score)
            accs["ps_perceptron"]["time"] = t2 - t1

    if "svm" in models:
        # Get inner products for precomputed kernel matrix
        t1 = time.time()
        train_ips = pm.manifold.component_inner(X_train[:, None], X_train[None, :]).sum(dim=-1)
        train_test_ips = pm.manifold.component_inner(X_test[:, None], X_train[None, :]).sum(dim=-1)

        # Convert to numpy
        train_ips = train_ips.detach().cpu().numpy()
        train_test_ips = train_test_ips.detach().cpu().numpy()

        # Train SVM on precomputed inner products
        svm = svm_class(kernel="precomputed", max_iter=10_000)
        # Need max_iter because it can hang. It can be large, since this doesn't happen often.
        t2 = time.time()
        svm.fit(train_ips, y_train_np)
        t3 = time.time()
        accs["svm"] = _score(train_test_ips, y_test_np, svm, use_torch=False, score=score)
        accs["svm"]["time"] = t3 - t1

    if "ps_svm" in models:
        try:
            ps_svm = ProductSpaceSVM(pm=pm, task=task, h_constraints=False, e_constraints=False)  # type: ignore
            t1 = time.time()
            ps_svm.fit(X_train, y_train)
            t2 = time.time()
            accs["ps_svm"] = _score(X_test, y_test_np, ps_svm, use_torch=False, score=score)
            accs["ps_svm"]["time"] = t2 - t1
        except Exception:
            pass
            #     accs["ps_svm"] = {"accuracy": 0.0, "f1-micro": 0.0, "time": 0.0}

    if "kappa_mlp" in models:
        assert isinstance(X_test_stereo, torch.Tensor)
        d = X_test_stereo.shape[1]  # Shape can't change between layers
        kappa_mlp = KappaGCN(
            pm=pm_stereo, hidden_dims=[d] * kappa_gcn_layers, task=task, output_dim=nn_outdim  # type: ignore
        ).to(device)
        t1 = time.time()
        if task == "link_prediction":
            kappa_mlp.fit(X_train_stereo, y_train, A=A_train, tqdm_prefix="kappa_mlp", **nn_train_kwargs)
        else:
            kappa_mlp.fit(X_train_stereo, y_train, A=None, tqdm_prefix="kappa_mlp", **nn_train_kwargs)
        t2 = time.time()
        y_pred = kappa_mlp.predict(X_test_stereo, A=None)
        if lp_test_idx:
            y_pred = y_pred[lp_test_idx]
        accs["kappa_mlp"] = _score(None, y_test_np, kappa_mlp, y_pred_override=y_pred, use_torch=True, score=score)
        accs["kappa_mlp"]["time"] = t2 - t1

    if "ambient_mlp" in models:
        ambient_mlp = KappaGCN(pm=pm_euc, **nn_kwargs).to(device)  # type: ignore
        t1 = time.time()
        ambient_mlp.fit(X_train, y_train, A=None, tqdm_prefix="ambient_mlp", **nn_train_kwargs)
        t2 = time.time()
        y_pred = ambient_mlp.predict(X_test, A=None)
        if lp_test_idx:
            y_pred = y_pred[lp_test_idx]
        accs["ambient_mlp"] = _score(None, y_test_np, ambient_mlp, y_pred_override=y_pred, use_torch=True, score=score)
        accs["ambient_mlp"]["time"] = t2 - t1

    if "tangent_mlp" in models:
        tangent_mlp = KappaGCN(pm=pm_euc, **nn_kwargs).to(device)  # type: ignore
        t1 = time.time()
        tangent_mlp.fit(X_train_tangent, y_train, A=None, tqdm_prefix="tangent_mlp", **nn_train_kwargs)
        t2 = time.time()
        y_pred = tangent_mlp.predict(X_test_tangent, A=None)
        if lp_test_idx:
            y_pred = y_pred[lp_test_idx]
        accs["tangent_mlp"] = _score(None, y_test_np, tangent_mlp, y_pred_override=y_pred, use_torch=True, score=score)
        accs["tangent_mlp"]["time"] = t2 - t1

    if "ambient_gcn" in models:
        ambient_gcn = KappaGCN(pm=pm_euc, **nn_kwargs).to(device)  # type: ignore
        t1 = time.time()
        ambient_gcn.fit(X_train, y_train, A=A_train, **nn_train_kwargs)
        t2 = time.time()
        y_pred = ambient_gcn.predict(X_test, A=A_test)
        if lp_test_idx:
            y_pred = y_pred[lp_test_idx]
        accs["ambient_gcn"] = _score(None, y_test_np, None, y_pred_override=y_pred, use_torch=True, score=score)
        accs["ambient_gcn"]["time"] = t2 - t1

    if "tangent_gcn" in models:
        tangent_gcn = KappaGCN(pm=pm_euc, **nn_kwargs).to(device)  # type: ignore
        t1 = time.time()
        tangent_gcn.fit(X_train_tangent, y_train, A=A_train, tqdm_prefix="tangent_gcn", **nn_train_kwargs)
        t2 = time.time()
        y_pred = tangent_gcn.predict(X_test_tangent, A=A_test)
        if lp_test_idx:
            y_pred = y_pred[lp_test_idx]
        accs["tangent_gcn"] = _score(None, y_test_np, None, y_pred_override=y_pred, use_torch=True, score=score)
        accs["tangent_gcn"]["time"] = t2 - t1

    if "kappa_gcn" in models:
        assert isinstance(X_test_stereo, torch.Tensor)
        d = X_test_stereo.shape[1]  # Shape can't change between layers
        kappa_gcn = KappaGCN(pm=pm_stereo, hidden_dims=[d] * kappa_gcn_layers, task=task, output_dim=nn_outdim).to(
            device
        )  # type: ignore
        t1 = time.time()
        kappa_gcn.fit(X_train_stereo, y_train, A=A_train, tqdm_prefix="kappa_gcn", **nn_train_kwargs)
        t2 = time.time()
        y_pred = kappa_gcn.predict(X_test_stereo, A=A_test)
        if lp_test_idx:
            y_pred = y_pred[lp_test_idx]
        accs["kappa_gcn"] = _score(None, y_test_np, None, y_pred_override=y_pred, use_torch=True, score=score)
        accs["kappa_gcn"]["time"] = t2 - t1

    if "kappa_mlr" in models:
        kappa_mlr = KappaGCN(pm=pm_stereo, hidden_dims=[], task=task, output_dim=nn_outdim).to(device)  # type: ignore
        t1 = time.time()
        kappa_mlr.fit(X_train_stereo, y_train, A=None, tqdm_prefix="kappa_mlr", **nn_train_kwargs)
        t2 = time.time()
        y_pred = kappa_mlr.predict(X_test_stereo, A=None)
        if lp_test_idx:
            y_pred = y_pred[lp_test_idx]
        accs["kappa_mlr"] = _score(None, y_test_np, None, y_pred_override=y_pred, use_torch=True, score=score)
        accs["kappa_mlr"]["time"] = t2 - t1

    if "tangent_mlr" in models:
        tangent_mlr = KappaGCN(pm=pm_euc, hidden_dims=[], task=task, output_dim=nn_outdim).to(device)  # type: ignore
        t1 = time.time()
        tangent_mlr.fit(X_train_tangent, y_train, A=None, tqdm_prefix="tangent_mlr", **nn_train_kwargs)
        t2 = time.time()
        y_pred = tangent_mlr.predict(X_test_tangent, A=None)
        if lp_test_idx:
            y_pred = y_pred[lp_test_idx]
        accs["tangent_mlr"] = _score(None, y_test_np, None, y_pred_override=y_pred, use_torch=True, score=score)
        accs["tangent_mlr"]["time"] = t2 - t1

    if "ambient_mlr" in models:
        ambient_mlr = KappaGCN(pm=pm_euc, hidden_dims=[], task=task, output_dim=nn_outdim).to(device)  # type: ignore
        t1 = time.time()
        ambient_mlr.fit(X_train, y_train, A=None, tqdm_prefix="ambient_mlr", **nn_train_kwargs)
        t2 = time.time()
        y_pred = ambient_mlr.predict(X_test, A=None)
        if lp_test_idx:
            y_pred = y_pred[lp_test_idx]
        accs["ambient_mlr"] = _score(None, y_test_np, None, y_pred_override=y_pred, use_torch=True, score=score)
        accs["ambient_mlr"]["time"] = t2 - t1

    # return accs
    return {
        **{
            f"{model}_{metric}": value
            for model, metrics in accs.items()
            if isinstance(metrics, dict)
            for metric, value in metrics.items()
        },
        **{k: v for k, v in accs.items() if not isinstance(v, dict)},  # type: ignore
    }
