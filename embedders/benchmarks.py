"""Implementation for benchmarking different product space machine learning methods"""

from typing import List, Literal, Dict, Optional
import time
from torchtyping import TensorType as TT

import torch
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.svm import SVC, SVR

from .manifolds import ProductManifold

from .predictors.tree_new import ProductSpaceDT, ProductSpaceRF
from .predictors.perceptron import ProductSpacePerceptron
from .predictors.svm import ProductSpaceSVM
from .predictors.mlp import MLP
from .predictors.gnn import GNN, get_nonzero
from .predictors.kappa_gcn import KappaGCN, get_A_hat

def benchmark(
    X: TT["batch", "dim"],
    y: TT["batch"],
    pm: ProductManifold,
    device: Literal["cpu", "cuda", "mps"] = "cpu",
    score: List[Literal["accuracy", "f1-micro", "mse", "percent_rmse"]] = ["accuracy", "f1-micro"],
    models: List[str] = [
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
        "tangent_mlp",
        "ambient_mlp",
        "tangent_gnn",
        "ambient_gnn",
        "kappa_gcn",
        "product_mlr",
    ],
    max_depth: int = 3,
    n_estimators: int = 12,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    task: Literal["classification", "regression"] = "classification",
    seed: Optional[int] = None,
    use_special_dims: bool = False,
    n_features: Literal["d", "d_choose_2"] = "d",
    X_train=None,
    X_test=None,
    y_train=None,
    y_test=None,
    batch_size=None,
    adj=None,
    hidden_dims=[128, 128],
    epochs=400,
    lr=1e-2,
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
    # Input validation on (task, score) pairing
    if task == "classification":
        assert all(s in ["accuracy", "f1-micro", "time"] for s in score)
    elif task == "regression":
        assert all(s in ["mse", "rmse", "percent_rmse", "time"] for s in score)

    # Make sure classification labels are formatted correctly
    if task == "classification":
        if y is not None:
            y = torch.unique(y, return_inverse=True)[1]
        if y_train is not None and y_test is not None:
            y_concat = torch.cat([y_train, y_test])
            y_train = torch.unique(y_concat, return_inverse=True)[1][: len(y_train)]
            y_test = torch.unique(y_concat, return_inverse=True)[1][len(y_train) :]

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

    else:
        # Coerce to tensor as needed
        if not torch.is_tensor(X):
            X = torch.tensor(X)
        if not torch.is_tensor(y):
            y = torch.tensor(y)

        X = X.to(device)
        y = y.to(device)

        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(X, y, np.arange(len(X)), test_size=0.2)

    # If we use X_train and X_test, we should make a fake X and y
    if X is None and y is None:
        X = torch.cat([X_train, X_test])
        y = torch.cat([y_train, y_test])
        train_idx = np.arange(len(X_train))
        test_idx = np.arange(len(X_train), len(X))

    # Get pdists
    pdists = pm.pdist(X).detach()

    # Get tangent plane
    X_tangent = pm.logmap(X).detach()
    X_train_tangent = pm.logmap(X_train).detach()
    X_test_tangent = pm.logmap(X_test).detach()

    # Get numpy versions
    X_train_np, X_test_np = X_train.detach().cpu().numpy(), X_test.detach().cpu().numpy()
    y_train_np, y_test_np = y_train.detach().cpu().numpy(), y_test.detach().cpu().numpy()
    X_train_tangent_np, X_test_tangent_np = X_train_tangent.cpu().numpy(), X_test_tangent.cpu().numpy()

    # Get stereographic version
    pm_stereo, X_train_stereo, X_test_stereo = pm.stereographic(X_train, X_test)

    # Also euclidean """PM"""
    pm_euc = ProductManifold(signature=[(0, X.shape[1])], device=device, stereographic=True)

    # Get A_hat
    if adj is not None:
        A_hat = get_A_hat(adj).detach()
    else:
        dists = pdists**2
        dists /= dists.max()
        A_hat = get_A_hat(dists).detach()
    A_hat = A_hat.to(device)
    A_train = A_hat[train_idx][:, train_idx]
    A_test = A_hat[test_idx][:, test_idx]

    def _score(_X, _y, model, y_pred_override=None, torch=False):
        # Override y_pred
        if y_pred_override is not None:
            y_pred = y_pred_override
        else:
            y_pred = model.predict(_X)

        # Convert to numpy
        if torch:
            y_pred = y_pred.detach().cpu().numpy()

        # Score handling
        out = {}
        for s in score:
            if s == "accuracy":
                out[s] = accuracy_score(_y, y_pred)
            elif s == "f1-micro":
                out[s] = f1_score(_y, y_pred, average="micro")
            elif s == "mse":
                out[s] = mean_squared_error(_y, y_pred)
            elif s == "rmse":
                out[s] = root_mean_squared_error(_y, y_pred)
            elif s == "percent_rmse":
                out[s] = (root_mean_squared_error(_y, y_pred, multioutput="raw_values") / np.abs(_y)).mean()
            else:
                raise ValueError(f"Unknown score: {s}")
        return out

    # Aggregate arguments
    tree_kwargs = {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "min_samples_split": min_samples_split}
    prod_kwargs = {"use_special_dims": use_special_dims, "n_features": n_features, "batch_size": batch_size}
    rf_kwargs = {"n_estimators": n_estimators, "n_jobs": -1, "random_state": seed}
    nn_outdim = 1 if task == "regression" else len(torch.unique(y))
    nn_kwargs = {"task": task, "output_dim": nn_outdim, "hidden_dims": hidden_dims}
    nn_train_kwargs = {"epochs": epochs, "lr": lr}

    # Define your models
    if task == "classification":
        dt_class = DecisionTreeClassifier
        rf_class = RandomForestClassifier
        knn_class = KNeighborsClassifier
        svm_class = SVC
        perceptron_class = SGDClassifier
    
    elif task == "regression":
        dt_class = DecisionTreeRegressor
        rf_class = RandomForestRegressor
        knn_class = KNeighborsRegressor
        svm_class = SVR
        perceptron_class = SGDRegressor

    # Evaluate sklearn
    accs = {}
    if "sklearn_dt" in models:
        dt = dt_class(**tree_kwargs)
        t1 = time.time()
        dt.fit(X_train_np, y_train_np)
        t2 = time.time()
        accs["sklearn_dt"] = _score(X_test_np, y_test_np, dt, torch=False)
        accs["sklearn_dt"]["time"] = t2 - t1

    if "sklearn_rf" in models:
        rf = rf_class(**tree_kwargs, **rf_kwargs)
        t1 = time.time()
        rf.fit(X_train_np, y_train_np)
        t2 = time.time()
        accs["sklearn_rf"] = _score(X_test_np, y_test_np, rf, torch=False)
        accs["sklearn_rf"]["time"] = t2 - t1

    if "product_dt" in models:
        psdt = ProductSpaceDT(
            pm=pm,
            task=task,
            **tree_kwargs,
            **prod_kwargs,
        )
        t1 = time.time()
        psdt.fit(X_train, y_train)
        t2 = time.time()
        accs["product_dt"] = _score(X_test, y_test_np, psdt, torch=True)
        accs["product_dt"]["time"] = t2 - t1

    if "product_rf" in models:
        psrf = ProductSpaceRF(
            pm=pm,
            task=task,
            **tree_kwargs,
            **rf_kwargs,
            **prod_kwargs,
        )
        t1 = time.time()
        psrf.fit(X_train, y_train)
        t2 = time.time()
        accs["product_rf"] = _score(X_test, y_test_np, psrf, torch=True)
        accs["product_rf"]["time"] = t2 - t1

    if "tangent_dt" in models:
        tdt = dt_class(**tree_kwargs)
        t1 = time.time()
        tdt.fit(X_train_tangent_np, y_train_np)
        t2 = time.time()
        accs["tangent_dt"] = _score(X_test_tangent_np, y_test_np, tdt, torch=False)
        accs["tangent_dt"]["time"] = t2 - t1

    if "tangent_rf" in models:
        trf = rf_class(**tree_kwargs, **rf_kwargs)
        t1 = time.time()
        trf.fit(X_train_tangent_np, y_train_np)
        t2 = time.time()
        accs["tangent_rf"] = _score(X_test_tangent_np, y_test_np, trf, torch=False)
        accs["tangent_rf"]["time"] = t2 - t1

    if "knn" in models:
        # Get dists - max imputation is a workaround for some nan values we occasionally get
        t1 = time.time()
        train_dists = pm.pdist(X_train)
        train_dists = torch.nan_to_num(train_dists, nan=train_dists[~train_dists.isnan()].max().item())
        train_test_dists = pm.dist(X_test, X_train)
        train_test_dists = torch.nan_to_num(
            train_test_dists, nan=train_test_dists[~train_test_dists.isnan()].max().item()
        )

        # Convert to numpy
        train_dists = train_dists.detach().cpu().numpy()
        train_test_dists = train_test_dists.detach().cpu().numpy()

        # Train classifier on distances
        knn = knn_class(metric="precomputed")
        t2 = time.time()
        knn.fit(train_dists, y_train_np)
        t3 = time.time()
        accs["knn"] = _score(train_test_dists, y_test_np, knn, torch=False)
        accs["knn"]["time"] = t3 - t1

    if "perceptron" in models:
        loss = "perceptron" if task == "classification" else "squared_error"
        ptron = perceptron_class(
            loss=loss, learning_rate="constant", fit_intercept=False, eta0=1.0, max_iter=10_000
        )  # fit_intercept must be false for ambient coordinates
        t1 = time.time()
        ptron.fit(X_train_np, y_train_np)
        t2 = time.time()
        accs["perceptron"] = _score(X_test_np, y_test_np, ptron, torch=False)
        accs["perceptron"]["time"] = t2 - t1

    if "ps_perceptron" in models:
        if task == "classification":
            ps_per = ProductSpacePerceptron(pm=pm)
            t1 = time.time()
            ps_per.fit(X_train, y_train)
            t2 = time.time()
            accs["ps_perceptron"] = _score(X_test, y_test_np, ps_per, torch=True)
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
        accs["svm"] = _score(train_test_ips, y_test_np, svm, torch=False)
        accs["svm"]["time"] = t3 - t1

    if "ps_svm" in models:
        try:
            ps_svm = ProductSpaceSVM(pm=pm, task=task, h_constraints=False, e_constraints=False)
            t1 = time.time()
            ps_svm.fit(X_train, y_train)
            t2 = time.time()
            accs["ps_svm"] = _score(X_test, y_test_np, ps_svm, torch=False)
            accs["ps_svm"]["time"] = t2 - t1
        except Exception:
            pass
            #     accs["ps_svm"] = {"accuracy": 0.0, "f1-micro": 0.0, "time": 0.0}

    # if "tangent_mlp" in models:
    #     tangent_mlp = MLP(pm=pm, tangent=True, **nn_kwargs).to(device)
    #     t1 = time.time()
    #     tangent_mlp.fit(X_train_tangent, y_train, **nn_train_kwargs)
    #     t2 = time.time()
    #     accs["tangent_mlp"] = _score(X_test_tangent, y_test_np, tangent_mlp, torch=True)
    #     accs["tangent_mlp"]["time"] = t2 - t1

    # if "ambient_mlp" in models:
    #     ambient_mlp = MLP(pm=pm, tangent=False, **nn_kwargs).to(device)
    #     t1 = time.time()
    #     ambient_mlp.fit(X_train, y_train, **nn_train_kwargs)
    #     t2 = time.time()
    #     accs["ambient_mlp"] = _score(X_test, y_test_np, ambient_mlp, torch=True)
    #     accs["ambient_mlp"]["time"] = t2 - t1

    # if "tangent_gnn" in models:
    #     tangent_gnn = GNN(pm=pm, tangent=True, **nn_kwargs).to(device)
    #     t1 = time.time()
    #     tangent_gnn.fit(X_tangent, y, adj=A_hat, train_idx=train_idx, **nn_train_kwargs)
    #     t2 = time.time()
    #     y_pred = tangent_gnn.predict(X_tangent, adj=A_hat, test_idx=test_idx)
    #     accs["tangent_gnn"] = _score(None, y_test_np, tangent_gnn, y_pred_override=y_pred, torch=True)
    #     accs["tangent_gnn"]["time"] = t2 - t1

    if "ambient_mlp" in models:
        ambient_mlp = KappaGCN(pm=pm_euc, **nn_kwargs).to(device)
        t1 = time.time()
        ambient_mlp.fit(X_train, y_train, A=None, **nn_train_kwargs)
        t2 = time.time()
        y_pred = ambient_mlp.predict(X_test, A=None)
        accs["ambient_mlp"] = _score(None, y_test_np, ambient_mlp, y_pred_override=y_pred, torch=True)
        accs["ambient_mlp"]["time"] = t2 - t1
    
    if "tangent_mlp" in models:
        tangent_mlp = KappaGCN(pm=pm_euc, **nn_kwargs).to(device)
        t1 = time.time()
        tangent_mlp.fit(X_train_tangent, y_train, A=None, **nn_train_kwargs)
        t2 = time.time()
        y_pred = tangent_mlp.predict(X_test_tangent, A=None)
        accs["tangent_mlp"] = _score(None, y_test_np, tangent_mlp, y_pred_override=y_pred, torch=True)
        accs["tangent_mlp"]["time"] = t2 - t1

    if "ambient_gnn" in models:
        ambient_gnn = KappaGCN(pm=pm_euc, **nn_kwargs).to(device)
        t1 = time.time()
        ambient_gnn.fit(X_train, y_train, A=A_train, **nn_train_kwargs)
        t2 = time.time()
        y_pred = ambient_gnn.predict(X_test, A=A_test)
        accs["ambient_gnn"] = _score(None, y_test_np, ambient_gnn, y_pred_override=y_pred, torch=True)
        accs["ambient_gnn"]["time"] = t2 - t1

    if "tangent_gnn" in models:
        tangent_gnn = KappaGCN(pm=pm_euc, **nn_kwargs).to(device)
        t1 = time.time()
        tangent_gnn.fit(X_train_tangent, y_train, A=A_train, **nn_train_kwargs)
        t2 = time.time()
        y_pred = tangent_gnn.predict(X_test_tangent, A=A_test)
        accs["ambient_gnn"] = _score(None, y_test_np, tangent_gnn, y_pred_override=y_pred, torch=True)
        accs["ambient_gnn"]["time"] = t2 - t1

    if "kappa_gcn" in models:
        d = X_test_stereo.shape[1] # Shape can't change between layers
        kappa_gcn = KappaGCN(pm=pm_stereo, hidden_dims=[d, d], task=task, output_dim=nn_outdim).to(device)
        t1 = time.time()
        kappa_gcn.fit(X_train_stereo, y_train, A=A_train, **nn_train_kwargs)
        t2 = time.time()
        y_pred = kappa_gcn.predict(X_test_stereo, A=A_test)
        accs["kappa_gcn"] = _score(None, y_test_np, None, y_pred_override=y_pred, torch=True)
        accs["kappa_gcn"]["time"] = t2 - t1
    
    if "product_mlr" in models:
        mlr_model = KappaGCN(pm=pm_stereo, hidden_dims=[], task=task, output_dim=nn_outdim).to(device)
        t1 = time.time()
        mlr_model.fit(X_train_stereo, y_train, A=A_train, **nn_train_kwargs)
        t2 = time.time()
        y_pred = mlr_model.predict(X_test_stereo, A=A_test)
        accs["product_mlr"] = _score(None, y_test_np, None, y_pred_override=y_pred, torch=True)
        accs["product_mlr"]["time"] = t2 - t1

    # return accs
    return {
        **{f"{model}_{metric}": value for model, metrics in accs.items() if isinstance(metrics, dict) for metric, value in metrics.items()},
        **{k: v for k, v in accs.items() if not isinstance(v, dict)}
    }
