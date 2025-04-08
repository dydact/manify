from sklearn.metrics import f1_score
from tqdm.notebook import tqdm
import time
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import torch
import manify

DEVICE = torch.device("cuda", 1)  # Use the 2nd GPU

DATASETS = ["dolphins", "football", "karate_club", "lesmis", "polbooks", "adjnoun"]
SIGNATURE = [(-1, 2), (0, 2), (1, 2)]
N_TRIALS = 10
TOTAL_ITERATIONS = 1_000
USE_DISTS = True
USE_TQDM = True
MODELS = [
    "sklearn_dt",
    "sklearn_rf",
    "product_dt",
    "product_rf",
    "tangent_dt",
    "tangent_rf",
    "single_manifold_rf",
    "knn",
    "ps_perceptron",
    # "ambient_mlp",
    # "ambient_gnn",
    # "kappa_gcn",
    # "product_mlr",
]
LR = 1e-4
EPOCHS = 4_000

results = []

# Copied from notebook 22


def make_link_prediction_dataset(X_embed, pm, adj, add_dists=True):
    # Stack embeddings
    emb = []
    for i in range(len(X_embed)):
        for j in range(len(X_embed)):
            joint_embed = torch.cat([X_embed[i], X_embed[j]])
            emb.append(joint_embed)

    X = torch.stack(emb)

    # Add distances
    if add_dists:
        dists = pm.pdist(X_embed)
        X = torch.cat([X, dists.flatten().unsqueeze(1)], dim=1)

    # y = torch.tensor(adj.flatten())
    if not torch.is_tensor(adj):
        adj = torch.tensor(adj)
    y = adj.flatten()

    # Make a new signature
    new_sig = pm.signature + pm.signature
    if add_dists:
        new_sig.append((0, 1))
    new_pm = manify.manifolds.ProductManifold(signature=new_sig)

    return X, y, new_pm


# for dataset in ["karate_club"]:
my_tqdm = tqdm(total=N_TRIALS * len(DATASETS))
for i, dataset in enumerate(DATASETS):
    dists, _, adj = manify.utils.dataloaders.load(dataset)
    dists, adj = dists.to(DEVICE), adj.to(DEVICE)
    dists = dists / dists[dists.isfinite()].max()

    # while len(results) < N_TRIALS:
    for seed in range(N_TRIALS):
        seed = seed + i * N_TRIALS  # Unique
        pm = manify.manifolds.ProductManifold(signature=SIGNATURE, device=DEVICE)
        X, _ = manify.embedders.coordinate_learning.train_coords(
            pm=pm,
            dists=dists,
            burn_in_iterations=int(0.1 * TOTAL_ITERATIONS),
            training_iterations=int(0.9 * TOTAL_ITERATIONS),
            scale_factor_learning_rate=0.02,
            device=DEVICE,
        )
        assert not torch.isnan(X).any()

        # Get data for classification variants
        XX, yy, pm_new = make_link_prediction_dataset(X, pm, adj, add_dists=USE_DISTS)
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            XX, yy, list(range(len(yy))), test_size=0.2
        )
        X_train = X_train[:1000]
        X_test = X_test[:1000]
        y_train = y_train[:1000]
        y_test = y_test[:1000]
        idx_train = idx_train[:1000]
        idx_test = idx_test[:1000]
        res = manify.utils.benchmarks.benchmark(
            # XX,
            # yy,
            None,
            None,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            pm=pm_new,
            task="classification",
            score=["accuracy", "f1-micro"],
            device=DEVICE,
            models=MODELS,
            seed=seed,
        )

        # Other manifolds we'll need
        pm_stereo, X_stereo = pm.stereographic(X)
        pm_stereo_euc = manify.manifolds.ProductManifold(signature=[(0, X.shape[1])], stereographic=True, device=DEVICE)

        # Get an adjacency matrix that's not leaky
        dists = pm.pdist2(X)
        max_dist = dists[dists.isfinite()].max()
        dists /= max_dist
        A = torch.exp(-dists)
        A_hat = manify.predictors.kappa_gcn.get_A_hat(A).float().to(DEVICE)

        # Ambient GCN
        agnn = manify.predictors.kappa_gcn.KappaGCN(
            pm=pm_stereo_euc, output_dim=1, hidden_dims=[pm_stereo_euc.dim], task="link_prediction"
        ).to(DEVICE)
        t1 = time.time()
        agnn.fit(X=X, y=y_train, A=A_hat, lr=LR, epochs=EPOCHS, lp_indices=idx_train, use_tqdm=USE_TQDM)
        t2 = time.time()
        y_pred = agnn.predict(X, A_hat)[idx_test]
        res["ambient_gcn_accuracy"] = (y_pred == y_test).float().mean().item()
        res["ambient_gcn_f1_micro"] = f1_score(y_test.cpu(), y_pred.cpu(), average="micro")
        res["ambient_gcn_time"] = t2 - t1

        # Tangent GCN
        tgcn = manify.predictors.kappa_gcn.KappaGCN(
            pm=pm_stereo, output_dim=1, hidden_dims=[pm_stereo.dim], task="link_prediction"
        ).to(DEVICE)
        t1 = time.time()
        tgcn.fit(
            X=pm.logmap(X).detach(), y=y_train, A=A_hat, lr=LR, epochs=EPOCHS, lp_indices=idx_train, use_tqdm=USE_TQDM
        )
        t2 = time.time()
        y_pred = tgcn.predict(pm.logmap(X).detach(), A_hat)[idx_test]
        res["tangent_gcn_accuracy"] = (y_pred == y_test).float().mean().item()
        res["tangent_gcn_f1_micro"] = f1_score(y_test.cpu(), y_pred.cpu(), average="micro")
        res["tangent_gcn_time"] = t2 - t1

        # Kappa GCN
        kgcn = manify.predictors.kappa_gcn.KappaGCN(
            pm=pm_stereo, output_dim=1, hidden_dims=[pm_stereo.dim], task="link_prediction"
        ).to(DEVICE)
        t1 = time.time()
        kgcn.fit(X=X_stereo, y=y_train, A=A_hat, lr=LR, epochs=EPOCHS, lp_indices=idx_train, use_tqdm=USE_TQDM)
        t2 = time.time()
        y_pred = kgcn.predict(X_stereo, A_hat)[idx_test]
        res["kappa_gcn_accuracy"] = (y_pred == y_test).float().mean().item()
        res["kappa_gcn_f1_micro"] = f1_score(y_test.cpu(), y_pred.cpu(), average="micro")
        res["kappa_gcn_time"] = t2 - t1

        # Ambient MLP
        amlp = manify.predictors.kappa_gcn.KappaGCN(
            pm=pm_stereo, output_dim=1, hidden_dims=[pm_stereo.dim], task="link_prediction"
        ).to(DEVICE)
        t1 = time.time()
        amlp.fit(
            X=X,
            y=y_train,
            A=torch.eye(len(X), device=DEVICE),
            lr=LR,
            epochs=EPOCHS,
            lp_indices=idx_train,
            use_tqdm=USE_TQDM,
        )
        t2 = time.time()
        y_pred = amlp.predict(X, torch.eye(len(X), device=DEVICE))[idx_test]
        res["ambient_mlp_accuracy"] = (y_pred == y_test).float().mean().item()
        res["ambient_mlp_f1_micro"] = f1_score(y_test.cpu(), y_pred.cpu(), average="micro")
        res["ambient_mlp_time"] = t2 - t1

        # Tangent MLP
        tmlp = manify.predictors.kappa_gcn.KappaGCN(
            pm=pm_stereo, output_dim=1, hidden_dims=[pm_stereo.dim], task="link_prediction"
        ).to(DEVICE)
        t1 = time.time()
        tmlp.fit(
            X=pm.logmap(X).detach(),
            y=y_train,
            A=torch.eye(len(X), device=DEVICE),
            lr=LR,
            epochs=EPOCHS,
            lp_indices=idx_train,
            use_tqdm=USE_TQDM,
        )
        t2 = time.time()
        y_pred = tmlp.predict(pm.logmap(X).detach(), torch.eye(len(X), device=DEVICE))[idx_test]
        res["tangent_mlp_accuracy"] = (y_pred == y_test).float().mean().item()
        res["tangent_mlp_f1_micro"] = f1_score(y_test.cpu(), y_pred.cpu(), average="micro")
        res["tangent_mlp_time"] = t2 - t1

        # Ambient MLR
        amlr = manify.predictors.kappa_gcn.KappaGCN(
            pm=pm_stereo, output_dim=1, hidden_dims=[], task="link_prediction"
        ).to(DEVICE)
        t1 = time.time()
        amlr.fit(
            X=X,
            y=y_train,
            A=torch.eye(len(X), device=DEVICE),
            lr=LR,
            epochs=EPOCHS,
            lp_indices=idx_train,
            use_tqdm=USE_TQDM,
        )
        t2 = time.time()
        y_pred = amlr.predict(X, torch.eye(len(X), device=DEVICE))[idx_test]
        res["ambient_mlr_accuracy"] = (y_pred == y_test).float().mean().item()
        res["ambient_mlr_f1_micro"] = f1_score(y_test.cpu(), y_pred.cpu(), average="micro")
        res["ambient_mlr_time"] = t2 - t1

        # Tangent MLR
        tmlr = manify.predictors.kappa_gcn.KappaGCN(
            pm=pm_stereo, output_dim=1, hidden_dims=[], task="link_prediction"
        ).to(DEVICE)
        t1 = time.time()
        tmlr.fit(
            X=pm.logmap(X).detach(),
            y=y_train,
            A=torch.eye(len(X)),
            lr=LR,
            epochs=EPOCHS,
            lp_indices=idx_train,
            use_tqdm=USE_TQDM,
        )
        t2 = time.time()
        y_pred = tmlr.predict(pm.logmap(X).detach(), torch.eye(len(X), device=DEVICE))[idx_test]
        res["tangent_mlr_accuracy"] = (y_pred == y_test).float().mean().item()
        res["tangent_mlr_f1_micro"] = f1_score(y_test.cpu(), y_pred.cpu(), average="micro")
        res["tangent_mlr_time"] = t2 - t1

        # Product MLR
        mlr = manify.predictors.kappa_gcn.KappaGCN(
            pm=pm_stereo, output_dim=1, hidden_dims=[], task="link_prediction"
        ).to(DEVICE)
        t1 = time.time()
        kgcn.fit(
            X=X_stereo,
            y=y_train,
            A=torch.eye(len(X_stereo), device=DEVICE),
            lr=LR,
            epochs=EPOCHS,
            lp_indices=idx_train,
            use_tqdm=USE_TQDM,
        )
        t2 = time.time()
        y_pred = kgcn.predict(X_stereo, torch.eye(len(X_stereo), device=DEVICE))[idx_test]
        res["kappa_mlr_accuracy"] = (y_pred == y_test).float().mean().item()
        res["kappa_mlr_f1_micro"] = f1_score(y_test.cpu(), y_pred.cpu(), average="micro")
        res["kappa_mlr_time"] = t2 - t1

        # Other details
        res["d_avg"] = manify.embedders.losses.d_avg(pm.pdist(X), dists).item()
        res["dataset"] = dataset

        results.append(res)
        my_tqdm.update(1)

results_df = pd.DataFrame(results)
results_df.to_csv("/home/phil/manify/data/results_icml_revision/link_prediction.tsv", sep="\t", index=False)
