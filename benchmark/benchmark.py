import torch
import numpy as np
import pandas as pd
import yaml
import wandb
import sys
from tqdm.notebook import tqdm
from pathlib import Path
from manify.utils.dataloaders_old import load
from manify.utils.benchmarks import benchmark
from manify.manifolds import ProductManifold
from manify.embedders.coordinate_learning import train_coords
from manify.utils.link_prediction import make_link_prediction_dataset, split_dataset
from manify.predictors.kappa_gcn import get_A_hat
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Load configuration from YAML file
path_to_config = sys.argv[1]
with open(path_to_config) as f:
    cfg = yaml.safe_load(f)

wandb.init(project="manifold_benchmarks", config=cfg)

OUTPUT_DIR = Path(cfg["OUTPUT_DIR"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device(cfg["DEVICE"])


# Common Benchmark Function
def run_benchmark(pm, X, y, task, signature, dataset_name, seed, extra_kwargs=None):
    print(f"Running benchmark for {dataset_name} | {signature} | {seed}")
    try:
        scores = (
            ["f1-micro", "f1-macro", "accuracy"] if task in ["classification", "link_prediction"] else ["mse", "rmse"]
        )

        result = benchmark(
            X,
            y,
            pm,
            task=task,
            seed=seed,
            device=device,
            score=scores,
            **cfg["BENCHMARK_KWARGS"],
            **(extra_kwargs or {}),
        )

        result.update({"dataset": dataset_name, "signature": signature, "seed": seed})
        wandb.log(result)
        wandb.run.summary.update(result)
        return result
    except Exception as e:
        wandb.log({"error": str(e), "dataset": dataset_name, "signature": signature, "seed": seed})
        print(f"Benchmark error [{dataset_name} | {signature}]: {e}")
        return None


def reorder_columns(df):
    first_cols = ["dataset", "signature", "seed"]
    metrics_order = ["f1-micro", "f1-macro", "accuracy", "mse", "rmse", "time"]

    other_cols = [col for col in df.columns if col not in first_cols]

    ordered_metrics = []
    for metric in metrics_order:
        ordered_metrics.extend(sorted([col for col in other_cols if col.endswith(f"_{metric}")]))

    # Include any remaining columns not covered by the metrics_order
    remaining_cols = [col for col in other_cols if col not in ordered_metrics]

    return df[first_cols + ordered_metrics + remaining_cols]


# Helper to save results
def save_results(results, filename):
    results_df = pd.DataFrame([r for r in results if r])
    results_df = reorder_columns(results_df)
    results_df.to_csv(OUTPUT_DIR / filename, sep="\t", index=False)


# Execute benchmarks
seed = cfg["RUNNING_SEED"]
for bench in cfg["BENCHMARKS"]:
    datasets, signatures, sig_strs = bench["datasets"], bench["signatures"], bench["signature_str"]
    tasks = bench["task"]

    # Convert single string to list for task/dataset
    if isinstance(tasks, str):
        tasks = [tasks] * len(signatures)
    elif len(tasks) != len(signatures):
        raise ValueError("Length of tasks and signatures must match, or task must be a single string.")
    if isinstance(datasets, str):
        datasets = [datasets] * len(signatures)
    elif len(datasets) != len(signatures):
        raise ValueError("Length of datasets and signatures must match, or dataset must be a single string.")

    tqdm_desc = f"Benchmarking {bench['type']}"
    results = []

    if bench["type"] == "gaussian":
        n_trials = cfg["N_TRIALS"]["gaussian"]
    else:
        n_trials = cfg["N_TRIALS"]["default"]

    with tqdm(total=len(datasets) * n_trials, desc=tqdm_desc) as pbar:
        for dataset, sig, sigstr, task in zip(datasets, signatures, sig_strs, tasks):
            for trial in range(n_trials):
                seed += 1
                pm = ProductManifold(signature=sig, device=device)

                X, y = None, None
                extra_kwargs = {}

                if bench["type"] == "gaussian":
                    X, y = pm.gaussian_mixture(seed=seed, task=task, **cfg["GAUSSIAN_KWARGS"])

                elif bench["type"] == "graph":
                    _, y, adj = load(dataset, bypassed=cfg["BYPASS_TOP_CC"])
                    adj = adj.float().to(device)
                    A_hat = get_A_hat(adj)
                    data = torch.load(f"data/graphs/embeddings/{dataset}/{sigstr}_{trial}.h5", weights_only=True)

                    # Get train and test indices
                    idx_test = data["test_idx"]
                    idx_train = [i for i in range(A_hat.shape[0]) if i not in idx_test]

                    extra_kwargs = {
                        "X_train": data["X_train"],
                        "y_train": data["y_train"],
                        "X_test": data["X_test"],
                        "y_test": data["y_test"],
                        "A_train": A_hat[idx_train][:, idx_train],
                        "A_test": A_hat[idx_test][:, idx_test],
                    }

                elif bench["type"] == "vae":
                    X_train = torch.tensor(np.load(f"data/{dataset}/embeddings/X_train_{trial}.npy"), device=device)
                    y_train = torch.tensor(np.load(f"data/{dataset}/embeddings/y_train_{trial}.npy"), device=device)
                    X_test = torch.tensor(np.load(f"data/{dataset}/embeddings/X_test_{trial}.npy"), device=device)
                    y_test = torch.tensor(np.load(f"data/{dataset}/embeddings/y_test_{trial}.npy"), device=device)

                    extra_kwargs = {
                        "X_train": X_train[: cfg["N_POINTS"]],
                        "y_train": y_train[: cfg["N_POINTS"]],
                        "X_test": X_test[: cfg["N_POINTS"]],
                        "y_test": y_test[: cfg["N_POINTS"]],
                    }

                elif bench["type"] == "empirical":
                    X, y, _ = load(dataset, seed=seed)

                elif bench["type"] == "link_prediction":
                    dists, _, adj = load(dataset, bypassed=cfg["BYPASS_TOP_CC"])
                    dists = (dists / dists.max()).to(device)
                    X_embed, _ = train_coords(pm, dists, **cfg["LINK_PREDICTION_KWARGS"], device=device)

                    X, y, pm = make_link_prediction_dataset(X_embed, pm, adj)

                # Consistent train-test split
                if X is not None and y is not None:
                    if bench["type"] == "link_prediction":
                        X_train, X_test, y_train, y_test, idx_train, idx_test = split_dataset(
                            X, y, test_size=0.2, random_state=seed
                        )
                        extra_kwargs.update(
                            {
                                "lp_train_idx": idx_train,
                                "lp_test_idx": idx_test,
                            }
                        )
                    else:
                        if len(X) > cfg["N_POINTS"]:
                            idx = np.random.choice(len(X), cfg["N_POINTS"], replace=False)
                            X, y = X[idx], y[idx]

                        X_cpu, y_cpu = X.cpu().numpy(), y.cpu().numpy()

                        X_train, X_test, y_train, y_test = train_test_split(
                            X_cpu,
                            y_cpu,
                            test_size=0.2,
                            random_state=seed,
                            stratify=y_cpu if task == "classification" else None,
                        )

                        X_train = torch.tensor(X_train, device=device)
                        X_test = torch.tensor(X_test, device=device)
                        y_train = torch.tensor(y_train, device=device)
                        y_test = torch.tensor(y_test, device=device)

                    extra_kwargs.update({"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test})

                res = run_benchmark(
                    pm,
                    None,
                    None,
                    task,
                    sigstr,
                    dataset,
                    seed,
                    extra_kwargs=extra_kwargs,
                )

                results.append(res)
                pbar.update(1)

    save_results(results, bench["output_file"])
