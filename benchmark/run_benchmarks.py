import argparse
import os
import sys
import yaml
import json
import time
import re
import logging
import traceback  # Import traceback for detailed error reporting
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

# --- Configure Logging ---
# Configure logging early
logging.basicConfig(
    level=logging.INFO,  # Default level, can be overridden by args
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],  # Use stdout
    # force=True # Uncomment if needed (Python 3.8+)
)
logger = logging.getLogger("manify_benchmark_single")
# --- End Logging Configuration ---

import numpy as np
import pandas as pd
import torch
import wandb

# --- Project Setup ---
# Add parent directory to path for local imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import manify components AFTER potentially modifying sys.path
try:
    import manify
    from manify.manifolds import ProductManifold
    from manify.utils.benchmarks import benchmark

    # Optional: Keep get_A_hat if kappa_gcn models are used
    try:
        from manify.predictors.kappa_gcn import get_A_hat
    except ImportError:
        logger.warning("Could not import get_A_hat from manify.predictors.kappa_gcn. KappaGCN models may fail.")
        get_A_hat = None  # Define as None to avoid NameErrors

    from manify.utils.link_prediction import make_link_prediction_dataset
    from manify.utils import dataloaders
except ImportError as e:
    logger.error(
        f"Failed to import manify components: {e}. Ensure manify is installed and project structure is correct."
    )
    sys.exit(1)
# --- End Project Setup ---


def load_config(config_path: str) -> Dict[str, Any]:
    """Load benchmark configuration with inheritance (simplified)."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Ensure 'common' exists for defaults
    if "common" not in config:
        config["common"] = {}

    # Define default common parameters if missing in common section
    common_defaults = {
        "n_trials": 10,
        "n_points": 1000,
        "device": "cpu",  # Default to CPU if not specified anywhere
        "output_dir": "results",
        "models": ["sklearn_dt", "sklearn_rf", "product_dt", "product_rf"],
        "classification_metrics": ["accuracy", "f1-micro"],
        "regression_metrics": ["rmse"],
        "max_depth": 5,  # Default max_depth
        "task": "classification",  # Default task
    }
    for key, value in common_defaults.items():
        config["common"].setdefault(key, value)

    # Define default gaussian parameters if missing in gaussian section
    if "gaussian" not in config:
        config["gaussian"] = {}
    gaussian_defaults = {
        "n_classes": 8,
        "n_clusters": 32,
        "cov_scale_means": 1.0,
        "cov_scale_points": 1.0,
        "n_trials_gaussian": config["common"]["n_trials"],  # Default to common n_trials
    }
    for key, value in gaussian_defaults.items():
        config["gaussian"].setdefault(key, value)

    common_params = list(common_defaults.keys())
    gaussian_params = list(gaussian_defaults.keys())

    # Propagate common params to specific benchmarks if not overridden
    for benchmark_type in config:
        if benchmark_type not in ["common", "gaussian", "wandb"]:
            if not isinstance(config[benchmark_type], dict):
                config[benchmark_type] = {}
            for param in common_params:
                config[benchmark_type].setdefault(param, config["common"][param])

    # Propagate gaussian params to relevant benchmarks if not overridden
    for benchmark_type in ["single_curvature", "signature_gaussian"]:
        if benchmark_type in config:
            if not isinstance(config[benchmark_type], dict):
                config[benchmark_type] = {}
            for param in gaussian_params:
                config[benchmark_type].setdefault(param, config["gaussian"][param])
            # Special case: ensure n_trials_gaussian is set
            config[benchmark_type].setdefault("n_trials_gaussian", config["gaussian"]["n_trials_gaussian"])

    # Final check: ensure 'models' is a list in common and propagated
    if not isinstance(config["common"]["models"], list):
        config["common"]["models"] = common_defaults["models"]  # Fallback
    for benchmark_type in config:
        if benchmark_type not in ["common", "gaussian", "wandb"] and "models" not in config[benchmark_type]:
            config[benchmark_type]["models"] = config["common"]["models"]

    return config


def setup_wandb(config: Dict[str, Any], config_path: str, args: argparse.Namespace) -> bool:
    """Initialize wandb for experiment tracking. Returns True if enabled."""
    if args.dry_run or args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
        logger.info("WandB logging is disabled.")
        return False
    else:
        os.environ["WANDB_MODE"] = "online"

    wandb_config = config.get("wandb", {})
    run_name = wandb_config.get("name", f"benchmark-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    try:
        wandb.init(
            project=wandb_config.get("project", "manify-benchmarks-single"),  # Adjust project name
            entity=wandb_config.get("entity"),
            name=run_name,
            config=config,  # Log the effective, potentially overridden config
            # No need for special start_method settings in single process
        )
        logger.info(f"WandB run '{wandb.run.name}' initialized (ID: {wandb.run.id})")

        # --- Artifact logging ---
        config_yaml_path = os.path.join(wandb.run.dir, "effective_config.yaml")
        with open(config_yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        yaml_artifact = wandb.Artifact(
            name=f"config-{Path(config_path).stem}-{wandb.run.id}",
            type="config_file",
            description=f"Benchmark configuration for run {wandb.run.id}",
        )
        yaml_artifact.add_file(config_path, name="original_config.yaml")
        yaml_artifact.add_file(config_yaml_path, name="effective_config.yaml")
        wandb.log_artifact(yaml_artifact)
        logger.info("Logged configuration artifact to WandB.")

        # --- Update wandb.config with key parameters ---
        important_params = {
            "command_line_args": vars(args),
            "effective_benchmark_types": args.benchmark_types,
            "global_seed": args.seed,
            "effective_device": config.get("common", {}).get("device"),  # Log the device actually used
            "common_models": config.get("common", {}).get("models", []),
            "common_n_trials": config.get("common", {}).get("n_trials"),
        }
        wandb.config.update(important_params, allow_val_change=True)

        # --- Config Summary Table ---
        summary_items = []
        for section, params in config.items():
            if section != "wandb" and isinstance(params, dict):
                for key, value in params.items():
                    if not isinstance(value, (dict)) and (not isinstance(value, (list, tuple)) or len(value) < 10):
                        summary_items.append({"section": section, "parameter": key, "value": str(value)})
        if summary_items:
            try:
                summary_df = pd.DataFrame(summary_items)
                wandb.log({"config_summary": wandb.Table(dataframe=summary_df)}, step=0)
                logger.info("Logged config summary table to WandB.")
            except Exception as e:
                logger.error(f"Failed to log config summary table: {e}")

        return True  # WandB is enabled

    except Exception as e:
        logger.error(f"Failed to initialize WandB: {e}")
        os.environ["WANDB_MODE"] = "disabled"  # Disable if init fails
        return False


def get_device(device_str: str) -> torch.device:
    """Get torch device based on a string identifier (e.g., 'cuda:0', 'mps', 'cpu')."""
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        try:
            # Check if specific device index is requested
            if ":" in device_str:
                gpu_id = int(device_str.split(":")[1])
                if gpu_id < torch.cuda.device_count():
                    device = torch.device(device_str)
                    device_name = torch.cuda.get_device_name(device)
                    logger.info(f"Using specified CUDA device {gpu_id}: {device_name}")
                    return device
                else:
                    logger.warning(
                        f"Requested GPU {gpu_id} not available ({torch.cuda.device_count()} devices detected). Falling back to default CUDA device."
                    )
                    device = torch.device("cuda")  # Fallback to cuda:0
            else:
                device = torch.device("cuda")  # Default cuda (usually cuda:0)

            # Log the device being used
            current_idx = torch.cuda.current_device() if device.index is None else device.index
            device_name = torch.cuda.get_device_name(current_idx)
            logger.info(f"Using CUDA device {current_idx}: {device_name}")
            return device

        except Exception as e:
            logger.error(f"Error setting CUDA device '{device_str}': {e}. Falling back to CPU.")
            return torch.device("cpu")

    elif device_str == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Using MPS device")
            return torch.device("mps")
        else:
            logger.warning("MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")
    else:
        if device_str != "cpu":
            logger.warning(f"Unrecognized device string '{device_str}'. Using CPU.")
        logger.info("Using CPU device")
        return torch.device("cpu")


import re  # Keep import


def log_wandb_results(base_metadata: Dict[str, Any], benchmark_results: Dict[str, Any]):
    """
    Logs results for each model in a trial to WandB, handling the flat
    dictionary structure returned by the benchmark function. CORRECTED PARSING.
    """
    logger.info(f"DEBUG_CHECK: Entered log_wandb_results for trial.")
    logger.info(f"DEBUG_CHECK: benchmark_results keys received: {list(benchmark_results.keys())}")
    # logger.info(f"DEBUG_CHECK: benchmark_results content: {benchmark_results}") # Optional

    if os.environ.get("WANDB_MODE") == "disabled":
        logger.debug("WANDB_LOG: Skipping log_wandb_results (disabled).")
        return
    if not benchmark_results or all(
        not isinstance(v, (int, float, np.number)) for k, v in benchmark_results.items() if k != "task"
    ):
        logger.warning(
            f"DEBUG_CHECK: benchmark_results dictionary seems empty or has no numeric results: {benchmark_results}. Skipping logging."
        )
        return

    grouped_results = {}
    # Define normalized metric names we expect to find at the end of keys
    # These should match the keys *after* replacing '-' with '_' in the original metric name
    known_metrics_normalized = ["accuracy", "f1", "f1_micro", "f1_macro", "auc", "mse", "rmse", "time"]

    for key, value in benchmark_results.items():
        if key == "task" or value is None:
            continue

        model_name_found = None
        metric_name_found = None

        # Iterate through KNOWN normalized metrics to see if the key ENDS with one
        for known_metric in known_metrics_normalized:
            # Construct the possible suffixes (_metric and -metric before normalization)
            suffix_underscore = "_" + known_metric
            suffix_hyphen = "_" + known_metric.replace("_", "-")  # Reconstruct potential original hyphen

            if key.endswith(suffix_underscore):
                # Found a match like "model_name_f1_micro"
                model_name_found = key[: -len(suffix_underscore)]
                metric_name_found = known_metric  # Use the normalized version
                break  # Stop checking metrics for this key
            elif key.endswith(suffix_hyphen):
                # Found a match like "model_name_f1-micro"
                model_name_found = key[: -len(suffix_hyphen)]
                metric_name_found = known_metric  # Use the normalized version
                break  # Stop checking metrics for this key

        if model_name_found is None or metric_name_found is None:
            logger.warning(f"DEBUG_CHECK: Could not parse model/metric from key '{key}'. Skipping.")
            continue

        # --- Initialize dict for model if needed ---
        if model_name_found not in grouped_results:
            grouped_results[model_name_found] = base_metadata.copy()
            # Add known numeric conversions from base_metadata (same as before)
            for k_meta, v_meta in base_metadata.items():
                if k_meta in ["seed", "trial", "curvature", "n_points", "n_classes", "n_clusters"]:
                    try:
                        grouped_results[model_name_found][k_meta] = float(v_meta)
                    except (ValueError, TypeError):
                        grouped_results[model_name_found][k_meta] = v_meta
                elif k_meta == "signature" and "signature_name" in base_metadata:
                    grouped_results[model_name_found]["signature_name"] = base_metadata["signature_name"]
                elif k_meta == "signature":
                    grouped_results[model_name_found][k_meta] = str(v_meta)
                else:
                    grouped_results[model_name_found][k_meta] = v_meta
            grouped_results[model_name_found]["model"] = model_name_found  # Add model name
        # --- End Initialization ---

        # --- Add the metric value (ensure numeric) ---
        try:
            numeric_value = float(value) if value is not None else None
            # Use the ALREADY NORMALIZED metric name as the key
            grouped_results[model_name_found][metric_name_found] = numeric_value
            logger.debug(
                f"DEBUG_CHECK: Parsed '{key}' -> model='{model_name_found}', metric='{metric_name_found}', value={numeric_value}"
            )
        except (ValueError, TypeError):
            logger.warning(
                f"Could not convert metric {metric_name_found}={value} to float for model {model_name_found}."
            )
            grouped_results[model_name_found][metric_name_found] = None
        # --- End Add Metric ---

    # --- Log one row per model ---
    if not grouped_results:
        logger.warning("DEBUG_CHECK: No models with metrics were successfully grouped. No data logged.")
        return

    # Define standard metrics with normalized names for padding
    standard_metrics_normalized = ["accuracy", "f1", "f1_micro", "f1_macro", "auc", "mse", "rmse", "time"]
    for model_name, wandb_row in grouped_results.items():
        # Ensure standard metrics exist in the row, even if None
        for std_metric_norm in standard_metrics_normalized:
            if std_metric_norm not in wandb_row:
                wandb_row[std_metric_norm] = None  # Or float('nan')

        try:
            logger.info(
                f"DEBUG_CHECK: About to call wandb.log for grouped model '{model_name}'. Keys: {list(wandb_row.keys())}"
            )
            wandb.log(wandb_row)
        except Exception as e:
            logger.error(
                f"Failed to log row to WandB for model {model_name}: {e} - Data Keys: {list(wandb_row.keys())}"
            )

    logger.info("DEBUG_CHECK: Finished iterating through grouped results.")


# --- Trial Generation (Unchanged from previous version) ---
def generate_trials(benchmark_type: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate trials for a specific benchmark type."""
    trials = []
    n_trials_common = config.get("n_trials", 10)  # Default common trials

    if benchmark_type == "single_curvature":
        curvatures = config.get("curvatures", [-2, -1, -0.5, 0, 0.5, 1, 2])
        n_trials = config.get("n_trials_gaussian", n_trials_common)
        tasks = config.get("tasks", ["classification", "regression"])

        for task in tasks:
            for K in curvatures:  # Use simple loop var
                for seed_idx in range(n_trials):
                    seed_value = seed_idx  # Use index as seed value
                    trials.append(
                        {"curvature": K, "seed": seed_value, "task": task, "trial_id": f"sc-{task}-k{K}-s{seed_value}"}
                    )

    elif benchmark_type == "signature_gaussian":
        signatures = config.get("signatures", [[(-1, 2), (0, 2)], [(-1, 2), (1, 2)], [(0, 2), (1, 2)]])
        signature_names = config.get("signature_names", ["HE", "HS", "ES"])  # Make sure names match signatures
        n_trials = config.get("n_trials_gaussian", n_trials_common)
        tasks = config.get("tasks", ["classification", "regression"])

        # Ensure names and signatures align
        if len(signature_names) != len(signatures):
            logger.warning(
                f"Mismatch between signature_names ({len(signature_names)}) and signatures ({len(signatures)}) in config for signature_gaussian. Using default names."
            )
            signature_names = [f"sig{i}" for i in range(len(signatures))]

        for task in tasks:
            for sig, sig_name in zip(signatures, signature_names):
                for seed_idx in range(n_trials):
                    seed_value = seed_idx
                    trials.append(
                        {
                            "signature": sig,
                            "signature_name": sig_name,
                            "seed": seed_value,
                            "task": task,
                            "trial_id": f"sg-{task}-{sig_name}-s{seed_value}",
                        }
                    )

    elif benchmark_type == "graph_embedding":
        embedding_names = config.get("embedding_names", ["polblogs", "cs_phds", "cora", "citeseer"])
        # Make signatures mapping more explicit
        default_sig_map = {"polblogs": "SS", "cora": "H", "citeseer": "HS", "cs_phds": "H"}
        signatures_map = config.get("signatures_map", default_sig_map)  # Allow overriding map
        n_trials = config.get("n_trials", n_trials_common)
        default_task = config.get("task", "classification")
        task_overrides = config.get("task_overrides", {})

        for emb_name in embedding_names:
            sig = signatures_map.get(emb_name)
            if sig is None:
                logger.warning(f"No signature mapped for graph embedding '{emb_name}'. Skipping.")
                continue  # Skip if no signature is defined

            task = task_overrides.get(emb_name, default_task)

            for trial_idx in range(n_trials):
                trials.append(
                    {
                        "embedding": emb_name,
                        "signature": sig,  # Store the string signature (e.g., "HS")
                        "trial": trial_idx,  # Keep 'trial' as the seed concept here
                        "task": task,
                        "trial_id": f"ge-{emb_name}-{sig}-t{trial_idx}",
                    }
                )

    elif benchmark_type == "vae_embedding":
        embedding_names = config.get("embedding_names", ["blood_cell_scrna", "lymphoma", "cifar_100", "mnist"])
        # Use a map for clarity
        default_sig_map = {
            "blood_cell_scrna": [(1, 2), (0, 2), (-1, 2), (-1, 2), (-1, 2)],
            "lymphoma": [(1, 2), (1, 2)],
            "cifar_100": [(1, 2), (1, 2), (1, 2), (1, 2)],
            "mnist": [(1, 2), (0, 2), (-1, 2)],
        }
        signatures_map = config.get("signatures_map", default_sig_map)
        n_trials = config.get("n_trials", n_trials_common)
        default_task = config.get("task", "classification")
        task_overrides = config.get("task_overrides", {})

        for emb_name in embedding_names:
            sig = signatures_map.get(emb_name)
            if sig is None:
                logger.warning(f"No signature mapped for VAE embedding '{emb_name}'. Skipping.")
                continue

            task = task_overrides.get(emb_name, default_task)

            for trial_idx in range(n_trials):
                trials.append(
                    {
                        "embedding": emb_name,
                        "signature": sig,  # Store the actual signature list
                        "trial": trial_idx,
                        "task": task,
                        "trial_id": f"vae-{emb_name}-t{trial_idx}",
                    }
                )

    elif benchmark_type == "link_prediction":
        graph_names = config.get("graph_names", ["karate_club", "lesmis", "dolphins", "polbooks"])
        # Allow multiple signatures to be tested per graph
        component_signatures = config.get("component_signatures", [[(-1, 2), (0, 2), (1, 2)]])
        component_names = config.get("component_names", ["HES"])
        n_trials = config.get("n_trials", n_trials_common)
        task = "link_prediction"

        if len(component_names) != len(component_signatures):
            logger.warning(
                f"Mismatch between component_names and component_signatures for link_prediction. Using default names."
            )
            component_names = [f"sig{i}" for i in range(len(component_signatures))]

        for graph_name in graph_names:
            for sig, sig_name in zip(component_signatures, component_names):
                for trial_idx in range(n_trials):
                    trials.append(
                        {
                            "graph": graph_name,
                            "signature": sig,
                            "signature_name": sig_name,
                            "trial": trial_idx,
                            "task": task,
                            "trial_id": f"lp-{graph_name}-{sig_name}-t{trial_idx}",
                        }
                    )

    elif benchmark_type == "empirical_benchmarks":
        dataset_names = config.get("dataset_names", ["landmasses", "neuron_33", "neuron_46", "temperature", "traffic"])
        # Use maps for signatures and tasks
        default_sig_map = {
            "landmasses": [(1, 2)],
            "neuron_33": [(1, 1)] * 10,
            "neuron_46": [(1, 1)] * 10,
            "temperature": [(1, 2), (1, 1)],
            "traffic": [(0, 1)] + [(1, 1)] * 4,
        }
        default_task_map = {
            "landmasses": "classification",
            "neuron_33": "classification",
            "neuron_46": "classification",
            "temperature": "regression",
            "traffic": "regression",
        }
        signatures_map = config.get("signatures_map", default_sig_map)
        tasks_map = config.get("tasks_map", default_task_map)
        n_trials = config.get("n_trials", n_trials_common)

        for dataset in dataset_names:
            sig = signatures_map.get(dataset)
            task = tasks_map.get(dataset)
            if sig is None or task is None:
                logger.warning(f"Missing signature or task mapping for empirical dataset '{dataset}'. Skipping.")
                continue

            for seed_idx in range(n_trials):
                trials.append(
                    {
                        "dataset": dataset,
                        "signature": sig,
                        "task": task,
                        "seed": seed_idx,
                        "trial_id": f"emp-{dataset}-{task}-s{seed_idx}",
                    }
                )

    else:
        logger.warning(f"Trial generation not implemented for benchmark type: {benchmark_type}")

    # Apply max_trials limit if specified
    max_trials_limit = config.get(f"max_trials_{benchmark_type}", config.get("max_trials"))
    if max_trials_limit is not None and len(trials) > max_trials_limit:
        logger.info(f"Limiting trials for {benchmark_type} from {len(trials)} to {max_trials_limit}")
        # Maybe randomize selection instead of just taking the first ones?
        # np.random.shuffle(trials) # Uncomment to randomize before limiting
        trials = trials[:max_trials_limit]

    return trials


# --- Individual Trial Runners (Adapted for Single Process) ---
# Keep the 'device' argument for clarity, passed from main loop.
# Remove logging messages that were specific to multi-GPU setups.


def run_single_curvature_trial(
    trial_params: Dict[str, Any], config: Dict[str, Any], models: List[str], device: torch.device
) -> Dict[str, Any]:
    """Run a single curvature benchmark trial."""
    K = trial_params["curvature"]
    seed = trial_params["seed"]
    task = trial_params["task"]
    logger.debug(f"Running single curvature trial: K={K}, seed={seed}, task={task}")

    dim = config.get("dimension", 2)
    n_points = config.get("n_points", 1000)
    n_classes = config.get("n_classes", 8)
    n_clusters = config.get("n_clusters", 32)
    cov_scale_means = config.get("cov_scale_means", 1.0)
    cov_scale_points = config.get("cov_scale_points", 1.0)
    score = config.get("classification_metrics") if task == "classification" else config.get("regression_metrics")

    try:
        # Manifold and data generation should happen on the target device
        pm = ProductManifold(signature=[(K, dim)], device=device)
        X, y = pm.gaussian_mixture(
            seed=seed,
            num_points=n_points,
            num_classes=n_classes,
            num_clusters=n_clusters,
            cov_scale_means=cov_scale_means / dim if dim > 0 else cov_scale_means,
            cov_scale_points=cov_scale_points / dim if dim > 0 else cov_scale_points,
            task=task,
        )
        # X, y are already on 'device' from gaussian_mixture if pm is on device
    except Exception as e:
        logger.error(f"Error during data generation for K={K}, seed={seed}: {e}", exc_info=True)
        return {"error": f"Data generation failed: {e}", "traceback": traceback.format_exc()}

    try:
        result = benchmark(X=X, y=y, pm=pm, task=task, score=score, seed=seed, device=device, models=models)
        result["task"] = task  # Ensure task is in result
        return result
    except Exception as e:
        logger.error(f"Error during benchmark execution for K={K}, seed={seed}: {e}", exc_info=True)
        return {"error": f"Benchmark execution failed: {e}", "traceback": traceback.format_exc()}


def run_signature_gaussian_trial(
    trial_params: Dict[str, Any], config: Dict[str, Any], models: List[str], device: torch.device
) -> Dict[str, Any]:
    """Run a signature gaussian benchmark trial."""
    signature = trial_params["signature"]
    signature_name = trial_params["signature_name"]
    seed = trial_params["seed"]
    task = trial_params["task"]
    logger.debug(f"Running signature gaussian trial: sig={signature_name}, seed={seed}, task={task}")

    n_points = config.get("n_points", 1000)
    n_classes = config.get("n_classes", 8)
    n_clusters = config.get("n_clusters", 32)
    cov_scale_means = config.get("cov_scale_means", 1.0)
    cov_scale_points = config.get("cov_scale_points", 1.0)
    score = config.get("classification_metrics") if task == "classification" else config.get("regression_metrics")

    try:
        pm = ProductManifold(signature=signature, device=device)
        total_dim = pm.dim
        X, y = pm.gaussian_mixture(
            seed=seed,
            num_points=n_points,
            num_classes=n_classes,
            num_clusters=n_clusters,
            cov_scale_means=cov_scale_means / total_dim if total_dim > 0 else cov_scale_means,
            cov_scale_points=cov_scale_points / total_dim if total_dim > 0 else cov_scale_points,
            task=task,
        )
    except Exception as e:
        logger.error(f"Error during data generation for sig={signature_name}, seed={seed}: {e}", exc_info=True)
        return {"error": f"Data generation failed: {e}", "traceback": traceback.format_exc()}

    try:
        result = benchmark(X=X, y=y, pm=pm, task=task, score=score, seed=seed, device=device, models=models)
        result["task"] = task
        return result
    except Exception as e:
        logger.error(f"Error during benchmark execution for sig={signature_name}, seed={seed}: {e}", exc_info=True)
        return {"error": f"Benchmark execution failed: {e}", "traceback": traceback.format_exc()}


def run_graph_embedding_trial(
    trial_params: Dict[str, Any], config: Dict[str, Any], models: List[str], device: torch.device
) -> Dict[str, Any]:
    """Run a graph embedding benchmark trial."""
    embedding_name = trial_params["embedding"]
    signature_str = trial_params["signature"]  # e.g., "HS"
    trial = trial_params["trial"]  # Seed
    task = trial_params["task"]
    logger.debug(
        f"Running graph embedding trial: dataset={embedding_name}, sig={signature_str}, trial={trial}, task={task}"
    )

    score = config.get("classification_metrics") if task == "classification" else config.get("regression_metrics")

    # Use data_dir and embeddings_dir from the specific benchmark config or common config
    data_root = Path(config.get("data_dir", project_root / "data"))  # Use common data_dir as root
    embeddings_dir = Path(config.get("embeddings_dir", data_root / "graphs" / "embeddings"))
    graph_data_dir = Path(config.get("graph_data_dir", data_root / "graphs"))  # Dir for .edges file

    data_path = embeddings_dir / embedding_name / f"{signature_str}_{trial}.h5"

    if not data_path.exists():
        logger.error(f"Embedding file not found: {data_path}")
        return {"error": f"Embedding file not found: {data_path}"}

    try:
        data = torch.load(data_path, map_location=device)  # Load directly to target device

        # --- Data Validation and Preparation ---
        required_keys = ["X_train", "X_test", "y_train", "y_test", "test_idx"]
        if not all(key in data for key in required_keys):
            missing = [k for k in required_keys if k not in data]
            raise KeyError(f"Embedding file missing keys: {missing}")

        X_train = data["X_train"].to(device)
        X_test = data["X_test"].to(device)
        y_train = data["y_train"].to(device)
        y_test = data["y_test"].to(device)
        test_idx = data["test_idx"]
        if isinstance(test_idx, torch.Tensor):
            test_idx = test_idx.cpu().tolist()  # Ensure list for indexing

        num_total_nodes = len(X_train) + len(X_test)
        all_indices = set(range(num_total_nodes))
        test_indices_set = set(test_idx)
        train_idx = sorted(list(all_indices - test_indices_set))
        # --- End Data Validation ---

        # --- Adjacency Matrix Loading (Optional) ---
        A_train, A_test = None, None
        # Only load if get_A_hat was imported successfully and needed by a model
        if get_A_hat and any(m in models for m in ["kappa_gcn", "some_other_gcn"]):  # Add models needing A_hat
            adj_path = graph_data_dir / f"{embedding_name}.edges"  # Standard location?
            try:
                # Use dataloader to get adj matrix
                _, _, adj = manify.utils.dataloaders.load(embedding_name, data_dir=str(graph_data_dir))
                if adj is not None:
                    adj = adj.float().cpu()  # Process on CPU
                    A_hat = get_A_hat(adj)
                    A_train = A_hat[train_idx][:, train_idx].to(device)
                    A_test = A_hat[test_idx][:, test_idx].to(device)
                else:
                    logger.warning(f"Adjacency matrix not loaded or is None for {embedding_name}.")
            except FileNotFoundError:
                logger.warning(
                    f"Adjacency file (.edges) not found for {embedding_name} at {graph_data_dir}. Skipping A_hat."
                )
            except Exception as e:
                logger.error(f"Error loading/processing adjacency matrix for {embedding_name}: {e}", exc_info=True)
        # --- End Adjacency Matrix ---

        # --- Manifold Setup ---
        sig_map = {  # Define the mapping from string to signature list
            "HH": [(-1, 2), (-1, 2)],
            "HE": [(-1, 2), (0, 2)],
            "HS": [(-1, 2), (1, 2)],
            "SS": [(1, 2), (1, 2)],
            "SE": [(1, 2), (0, 2)],
            "ES": [(0, 2), (1, 2)],
            "H": [(-1, 4)],
            "E": [(0, 4)],
            "S": [(1, 4)],
        }
        if signature_str not in sig_map:
            raise ValueError(f"Unknown signature string '{signature_str}' for graph embedding.")
        pm = ProductManifold(signature=sig_map[signature_str], device=device)
        # --- End Manifold Setup ---

        result = benchmark(
            X=None,
            y=None,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            pm=pm,
            device=device,
            A_train=A_train,
            A_test=A_test,
            task=task,
            score=score,
            models=models,
            seed=trial,
        )
        result["task"] = task
        return result

    except Exception as e:
        logger.error(f"Error during graph embedding trial {embedding_name}, trial {trial}: {e}", exc_info=True)
        return {"error": f"Benchmark execution failed: {e}", "traceback": traceback.format_exc()}


def run_vae_embedding_trial(
    trial_params: Dict[str, Any], config: Dict[str, Any], models: List[str], device: torch.device
) -> Dict[str, Any]:
    """Run a VAE embedding benchmark trial."""
    embedding_name = trial_params["embedding"]
    signature = trial_params["signature"]  # Actual signature list
    trial = trial_params["trial"]  # Seed
    task = trial_params["task"]
    logger.debug(f"Running VAE embedding trial: dataset={embedding_name}, trial={trial}, task={task}")

    n_points_train = config.get("n_points_train", config.get("n_points", 1000))
    n_points_test = config.get("n_points_test", config.get("n_points", 1000))
    score = config.get("classification_metrics") if task == "classification" else config.get("regression_metrics")

    data_root = Path(config.get("data_dir", project_root / "data"))
    # Default structure: data_root / dataset_name / embeddings / X_train_0.npy etc.
    data_dir = Path(config.get("vae_data_dir", data_root / embedding_name / "embeddings"))

    x_train_path = data_dir / f"X_train_{trial}.npy"
    y_train_path = data_dir / f"y_train_{trial}.npy"
    x_test_path = data_dir / f"X_test_{trial}.npy"
    y_test_path = data_dir / f"y_test_{trial}.npy"

    try:
        # --- File Check ---
        missing_files = []
        for p in [x_train_path, y_train_path, x_test_path, y_test_path]:
            if not p.exists():
                missing_files.append(str(p.name))
        if missing_files:
            raise FileNotFoundError(f"Missing VAE embedding files in {data_dir}: {missing_files}")
        # --- End File Check ---

        # --- Load Data (Numpy) ---
        X_train_np = np.load(x_train_path)
        y_train_np = np.load(y_train_path)
        X_test_np = np.load(x_test_path)
        y_test_np = np.load(y_test_path)
        # --- End Load Data ---

        # --- Subsampling (Reproducible) ---
        np.random.seed(trial)  # Seed for sampling
        if len(X_train_np) > n_points_train:
            idx_train = np.random.choice(X_train_np.shape[0], n_points_train, replace=False)
            X_train_np, y_train_np = X_train_np[idx_train], y_train_np[idx_train]
        if len(X_test_np) > n_points_test:
            idx_test = np.random.choice(X_test_np.shape[0], n_points_test, replace=False)
            X_test_np, y_test_np = X_test_np[idx_test], y_test_np[idx_test]
        # --- End Subsampling ---

        # --- Convert to Tensor and Move to Device ---
        X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
        y_dtype = torch.long if task == "classification" else torch.float32
        y_train = torch.tensor(y_train_np, dtype=y_dtype).to(device)
        y_test = torch.tensor(y_test_np, dtype=y_dtype).to(device)
        # --- End Conversion ---

        pm = ProductManifold(signature=signature, device=device)

        # --- Optional Adjacency Matrix (if needed) ---
        A_train, A_test = None, None
        if get_A_hat and config.get("generate_adj_for_vae", False) and any(m in models for m in ["kappa_gcn"]):
            logger.info(f"Generating adjacency matrices for VAE {embedding_name}, trial {trial}...")
            # Simplified placeholder - implement if needed, careful with memory
            A_train, A_test = None, None  # Implement actual calculation if required
            logger.warning("VAE Adjacency matrix generation not fully implemented in this version.")
        # --- End Adjacency Matrix ---

        result = benchmark(
            X=None,
            y=None,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            pm=pm,
            A_train=A_train,
            A_test=A_test,
            device=device,
            models=models,
            task=task,
            score=score,
            seed=trial,
        )
        result["task"] = task
        return result

    except Exception as e:
        logger.error(f"Error during VAE embedding trial {embedding_name}, trial {trial}: {e}", exc_info=True)
        return {"error": f"Benchmark execution failed: {e}", "traceback": traceback.format_exc()}


def run_link_prediction_trial(
    trial_params: Dict[str, Any], config: Dict[str, Any], models: List[str], device: torch.device
) -> Dict[str, Any]:
    """Run a link prediction benchmark trial."""
    graph_name = trial_params["graph"]
    signature = trial_params["signature"]
    signature_name = trial_params["signature_name"]
    trial = trial_params["trial"]  # Seed
    task = "link_prediction"
    logger.debug(f"Running link prediction trial: graph={graph_name}, sig={signature_name}, trial={trial}")

    # --- Config Params ---
    embedding_iterations = config.get("embedding_iterations", 5000)
    burn_in_ratio = config.get("burn_in_ratio", 0.1)
    test_size = config.get("test_size", 0.2)
    max_depth = config.get("max_depth", 3)
    add_dists_feature = config.get("add_dists_feature", True)
    learning_rate = config.get("learning_rate", 1e-4)
    score = config.get("classification_metrics", ["accuracy", "f1-micro"])  # AUC is good for LP
    n_points = config.get("downsample", 1000)
    # --- End Config ---

    data_root = Path(config.get("data_dir", project_root / "data"))

    try:
        # --- Load Graph Data ---
        dists_loaded, _, adj_loaded = manify.utils.dataloaders.load(graph_name)
        if dists_loaded is None or adj_loaded is None:
            raise ValueError(f"Dists or Adjacency matrix not loaded for graph {graph_name}")
        dists_tensor = dists_loaded.float().to(device)  # Ensure float
        adj = adj_loaded.float().to(device)
        n_nodes = adj.shape[0]
        # Optional distance normalization
        # ... (add if needed, same logic as before) ...
        # --- End Load Graph Data ---

        # --- Coordinate Learning ---
        torch.manual_seed(trial)
        np.random.seed(trial)
        pm = ProductManifold(signature=signature, device=device)
        from manify.embedders.coordinate_learning import train_coords  # Local import ok

        logger.info(f"Starting coordinate training for {graph_name}, trial {trial}...")
        X_embed, losses = train_coords(
            pm,
            dists_tensor,
            device=device,  # Pass device explicitly
            burn_in_iterations=int(burn_in_ratio * embedding_iterations),
            training_iterations=int((1.0 - burn_in_ratio) * embedding_iterations),
            scale_factor_learning_rate=0,
            burn_in_learning_rate=learning_rate * 0.1,
            learning_rate=learning_rate,
        )
        # logger.info(f"Finished coordinate training. Final loss: {losses[-1]:.4f}")
        logger.info(f"Finished coordinate training. Final loss: {losses['total'][-1]:.4f}")
        X_embed = X_embed.to(device)  # Ensure on device
        pm = pm.to(device)
        # --- End Coordinate Learning ---

        # --- Create Link Prediction Dataset ---
        logger.info(f"Making link prediction dataset (add_dists={add_dists_feature})...")
        X_link, y_link, pm_link = make_link_prediction_dataset(X_embed, pm, adj, add_dists=add_dists_feature)
        X_link = X_link.to(device)
        y_link = y_link.to(device)  # Should be LongTensor
        pm_link = pm_link.to(device)
        logger.info(f"Link prediction dataset created. X shape: {X_link.shape}, y shape: {y_link.shape}")
        # --- End Link Prediction Dataset ---

        # --- Subsampling while preserving pairs ---
        if len(X_link) > n_points:
            # Create random indices for sampling
            indices = torch.randperm(len(X_link))[:n_points]
            # Sample both X and y using the same indices to keep pairs together
            X_link, y_link = X_link[indices], y_link[indices]
            logger.info(f"Subsampled link prediction dataset from {len(indices)} to {n_points} points.")
        # --- End Subsampling ---

        # --- Train/Test Split ---
        from sklearn.model_selection import train_test_split

        # Split the generated pairs directly, stratifying by label
        y_link_cpu = y_link.cpu()  # Stratify requires CPU tensor/array
        if torch.numel(y_link_cpu) > 0 and len(torch.unique(y_link_cpu)) > 1:
            stratify_arg = y_link_cpu
        else:
            logger.warning(f"Cannot stratify link prediction split for {graph_name} (labels empty or uniform).")
            stratify_arg = None

        X_train, X_test, y_train, y_test = train_test_split(
            X_link, y_link, test_size=test_size, random_state=trial, stratify=stratify_arg
        )
        # Move back to device if split happened on CPU implicitly
        X_train, X_test = X_train.to(device), X_test.to(device)
        y_train, y_test = y_train.to(device), y_test.to(device)
        logger.info(
            f"Train shapes: X={X_train.shape}, y={y_train.shape}; Test shapes: X={X_test.shape}, y={y_test.shape}"
        )
        # --- End Train/Test Split ---

        result = benchmark(
            X=None,
            y=None,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            pm=pm_link,
            task="classification",
            score=score,
            models=models,
            device=device,
            max_depth=max_depth,
            seed=trial,
        )
        result["task"] = "link_prediction"
        return result

    except Exception as e:
        logger.error(f"Error during link prediction trial {graph_name}, trial {trial}: {e}", exc_info=True)
        return {"error": f"Benchmark execution failed: {e}", "traceback": traceback.format_exc()}


def run_empirical_trial(
    trial_params: Dict[str, Any], config: Dict[str, Any], models: List[str], device: torch.device
) -> Dict[str, Any]:
    """Run an empirical benchmark trial."""
    dataset = trial_params["dataset"]
    signature = trial_params["signature"]
    task = trial_params["task"]
    seed = trial_params["seed"]
    logger.debug(f"Running empirical trial: dataset={dataset}, task={task}, seed={seed}")

    n_points = config.get("n_points", config.get("downsample", 1000))
    score = config.get("classification_metrics") if task == "classification" else config.get("regression_metrics")

    data_root = Path(config.get("data_dir", project_root / "data"))
    empirical_data_dir = Path(config.get("empirical_data_dir", data_root / "empirical"))

    try:
        pm = ProductManifold(signature=signature, device=device)

        # --- Load Data ---
        X_loaded, y_loaded, _ = manify.utils.dataloaders.load(dataset)
        if X_loaded is None or y_loaded is None:
            raise ValueError(f"Data (X or y) not loaded for empirical dataset {dataset}")

        X_tensor = (
            torch.tensor(X_loaded, dtype=torch.float32) if not isinstance(X_loaded, torch.Tensor) else X_loaded.float()
        )
        y_dtype = torch.long if task == "classification" else torch.float32
        y_tensor = (
            torch.tensor(y_loaded, dtype=y_dtype) if not isinstance(y_loaded, torch.Tensor) else y_loaded.to(y_dtype)
        )
        # --- End Load Data ---

        # --- Subsampling ---
        np.random.seed(seed)
        torch.manual_seed(seed)
        if len(X_tensor) > n_points:
            indices = torch.randperm(len(X_tensor))[:n_points]
            X_sampled, y_sampled = X_tensor[indices], y_tensor[indices]
            logger.info(f"Subsampled empirical dataset {dataset} from {len(X_tensor)} to {n_points} points.")
        else:
            X_sampled, y_sampled = X_tensor, y_tensor
        # --- End Subsampling ---

        X = X_sampled.to(device)
        y = y_sampled.to(device)
        pm = pm.to(device)  # Ensure manifold is on device too

        result = benchmark(X=X, y=y, pm=pm, seed=seed, task=task, score=score, device=device, models=models)
        result["task"] = task
        return result

    except Exception as e:
        logger.error(f"Error during empirical trial {dataset}, seed {seed}: {e}", exc_info=True)
        return {"error": f"Benchmark execution failed: {e}", "traceback": traceback.format_exc()}


# --- Result Saving (Slightly adapted for single list) ---
def save_all_results(results_list: List[Dict], output_dir: str, run_name: Optional[str] = None) -> Optional[str]:
    """Save aggregated benchmark results (list of trial dicts) to TSV file."""
    if not results_list:
        logger.warning("No results to save.")
        return None

    # Filter out None or non-dict entries just in case
    valid_results = [r for r in results_list if isinstance(r, dict)]
    if not valid_results:
        logger.warning("No valid dictionary results found to save.")
        return None

    all_rows = []
    # Flatten results: one row per model per trial
    for trial_result in valid_results:
        base_info = {
            k: v for k, v in trial_result.items() if not isinstance(v, dict) and k not in ["error", "traceback"]
        }
        base_info["trial_error"] = trial_result.get("error")  # Add overall trial error if present

        if "error" in trial_result:
            # If the whole trial failed, create one row indicating the error
            row = base_info.copy()
            row["model"] = "TRIAL_ERROR"
            all_rows.append(row)
        else:
            # Iterate through models within the successful trial
            for model_name, model_data in trial_result.items():
                if isinstance(model_data, dict) and any(
                    k in model_data for k in ["time", "accuracy", "f1", "rmse", "mse", "auc"]
                ):
                    row = base_info.copy()
                    row["model"] = model_name
                    row["time"] = model_data.get("time")
                    # Add metrics
                    for metric_key, metric_value in model_data.items():
                        if metric_key != "time":
                            row[metric_key.replace("-", "_")] = metric_value
                    all_rows.append(row)

    if not all_rows:
        logger.warning("No model-specific result rows generated for saving.")
        return None

    try:
        df = pd.DataFrame(all_rows)
        # Optional: Reorder columns nicely
        # cols_order = ['benchmark_type', 'task', 'dataset', ...] # Define order
        # df = df[...]

    except Exception as e:
        logger.error(f"Error creating DataFrame from results: {e}")
        # Fallback to JSON dump of the original list
        fallback_path = (
            Path(output_dir) / f"results_{run_name or datetime.now().strftime('%Y%m%d-%H%M%S')}_fallback.json"
        )
        try:
            with open(fallback_path, "w") as f:
                # Use a custom encoder for potentially non-serializable items if needed
                json.dump(valid_results, f, indent=2, default=str)
            logger.info(f"Saved raw results fallback to {fallback_path}")
        except Exception as json_e:
            logger.error(f"Failed to save fallback JSON: {json_e}")
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"aggregated_results_{run_name or datetime.now().strftime('%Y%m%d-%H%M%S')}.tsv"
    full_output_path = output_path / filename

    try:
        df.to_csv(full_output_path, sep="\t", index=False, na_rep="NaN")
        logger.info(f"Aggregated results saved to {full_output_path}")
        return str(full_output_path)
    except Exception as e:
        logger.error(f"Failed to save results TSV: {e}")
        return None


# --- Main Execution Logic ---
def main():
    """Main entry point for the single-process benchmarking script."""
    parser = argparse.ArgumentParser(description="Run manify benchmarks (single process) with YAML configuration")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--output-dir", default=None, help="Directory to save results TSV (overrides config)")
    parser.add_argument("--device", default=None, help="Device to use (e.g., cuda:0, cpu, mps) (overrides config)")
    parser.add_argument("--dry-run", action="store_true", help="Prepare run but don't execute benchmarks")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity (WARNING level)")
    parser.add_argument("--max-trials", type=int, help="Global max trials per benchmark type (overrides config)")
    parser.add_argument("--models", type=str, help="Comma-separated models list (overrides config)")
    parser.add_argument("--seed", type=int, default=42, help="Global base random seed")
    parser.add_argument(
        "--benchmark-types",
        nargs="+",
        choices=[
            "all",
            "single_curvature",
            "signature_gaussian",
            "graph_embedding",
            "vae_embedding",
            "link_prediction",
            "empirical_benchmarks",
        ],
        default=["all"],
        help="Types of benchmark to run ('all' or a list)",
    )
    parser.add_argument("--data-root", default=None, help="Root directory for data/embeddings (overrides config paths)")

    args = parser.parse_args()

    # --- Setup Logging Level ---
    log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARNING
    logging.getLogger().setLevel(log_level)  # Set root logger level
    logger.setLevel(log_level)
    logging.getLogger("wandb").setLevel(logging.WARNING)  # Silence wandb logs unless debug
    if args.quiet:
        os.environ["TQDM_DISABLE"] = "1"
    logger.info(f"Starting benchmark script with args: {args}")
    # --- End Logging Setup ---

    # --- Seed ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logger.info(f"Set random seeds for numpy, torch, cuda to {args.seed}")
    # --- End Seed ---

    # --- Load Configuration ---
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}", exc_info=True)
        sys.exit(1)
    # --- End Load Config ---

    # --- Apply Command-Line Overrides ---
    # Ensure 'common' section exists
    if "common" not in config:
        config["common"] = {}

    # Device override
    effective_device_str = args.device or config["common"].get("device", "cpu")
    config["common"]["device"] = effective_device_str  # Store effective device string

    # Output directory override
    effective_output_dir = args.output_dir or config["common"].get("output_dir", "results")
    output_dir_path = Path(effective_output_dir)
    if not output_dir_path.is_absolute():
        output_dir_path = project_root / output_dir_path
    effective_output_dir = str(output_dir_path)
    config["common"]["output_dir"] = effective_output_dir

    # Max trials override
    if args.max_trials is not None:
        logger.info(f"Applying global max_trials override: {args.max_trials}")
        config["max_trials"] = args.max_trials  # Global setting for generate_trials
        # Also override specific known trial keys if they exist
        for section in config:
            if isinstance(config[section], dict):
                for key in ["n_trials", "n_trials_gaussian"]:
                    if key in config[section]:
                        config[section][key] = min(config[section].get(key, float("inf")), args.max_trials)

    # Models override
    if args.models:
        model_list = [m.strip() for m in args.models.split(",") if m.strip()]
        logger.info(f"Applying models override: {model_list}")
        config["common"]["models"] = model_list
        for section in config:
            if section not in ["wandb", "gaussian", "common"] and isinstance(config[section], dict):
                config[section]["models"] = model_list

    # Data root override (update common paths)
    if args.data_root:
        data_root_path = Path(args.data_root).resolve()  # Use absolute path
        logger.info(f"Applying data_root override: {data_root_path}")
        # Update common paths used as defaults by trial runners
        config["common"]["data_dir"] = str(data_root_path)  # General data root
        # Adjust these based on your expected structure relative to data_root
        config["common"]["graph_data_dir"] = str(data_root_path / "graphs")
        config["common"]["embeddings_dir"] = str(data_root_path / "graphs" / "embeddings")
        config["common"]["vae_data_dir"] = lambda name: str(
            data_root_path / name / "embeddings"
        )  # Example dynamic path
        config["common"]["empirical_data_dir"] = str(data_root_path / "empirical")
        # Note: Trial runners should check for specific config keys first, then common ones.

    logger.debug(f"Effective configuration after overrides:\n{yaml.dump(config)}")
    # --- End Overrides ---

    # --- Determine Device ---
    device = get_device(effective_device_str)
    # --- End Device Setup ---

    # --- Initialize WandB ---
    is_wandb_enabled = setup_wandb(config, args.config, args)
    run_name_for_saving = None
    if is_wandb_enabled and wandb.run:
        run_name_for_saving = wandb.run.name
    else:
        run_name_for_saving = f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    # --- End WandB Setup ---

    # --- Determine Benchmarks to Run ---
    all_benchmark_types = [
        "single_curvature",
        "signature_gaussian",
        "graph_embedding",
        "vae_embedding",
        "link_prediction",
        "empirical_benchmarks",
    ]
    requested_benchmark_types = (
        all_benchmark_types
        if "all" in args.benchmark_types
        else sorted(list(set(args.benchmark_types)), key=args.benchmark_types.index)
    )
    runnable_benchmark_types = [b for b in requested_benchmark_types if b in config and isinstance(config[b], dict)]
    skipped_types = set(requested_benchmark_types) - set(runnable_benchmark_types)
    if skipped_types:
        logger.warning(f"Skipping benchmark types not configured in {args.config}: {skipped_types}")
    if not runnable_benchmark_types:
        logger.error("No valid benchmark types to run based on config and selection.")
        if is_wandb_enabled and wandb.run:
            wandb.finish(exit_code=1)
        sys.exit(1)
    logger.info(f"Planning to run benchmarks: {runnable_benchmark_types}")
    # --- End Benchmark Selection ---

    # --- Dry Run ---
    if args.dry_run:
        logger.info("Dry run requested. Generating trials only.")
        total_dry_trials = 0
        for benchmark_type in runnable_benchmark_types:
            benchmark_config = config.get(benchmark_type, {})
            logger.info(f"--- Dry run: Generating trials for {benchmark_type} ---")
            try:
                trials = generate_trials(benchmark_type, benchmark_config)
                logger.info(f"Generated {len(trials)} potential trials for {benchmark_type}.")
                total_dry_trials += len(trials)
                for i, t in enumerate(trials[: min(3, len(trials))]):
                    logger.info(f"  Trial {i}: {t}")
            except Exception as e:
                logger.error(f"Error generating trials for {benchmark_type}: {e}", exc_info=True)
        logger.info(f"Dry run complete. Total potential trials: {total_dry_trials}")
        if is_wandb_enabled and wandb.run:
            wandb.finish()
        sys.exit(0)
    # --- End Dry Run ---

    # --- Main Benchmark Execution Loop ---
    all_results = []
    start_time = time.time()
    total_trials_run = 0
    total_trials_failed = 0

    for benchmark_type in runnable_benchmark_types:
        benchmark_config = config.get(benchmark_type, {})  # Get specific config section
        logger.info(f"--- Starting Benchmark: {benchmark_type} ---")

        # Generate trials for this specific benchmark type
        try:
            trials_to_run = generate_trials(benchmark_type, benchmark_config)
        except Exception as e:
            logger.error(f"Failed to generate trials for {benchmark_type}: {e}. Skipping.", exc_info=True)
            continue  # Skip to next benchmark type

        if not trials_to_run:
            logger.warning(f"No trials generated for {benchmark_type}. Skipping.")
            continue

        logger.info(f"Running {len(trials_to_run)} trials for {benchmark_type} on device '{device}'")

        # Determine models for this benchmark type
        default_models = config.get("common", {}).get("models", [])
        excluded_models = benchmark_config.get("exclude_models", [])
        models_to_run = [m for m in benchmark_config.get("models", default_models) if m not in excluded_models]
        logger.info(f"Using models: {models_to_run}")

        # Run trials sequentially
        for i, trial_params in enumerate(trials_to_run):
            trial_start_time = time.time()
            trial_id = trial_params.get("trial_id", f"{benchmark_type}-trial{i}")
            logger.info(f"Starting Trial {i+1}/{len(trials_to_run)} ({trial_id})...")
            total_trials_run += 1
            full_trial_result = None
            trial_failed = False

            # Prepare base metadata for WandB logging (can be reused for error logging)
            base_metadata = {"benchmark_type": benchmark_type}
            for k, v in trial_params.items():
                if isinstance(v, (str, int, float, bool)):
                    base_metadata[k] = v
                elif isinstance(v, (list, tuple)):
                    try:
                        base_metadata[k] = json.dumps(v)  # Serialize complex params like signatures
                    except TypeError:
                        base_metadata[k] = str(v)

            try:
                # --- Select and Run Trial Function ---
                if benchmark_type == "single_curvature":
                    trial_results = run_single_curvature_trial(trial_params, benchmark_config, models_to_run, device)
                elif benchmark_type == "signature_gaussian":
                    trial_results = run_signature_gaussian_trial(trial_params, benchmark_config, models_to_run, device)
                elif benchmark_type == "graph_embedding":
                    trial_results = run_graph_embedding_trial(trial_params, benchmark_config, models_to_run, device)
                elif benchmark_type == "vae_embedding":
                    trial_results = run_vae_embedding_trial(trial_params, benchmark_config, models_to_run, device)
                elif benchmark_type == "link_prediction":
                    trial_results = run_link_prediction_trial(trial_params, benchmark_config, models_to_run, device)
                elif benchmark_type == "empirical_benchmarks":
                    trial_results = run_empirical_trial(trial_params, benchmark_config, models_to_run, device)
                else:
                    logger.error(f"Unknown benchmark type '{benchmark_type}' encountered in loop.")
                    trial_results = {"error": f"Unknown benchmark type: {benchmark_type}"}
                # --- End Trial Function Call ---

                # --- Process Trial Result ---
                if trial_results and "error" in trial_results:
                    logger.error(f"Trial {trial_id} failed: {trial_results['error']}")
                    # Log error to wandb
                    error_payload = base_metadata.copy()
                    error_payload["model"] = "TRIAL_ERROR"  # Special model name for trial errors
                    error_payload["error"] = trial_results["error"][:1000]  # Truncate long errors
                    if is_wandb_enabled:
                        wandb.log(error_payload)
                    full_trial_result = trial_results  # Store the error dict
                    trial_failed = True
                elif trial_results:
                    # Successful trial, log results per model
                    logger.info(f"DEBUG_CHECK: Reached success block for {trial_id}. Preparing to log.")
                    log_wandb_results(base_metadata, trial_results)  # Log rows to WandB
                    # Add seed/trial and type info for saving
                    trial_results["seed_or_trial"] = trial_params.get("seed", trial_params.get("trial"))
                    trial_results["benchmark_type"] = benchmark_type
                    full_trial_result = trial_results  # Store the full success dict
                    logger.info(f"Trial {trial_id} completed successfully (took {time.time() - trial_start_time:.2f}s)")
                else:
                    # Should not happen if trial runners always return a dict
                    logger.error(f"Trial {trial_id} returned None or empty result.")
                    full_trial_result = {"error": "Trial returned no result", **base_metadata}
                    trial_failed = True

            except Exception as e:
                # Catch unexpected errors *within* the main loop/trial runner call
                tb = traceback.format_exc()
                logger.error(f"Uncaught exception in trial {trial_id}: {e}\n{tb}")
                error_payload = base_metadata.copy()
                error_payload["model"] = "UNCAUGHT_ERROR"
                error_payload["error"] = f"Uncaught Exception: {e}"[:1000]
                if is_wandb_enabled:
                    wandb.log(error_payload)
                full_trial_result = {  # Store error dict
                    "error": f"Uncaught Exception: {e}",
                    "traceback": tb,
                    **base_metadata,
                }
                trial_failed = True

            # Accumulate result (success or error dict)
            if full_trial_result:
                all_results.append(full_trial_result)
            if trial_failed:
                total_trials_failed += 1

        logger.info(f"--- Finished Benchmark: {benchmark_type} ---")
    # --- End Main Benchmark Loop ---

    # --- Save Final Aggregated Results ---
    final_results_df = None  # Store the DataFrame itself
    final_results_path = None
    if all_results:
        logger.info(f"Aggregating results from {len(all_results)} trials...")
        # Modify save_all_results to optionally return the DataFrame
        final_results_path, final_results_df = save_all_results(
            all_results, effective_output_dir, run_name=run_name_for_saving
        )
    # --- End Save Results ---

    # --- Finalize WandB ---
    if is_wandb_enabled and wandb.run:
        logger.info("Finalizing WandB run...")
        try:
            # Log summary metrics
            wandb.summary["total_trials_run"] = total_trials_run
            wandb.summary["total_trials_failed"] = total_trials_failed
            wandb.summary["total_execution_time_sec"] = time.time() - start_time

            # Log the final aggregated DataFrame as a W&B Table
            if final_results_df is not None and not final_results_df.empty:
                try:
                    logger.info("DEBUG_CHECK: Preparing to log final results_table.")
                    # Sanitize column names just in case (dots/slashes can be issues)
                    final_results_df.columns = [
                        "".join(c if c.isalnum() else "_" for c in str(x)) for x in final_results_df.columns
                    ]
                    wandb.log({"results_table": wandb.Table(dataframe=final_results_df)})
                    logger.info("Logged final results DataFrame as wandb.Table('results_table').")
                except Exception as table_e:
                    logger.error(f"Failed to log results DataFrame as wandb.Table: {table_e}")

            # Log the final aggregated TSV as an artifact if created
            if final_results_path and Path(final_results_path).exists():
                results_artifact = wandb.Artifact(f"aggregated_results-{wandb.run.id}", type="results")
                results_artifact.add_file(final_results_path)
                wandb.log_artifact(results_artifact)
                logger.info(f"Logged aggregated results artifact ({final_results_path}) to WandB.")

            wandb.finish()
            logger.info("WandB run finished.")
        except Exception as e:
            logger.error(f"Error during WandB finish: {e}")
    # --- End Finalize WandB ---

    total_time = time.time() - start_time
    logger.info(f"Benchmark script completed in {total_time:.2f} seconds.")
    logger.info(f"Total trials run: {total_trials_run}, Failed trials: {total_trials_failed}")
    if final_results_path:
        logger.info(f"Aggregated results saved to: {final_results_path}")


if __name__ == "__main__":
    main()
