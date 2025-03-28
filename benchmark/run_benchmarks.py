#!/usr/bin/env python3
"""
Configurable benchmarking script for manify.

This script runs benchmarks for different manifold configurations and models,
with support for:
- YAML configuration
- Weights & Biases (wandb) integration
- GPU parallelization
- Real-time result reporting

Benchmark types:
1. Single curvature Gaussian (synthetic data)
2. Signature Gaussian (synthetic data with mixed curvatures)
3. Graph embeddings (pre-computed embeddings for graph datasets)
4. VAE embeddings (pre-computed embeddings from VAE models)
5. Link prediction (graph-based link prediction with learned coordinates)
"""

import argparse
import os
import sys
import yaml
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime
import multiprocessing
# Set multiprocessing start method to 'spawn' for CUDA compatibility
multiprocessing.set_start_method('spawn', force=True)
from multiprocessing import Process, Queue

import numpy as np
import pandas as pd
import torch
import wandb

# Add parent directory to path so we can import manify
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import manify
from manify.manifolds import ProductManifold
from manify.utils.benchmarks import benchmark
from manify.predictors.kappa_gcn import get_A_hat
from manify.utils.link_prediction import make_link_prediction_dataset
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("manify_benchmark")

# Setup debug level - will be updated in main if --debug flag is used
debug_mode = False

# Type aliases
BenchmarkConfig = Dict[str, Any]
BenchmarkResult = Dict[str, Any]


def load_config(config_path: str) -> BenchmarkConfig:
    """Load benchmark configuration from a YAML file with inheritance."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply common settings to all benchmark types
    if "common" in config:
        common = config["common"]

        # Apply common settings to each benchmark type
        for benchmark_type in ["single_curvature", "signature_gaussian", "graph_embedding", "vae_embedding"]:
            if benchmark_type in config:
                # Apply common settings if not already defined in the benchmark config
                for key, value in common.items():
                    if key not in config[benchmark_type]:
                        config[benchmark_type][key] = value

    # Apply gaussian settings to gaussian-related benchmark types
    if "gaussian" in config:
        gaussian = config["gaussian"]
        for benchmark_type in ["single_curvature", "signature_gaussian"]:
            if benchmark_type in config:
                for key, value in gaussian.items():
                    if key not in config[benchmark_type]:
                        config[benchmark_type][key] = value

    return config


def setup_wandb(config: BenchmarkConfig, dry_run: bool = False) -> None:
    """Initialize wandb for experiment tracking."""
    if dry_run:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb_config = config.get("wandb", {})
    
    # Simple straightforward initialization
    wandb.init(
        project=wandb_config.get("project", "manify-benchmarks"),
        entity=wandb_config.get("entity"),
        name=wandb_config.get("name", f"benchmark-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
        config=config
    )


def get_available_gpus() -> List[int]:
    """Get list of available GPU IDs."""
    try:
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            logger.warning("No CUDA devices detected. Using CPU.")
            return []
        
        # Print device info for debugging
        for i in range(gpu_count):
            logger.info(f"Found CUDA device {i}: {torch.cuda.get_device_name(i)}")
        
        return list(range(gpu_count))
    except (ImportError, ModuleNotFoundError):
        try:
            # Try using nvidia-smi as fallback
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            lines = result.stdout.strip().split("\n")
            gpu_ids = []
            for line in lines:
                parts = line.split(",", 1)
                if len(parts) >= 1:
                    gpu_id = parts[0].strip()
                    try:
                        gpu_ids.append(int(gpu_id))
                        logger.info(f"Found GPU {gpu_id} via nvidia-smi")
                    except ValueError:
                        pass
            
            if not gpu_ids:
                logger.warning("No GPUs detected via nvidia-smi. Using CPU.")
                return []
            
            return gpu_ids
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("Could not determine available GPUs. Using CPU.")
            return []


def assign_gpu(gpu_id: int) -> None:
    """Set environment variable to use specific GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logger.info(f"Using GPU {gpu_id}")


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """Get appropriate torch device."""
    if gpu_id is not None:
        assign_gpu(gpu_id)
        
    try:
        if torch.cuda.is_available():
            # Check if CUDA_VISIBLE_DEVICES has been set
            visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
            if visible_devices:
                logger.info(f"Using GPU(s) defined in CUDA_VISIBLE_DEVICES: {visible_devices}")
            
            device = torch.device("cuda")
            # Print device being used for debug purposes
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            logger.info(f"Using CUDA device {current_device}: {device_name}")
            return device
        elif torch.backends.mps.is_available():
            logger.info("Using MPS device")
            return torch.device("mps")
        else:
            logger.info("Using CPU device")
            return torch.device("cpu")
    except Exception as e:
        logger.warning(f"Error setting device, falling back to CPU: {e}")
        return torch.device("cpu")


def run_single_curvature_benchmark(
    config: BenchmarkConfig,
    results_queue: Queue,
    gpu_id: Optional[int] = None,
    trial_indices: Optional[List[int]] = None,
    no_wandb: bool = False,
    wandb_group: Optional[str] = None,
    wandb_run_id: Optional[str] = None
) -> None:
    """Run benchmarks for single curvature Gaussian mixture data for both classification and regression."""
    # No wandb initialization in worker processes
        
    device = get_device(gpu_id)
    sample_device = torch.device("cpu") if device != torch.device("cuda") else device

    # Extract configuration
    benchmark_config = config.get("single_curvature", {})
    curvatures = benchmark_config.get("curvatures", [-2, -1, -0.5, 0, 0.5, 1, 2])
    dim = benchmark_config.get("dimension", 2)
    n_trials_gaussian = benchmark_config.get("n_trials_gaussian", 10)
    n_points = benchmark_config.get("n_points", 1000)
    n_classes = benchmark_config.get("n_classes", 8)
    n_clusters = benchmark_config.get("n_clusters", 32)
    cov_scale_means = benchmark_config.get("cov_scale_means", 1.0)
    cov_scale_points = benchmark_config.get("cov_scale_points", 1.0)
    
    # Default models plus any model exclusions for this specific run
    default_models = ["sklearn_dt", "sklearn_rf", "product_dt", "product_rf"]
    excluded_models = benchmark_config.get("exclude_models", [])
    
    # Get configured models or use defaults, filtering out excluded models
    models = [m for m in benchmark_config.get("models", default_models) if m not in excluded_models]
    
    logger.info(f"Using models for single_curvature benchmark: {models}")

    # Set up scoring metrics
    classification_metrics = benchmark_config.get("classification_metrics", ["accuracy", "f1-micro"])
    regression_metrics = benchmark_config.get("regression_metrics", ["rmse"])

    # Run for both classification and regression tasks
    tasks = ["classification", "regression"]

    # Generate trials based on all curvatures, seeds, and tasks
    trials = []
    for task in tasks:
        for i, K in enumerate(curvatures):
            for seed in range(n_trials_gaussian):
                # Ensure unique seed per trial
                seed_value = seed + n_trials_gaussian * i
                trials.append((K, seed_value, task))

    # Filter trials based on trial_indices if provided
    if trial_indices is not None:
        trials = [trials[i] for i in trial_indices]

    results = []
    for K, seed, task in trials:
        logger.info(f"Running single curvature benchmark with K={K}, seed={seed}, task={task}")

        # Set up product manifold with single curvature
        pm = ProductManifold(signature=[(K, dim)]).to(sample_device)

        # Generate synthetic data
        X, y = pm.gaussian_mixture(
            seed=seed,
            num_points=n_points,
            num_classes=n_classes,
            num_clusters=n_clusters,
            cov_scale_means=cov_scale_means / dim,
            cov_scale_points=cov_scale_points / dim,
            task=task,
        )

        X = X.to(device)
        y = y.to(device)
        pm = pm.to(device)

        # Set up task-specific scoring metrics
        score = classification_metrics if task == "classification" else regression_metrics

        # Run benchmark with error handling
        try:
            model_results = benchmark(X, y, pm, task=task, score=score, seed=seed, device=device, models=models)
            
            # Add metadata
            model_results["curvature"] = K
            model_results["seed"] = seed
            model_results["task"] = task
            model_results["benchmark_type"] = "single_curvature"
            
            # Skip wandb logging in worker processes - we'll do it in the main process
                        
        except Exception as e:
            logger.error(f"Error in benchmark with K={K}, seed={seed}, task={task}: {str(e)}")
            # Create an empty result with error information
            model_results = {
                "curvature": K,
                "seed": seed,
                "task": task,
                "benchmark_type": "single_curvature",
                "error": str(e)
            }
            # Skip wandb logging in worker processes

        # Save this individual result immediately to disk
        try:
            # Create incremental results directory if needed
            incremental_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data/results_wandb/incremental')
            os.makedirs(incremental_dir, exist_ok=True)
            
            # Create an individual file for this result
            result_filename = f"single_curvature_K{K}_seed{seed}_{task}_{int(time.time())}.json"
            result_path = os.path.join(incremental_dir, result_filename)
            
            # Safe write to avoid data loss (write to temp file first, then rename)
            with open(result_path + ".tmp", 'w') as f:
                json.dump(model_results, f, indent=2)
            os.rename(result_path + ".tmp", result_path)
            
            logger.info(f"Saved individual result to {result_path}")
        except Exception as e:
            logger.error(f"ERROR SAVING INDIVIDUAL RESULT: {str(e)}")
            
        # Append to in-memory results collection
        results.append(model_results)
        logger.info(f"Completed single curvature benchmark with K={K}, seed={seed}, task={task}")

    # Don't finish the wandb run - just let the main process handle it
    # Put results in queue for main process
    results_queue.put(results)


def run_signature_gaussian_benchmark(
    config: BenchmarkConfig,
    results_queue: Queue,
    gpu_id: Optional[int] = None,
    trial_indices: Optional[List[int]] = None,
    no_wandb: bool = False,
    wandb_group: Optional[str] = None,
    wandb_run_id: Optional[str] = None
) -> None:
    # No wandb initialization in worker processes
    """Run benchmarks for mixed curvature Gaussian mixture data for both classification and regression."""
    device = get_device(gpu_id)
    sample_device = torch.device("cpu") if device != torch.device("cuda") else device

    # Extract configuration
    benchmark_config = config.get("signature_gaussian", {})
    signatures = benchmark_config.get(
        "signatures",
        [
            [(-1, 2), (0, 2)],  # HE
            [(-1, 2), (1, 2)],  # HS
            [(0, 2), (1, 2)],  # ES
            [(-1, 2), (-1, 2)],  # HH
            [(1, 2), (1, 2)],  # SS
        ],
    )
    signature_names = benchmark_config.get("signature_names", ["HE", "HS", "ES", "HH", "SS"])
    n_trials_gaussian = benchmark_config.get("n_trials_gaussian", 10)
    n_points = benchmark_config.get("n_points", 1000)
    n_classes = benchmark_config.get("n_classes", 8)
    n_clusters = benchmark_config.get("n_clusters", 32)
    cov_scale_means = benchmark_config.get("cov_scale_means", 1.0)
    cov_scale_points = benchmark_config.get("cov_scale_points", 1.0)
    
    # Default models plus any model exclusions for this specific run
    default_models = ["sklearn_dt", "sklearn_rf", "product_dt", "product_rf"]
    excluded_models = benchmark_config.get("exclude_models", [])
    
    # Get configured models or use defaults, filtering out excluded models
    models = [m for m in benchmark_config.get("models", default_models) if m not in excluded_models]
    
    logger.info(f"Using models for signature_gaussian benchmark: {models}")

    # Set up scoring metrics
    classification_metrics = benchmark_config.get("classification_metrics", ["accuracy", "f1-micro"])
    regression_metrics = benchmark_config.get("regression_metrics", ["rmse"])

    # Run for both classification and regression tasks
    tasks = ["classification", "regression"]

    # Generate trials based on all signatures, seeds, and tasks
    trials = []
    for task in tasks:
        for i, (sig, sig_name) in enumerate(zip(signatures, signature_names)):
            for seed in range(n_trials_gaussian):
                # Ensure unique seed per trial
                seed_value = seed + n_trials_gaussian * i
                trials.append((sig, sig_name, seed_value, task))

    # Filter trials based on trial_indices if provided
    if trial_indices is not None:
        trials = [trials[i] for i in trial_indices]

    results = []
    for signature, signature_name, seed, task in trials:
        logger.info(f"Running signature Gaussian benchmark with signature={signature_name}, seed={seed}, task={task}")

        # Set up product manifold with mixed curvature signature
        pm = ProductManifold(signature=signature).to(sample_device)

        # Calculate total dimension for scaling covariance
        total_dim = sum(dim for _, dim in signature)

        # Generate synthetic data
        X, y = pm.gaussian_mixture(
            seed=seed,
            num_points=n_points,
            num_classes=n_classes,
            num_clusters=n_clusters,
            cov_scale_means=cov_scale_means / total_dim,
            cov_scale_points=cov_scale_points / total_dim,
            task=task,
        )

        X = X.to(device)
        y = y.to(device)
        pm = pm.to(device)

        # Set up task-specific scoring metrics
        score = classification_metrics if task == "classification" else regression_metrics

        # Run benchmark with error handling
        try:
            model_results = benchmark(X, y, pm, task=task, score=score, seed=seed, device=device, models=models)
            
            # Add metadata
            model_results["signature"] = signature_name
            model_results["seed"] = seed
            model_results["task"] = task
            model_results["benchmark_type"] = "signature_gaussian"
            
            # Skip wandb logging in worker processes - we'll do it in the main process
                        
        except Exception as e:
            logger.error(f"Error in benchmark with signature={signature_name}, seed={seed}, task={task}: {str(e)}")
            # Create an empty result with error information
            model_results = {
                "signature": signature_name,
                "seed": seed,
                "task": task,
                "benchmark_type": "signature_gaussian",
                "error": str(e)
            }
            # Skip wandb logging in worker processes

        # Save this individual result immediately to disk
        try:
            # Create incremental results directory if needed
            incremental_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data/results_wandb/incremental')
            os.makedirs(incremental_dir, exist_ok=True)
            
            # Create an individual file for this result
            result_filename = f"signature_gaussian_{signature_name}_seed{seed}_{task}_{int(time.time())}.json"
            result_path = os.path.join(incremental_dir, result_filename)
            
            # Safe write to avoid data loss (write to temp file first, then rename)
            with open(result_path + ".tmp", 'w') as f:
                json.dump(model_results, f, indent=2)
            os.rename(result_path + ".tmp", result_path)
            
            logger.info(f"Saved individual result to {result_path}")
        except Exception as e:
            logger.error(f"ERROR SAVING INDIVIDUAL RESULT: {str(e)}")
            
        # Append to in-memory results collection
        results.append(model_results)
        logger.info(f"Completed signature Gaussian benchmark with signature={signature_name}, seed={seed}, task={task}")

    # Don't finish the wandb run - just let the main process handle it
    # Put results in queue for main process
    results_queue.put(results)


def run_link_prediction_benchmark(
    config: BenchmarkConfig,
    results_queue: Queue,
    gpu_id: Optional[int] = None,
    trial_indices: Optional[List[int]] = None,
    no_wandb: bool = False,
    wandb_group: Optional[str] = None,
    wandb_run_id: Optional[str] = None
) -> None:
    """Run link prediction benchmarks on graph datasets."""
    # No wandb initialization in worker processes
    device = get_device(gpu_id)

    # Extract configuration
    benchmark_config = config.get("link_prediction", {})
    
    # Resolve paths relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, benchmark_config.get("data_dir", "data/graphs"))
    
    # Default datasets to run link prediction on
    graph_names = benchmark_config.get("graph_names", ["karate", "lesmis", "dolphins", "polbooks", "adjnoun", "football"])
    
    # Component signatures for embedding the graphs
    component_signatures = benchmark_config.get("component_signatures", [
        [(-1, 2), (0, 2), (1, 2)]  # Default: Hyperbolic + Euclidean + Spherical
    ])
    component_names = benchmark_config.get("component_names", ["HES"])
    
    # Training parameters
    embedding_iterations = benchmark_config.get("embedding_iterations", 5000) 
    test_size = benchmark_config.get("test_size", 0.2)
    max_depth = benchmark_config.get("max_depth", 3)
    add_dists = benchmark_config.get("add_dists", True)
    learning_rate = benchmark_config.get("learning_rate", 1e-4)
    n_trials = benchmark_config.get("n_trials", 10)
    
    # Models to use
    default_models = ["sklearn_dt", "sklearn_rf", "product_dt", "product_rf", "tangent_dt", "tangent_rf", "knn"]
    excluded_models = benchmark_config.get("exclude_models", [])
    models = [m for m in benchmark_config.get("models", default_models) if m not in excluded_models]
    
    logger.info(f"Using models for link_prediction benchmark: {models}")
    
    # Set up scoring metrics
    classification_metrics = benchmark_config.get("classification_metrics", ["accuracy", "f1-micro", "auc"])
    
    # Generate trials based on graph datasets and signatures
    trials = []
    for graph_name in graph_names:
        for i, (sig, sig_name) in enumerate(zip(component_signatures, component_names)):
            for trial in range(n_trials):
                trials.append((graph_name, sig, sig_name, trial))
    
    # Filter trials based on trial_indices if provided
    if trial_indices is not None:
        trials = [trials[i] for i in trial_indices]
        
    results = []
    for graph_name, signature, signature_name, trial in trials:
        logger.info(f"Running link prediction benchmark with graph={graph_name}, signature={signature_name}, trial={trial}")
        
        try:
            # Load graph data
            try:
                # Try to load using manify.utils.dataloaders
                dists, labels, adj = manify.utils.dataloaders.load(graph_name)
                # Normalize distances
                dists = dists / dists.max()
            except Exception as e:
                logger.error(f"Error loading graph {graph_name}: {str(e)}")
                continue
                
            # Set random seed for reproducibility
            torch.manual_seed(trial)
                
            # Create product manifold
            pm = ProductManifold(signature=signature).to(device)
            
            # Train coordinates using the stochastic Riemannian optimizer
            try:
                # Import the coordinate learning module
                from manify.embedders.coordinate_learning import train_coords
                
                # Train coordinates on the distance matrix
                X_embed, losses = train_coords(
                    pm,
                    dists.to(device),
                    burn_in_iterations=int(0.1 * embedding_iterations),
                    training_iterations=int(0.9 * embedding_iterations),
                    scale_factor_learning_rate=0,  # Fixed scale factor
                    burn_in_learning_rate=learning_rate * 0.1,
                    learning_rate=learning_rate,
                )
                
                logger.info(f"Trained coordinates for {graph_name}, final loss: {losses[-1]:.4f}")
                
            except Exception as e:
                logger.error(f"Error training coordinates for {graph_name}: {str(e)}")
                continue
                
            # Create link prediction dataset
            try:
                # Generate pairwise features and labels
                X, y, pm_new = make_link_prediction_dataset(
                    X_embed, 
                    pm, 
                    adj.to(device), 
                    add_dists=add_dists
                )
                
                # Split into train and test sets
                n_pairs, n_dims = X.shape
                n_nodes = int(n_pairs**0.5)
                
                # Reshape tensors to node x node format
                X_reshaped = X.view(n_nodes, n_nodes, -1)
                y_reshaped = y.view(n_nodes, n_nodes)
                
                # Take test_size % of the nodes as test nodes
                idx = list(range(n_nodes))
                idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=trial)
                
                # Extract train and test sets
                X_train = X_reshaped[idx_train][:, idx_train].reshape(-1, n_dims)
                y_train = y_reshaped[idx_train][:, idx_train].reshape(-1)
                
                X_test = X_reshaped[idx_test][:, idx_test].reshape(-1, n_dims)
                y_test = y_reshaped[idx_test][:, idx_test].reshape(-1)
                
                logger.info(f"Created link prediction dataset: X_train: {X_train.shape}, y_train: {y_train.shape}")
                
            except Exception as e:
                logger.error(f"Error creating link prediction dataset for {graph_name}: {str(e)}")
                continue
                
            # Run benchmark with error handling
            try:
                model_results = benchmark(
                    X=None, 
                    y=None,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    pm=pm_new,
                    task="classification",
                    score=classification_metrics,
                    models=models,
                    device=device,
                    max_depth=max_depth
                )
                
                # Add metadata
                model_results["graph"] = graph_name
                model_results["signature"] = signature_name
                model_results["trial"] = trial
                model_results["task"] = "classification"  # Link prediction is always classification
                model_results["benchmark_type"] = "link_prediction"
                
                # Skip wandb logging in worker processes - main process will handle it
                
            except Exception as e:
                logger.error(f"Error in benchmark with graph={graph_name}, signature={signature_name}, trial={trial}: {str(e)}")
                # Create an empty result with error information
                model_results = {
                    "graph": graph_name,
                    "signature": signature_name,
                    "trial": trial,
                    "task": "classification",
                    "benchmark_type": "link_prediction",
                    "error": str(e)
                }
            
            # Save this individual result immediately to disk
            try:
                # Create incremental results directory if needed
                incremental_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data/results_wandb/incremental')
                os.makedirs(incremental_dir, exist_ok=True)
                
                # Create an individual file for this result
                result_filename = f"link_prediction_{graph_name}_{signature_name}_trial{trial}_{int(time.time())}.json"
                result_path = os.path.join(incremental_dir, result_filename)
                
                # Safe write to avoid data loss (write to temp file first, then rename)
                with open(result_path + ".tmp", 'w') as f:
                    json.dump(model_results, f, indent=2)
                os.rename(result_path + ".tmp", result_path)
                
                logger.info(f"Saved individual result to {result_path}")
            except Exception as e:
                logger.error(f"ERROR SAVING INDIVIDUAL RESULT: {str(e)}")
            
            # Append to in-memory results collection
            results.append(model_results)
            logger.info(f"Completed link prediction benchmark with graph={graph_name}, signature={signature_name}, trial={trial}")
            
        except Exception as e:
            logger.error(f"Error processing graph {graph_name} with signature {signature_name}, trial {trial}: {e}")
    
    # Don't finish the wandb run - just let the main process handle it
    # Put results in queue for main process
    results_queue.put(results)


def run_graph_embedding_benchmark(
    config: BenchmarkConfig,
    results_queue: Queue,
    gpu_id: Optional[int] = None,
    trial_indices: Optional[List[int]] = None,
    no_wandb: bool = False,
    wandb_group: Optional[str] = None,
    wandb_run_id: Optional[str] = None
) -> None:
    # No wandb initialization in worker processes
    """Run benchmarks for pre-computed graph embeddings."""
    device = get_device(gpu_id)

    # Extract configuration
    benchmark_config = config.get("graph_embedding", {})
    
    # Resolve paths relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    embeddings_dir = os.path.join(project_root, benchmark_config.get("embeddings_dir", "data/graphs/embeddings"))
    data_dir = os.path.join(project_root, benchmark_config.get("data_dir", "data/graphs"))
    
    embedding_names = benchmark_config.get("embedding_names", ["polblogs", "cs_phds", "cora", "citeseer"])
    signatures = benchmark_config.get("signatures", ["SS", "H", "HS"])  # Default to pre-selected optimal signatures
    n_trials = benchmark_config.get("n_trials", 10)
    
    # Default models plus any model exclusions for this specific run
    default_models = ["sklearn_dt", "sklearn_rf", "product_dt", "product_rf"]
    excluded_models = benchmark_config.get("exclude_models", [])
    
    # Get configured models or use defaults, filtering out excluded models
    models = [m for m in benchmark_config.get("models", default_models) if m not in excluded_models]
    
    logger.info(f"Using models for graph_embedding benchmark: {models}")
    
    task_overrides = benchmark_config.get("task_overrides", {})

    # Set up scoring metrics
    classification_metrics = benchmark_config.get("classification_metrics", ["accuracy", "f1-micro"])
    regression_metrics = benchmark_config.get("regression_metrics", ["rmse"])

    # Generate trials based on embedding datasets, signatures, and trial numbers
    # Each dataset has a specific signature from the config
    trials = []
    dataset_signature_map = {
        'polblogs': 'SS',
        'cora': 'H',
        'citeseer': 'HS', 
        'cs_phds': 'H'
    }
    
    # Override with values from config if specified
    if len(embedding_names) == len(signatures):
        for i, emb_name in enumerate(embedding_names):
            sig = signatures[i]
            for trial in range(n_trials):
                trials.append((emb_name, sig, trial))
    else:
        # Use default mapping if lengths don't match
        for emb_name in embedding_names:
            sig = dataset_signature_map.get(emb_name, signatures[0])  # Fallback to first signature
            for trial in range(n_trials):
                trials.append((emb_name, sig, trial))

    # Filter trials based on trial_indices if provided
    if trial_indices is not None:
        trials = [trials[i] for i in trial_indices]

    results = []
    for embedding_name, signature, trial in trials:
        logger.info(
            f"Running graph embedding benchmark with dataset={embedding_name}, signature={signature}, trial={trial}"
        )

        # Determine the task type based on dataset or override
        default_task = benchmark_config.get("task", "classification")
        task = task_overrides.get(embedding_name, default_task)

        # Set up task-specific scoring metrics
        score = classification_metrics if task == "classification" else regression_metrics

        # Load pre-computed embeddings
        try:
            data_path = os.path.join(embeddings_dir, embedding_name, f"{signature}_{trial}.h5")
            logger.info(f"Looking for embedding file: {data_path}")
            
            # Check if embedding file exists
            if not os.path.exists(data_path):
                logger.error(f"Embedding file not found: {data_path}")
                continue
            
            # Actually try to load the file
            try:
                logger.info(f"Loading embedding from {data_path}")
                data = torch.load(data_path, map_location=device, weights_only=True)
            except Exception as e:
                logger.error(f"Error loading embedding {data_path}: {str(e)}")
                continue
                
            # Verify data has required keys
            required_keys = ["X_train", "X_test", "y_train", "y_test", "test_idx"]
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                logger.error(f"Embedding file {data_path} missing required keys: {missing_keys}")
                continue

            # Extract embedding data
            X_train = data["X_train"].to(device)
            X_test = data["X_test"].to(device)
            y_train = data["y_train"].to(device)
            y_test = data["y_test"].to(device)
            test_idx = data["test_idx"]
            train_idx = [i for i in range(len(X_train) + len(X_test)) if i not in test_idx]

            # Load adjacency matrix if needed for graph-based models
            adj_path = os.path.join(data_dir, f"{embedding_name}.edges")
            if os.path.exists(adj_path):
                _, _, adj = manify.utils.dataloaders.load(embedding_name)
                A_hat = get_A_hat(adj.float())
                A_train = A_hat[train_idx][:, train_idx].to(device)
                A_test = A_hat[test_idx][:, test_idx].to(device)
            else:
                A_train = None
                A_test = None

            # Set up product manifold from signature string
            sig_map = {
                "HH": [(-1, 2), (-1, 2)],
                "HE": [(-1, 2), (0, 2)],
                "HS": [(-1, 2), (1, 2)],
                "SS": [(1, 2), (1, 2)],
                "SE": [(1, 2), (0, 2)],
                "H": [(-1, 4)],
                "E": [(0, 4)],
                "S": [(1, 4)],
            }

            pm = ProductManifold(signature=sig_map[signature], device=device)

            # Run benchmark with error handling
            try:
                model_results = benchmark(
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
                )
                
                # Add metadata
                model_results["embedding"] = embedding_name
                model_results["signature"] = signature
                model_results["trial"] = trial
                model_results["task"] = task
                model_results["benchmark_type"] = "graph_embedding"
                
                # Skip wandb logging in worker processes - we'll do it in the main process
                            
            except Exception as e:
                logger.error(f"Error in benchmark with embedding={embedding_name}, signature={signature}, trial={trial}: {str(e)}")
                # Create an empty result with error information
                model_results = {
                    "embedding": embedding_name,
                    "signature": signature,
                    "trial": trial,
                    "task": task,
                    "benchmark_type": "graph_embedding",
                    "error": str(e)
                }
                # Skip wandb logging in worker processes

            # Save this individual result immediately to disk
            try:
                # Create incremental results directory if needed
                incremental_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data/results_wandb/incremental')
                os.makedirs(incremental_dir, exist_ok=True)
                
                # Create an individual file for this result
                result_filename = f"graph_embedding_{embedding_name}_{signature}_trial{trial}_{int(time.time())}.json"
                result_path = os.path.join(incremental_dir, result_filename)
                
                # Safe write to avoid data loss (write to temp file first, then rename)
                with open(result_path + ".tmp", 'w') as f:
                    json.dump(model_results, f, indent=2)
                os.rename(result_path + ".tmp", result_path)
                
                logger.info(f"Saved individual result to {result_path}")
            except Exception as e:
                logger.error(f"ERROR SAVING INDIVIDUAL RESULT: {str(e)}")
            
            # Append to in-memory results collection
            results.append(model_results)
            logger.info(
                f"Completed graph embedding benchmark with dataset={embedding_name}, signature={signature}, trial={trial}"
            )

        except Exception as e:
            logger.error(f"Error processing embedding {embedding_name}, signature {signature}, trial {trial}: {e}")

    # Don't finish the wandb run - just let the main process handle it
    # Put results in queue for main process
    results_queue.put(results)


def run_vae_embedding_benchmark(
    config: BenchmarkConfig,
    results_queue: Queue,
    gpu_id: Optional[int] = None,
    trial_indices: Optional[List[int]] = None,
    no_wandb: bool = False,
    wandb_group: Optional[str] = None,
    wandb_run_id: Optional[str] = None
) -> None:
    # No wandb initialization in worker processes
    """Run benchmarks for pre-computed VAE embeddings."""
    device = get_device(gpu_id)

    # Extract configuration
    benchmark_config = config.get("vae_embedding", {})
    
    # Resolve paths relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, benchmark_config.get("data_dir", "data"))
    
    embedding_names = benchmark_config.get("embedding_names", ["blood_cell_scrna", "lymphoma", "cifar_100", "mnist"])
    signatures = benchmark_config.get(
        "signatures",
        [
            [(1, 2), (0, 2), (-1, 2), (-1, 2), (-1, 2)],  # blood_cell_scrna
            [(1, 2), (1, 2)],  # lymphoma
            [(1, 2), (1, 2), (1, 2), (1, 2)],  # cifar_100
            [(1, 2), (0, 2), (-1, 2)],  # mnist
        ],
    )
    n_trials = benchmark_config.get("n_trials", 10)
    n_samples = benchmark_config.get("n_samples", 1000)  # Max samples to use (for large datasets)
    
    # Default models plus any model exclusions for this specific run
    default_models = ["sklearn_dt", "sklearn_rf", "product_dt", "product_rf"]
    excluded_models = benchmark_config.get("exclude_models", [])
    
    # Get configured models or use defaults, filtering out excluded models
    models = [m for m in benchmark_config.get("models", default_models) if m not in excluded_models]
    
    logger.info(f"Using models for vae_embedding benchmark: {models}")
    
    task = benchmark_config.get("task", "classification")

    # Set up scoring metrics
    classification_metrics = benchmark_config.get("classification_metrics", ["accuracy", "f1-micro"])

    # Generate trials based on embedding datasets and trial numbers
    trials = []
    for i, emb_name in enumerate(embedding_names):
        if i < len(signatures):
            sig = signatures[i]
        else:
            logger.warning(f"No signature provided for {emb_name}, using default signature")
            sig = [(1, 2), (1, 2)]

        for trial in range(n_trials):
            trials.append((emb_name, sig, trial))

    # Filter trials based on trial_indices if provided
    if trial_indices is not None:
        trials = [trials[i] for i in trial_indices]

    results = []
    for embedding_name, signature, trial in trials:
        logger.info(f"Running VAE embedding benchmark with dataset={embedding_name}, trial={trial}")

        try:
            # Load pre-computed embeddings
            x_train_path = f"{data_dir}/{embedding_name}/embeddings/X_train_{trial}.npy"
            y_train_path = f"{data_dir}/{embedding_name}/embeddings/y_train_{trial}.npy"
            x_test_path = f"{data_dir}/{embedding_name}/embeddings/X_test_{trial}.npy"
            y_test_path = f"{data_dir}/{embedding_name}/embeddings/y_test_{trial}.npy"
            
            # Check if all required files exist
            missing_files = []
            for path in [x_train_path, y_train_path, x_test_path, y_test_path]:
                if not os.path.exists(path):
                    missing_files.append(path)
            
            if missing_files:
                logger.error(f"Missing embedding files for {embedding_name} trial {trial}: {missing_files}")
                continue
                
            logger.info(f"Loading VAE embeddings for {embedding_name} trial {trial}")
            
            try:
                X_train = np.load(x_train_path)
                y_train = np.load(y_train_path)
                X_test = np.load(x_test_path)
                y_test = np.load(y_test_path)
            except Exception as e:
                logger.error(f"Error loading VAE embedding files for {embedding_name} trial {trial}: {str(e)}")
                continue

            # Randomly subsample large datasets
            if len(X_train) > n_samples:
                idx = np.random.choice(X_train.shape[0], n_samples, replace=False)
                X_train = X_train[idx]
                y_train = y_train[idx]

            if len(X_test) > n_samples:
                idx = np.random.choice(X_test.shape[0], n_samples, replace=False)
                X_test = X_test[idx]
                y_test = y_test[idx]

            # Convert to torch tensors
            X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
            y_train = torch.tensor(y_train, dtype=torch.long, device=device)
            X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
            y_test = torch.tensor(y_test, dtype=torch.long, device=device)

            # Set up product manifold
            pm = ProductManifold(signature=signature, device=device)

            # Generate adjacency matrices for graph-based models
            D_train = pm.pdist2(X_train)
            max_train_dist = D_train[D_train.isfinite()].max()
            D_train /= max_train_dist
            A_train = get_A_hat(torch.exp(-D_train))
            A_test = get_A_hat(torch.exp(-pm.pdist2(X_test) / max_train_dist))

            # Set up scoring metrics
            score = classification_metrics

            # Run benchmark with error handling
            try:
                model_results = benchmark(
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
                )
                
                # Add metadata
                model_results["embedding"] = embedding_name
                model_results["trial"] = trial
                model_results["task"] = task
                model_results["benchmark_type"] = "vae_embedding"
                
                # Skip wandb logging in worker processes - we'll do it in the main process
                            
            except Exception as e:
                logger.error(f"Error in benchmark with embedding={embedding_name}, trial={trial}: {str(e)}")
                # Create an empty result with error information
                model_results = {
                    "embedding": embedding_name,
                    "trial": trial,
                    "task": task,
                    "benchmark_type": "vae_embedding",
                    "error": str(e)
                }
                # Skip wandb logging in worker processes

            # Save this individual result immediately to disk
            try:
                # Create incremental results directory if needed
                incremental_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data/results_wandb/incremental')
                os.makedirs(incremental_dir, exist_ok=True)
                
                # Create an individual file for this result
                result_filename = f"vae_embedding_{embedding_name}_trial{trial}_{int(time.time())}.json"
                result_path = os.path.join(incremental_dir, result_filename)
                
                # Safe write to avoid data loss (write to temp file first, then rename)
                with open(result_path + ".tmp", 'w') as f:
                    json.dump(model_results, f, indent=2)
                os.rename(result_path + ".tmp", result_path)
                
                logger.info(f"Saved individual result to {result_path}")
            except Exception as e:
                logger.error(f"ERROR SAVING INDIVIDUAL RESULT: {str(e)}")
            
            # Append to in-memory results collection
            results.append(model_results)
            logger.info(f"Completed VAE embedding benchmark with dataset={embedding_name}, trial={trial}")

        except Exception as e:
            logger.error(f"Error processing VAE embedding {embedding_name}, trial {trial}: {e}")

    # Don't finish the wandb run - just let the main process handle it
    # Put results in queue for main process
    results_queue.put(results)


def allocate_gpus(benchmark_types: List[str], available_gpus: List[int]) -> Dict[str, List[int]]:
    """Allocate GPUs to benchmark tasks."""
    allocation = {}
    n_benchmarks = len(benchmark_types)
    n_gpus = len(available_gpus)

    # Simple allocation strategy: divide GPUs evenly among benchmarks
    gpus_per_benchmark = max(1, n_gpus // n_benchmarks)

    for i, benchmark_type in enumerate(benchmark_types):
        start_idx = (i * gpus_per_benchmark) % n_gpus
        # If this is the last benchmark, give it all remaining GPUs
        if i == n_benchmarks - 1:
            allocated_gpus = available_gpus[start_idx:] + available_gpus[:start_idx]
        else:
            end_idx = (start_idx + gpus_per_benchmark) % n_gpus
            if end_idx > start_idx:
                allocated_gpus = available_gpus[start_idx:end_idx]
            else:
                allocated_gpus = available_gpus[start_idx:] + available_gpus[:end_idx]

        allocation[benchmark_type] = allocated_gpus

    return allocation


def partition_trials(n_trials: int, n_gpus: int) -> List[List[int]]:
    """Partition trial indices among available GPUs for parallel execution."""
    indices = list(range(n_trials))
    partitions = [[] for _ in range(n_gpus)]

    for i, idx in enumerate(indices):
        partitions[i % n_gpus].append(idx)

    return partitions


def save_results(results: List[Dict], output_dir: str, benchmark_type: str) -> None:
    """Save benchmark results to a file."""
    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Ensure output_dir is an absolute path
    if not os.path.isabs(output_dir):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        output_dir = os.path.join(project_root, output_dir)
        logger.info(f"Converting relative output path to absolute: {output_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Save as TSV
    output_path = os.path.join(output_dir, f"{benchmark_type}_{timestamp}.tsv")
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Results saved to {output_path}")

    return output_path


def main():
    """Main entry point for the benchmarking script."""
    global debug_mode
    global original_stdout
    original_stdout = None
    
    parser = argparse.ArgumentParser(description="Run manify benchmarks with YAML configuration")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--output-dir", default=None, help="Directory to save benchmark results (overrides config)")
    parser.add_argument("--dry-run", action="store_true", help="Perform a run without saving results to disk (will still log to wandb)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use (default: all available)")
    parser.add_argument("--gpu-ids", type=str, help="Comma-separated list of specific GPU IDs to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--max-trials", type=int, help="Maximum number of trials to run per benchmark (for quick testing)")
    parser.add_argument("--models", type=str, help="Comma-separated list of models to use (overrides config)")
    parser.add_argument(
        "--benchmark-types",
        nargs="+",
        choices=["single_curvature", "signature_gaussian", "graph_embedding", "vae_embedding", "all"],
        default=["all"],
        help="Type(s) of benchmark to run",
    )
    parser.add_argument("--gpu-allocation", type=str, help="JSON string mapping benchmark types to GPU IDs")

    args = parser.parse_args()

    # Set logging level based on flags
    if args.debug:
        debug_mode = True
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    elif args.quiet:
        logger.setLevel(logging.WARNING)
        # Set environment variables to suppress tqdm progress bars and torch outputs
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["TORCH_SHOW_WARNINGS"] = "0"
        # Suppress other libraries' logging
        logging.getLogger("torch").setLevel(logging.ERROR)
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("PIL").setLevel(logging.ERROR)
        # Redirect stdout temporarily during neural network training in benchmark to reduce output
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        logger.warning("Quiet mode enabled - reduced output verbosity")

    # Load configuration
    config = load_config(args.config)
    
    # Apply max-trials limit if specified
    if args.max_trials is not None:
        logger.info(f"Limiting to maximum {args.max_trials} trials per benchmark type")
        for section in ["gaussian"] + list(config.keys()):
            if section in config:
                if "n_trials_gaussian" in config[section]:
                    config[section]["n_trials_gaussian"] = min(config[section]["n_trials_gaussian"], args.max_trials)
                    logger.info(f"Set {section}.n_trials_gaussian = {config[section]['n_trials_gaussian']}")
                if "n_trials" in config[section]:
                    config[section]["n_trials"] = min(config[section]["n_trials"], args.max_trials)
                    logger.info(f"Set {section}.n_trials = {config[section]['n_trials']}")
    
    # Override models if specified
    if args.models:
        model_list = [m.strip() for m in args.models.split(",")]
        logger.info(f"Overriding models with: {model_list}")
        # Update common config if it exists
        if "common" in config and "models" in config["common"]:
            config["common"]["models"] = model_list
        # Update each benchmark type directly
        for section in ["single_curvature", "signature_gaussian", "graph_embedding", "vae_embedding"]:
            if section in config:
                config[section]["models"] = model_list

    # Override output directory if specified
    if args.output_dir:
        # Ensure the provided output directory is an absolute path
        output_dir = args.output_dir
        if not os.path.isabs(output_dir):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            output_dir = os.path.join(project_root, output_dir)
            logger.info(f"Converting relative command-line output path to absolute: {output_dir}")
        
        # Update all relevant sections in the config
        for section in ["common"] + list(config.keys()):
            if section in config and "output_dir" in config[section]:
                config[section]["output_dir"] = output_dir

    # Determine available GPUs
    if args.gpu_ids:
        available_gpus = [int(gpu_id) for gpu_id in args.gpu_ids.split(",")]
    else:
        available_gpus = get_available_gpus()
        if args.gpus is not None and args.gpus < len(available_gpus):
            available_gpus = available_gpus[: args.gpus]

    n_gpus = len(available_gpus)
    logger.info(f"Using {n_gpus} GPUs: {available_gpus}")

    # Initialize wandb if enabled
    if not args.no_wandb and not args.dry_run:
        setup_wandb(config, dry_run=False)
    elif not args.no_wandb and args.dry_run:
        setup_wandb(config, dry_run=True)

    # Determine which benchmarks to run
    all_benchmark_types = ["single_curvature", "signature_gaussian", "graph_embedding", "vae_embedding", "link_prediction"]
    benchmark_types = []

    if "all" in args.benchmark_types:
        benchmark_types = all_benchmark_types
    else:
        benchmark_types = args.benchmark_types

    # Filter benchmark types based on config
    valid_benchmark_types = []
    for benchmark_type in benchmark_types:
        if benchmark_type in config:
            valid_benchmark_types.append(benchmark_type)
        else:
            logger.warning(f"No configuration found for {benchmark_type}, skipping")

    if not valid_benchmark_types:
        logger.error("No valid benchmark types to run")
        sys.exit(1)

    logger.info(f"Running benchmarks: {valid_benchmark_types}")

    # Allocate GPUs to benchmarks
    if args.gpu_allocation:
        gpu_allocation = json.loads(args.gpu_allocation)
    else:
        gpu_allocation = allocate_gpus(valid_benchmark_types, available_gpus)

    logger.info(f"GPU allocation: {gpu_allocation}")

    # Mapping from benchmark type to function
    benchmark_funcs = {
        "single_curvature": run_single_curvature_benchmark,
        "signature_gaussian": run_signature_gaussian_benchmark,
        "graph_embedding": run_graph_embedding_benchmark,
        "vae_embedding": run_vae_embedding_benchmark,
        "link_prediction": run_link_prediction_benchmark,
    }

    # Run selected benchmarks
    all_results = []
    processes = []
    results_queues = {}

    # Initialize wandb group run ID
    if not args.no_wandb:
        wandb_run_id = wandb.run.id if wandb.run else None
        logger.info(f"Main wandb run ID: {wandb_run_id}")
    else:
        wandb_run_id = None
    
    # Start benchmark processes
    for benchmark_type in valid_benchmark_types:
        logger.info(f"Starting {benchmark_type} benchmark")

        # Get benchmark config and function
        benchmark_config = config[benchmark_type]
        benchmark_func = benchmark_funcs[benchmark_type]

        # Estimate total number of trials for this benchmark
        total_trials = 0
        if benchmark_type == "single_curvature":
            n_trials_gaussian = benchmark_config.get("n_trials_gaussian", 10)
            n_tasks = 2  # classification and regression
            total_trials = len(benchmark_config.get("curvatures", [])) * n_trials_gaussian * n_tasks
        elif benchmark_type == "signature_gaussian":
            n_trials_gaussian = benchmark_config.get("n_trials_gaussian", 10)
            n_tasks = 2  # classification and regression
            total_trials = len(benchmark_config.get("signatures", [])) * n_trials_gaussian * n_tasks
        elif benchmark_type == "graph_embedding":
            total_trials = (
                len(benchmark_config.get("embedding_names", []))
                * len(benchmark_config.get("signatures", []))
                * benchmark_config.get("n_trials", 10)
            )
        elif benchmark_type == "vae_embedding":
            total_trials = len(benchmark_config.get("embedding_names", [])) * benchmark_config.get("n_trials", 10)

        # Get GPUs allocated for this benchmark
        allocated_gpus = gpu_allocation.get(benchmark_type, available_gpus[:1] if available_gpus else [None])
        
        # If no GPUs are available, use CPU (None as gpu_id)
        if not allocated_gpus:
            allocated_gpus = [None]
            logger.warning(f"No GPUs available for {benchmark_type}, using CPU")
        
        num_gpus = len(allocated_gpus)
        logger.info(f"Allocated {num_gpus} GPU(s) for {benchmark_type}: {allocated_gpus}")
        
        # Partition trials among GPUs
        trial_partitions = partition_trials(total_trials, num_gpus)

        # Set up results queue
        results_queue = Queue()
        results_queues[benchmark_type] = results_queue

        # Start processes for this benchmark
        for i, gpu_id in enumerate(allocated_gpus):
            if i < len(trial_partitions) and trial_partitions[i]:  # Only start process if there are trials to run
                p = Process(
                    target=benchmark_func, 
                    args=(
                        config, 
                        results_queue, 
                        gpu_id, 
                        trial_partitions[i], 
                        args.no_wandb,
                    ),
                    kwargs={
                        "wandb_group": benchmark_type,
                        "wandb_run_id": wandb_run_id
                    }
                )
                p.start()
                processes.append((benchmark_type, p))
                logger.info(f"Started {benchmark_type} process on GPU {gpu_id} with {len(trial_partitions[i])} trials")

    # Wait for all processes to complete and collect results
    benchmark_results = {benchmark_type: [] for benchmark_type in valid_benchmark_types}

    # Set up output directory
    output_dir = config.get("common", {}).get("output_dir", "results")
    
    # Ensure output_dir is an absolute path
    if not os.path.isabs(output_dir):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        output_dir = os.path.join(project_root, output_dir)
        logger.info(f"Using absolute output directory path: {output_dir}")

    # Wait for all processes to finish
    try:
        # Process monitoring loop
        running_processes = True
        while running_processes:
            running_processes = False
            # Create a new list rather than modifying the original during iteration
            still_running = []
            
            for benchmark_type, process in processes:
                if process.is_alive():
                    running_processes = True
                    still_running.append((benchmark_type, process))
                else:
                    logger.info(f"{benchmark_type} process finished with exit code {process.exitcode}")
                    if process.exitcode != 0:
                        logger.error(f"{benchmark_type} process exited with code {process.exitcode}")
            
            # Replace the processes list with only those still running
            processes = still_running
            
            # Check for available results
            for benchmark_type, queue in results_queues.items():
                try:
                    while not queue.empty():
                        try:
                            results = queue.get_nowait()
                            if results:
                                benchmark_results[benchmark_type].extend(results)
                                logger.info(f"Collected {len(results)} results from {benchmark_type}")
                                
                                # Save results immediately to disk as they arrive in main process
                                try:
                                    # Create a timestamp for unique filenames
                                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                                    
                                    # Ensure output_dir exists
                                    output_dir = config.get("common", {}).get("output_dir", "data/results_wandb")
                                    if not os.path.isabs(output_dir):
                                        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                                        output_dir = os.path.join(project_root, output_dir)
                                    os.makedirs(output_dir, exist_ok=True)
                                    
                                    # Immediately save these new results
                                    incremental_output_path = os.path.join(output_dir, f"{benchmark_type}_incremental_{timestamp}.tsv")
                                    
                                    # Save these specific results
                                    df = pd.DataFrame(results)
                                    df.to_csv(incremental_output_path, sep="\t", index=False)
                                    logger.info(f"SAVED {len(results)} NEW RESULTS TO {incremental_output_path}")
                                    
                                    # Also append to cumulative file
                                    cumulative_path = os.path.join(output_dir, f"{benchmark_type}_cumulative.tsv")
                                    
                                    if os.path.exists(cumulative_path):
                                        # Append to existing cumulative file
                                        df.to_csv(cumulative_path, sep="\t", index=False, mode='a', header=False)
                                    else:
                                        # Create new cumulative file
                                        df.to_csv(cumulative_path, sep="\t", index=False)
                                    
                                    logger.info(f"Updated cumulative results file: {cumulative_path}")
                                except Exception as e:
                                    logger.error(f"ERROR SAVING REAL-TIME RESULTS: {str(e)}")
                                
                                # Real-time logging to wandb if enabled
                                if not args.no_wandb and not args.dry_run:
                                    # Log each result immediately as it comes in
                                    for result in results:
                                        # Create a clean unique ID for this result
                                        if "benchmark_type" in result and result["benchmark_type"] == "single_curvature":
                                            result_id = f"K{result.get('curvature', 'unknown')}_seed{result.get('seed', 'unknown')}_{result.get('task', 'unknown')}"
                                        elif "benchmark_type" in result and result["benchmark_type"] == "signature_gaussian":
                                            result_id = f"{result.get('signature', 'unknown')}_seed{result.get('seed', 'unknown')}_{result.get('task', 'unknown')}"
                                        elif "benchmark_type" in result and result["benchmark_type"] == "graph_embedding":
                                            result_id = f"{result.get('embedding', 'unknown')}_{result.get('signature', 'unknown')}_trial{result.get('trial', 'unknown')}"
                                        elif "benchmark_type" in result and result["benchmark_type"] == "vae_embedding":
                                            result_id = f"{result.get('embedding', 'unknown')}_trial{result.get('trial', 'unknown')}"
                                        else:
                                            result_id = f"result_{hash(str(result))}"
                                        
                                        # Log the results with metadata using a flat structure
                                        for model_name, metrics in result.items():
                                            if isinstance(metrics, dict) and model_name not in ["benchmark_type", "error"]:
                                                # Create a flat log entry with all metrics and metadata
                                                log_data = {}
                                                
                                                # Add metadata first (consistent for all logs)
                                                log_data["benchmark_type"] = benchmark_type
                                                log_data["model"] = model_name
                                                log_data["result_id"] = result_id
                                                
                                                # Add all relevant result metadata
                                                for k, v in result.items():
                                                    if k not in ["benchmark_type", model_name, "error"] and isinstance(v, (int, float, bool, str)):
                                                        log_data[k] = v
                                                
                                                # Add model metrics with simple naming
                                                for k, v in metrics.items():
                                                    if isinstance(v, (int, float, bool, str)):
                                                        # Use simple flat names for metrics
                                                        log_data[k] = v
                                                
                                                # Generate a unique step ID to ensure separate points in wandb
                                                step_id = int(time.time() * 1000) % 1000000
                                                wandb.log(log_data, step=step_id, commit=True)
                        except Exception as e:
                            logger.error(f"Error getting results from queue for {benchmark_type}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error accessing queue for {benchmark_type}: {str(e)}")
            
            # Short sleep to prevent CPU spinning
            time.sleep(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user, terminating processes...")
        for benchmark_type, process in processes:
            if process.is_alive():
                logger.info(f"Terminating {benchmark_type} process")
                process.terminate()
        sys.exit(1)

    # Make sure we get any remaining results
    for benchmark_type, queue in results_queues.items():
        while not queue.empty():
            try:
                results = queue.get_nowait()
                benchmark_results[benchmark_type].extend(results)
            except Exception as e:
                logger.error(f"Error getting final results from queue: {e}")

    # Save results and log to wandb if not dry run
    if not args.dry_run:
        for benchmark_type, results in benchmark_results.items():
            if results:
                output_path = save_results(results, output_dir, benchmark_type)

                # Log results to wandb if enabled
                if not args.no_wandb:
                    # Create a clean flattened pandas DataFrame for easier visualization
                    flattened_data = []
                    for result in results:
                        # Extract metadata
                        metadata = {k: v for k, v in result.items() 
                                    if k not in ["error"] and not isinstance(v, dict)}
                        
                        # Extract metrics from each model
                        for model_name, metrics in result.items():
                            if isinstance(metrics, dict):
                                # Create a record with metadata plus model metrics
                                record = metadata.copy()
                                record["model"] = model_name
                                # Add metrics with simple column names
                                for metric_name, metric_value in metrics.items():
                                    if isinstance(metric_value, (int, float, bool, str)):
                                        record[metric_name] = metric_value
                                flattened_data.append(record)
                    
                    # Convert to DataFrame
                    if flattened_data:
                        results_df = pd.DataFrame(flattened_data)
                        
                        # Log as both table artifact (can be downloaded) and metrics
                        wandb.log({f"{benchmark_type}_results_table": wandb.Table(dataframe=results_df)})
                        
                        # Create summary table for each metric
                        try:
                            for metric in ["accuracy", "f1-micro", "rmse", "mse"]:
                                if metric in results_df.columns:
                                    # Group by relevant columns and compute mean
                                    group_cols = [col for col in ["model", "benchmark_type", "task", "curvature", 
                                                                  "signature", "embedding", "trial"] 
                                                 if col in results_df.columns]
                                    if group_cols:
                                        summary = results_df.groupby(group_cols)[metric].mean().reset_index()
                                        wandb.log({f"{benchmark_type}_{metric}_summary": wandb.Table(dataframe=summary)})
                        except Exception as e:
                            logger.warning(f"Error creating metric summaries: {e}")
                    
                    wandb.save(output_path)

    # Finish wandb run if enabled
    if not args.no_wandb:
        wandb.finish()
        
    # Restore stdout if it was redirected
    if args.quiet and hasattr(sys.stdout, 'close') and original_stdout is not None:
        sys.stdout.close()
        sys.stdout = original_stdout

    logger.info("All benchmarks completed")


if __name__ == "__main__":
    main()
