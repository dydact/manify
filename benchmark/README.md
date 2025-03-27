# Manify Benchmarking Framework

This directory contains a configurable script for running benchmarks across different types of data and manifold configurations. The benchmarking system is designed to be configurable via YAML, support parallel execution across GPUs, and integrate with Weights & Biases for experiment tracking.

## Features

- **YAML Configuration**: Define benchmark parameters in a simple YAML file
- **Weights & Biases Integration**: Track and compare benchmark runs
- **Parallel Execution**: Utilize multiple GPUs for faster benchmarking
- **Dry Run Mode**: Test configurations without saving results
- **Multiple Benchmark Types**: Support for different data types and manifold structures

## Benchmark Types

1. **Single Curvature Gaussian**: Synthetic data with a single curvature parameter (runs both classification and regression)
2. **Signature Gaussian**: Synthetic data with mixed curvature parameters (product manifolds, runs both classification and regression)
3. **Graph Embeddings**: Pre-computed graph embeddings with different manifold signatures
4. **VAE Embeddings**: Pre-computed VAE-generated embeddings with different manifold structures

## Usage

### Basic Usage

Run all benchmark types:

```bash
python run_benchmarks.py config.yaml
```

Run a specific benchmark type:

```bash
python run_benchmarks.py config.yaml --benchmark-types single_curvature
```

### Parallel Execution Across GPUs

The script automatically divides benchmarks across available GPUs:

```bash
python run_benchmarks.py config.yaml
```

Specify number of GPUs to use:

```bash
python run_benchmarks.py config.yaml --gpus 2
```

Specify specific GPU IDs:

```bash
python run_benchmarks.py config.yaml --gpu-ids 0,1
```

Run multiple benchmark types with custom GPU allocation:

```bash
python run_benchmarks.py config.yaml --benchmark-types single_curvature signature_gaussian --gpu-allocation '{"single_curvature": [0], "signature_gaussian": [1]}'
```

### Dry Run Mode

Test configuration without saving results:

```bash
python run_benchmarks.py config.yaml --dry-run
```

### Disable Weights & Biases

Run without logging to Weights & Biases:

```bash
python run_benchmarks.py config.yaml --no-wandb
```

## Configuration

The `config.yaml` file contains settings for each benchmark type with inheritance from common sections:

```yaml
# Common settings applied to all benchmark types
common:
  # Model selection 
  models:
    - sklearn_dt
    - sklearn_rf
    - product_dt
    - product_rf
    # other models...
  
  # Device settings
  device: cuda
  
  # Output settings
  output_dir: ../data/results_wandb
  
  # Default number of trials
  n_trials: 10

# Gaussian mixture model specific settings (for both single and signature)
gaussian:
  n_trials_gaussian: 10  # Specific number of trials for Gaussian benchmarks
  n_classes: 8
  n_clusters: 32
  cov_scale_means: 1.0
  cov_scale_points: 1.0

# Benchmark type specific settings
single_curvature:
  curvatures: [-2, -1, -0.5, 0, 0.5, 1, 2]
  dimension: 2
  # Inherits settings from common and gaussian sections

signature_gaussian:
  signatures:
    - [[-1, 2], [0, 2]]   # HE
    - [[-1, 2], [1, 2]]   # HS
    # other signatures...
  # Inherits settings from common and gaussian sections

# Other benchmark types...
```

## Output

Results are saved in TSV format to the specified output directory. Each benchmark run creates a timestamped file with detailed results for each model, parameter configuration, and trial.

When using Weights & Biases, results are also logged as tables and metrics for easier visualization and comparison.

## Benchmarking Pipeline

1. **Configuration**: Define parameters in the YAML file
2. **Execution**: Run benchmarks on specified data and manifold configurations
3. **Parallelization**: Distribute trials across available GPUs
4. **Results Collection**: Gather metrics from all trials and models
5. **Analysis**: Analyze results through W&B or custom analysis scripts

## Dependencies

- PyTorch
- NumPy
- Pandas
- PyYAML
- wandb (optional)

All dependencies should be included with the main manify package.