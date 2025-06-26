# Manify 🪐
> A Python Library for Learning Non-Euclidean Representations

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/pchlenski/manify)](https://github.com/pchlenski/manify/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/manify.svg)](https://badge.fury.io/py/manify)
[![Tests](https://github.com/pchlenski/manify/actions/workflows/test.yml/badge.svg)](https://github.com/pchlenski/manify/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/pchlenski/manify/branch/main/graph/badge.svg)](https://codecov.io/gh/pchlenski/manify)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


Manify is a Python library for generating graph/data embeddings and performing machine learning in product spaces with mixed curvature (hyperbolic, Euclidean, and spherical spaces). It provides tools for manifold creation, curvature estimation, embedding generation, and predictive modeling that respects the underlying geometry of complex data.

You can read our manuscript here: [Manify: A Python Library for Learning Non-Euclidean Representations](https://arxiv.org/abs/2503.09576)

📖 **Documentation**: [manify.readthedocs.io](https://manify.readthedocs.io)

## Key Features
- Create and manipulate manifolds with different curvatures (hyperbolic, Euclidean, spherical)
- Build product manifolds by combining multiple spaces with different geometric properties
- Learn embeddings of data in these manifolds
- Train machine learning models that respect the geometry of the embedding space
- Generate synthetic data with known geometric properties for benchmarking

## Installation

There are two ways to install `manify`:

1. **From PyPI**:
   ```bash
   pip install manify
   ```

2. **From GitHub**:
   ```bash
   pip install git+https://github.com/pchlenski/manify
   ```

## Quick Example

```python
import torch
from manify.manifolds import ProductManifold
from manify.embedders import CoordinateLearning
from manify.predictors.decision_tree import ProductSpaceDT
from manify.utils.dataloaders import load_hf
from sklearn.model_selection import train_test_split

# Load graph data
features, dists, adj, labels = load_hf("polblogs")

# Create product manifold
pm = ProductManifold(signature=[(1, 4)])  # S^4_1

# Learn embeddings (Gu et al (2018) method)
embedder = CoordinateLearning(pm=pm)
embedder.fit(X=None, D=dists)
X_embedded = embedder.transform()

# Train and evaluate classifier (Chlenski et al (2025) method)
X_train, X_test, y_train, y_test = train_test_split(X_embedded, labels)
tree = ProductSpaceDT(pm=pm, max_depth=3, task="classification")
tree.fit(X_train, y_train)
print(tree.score(X_test, y_test))
```

## Modules

**Manifold Operations**
- `manify.manifolds` - Tools for generating Riemannian manifolds and product manifolds

**Curvature Estimation**
- `manify.curvature_estimation.delta_hyperbolicity` - Compute delta-hyperbolicity of a metric space
- `manify.curvature_estimation.greedy_method` - Greedy selection of signatures
- `manify.curvature_estimation.sectional_curvature` - Sectional curvature estimation using Toponogov's theorem

**Embedders**
- `manify.embedders.coordinate_learning` - Coordinate learning and optimization
- `manify.embedders.siamese` - Siamese network embedder
- `manify.embedders.vae` - Product space variational autoencoder

**Predictors**
- `manify.predictors.decision_tree` - Decision tree and random forest predictors
- `manify.predictors.kappa_gcn` - Kappa GCN
- `manify.predictors.perceptron` - Product space perceptron
- `manify.predictors.svm` - Product space SVM

**Clustering**
- `manify.clustering.fuzzy_kmeans` - Riemannian fuzzy K-Means for clustering

**Optimizers**
- `manify.optimizers.radan` - Riemannian version of Adan optimizer

**Utilities**
- `manify.utils.benchmarks` - Tools for benchmarking
- `manify.utils.dataloaders` - Loading datasets
- `manify.utils.link_prediction` - Preprocessing graphs with link prediction
- `manify.utils.visualization` - Tools for visualization

## Research Background
Manify implements geometric machine learning approaches described in academic papers, particularly focusing on handling data with mixed geometric properties. It's especially suited for data that naturally lives in non-Euclidean spaces, such as hierarchical data, networks, and certain types of biological data.

## Citation
If you use our work, please cite the `Manify` paper:
```bibtex
@misc{chlenski2025manifypythonlibrarylearning,
      title={Manify: A Python Library for Learning Non-Euclidean Representations}, 
      author={Philippe Chlenski and Kaizhu Du and Dylan Satow and Itsik Pe'er},
      year={2025},
      eprint={2503.09576},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.09576}, 
}
```
