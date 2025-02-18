# `manify`
Generate graph/data embeddings and perform product space machine learning in multiple ways.

## Installation

There are two ways to install `manify`:

1. **GitHub**

   ```bash
   pip install git+https://github.com/pchlenski/manify

2. **pypi** 

   ```bash
   pip install manify
## What is done so far

**curvature estimation**

-  ``manify.curvature_estimation.delta_hyperbolicity.py`` - compute delta-hyperbolicity of a metric space.
-  ``manify.curvature_estimation.greedy_method.py`` -greedy selection of signatures
-  ``manify.curvature_estimation.sectional_curvature.py`` - sectional curvature estimation using Toponogov’s theorem

**embedders**

- ``manify.embedders.coordinate_learning.py`` - coordinate learning and optimization
- ``manify.embedders.losses.py`` - different measurement metrics
- ``manify.embedders.siamese.py`` - siamese network embedder
- ``manify.embedders.siamese.py`` - product space variational autoencoder

**predictors**

- ``manify.predictors.decision_tree.py`` - decision tree and random forest predictors for product space manifolds.
- ``manify.predictors.kappa_gcn.py`` - kappa gcn
- ``manify.predictors.kernel.py`` - kernel matrix calculation
- ``manify.predictors.midpoint.py`` - angular midpoints calculation
- ``manify.predictors.perceptron.py`` - product space perception
- ``manify.predictors.svm.py`` - product space svm

**manifold**

- ``manify.manifolds.py`` -tools for generating Riemannian manifolds and product manifolds.

**utils**

- ``manify.utils.benchmarks.py`` - tools for benchmarking.
- ``manify.utils.dataloaders.py`` - loading datasets.
- ``manify.utils.link_prediction.py`` - preprocessing graphs with link prediction
- ``manify.utils.visualization.py`` - tools for visualization.




