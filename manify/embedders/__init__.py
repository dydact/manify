"""Tools for embedding data into Riemannian manifolds and product spaces.

The embedders module provides various ways to embed data into manifolds of constant
or mixed curvature. The module includes:

* `coordinate_learning`: Direct optimization of coordinates in a product manifold.
* `siamese`: Siamese network-based embedding for metric learning.
* `vae`: Variational autoencoders for learning representations in product manifolds.
* `_losses`: Loss functions for measuring embedding quality.
"""

import manify.embedders.coordinate_learning
import manify.embedders.siamese
import manify.embedders.vae
