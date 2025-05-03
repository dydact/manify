"""Siamese network implementation for manifold embedding.

This module provides a Siamese network architecture that can be used for embedding data into product manifolds. Siamese
networks are particularly useful for metric learning tasks, where the goal is to learn a distance-preserving embedding,
while also encoding a set of features.

The SiameseNetwork class supports both encoding (embedding) data into a manifold space and optionally decoding
(reconstructing) from the embedding space back to the original data space.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from jaxtyping import Float

from ..manifolds import ProductManifold
from ._base import BaseEmbedder
from ._losses import distortion_loss

# TQDM: notebook or regular
if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class SiameseNetwork(BaseEmbedder, torch.nn.Module):
    """Siamese network for embedding data into a product manifold space.

    A Siamese network consists of an encoder network that maps input data to a latent representation in a product
    manifold, and optionally a decoder network that maps the latent representation back to the original feature space.

    Attributes:
        pm: Product manifold defining the structure of the latent space.
        random_state: Random state for reproducibility.
        encoder: Neural network that maps inputs to latent embeddings.
        decoder: Neural network that reconstructs inputs from latent embeddings.
        beta: Weight for the distortion term in the loss function.
        device: Device for tensor computations.
        reconstruction_loss: Type of reconstruction loss to use.
        

    Args:
        pm: Product manifold defining the structure of the latent space.
        encoder: Neural network module that maps inputs to the manifold's intrinsic dimension.
            The output dimension should match the intrinsic dimension of the product manifold.
        decoder: Neural network module that maps latent representations back to the input space.
        random_state: Optional random state for reproducibility.
        device: Optional device for tensor computations.
        beta: Weight of the distortion term in the loss function.
        reconstruction_loss: Type of reconstruction loss to use.
    """

    def __init__(
        self,
        pm: ProductManifold,
        encoder: torch.nn.Module,
        decoder: Optional[torch.nn.Module] = None,
        reconstruction_loss: str = "mse",
    ):
        super().__init__()
        self.pm = pm
        self.encoder = encoder

        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = torch.nn.Identity()
            self.decoder.requires_grad_(False)
            self.decoder.to(pm.device)

        if reconstruction_loss == "mse":
            self.reconstruction_loss = torch.nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unknown reconstruction loss: {reconstruction_loss}")

    def encode(self, x: Float[torch.Tensor, "batch_size n_features"]) -> Float[torch.Tensor, "batch_size n_latent"]:
        """Encodes input data into the manifold embedding space.

        Takes a batch of input data and passes it through the encoder network to obtain embeddings in the manifold.

        Args:
            x: Input data tensor..

        Returns:
            embeddings: Tensor containing the embeddings in the manifold space.
        """
        return self.encoder(x)

    def decode(self, z: Float[torch.Tensor, "batch_size n_latent"]) -> Float[torch.Tensor, "batch_size n_features"]:
        """Decodes manifold embeddings back to the original input space.

        Takes a batch of embeddings from the manifold space and passes them through
        the decoder network to reconstruct the original input data.

        Args:
            z: Embedding tensor from the manifold space.

        Returns:
            reconstructed: Tensor containing the reconstructed input data.
        """
        return self.decoder(z)
