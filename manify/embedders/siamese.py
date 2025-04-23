"""Siamese network implementation for manifold embedding.

This module provides a Siamese network architecture that can be used for embedding data into product manifolds. Siamese
networks are particularly useful for metric learning tasks, where the goal is to learn a distance-preserving embedding,
while also encoding a set of features.

The SiameseNetwork class supports both encoding (embedding) data into a manifold space and optionally decoding
(reconstructing) from the embedding space back to the original data space.
"""

from __future__ import annotations

from typing import Optional

import torch
from jaxtyping import Float

from ..manifolds import ProductManifold


class SiameseNetwork(torch.nn.Module):
    """Siamese network for embedding data into a product manifold space.

    A Siamese network consists of an encoder network that maps input data to a latent representation in a product
    manifold, and optionally a decoder network that maps the latent representation back to the original feature space.

    Attributes:
        pm: The product manifold object defining the embedding space.
        encoder: Neural network module that maps input data to the embedding space.
        decoder: Optional neural network module for reconstructing input data from embeddings.
        reconstruction_loss: Loss function for measuring reconstruction quality.

    Args:
        pm: Product manifold object defining the target embedding space.
        encoder: Neural network module that maps inputs to the embedding space.
        decoder: Optional neural network module that maps embeddings back to input space.
            If None, a no-op identity module is used. Defaults to None.
        reconstruction_loss: Type of reconstruction loss to use.
            Currently only "mse" (mean squared error) is supported. Defaults to "mse".

    Raises:
        ValueError: If an unsupported reconstruction_loss is specified.
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
