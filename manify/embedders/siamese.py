"""Siamese network embedder"""

from typing import List, Optional, Literal
from torchtyping import TensorType as TT
import torch

from ..manifolds import ProductManifold


class SiameseNetwork(torch.nn.Module):
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
            self.decoder = lambda x: x
            self.decoder.requires_grad_(False)
            self.decoder.to(pm.device)

        if reconstruction_loss == "mse":
            self.reconstruction_loss = torch.nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unknown reconstruction loss: {reconstruction_loss}")

    def encode(self, x: TT["batch_size", "n_features"]) -> TT["batch_size", "n_latent"]:
        """Encodes the input tensor into a latent representation.

        Args:
            x (TensorType["batch_size", "n_features"]): The input tensor.

        Returns:
            TensorType["batch_size", "n_latent"]: The encoded latent representation.
        """
        return self.encoder(x)

    def decode(self, z: TT["batch_size", "n_latent"]) -> TT["batch_size", "n_features"]:
        """Decodes the latent representation back to the input space.

        Args:
            z (TensorType["batch_size", "n_latent"]): The latent representation.

        Returns:
            TensorType["batch_size", "n_features"]: The reconstructed input tensor.
        """
        return self.decoder(z)
