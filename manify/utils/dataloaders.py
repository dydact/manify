"""Dataloaders submodule.


| Dataset     | Type      | Task          | Distance Matrix | Features | Classification Labels | Regression Labels | Adjacency Matrix | Source/Citation |
|-------------|-----------|---------------|-----------------|----------|----------------------|------------------|------------------|-----------------|
| cities      | distance  | none          | ✅              | ❌       | ❌                   | ❌                | ❌               | [Citation]      |
| cs_phds     | distance  | regression    | ✅              | ❌       | ❌                   | ✅                | ✅               | [Citation]      |
| polblogs    | distance  | classification| ✅              | ❌       | ✅                   | ❌                | ✅               | [Citation]      |
| polbooks    | distance  | classification| ✅              | ❌       | ✅                   | ❌                | ✅               | [Citation]      |
| cora        | distance  | classification| ✅              | ❌       | ✅                   | ❌                | ✅               | [Citation]      |
| citeseer    | distance  | classification| ✅              | ❌       | ✅                   | ❌                | ✅               | [Citation]      |
| karate_club | distance  | none          | ✅              | ❌       | ❌                   | ❌                | ✅               | [Citation]      |
| lesmis      | distance  | none          | ✅              | ❌       | ❌                   | ❌                | ✅               | [Citation]      |
| adjnoun     | distance  | none          | ✅              | ❌       | ❌                   | ❌                | ✅               | [Citation]      |
| football    | distance  | none          | ✅              | ❌       | ❌                   | ❌                | ✅               | [Citation]      |
| dolphins    | distance  | none          | ✅              | ❌       | ❌                   | ❌                | ✅               | [Citation]      |
| blood_cells | feature   | classification| ❌              | ✅       | ✅                   | ❌                | ❌               | [Citation]      |
| lymphoma    | feature   | classification| ❌              | ✅       | ✅                   | ❌                | ❌               | [Citation]      |
| cifar_100   | feature   | classification| ❌              | ✅       | ✅                   | ❌                | ❌               | [Citation]      |
| mnist       | feature   | classification| ❌              | ✅       | ✅                   | ❌                | ❌               | [Citation]      |
| temperature | feature   | regression    | ❌              | ✅       | ❌                   | ✅                | ❌               | [Citation]      |
| landmasses  | feature   | classification| ❌              | ✅       | ✅                   | ❌                | ❌               | [Citation]      |
| neuron_33   | feature   | classification| ❌              | ✅       | ✅                   | ❌                | ❌               | [Citation]      |
| neuron_46   | feature   | classification| ❌              | ✅       | ✅                   | ❌                | ❌               | [Citation]      |
| traffic     | feature   | regression    | ❌              | ✅       | ❌                   | ✅                | ❌               | [Citation]      |

## Dataset Characteristics Legend:
- Type: Primary data representation (distance or feature matrices)
- Task: Machine learning task associated with the dataset (classification, regression, or none)
- Distance Matrix: ✅ = Present, ❌ = Not present
- Features: ✅ = Present, ❌ = Not present
- Classification Labels: ✅ = Present, ❌ = Not present
- Regression Labels: ✅ = Present, ❌ = Not present
- Adjacency Matrix: ✅ = Present, ❌ = Not present
- Source/Citation: Reference to the dataset's original source
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from datasets import load_dataset
from jaxtyping import Float


def load_hf(name: str, namespace: str = "manify") -> Tuple[
    Optional[Float[torch.Tensor, "n_points n_features"]],  # features
    Optional[Float[torch.Tensor, "n_points n_points"]],  # pairwise dists
    Optional[Float[torch.Tensor, "n_points n_points"]],  # adjacency labels
    Optional[Float[torch.Tensor, "n_points,"]],  # labels
]:
    """
    Load a dataset from HuggingFace Hub at {namespace}/{name}.
    Returns:
      - if distance‑based:  (dists, labels, adj)
      - if feature‑based:   (feats, labels, adj)
    Each is a torch.Tensor (or None if absent).
    """
    # 1) fetch the single‑row dataset
    ds = load_dataset(f"{namespace}/{name}")
    data = ds["train"] if "train" in ds else ds
    row = data[0]

    # 2) helper to turn lists → torch (or None)
    def to_tensor(key, dtype):
        vals = row.get(key, [])
        if not vals:
            return None
        return torch.tensor(vals, dtype=dtype)

    # 3) reconstruct everything
    dists = to_tensor("distances", torch.float32)
    feats = to_tensor("features", torch.float32)
    adj = to_tensor("adjacency", torch.float32)

    cls_ls = row.get("classification_labels", [])
    reg_ls = row.get("regression_labels", [])
    if cls_ls:
        labels = torch.tensor(cls_ls, dtype=torch.int64)
    elif reg_ls:
        labels = torch.tensor(reg_ls, dtype=torch.float32)
    else:
        labels = None

    return feats, dists, adj, labels
