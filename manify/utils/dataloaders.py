"""Dataloaders submodule
======================

The dataloaders module allows users to load datasets from Manify's datasets repo `on Hugging Face
<https://huggingface.co/manify>`_.
We provide a summary of the data types available, and their original sources, here.

Earlier versions of Manify included scripts to process raw data, which we have replaced with a single, centralized
Hugging Face repo and the function `load_hf`. For transparency, we have preserved the data generation code in
`the Dataset-Generation branch of Manify <https://github.com/pchlenski/manify/tree/Dataset-Generation>`.

.. list-table::
   :header-rows: 1
   :widths: 12 10 15 10 8 15 15 15 20

   * - Dataset
     - Type
     - Task
     - Distance Matrix
     - Features
     - Classification Labels
     - Regression Labels
     - Adjacency Matrix
     - Source/Citation
   * - cities
     - distance
     - none
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - `Network Repository: Cities <https://networkrepository.com/Cities.php>`_
   * - cs_phds
     - distance
     - regression
     - ✅
     - ❌
     - ❌
     - ✅
     - ✅
     - `Network Repository: CS PhDs <https://networkrepository.com/CSphd.php>`_
   * - polblogs
     - distance
     - classification
     - ✅
     - ❌
     - ✅
     - ❌
     - ✅
     - `Network Repository: Polblogs <https://networkrepository.com/polblogs.php>`_
   * - polbooks
     - distance
     - classification
     - ✅
     - ❌
     - ✅
     - ❌
     - ✅
     - `Network Repository: Polbooks <https://networkrepository.com/polbooks.php>`_
   * - cora
     - distance
     - classification
     - ✅
     - ❌
     - ✅
     - ❌
     - ✅
     - `Network Repository: Cora <https://networkrepository.com/cora.php>`_
   * - citeseer
     - distance
     - classification
     - ✅
     - ❌
     - ✅
     - ❌
     - ✅
     - `Network Repository: Citeseer <https://networkrepository.com/citeseer.php>`_
   * - karate_club
     - distance
     - none
     - ✅
     - ❌
     - ❌
     - ❌
     - ✅
     - `Network Repository: Karate <https://networkrepository.com/karate.php>`_
   * - lesmis
     - distance
     - none
     - ✅
     - ❌
     - ❌
     - ❌
     - ✅
     - `Network Repository: Lesmis <https://networkrepository.com/lesmis.php>`_
   * - adjnoun
     - distance
     - none
     - ✅
     - ❌
     - ❌
     - ❌
     - ✅
     - `Network Repository: Adjnoun <https://networkrepository.com/adjnoun.php>`_
   * - football
     - distance
     - none
     - ✅
     - ❌
     - ❌
     - ❌
     - ✅
     - `Network Repository: Football <https://networkrepository.com/football.php>`_
   * - dolphins
     - distance
     - none
     - ✅
     - ❌
     - ❌
     - ❌
     - ✅
     - `Network Repository: Dolphins <https://networkrepository.com/dolphins.php>`_
   * - blood_cells
     - feature
     - classification
     - ❌
     - ✅
     - ✅
     - ❌
     - ❌
     - See datasets from Zheng et al (2017): Massively parallel digital transcriptional profiling of single cells.
       - `CD8+ Cytotoxic T-cells <https://www.10xgenomics.com/datasets/cd-8-plus-cytotoxic-t-cells-1-standard-1-1-0>`_
       - `CD8+/CD45RA+ Naive Cytotoxic T Cells <https://www.10xgenomics.com/datasets/cd-8-plus-cd-45-r-aplus-naive-cytotoxic-t-cells-1-standard-1-1-0>`_
       - `CD56+ Natural Killer Cells <https://www.10xgenomics.com/datasets/cd-56-plus-natural-killer-cells-1-standard-1-1-0>`_
       - `CD4+ Helper T Cells <https://www.10xgenomics.com/datasets/cd-4-plus-helper-t-cells-1-standard-1-1-0>`_
       - `CD4+/CD45RO+ Memory T Cells <https://www.10xgenomics.com/datasets/cd-4-plus-cd-45-r-oplus-memory-t-cells-1-standard-1-1-0>`_
       - `CD4+/CD45RA+/CD25- Naive T Cells <https://www.10xgenomics.com/datasets/cd-4-plus-cd-45-r-aplus-cd-25-naive-t-cells-1-standard-1-1-0>`_
       - `CD4+/CD25+ Regulatory T Cells <https://www.10xgenomics.com/datasets/cd-4-plus-cd-25-plus-regulatory-t-cells-1-standard-1-1-0>`_
       - `CD34+ Cells <https://www.10xgenomics.com/datasets/cd-34-plus-cells-1-standard-1-1-0>`_
       - `CD19+ B Cells <https://www.10xgenomics.com/datasets/cd-19-plus-b-cells-1-standard-1-1-0>`_
       - `CD14+ Monocytes <https://www.10xgenomics.com/datasets/cd-14-plus-monocytes-1-standard-1-1-0>`_
   * - lymphoma
     - feature
     - classification
     - ❌
     - ✅
     - ✅
     - ❌
     - ❌
    - See datasets from 10x Genomics:
        - `Hodgkin's Lymphoma <https://www.10xgenomics.com/datasets/hodgkins-lymphoma-dissociated-tumor-targeted-immunology-panel-3-1-standard-4-0-0>`_
        - `Healthy Donor PBMCs <https://www.10xgenomics.com/datasets/pbm-cs-from-a-healthy-donor-targeted-compare-immunology-panel-3-1-standard-4-0-0>`_
    * - cifar_100
     - feature
     - classification
     - ❌
     - ✅
     - ✅
     - ❌
     - ❌
     - `Hugging Face Datasets: CIFAR-100 <https://huggingface.co/datasets/uoft-cs/cifar100>`_
   * - mnist
     - feature
     - classification
     - ❌
     - ✅
     - ✅
     - ❌
     - ❌
     - `Hugging Face Datasets: MNIST <https://huggingface.co/datasets/ylecun/mnist>`_
   * - temperature
     - feature
     - regression
     - ❌
     - ✅
     - ❌
     - ✅
     - ❌
     - [Citation]
   * - landmasses
     - feature
     - classification
     - ❌
     - ✅
     - ✅
     - ❌
     - ❌
     - Generated using `basemap.is_land <https://matplotlib.org/basemap/stable/api/basemap_api.html#mpl_toolkits.basemap.Basemap.is_land>`_
   * - neuron_33
     - feature
     - classification
     - ❌
     - ✅
     - ✅
     - ❌
     - ❌
     - `Allen Brain Atlas <https://celltypes.brain-map.org/experiment/electrophysiology/623474400>`_
   * - neuron_46
     - feature
     - classification
     - ❌
     - ✅
     - ✅
     - ❌
     - ❌
     - `Allen Brain Atlas <https://celltypes.brain-map.org/experiment/electrophysiology/623474400>`
   * - traffic
     - feature
     - regression
     - ❌
     - ✅
     - ❌
     - ✅
     - ❌
     - `Kaggle: Traffic Prediction Dataset <https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset>`_
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
    def to_tensor(key: str, dtype: torch.dtype) -> Optional[torch.Tensor]:
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
