"""Methods for estimating sectional curvature of graphs.

This module provides functions to estimate the sectional curvature of graphs based on the approach from:
Gu et al. "Learning mixed-curvature representations in product spaces." ICLR 2019.

The implementation builds upon code from the HazyResearch/hyperbolics repository:
https://github.com/HazyResearch/hyperbolics
"""

from __future__ import annotations

import random

import networkx as nx
import numpy as np


def sectional_curvature(G: nx.Graph) -> np.ndarray:
    r"""Estimates the sectional curvature of a graph.

    This function implements the graph sectional curvature estimation described in Gu et al. 2019.
    For a graph $\mathcal{G}$ and vertex $v$, the sectional curvature at $v$ is computed based on the geometry
    of triangles formed with its neighbors.

    Args:
        G: NetworkX graph to analyze.

    Returns:
        curvature_estimates: Array of curvature estimates for each node in the graph.

    Note:
        This function is not yet implemented.
    """
    raise NotImplementedError


# def Ka(D: np.ndarray, m: int, b: int, c: int, a: int) -> float:
#     if a == m:
#         return 0.0
#     k = D[a][m] ** 2 + D[b][c] ** 2 / 4.0 - (D[a][b] ** 2 + D[a][c] ** 2) / 2.0
#     k /= 2 * D[a][m]
#     return float(k)


# def K(D: np.ndarray, n: int, m: int, b: int, c: int) -> float:
#     ks = [Ka(D, m, b, c, a) for a in range(n)]
#     return float(np.mean(ks))


# def ref(D: np.ndarray, size: int, n: int, m: int, b: int, c: int) -> float:
#     ks = []
#     for i in range(n):
#         a = random.randint(0, size - 1)
#         if a == b or a == c:
#             continue
#         else:
#             ks.append(Ka(D, m, b, c, a))
#     return float(np.mean(ks))


# def estimate_curvature(G: nx.Graph, D: np.ndarray, n: int) -> None:
#     for m in range(n):
#         ks = []
#         edges = list(G.edges(m))
#         for i in range(len(edges)):
#             for j in range(i + 1, len(edges)):
#                 b = edges[i]
#                 c = edges[j]
#                 ks.append(K(D, n, m, b, c))
#     return None


# def sample(D: np.ndarray, size: int, n_samples: int = 100) -> np.ndarray:
#     samples = []
#     _cnt = 0
#     while _cnt < n_samples:
#         a, b, c, m = random.sample(range(0, size), 4)
#         k = Ka(D, m, b, c, a)
#         samples.append(k)

#         _cnt += 1

#     return np.array(samples)


# def estimate(D: np.ndarray, size: int, n_samples: int) -> np.ndarray:
#     samples = sample(D, size, n_samples)
#     m1 = np.mean(samples)
#     m2 = np.mean(samples**2)
#     return samples


# def estimate_diff(D: np.ndarray, size: int, n_sample: int, num: int) -> np.ndarray:
#     samples = []
#     _cnt = 0
#     while _cnt < n_sample:
#         b, c, m = random.sample(range(0, size), 3)
#         k = ref(D, size, num, m, b, c)
#         # k=K(D, n, m, b, c)
#         samples.append(k)
#         _cnt += 1
#     return np.array(samples)
