import random

import numpy as np


# The next couple functions are taken from this repo:
# https://github.com/HazyResearch/hyperbolics
# Paper: https://openreview.net/pdf?id=HJxeWnCcF7


def Ka(D, m, b, c, a):
    if a == m:
        return 0.0
    k = D[a][m] ** 2 + D[b][c] ** 2 / 4.0 - (D[a][b] ** 2 + D[a][c] ** 2) / 2.0
    k /= 2 * D[a][m]
    return k


def K(D, n, m, b, c):
    ks = [Ka(D, m, b, c, a) for a in range(n)]
    return np.mean(ks)


def ref(D, size, n, m, b, c):
    ks = []
    for i in range(n):
        a = random.randint(0, size - 1)
        if a == b or a == c:
            continue
        else:
            ks.append(Ka(D, m, b, c, a))
    return np.mean(ks)


def estimate_curvature(G, D, n):
    for m in range(n):
        ks = []
        edges = list(G.edges(m))
        for i in range(len(edges)):
            for j in range(b, len(edges)):
                b = edges[i]
                c = edges[j]
                ks.append(K(D, n, b, c))
    return None


def sample(D, size, n_samples=100):
    samples = []
    _cnt = 0
    while _cnt < n_samples:
        a, b, c, m = random.sample(range(0, size), 4)
        k = Ka(D, m, b, c, a)
        samples.append(k)

        _cnt += 1

    return np.array(samples)


def estimate(D, size, n_samples):
    samples = sample(D, size, n_samples)
    m1 = np.mean(samples)
    m2 = np.mean(samples**2)
    return samples


def estimate_diff(D, size, n_sample, num):
    samples = []
    _cnt = 0
    while _cnt < n_sample:
        b, c, m = random.sample(range(0, size), 3)
        k = ref(D, size, num, m, b, c)
        # k=K(D, n, m, b, c)
        samples.append(k)
        _cnt += 1
    return np.array(samples)
