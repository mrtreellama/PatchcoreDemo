"""Shared PatchCore math for the Python reference.

Every function here has a TypeScript twin in src/patchcore/ and must stay
bit-for-bit (or tolerance-for-tolerance) compatible with it:

- mulberry32 / rademacher_projection  -> src/patchcore/prng.ts
- greedy_coreset                       -> src/patchcore/coreset.ts
- knn_min_dist                         -> src/patchcore/knn.ts
- upsample_bilinear / gaussian_blur    -> src/patchcore/heatmap.ts
"""
from __future__ import annotations

import math

import numpy as np

CORESET_PROJ_DIM = 128
CORESET_SEED = 0
GAUSSIAN_SIGMA = 4.0
GAUSSIAN_TRUNCATE = 4.0
DEFAULT_MARGIN = 1.15


def _imul(a: int, b: int) -> int:
    return (a * b) & 0xFFFFFFFF


class Mulberry32:
    """Same generator as the JS one-liner, so seeded streams match exactly."""

    def __init__(self, seed: int):
        self.a = seed & 0xFFFFFFFF

    def next(self) -> float:
        self.a = (self.a + 0x6D2B79F5) & 0xFFFFFFFF
        a = self.a
        t = _imul(a ^ (a >> 15), 1 | a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0


def rademacher_projection(rng: Mulberry32, d_in: int, d_out: int) -> np.ndarray:
    """[d_in, d_out] matrix of +-1, drawn row-major. Exact in both languages."""
    r = np.empty((d_in, d_out), dtype=np.float64)
    for i in range(d_in):
        for j in range(d_out):
            r[i, j] = -1.0 if rng.next() < 0.5 else 1.0
    return r


def coreset_target(total: int) -> int:
    """10% of teach patches, clamped to 1,000-10,000 rows (never more than we have)."""
    return min(total, max(1000, min(10000, math.floor(total * 0.1 + 0.5))))


def greedy_coreset(features: np.ndarray, k: int, seed: int = CORESET_SEED) -> np.ndarray:
    """Greedy k-center selection on a seeded 128-d Rademacher projection."""
    n, d = features.shape
    rng = Mulberry32(seed)
    proj = rademacher_projection(rng, d, CORESET_PROJ_DIM)
    p = features.astype(np.float64) @ proj
    start = int(math.floor(rng.next() * n))
    selected = [start]
    min_d = ((p - p[start]) ** 2).sum(1)
    for _ in range(1, k):
        j = int(np.argmax(min_d))
        selected.append(j)
        min_d = np.minimum(min_d, ((p - p[j]) ** 2).sum(1))
    return np.asarray(selected, dtype=np.int32)


def knn_min_dist(features: np.ndarray, bank: np.ndarray, bank_sqnorm: np.ndarray) -> np.ndarray:
    f = features.astype(np.float64)
    b = bank.astype(np.float64)
    d2 = (f * f).sum(1)[:, None] - 2.0 * (f @ b.T) + bank_sqnorm.astype(np.float64)[None, :]
    return np.sqrt(np.maximum(d2.min(1), 0.0))


def _bilinear_axis(src_len: int, dst_len: int):
    o = np.arange(dst_len, dtype=np.float64)
    s = (o + 0.5) * (src_len / dst_len) - 0.5
    s = np.maximum(s, 0.0)
    i0 = np.floor(s).astype(np.int64)
    i0 = np.minimum(i0, src_len - 1)
    i1 = np.minimum(i0 + 1, src_len - 1)
    w = s - i0
    return i0, i1, w


def upsample_bilinear(grid: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """align_corners=False bilinear, same as torch.nn.functional.interpolate."""
    h, w = grid.shape
    y0, y1, wy = _bilinear_axis(h, out_h)
    x0, x1, wx = _bilinear_axis(w, out_w)
    g = grid.astype(np.float64)
    rows = g[y0] * (1 - wy)[:, None] + g[y1] * wy[:, None]
    return rows[:, x0] * (1 - wx)[None, :] + rows[:, x1] * wx[None, :]


def gaussian_kernel(sigma: float = GAUSSIAN_SIGMA, truncate: float = GAUSSIAN_TRUNCATE) -> np.ndarray:
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _reflect_index(i: np.ndarray, n: int) -> np.ndarray:
    # scipy 'reflect' (half-sample symmetric): d c b a | a b c d | d c b a
    period = 2 * n
    i = np.mod(i, period)
    return np.where(i >= n, period - 1 - i, i)


def gaussian_blur(img: np.ndarray, sigma: float = GAUSSIAN_SIGMA) -> np.ndarray:
    k = gaussian_kernel(sigma)
    r = len(k) // 2
    h, w = img.shape
    out = np.zeros((h, w), dtype=np.float64)
    for t, kv in enumerate(k):
        out += kv * img[_reflect_index(np.arange(h) + t - r, h), :]
    out2 = np.zeros((h, w), dtype=np.float64)
    for t, kv in enumerate(k):
        out2 += kv * out[:, _reflect_index(np.arange(w) + t - r, w)]
    return out2


def score_map(min_dist: np.ndarray, grid: int, size: int) -> np.ndarray:
    """Patch distances -> smoothed anomaly map at backbone input resolution."""
    return gaussian_blur(upsample_bilinear(min_dist.reshape(grid, grid), size, size))


def holdout_count(frames: int) -> int:
    return max(2, math.floor(frames * 0.2 + 0.5))
