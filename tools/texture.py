"""Procedural carbon-fibre twill texture with planted composite defects.

Everything is seeded so fixtures are reproducible. No third-party imagery.
"""
from __future__ import annotations

import numpy as np

SCENE_SIZE = 448


def _base_coords(size: int, rng: np.random.Generator):
    angle = np.deg2rad(rng.uniform(-2.0, 2.0))
    ox, oy = rng.uniform(0, 64, size=2)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    xr = np.cos(angle) * x - np.sin(angle) * y + ox
    yr = np.sin(angle) * x + np.cos(angle) * y + oy
    return xr, yr


def weave(size: int, rng: np.random.Generator, warp: np.ndarray | None = None,
          drop_tow: int | None = None) -> np.ndarray:
    """2x2 twill of ~16 px tows. Returns float RGB [size, size, 3] in 0..1."""
    tow = 16.0
    xr, yr = _base_coords(size, rng)
    if warp is not None:
        xr = xr + warp[0]
        yr = yr + warp[1]
    u, v = xr / tow, yr / tow
    i, j = np.floor(u), np.floor(v)
    fu, fv = u - i, v - j
    warp_on_top = np.mod(i - j, 4) < 2

    # Tow cross-section shading and fibre striations along the tow direction.
    stri_w = 0.04 * np.sin(2 * np.pi * (fu * 7.0 + 0.3 * np.sin(i * 1.7)))
    stri_f = 0.04 * np.sin(2 * np.pi * (fv * 7.0 + 0.3 * np.sin(j * 2.3)))
    shade_w = 0.55 + 0.45 * np.sin(np.pi * np.clip(fu, 0.02, 0.98)) + stri_w
    shade_f = 0.55 + 0.45 * np.sin(np.pi * np.clip(fv, 0.02, 0.98)) + stri_f
    # Anisotropic sheen: warp tows reflect more than weft tows under the rig light.
    lum = np.where(warp_on_top, 0.30 * shade_w, 0.20 * shade_f)

    if drop_tow is not None:
        gap = (i == drop_tow)
        # Missing warp tow: the weft below shows through, recessed and darker.
        lum = np.where(gap, 0.08 + 0.05 * shade_f, lum)

    # Gentle lighting gradient and sensor noise.
    gx, gy = rng.uniform(-0.04, 0.04, size=2)
    yy, xx = np.mgrid[0:size, 0:size] / size
    lum = lum * (1.0 + gx * (xx - 0.5) + gy * (yy - 0.5))
    lum = lum + rng.normal(0, 0.008, size=lum.shape)
    tint = np.array([0.95, 0.97, 1.05])
    return np.clip(lum[..., None] * tint[None, None, :] + 0.04, 0, 1)


def _defect_centre(rng: np.random.Generator, size: int):
    lo, hi = int(size * 0.3), int(size * 0.7)
    return rng.integers(lo, hi), rng.integers(lo, hi)


def foreign_patch(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    size = img.shape[0]
    cy, cx = _defect_centre(rng, size)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    ang = np.arctan2(y - cy, x - cx)
    r = 18 + 4 * np.sin(3 * ang + rng.uniform(0, 6)) + 3 * np.sin(5 * ang)
    inside = np.hypot(y - cy, x - cx) < r
    colour = np.array([0.85, 0.80, 0.55])  # backing paper / release film
    out = img.copy()
    out[inside] = colour + rng.normal(0, 0.02, size=(inside.sum(), 3))
    return np.clip(out, 0, 1)


def wrinkle_warp(rng: np.random.Generator, size: int):
    cy, cx = _defect_centre(rng, size)
    theta = rng.uniform(0, np.pi)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    # Signed distance from a line through the centre, limited in length.
    n = np.array([np.cos(theta), np.sin(theta)])
    d = (x - cx) * n[0] + (y - cy) * n[1]
    along = -(x - cx) * n[1] + (y - cy) * n[0]
    env = np.exp(-(along / 70.0) ** 2)
    amp = 9.0 * env * np.exp(-(d / 10.0) ** 2)
    warp = (amp * n[0], amp * n[1])
    crease = env * np.exp(-(d / 3.0) ** 2)
    return warp, crease


def scene(kind: str, seed: int, size: int = SCENE_SIZE) -> np.ndarray:
    """kind in {good, dropped_tow, foreign_patch, wrinkle}; returns uint8 RGB."""
    rng = np.random.default_rng(seed)
    if kind == "good":
        img = weave(size, rng)
    elif kind == "dropped_tow":
        # A tow index near the centre of the scene (coordinate offset is <= 64 px).
        tow = int(np.floor((size * 0.5 + 32) / 16.0)) + int(np.random.default_rng(seed + 20_000).integers(-3, 3))
        img = weave(size, rng, drop_tow=tow)
    elif kind == "foreign_patch":
        img = foreign_patch(weave(size, rng), rng)
    elif kind == "wrinkle":
        sub = np.random.default_rng(seed + 10_000)
        warp, crease = wrinkle_warp(sub, size)
        img = weave(size, rng, warp=warp)
        img = np.clip(img + 0.35 * crease[..., None], 0, 1)
    else:
        raise ValueError(kind)
    return (img * 255 + 0.5).astype(np.uint8)
