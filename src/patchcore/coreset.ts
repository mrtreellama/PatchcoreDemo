import { mulberry32, rademacherProjection } from './prng';

export const CORESET_PROJ_DIM = 128;
export const CORESET_SEED = 0;

/** 10% of teach patches, clamped to 1,000-10,000 rows (never more than we have). */
export function coresetTarget(total: number): number {
  return Math.min(total, Math.max(1000, Math.min(10000, Math.round(total * 0.1))));
}

/**
 * Greedy k-center coreset (PatchCore). Distances are computed on a seeded
 * 128-d Rademacher projection for speed; callers keep the selected rows at full D.
 * Mirrors tools/common.py:greedy_coreset exactly.
 */
export function greedyCoreset(
  features: Float32Array,
  n: number,
  d: number,
  k: number,
  seed = CORESET_SEED,
  onProgress?: (done: number, total: number) => void,
): Int32Array {
  const P = CORESET_PROJ_DIM;
  const rng = mulberry32(seed);
  const proj = rademacherProjection(rng, d, P);

  // Project in float64: p[i] = features[i] @ proj
  const p = new Float64Array(n * P);
  const acc = new Float64Array(P);
  for (let i = 0; i < n; i++) {
    acc.fill(0);
    const fo = i * d;
    for (let c = 0; c < d; c++) {
      const v = features[fo + c];
      if (v === 0) continue;
      const ro = c * P;
      for (let j = 0; j < P; j++) acc[j] += v * proj[ro + j];
    }
    p.set(acc, i * P);
  }

  const start = Math.floor(rng() * n);
  const selected = new Int32Array(k);
  selected[0] = start;
  const minD = new Float64Array(n);
  updateMin(p, n, P, start, minD, true);
  const step = Math.max(1, Math.floor(k / 50));
  for (let s = 1; s < k; s++) {
    let best = 0;
    let bestV = -1;
    for (let i = 0; i < n; i++) {
      if (minD[i] > bestV) {
        bestV = minD[i];
        best = i;
      }
    }
    selected[s] = best;
    updateMin(p, n, P, best, minD, false);
    if (onProgress && s % step === 0) onProgress(s, k);
  }
  onProgress?.(k, k);
  return selected;
}

function updateMin(p: Float64Array, n: number, P: number, j: number, minD: Float64Array, init: boolean) {
  const jo = j * P;
  for (let i = 0; i < n; i++) {
    const io = i * P;
    let s = 0;
    for (let c = 0; c < P; c++) {
      const diff = p[io + c] - p[jo + c];
      s += diff * diff;
    }
    if (init || s < minD[i]) minD[i] = s;
  }
}

/** Gather selected rows into a new [k, d] bank. */
export function gatherRows(features: Float32Array, d: number, idx: Int32Array): Float32Array {
  const out = new Float32Array(idx.length * d);
  for (let r = 0; r < idx.length; r++) out.set(features.subarray(idx[r] * d, idx[r] * d + d), r * d);
  return out;
}
