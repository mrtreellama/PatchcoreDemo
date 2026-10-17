/** Squared L2 norm of each row, computed in float64 and stored as float32. */
export function rowSqNorms(bank: Float32Array, d: number): Float32Array {
  const m = bank.length / d;
  const out = new Float32Array(m);
  for (let r = 0; r < m; r++) {
    let s = 0;
    const o = r * d;
    for (let c = 0; c < d; c++) s += bank[o + c] * bank[o + c];
    out[r] = s;
  }
  return out;
}

/**
 * Plain TypeScript twin of knn.onnx: for each feature row, the Euclidean
 * distance to its nearest bank row, via |f|^2 - 2 f.b + |b|^2 clamped at 0.
 * Reference implementation and CPU fallback.
 */
export function knnMinDist(
  features: Float32Array,
  n: number,
  bank: Float32Array,
  bankSqNorm: Float32Array,
  d: number,
): Float32Array {
  const m = bankSqNorm.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const fo = i * d;
    let fsq = 0;
    for (let c = 0; c < d; c++) fsq += features[fo + c] * features[fo + c];
    let best = Infinity;
    for (let r = 0; r < m; r++) {
      const bo = r * d;
      let dot = 0;
      for (let c = 0; c < d; c++) dot += features[fo + c] * bank[bo + c];
      const d2 = fsq - 2 * dot + bankSqNorm[r];
      if (d2 < best) best = d2;
    }
    out[i] = Math.sqrt(Math.max(best, 0));
  }
  return out;
}

/** Fold a chunk result into a running minimum (chunked kNN over the bank). */
export function runningMin(acc: Float32Array, chunk: Float32Array): void {
  for (let i = 0; i < acc.length; i++) if (chunk[i] < acc[i]) acc[i] = chunk[i];
}
