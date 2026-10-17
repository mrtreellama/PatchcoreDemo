/** Seeded PRNG shared bit-for-bit with tools/common.py (Mulberry32). */
export function mulberry32(seed: number): () => number {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = ((t + Math.imul(t ^ (t >>> 7), 61 | t)) | 0) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** [dIn, dOut] matrix of +-1 drawn row-major; exact in both languages. */
export function rademacherProjection(rng: () => number, dIn: number, dOut: number): Float64Array {
  const r = new Float64Array(dIn * dOut);
  for (let i = 0; i < r.length; i++) r[i] = rng() < 0.5 ? -1 : 1;
  return r;
}
