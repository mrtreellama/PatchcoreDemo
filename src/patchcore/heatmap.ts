/** Heatmap post-processing; mirrors tools/common.py (upsample, blur, score_map). */

export const GAUSSIAN_SIGMA = 4;
const GAUSSIAN_TRUNCATE = 4;

function bilinearAxis(srcLen: number, dstLen: number) {
  const i0 = new Int32Array(dstLen);
  const i1 = new Int32Array(dstLen);
  const w = new Float64Array(dstLen);
  for (let o = 0; o < dstLen; o++) {
    const s = Math.max((o + 0.5) * (srcLen / dstLen) - 0.5, 0);
    const a = Math.min(Math.floor(s), srcLen - 1);
    i0[o] = a;
    i1[o] = Math.min(a + 1, srcLen - 1);
    w[o] = s - a;
  }
  return { i0, i1, w };
}

/** align_corners=false bilinear upsample of an [h, w] grid to [outH, outW]. */
export function upsampleBilinear(grid: ArrayLike<number>, h: number, w: number, outH: number, outW: number): Float64Array {
  const ya = bilinearAxis(h, outH);
  const xa = bilinearAxis(w, outW);
  const rows = new Float64Array(outH * w);
  for (let y = 0; y < outH; y++) {
    const a = ya.i0[y] * w;
    const b = ya.i1[y] * w;
    const t = ya.w[y];
    for (let x = 0; x < w; x++) rows[y * w + x] = grid[a + x] * (1 - t) + grid[b + x] * t;
  }
  const out = new Float64Array(outH * outW);
  for (let y = 0; y < outH; y++) {
    const ro = y * w;
    for (let x = 0; x < outW; x++) {
      const t = xa.w[x];
      out[y * outW + x] = rows[ro + xa.i0[x]] * (1 - t) + rows[ro + xa.i1[x]] * t;
    }
  }
  return out;
}

export function gaussianKernel(sigma = GAUSSIAN_SIGMA): Float64Array {
  const radius = Math.floor(GAUSSIAN_TRUNCATE * sigma + 0.5);
  const k = new Float64Array(2 * radius + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-0.5 * (i / sigma) ** 2);
    k[i + radius] = v;
    sum += v;
  }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  return k;
}

/** scipy 'reflect' boundary: d c b a | a b c d | d c b a */
function reflect(i: number, n: number): number {
  const period = 2 * n;
  i = ((i % period) + period) % period;
  return i >= n ? period - 1 - i : i;
}

/** Separable Gaussian blur (rows first, then columns), float64. */
export function gaussianBlur(img: Float64Array, h: number, w: number, sigma = GAUSSIAN_SIGMA): Float64Array {
  const k = gaussianKernel(sigma);
  const r = (k.length - 1) / 2;
  const tmp = new Float64Array(h * w);
  // Python blurs axis 0 (y) first, then axis 1 (x); keep the same order.
  const yIdx: Int32Array[] = [];
  for (let t = 0; t < k.length; t++) {
    const idx = new Int32Array(h);
    for (let y = 0; y < h; y++) idx[y] = reflect(y + t - r, h);
    yIdx.push(idx);
  }
  for (let t = 0; t < k.length; t++) {
    const kv = k[t];
    const idx = yIdx[t];
    for (let y = 0; y < h; y++) {
      const so = idx[y] * w;
      const o = y * w;
      for (let x = 0; x < w; x++) tmp[o + x] += kv * img[so + x];
    }
  }
  const out = new Float64Array(h * w);
  for (let t = 0; t < k.length; t++) {
    const kv = k[t];
    const idx = new Int32Array(w);
    for (let x = 0; x < w; x++) idx[x] = reflect(x + t - r, w);
    for (let y = 0; y < h; y++) {
      const o = y * w;
      for (let x = 0; x < w; x++) out[o + x] += kv * tmp[o + idx[x]];
    }
  }
  return out;
}

/** Patch distances [grid*grid] -> smoothed anomaly map at backbone input size. */
export function scoreMap(minDist: ArrayLike<number>, grid: number, size: number): Float64Array {
  return gaussianBlur(upsampleBilinear(minDist, grid, grid, size, size), size, size);
}

export function maxOf(a: ArrayLike<number>): number {
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
  return m;
}

export function median(values: ArrayLike<number>): number {
  const s = Float64Array.from(values).sort();
  const n = s.length;
  if (n === 0) return NaN;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

// Turbo colormap (Google, Apache-2.0) polynomial approximation.
export function turbo(t: number): [number, number, number] {
  t = Math.min(1, Math.max(0, t));
  const r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  const b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  const c = (v: number) => Math.round(255 * Math.min(1, Math.max(0, v)));
  return [c(r), c(g), c(b)];
}

const TURBO_LUT = (() => {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) lut.set(turbo(i / 255), i * 3);
  return lut;
})();

/**
 * Colour-map a smoothed score map into RGBA: transparent below `floor`,
 * Turbo from floor to 1.5 x threshold, white contour pixels on the boundary
 * of regions above threshold.
 */
export function renderOverlay(
  map: ArrayLike<number>,
  size: number,
  floor: number,
  threshold: number,
  out: Uint8ClampedArray,
): void {
  const top = Math.max(1.5 * threshold, floor + 1e-6);
  const span = top - floor;
  for (let i = 0; i < size * size; i++) {
    const v = map[i];
    const o = i * 4;
    if (v <= floor) {
      out[o + 3] = 0;
      continue;
    }
    const t = Math.min(1, (v - floor) / span);
    const li = Math.round(t * 255) * 3;
    out[o] = TURBO_LUT[li];
    out[o + 1] = TURBO_LUT[li + 1];
    out[o + 2] = TURBO_LUT[li + 2];
    out[o + 3] = Math.round(90 + 165 * t);
  }
  // Contours: above-threshold pixels with a 4-neighbour below threshold.
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x;
      if (map[i] <= threshold) continue;
      const edge =
        x === 0 || y === 0 || x === size - 1 || y === size - 1 ||
        map[i - 1] <= threshold || map[i + 1] <= threshold ||
        map[i - size] <= threshold || map[i + size] <= threshold;
      if (edge) {
        const o = i * 4;
        out[o] = out[o + 1] = out[o + 2] = 255;
        out[o + 3] = 255;
      }
    }
  }
}
