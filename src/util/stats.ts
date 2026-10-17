export function percentile(values: number[], p: number): number {
  if (!values.length) return NaN;
  const s = [...values].sort((a, b) => a - b);
  const idx = Math.min(s.length - 1, Math.max(0, Math.ceil((p / 100) * s.length) - 1));
  return s[idx];
}

export const medianOf = (v: number[]) => {
  if (!v.length) return NaN;
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** Exponential moving average of frames per second. */
export class FpsMeter {
  private last = 0;
  fps = 0;
  tick(t = performance.now()): number {
    if (this.last) {
      const inst = 1000 / Math.max(1, t - this.last);
      this.fps = this.fps ? this.fps * 0.85 + inst * 0.15 : inst;
    }
    this.last = t;
    return this.fps;
  }
  reset() {
    this.last = 0;
    this.fps = 0;
  }
}

export const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
export const nextFrame = () => new Promise<void>((r) => requestAnimationFrame(() => r()));
