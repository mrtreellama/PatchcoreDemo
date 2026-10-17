import { describe, expect, it } from 'vitest';
import { coresetTarget, gatherRows, greedyCoreset } from '../../src/patchcore/coreset';
import { knnMinDist, rowSqNorms } from '../../src/patchcore/knn';
import { maxOf, median, scoreMap } from '../../src/patchcore/heatmap';
import { holdoutCount, thresholdFrom } from '../../src/patchcore/threshold';
import { mulberry32 } from '../../src/patchcore/prng';
import { decodeF16, encodeF16 } from '../../src/patchcore/f16';
import { loadF16, loadI32, meta } from './fixtures';

const { D, grid, inputSize } = meta.model;
const m = meta.math;

describe('prng', () => {
  it('matches the Python Mulberry32 stream', () => {
    // First values of Mulberry32(0) as produced by tools/common.py.
    const rng = mulberry32(0);
    const v = [rng(), rng(), rng()];
    expect(v[0]).toBeCloseTo(0.26642920868471265, 15);
    expect(v[1]).toBeCloseTo(0.0003297457005828619, 15);
    expect(v[2]).toBeCloseTo(0.2232720274478197, 15);
  });
});

describe('teach math vs Python fixtures', () => {
  const teach = loadF16('teach_bank_src.f16');
  const n = teach.length / D;
  const idx = greedyCoreset(teach, n, D, coresetTarget(n), m.coresetSeed);
  const bank = gatherRows(teach, D, idx);
  const sq = rowSqNorms(bank, D);

  it('holdout and coreset sizing', () => {
    expect(holdoutCount(7)).toBe(2);
    expect(holdoutCount(15)).toBe(3);
    expect(holdoutCount(50)).toBe(10);
    expect(n).toBe(m.teachPatches);
    expect(coresetTarget(n)).toBe(m.coresetK);
    expect(coresetTarget(200_000)).toBe(10_000);
    expect(coresetTarget(500)).toBe(500);
  });

  it('greedy coreset picks the same indices', () => {
    const ref = loadI32('coreset_idx.i32');
    expect(idx.length).toBe(ref.length);
    let same = 0;
    for (let i = 0; i < ref.length; i++) if (idx[i] === ref[i]) same++;
    expect(same).toBe(ref.length);
  });

  it('kNN distances, score maps and image scores match', () => {
    for (const t of m.tests) {
      const f = loadF16(`test_${t.name}.f16`);
      const d = knnMinDist(f, grid * grid, bank, sq, D);
      let worst = 0;
      for (let i = 0; i < d.length; i++) worst = Math.max(worst, Math.abs(d[i] - t.minDist[i]));
      expect(worst, t.name).toBeLessThan(1e-3);
      const smap = scoreMap(d, grid, inputSize);
      expect(maxOf(smap)).toBeCloseTo(t.imageScore, 3);
      const samplePts = [[0, 0], [17, 200], [112, 112], [223, 5], [223, 223]];
      samplePts.forEach(([y, x], k) => expect(smap[y * inputSize + x]).toBeCloseTo(t.smoothedSample[k], 3));
    }
  });

  it('threshold and heat floor from held-out frames', () => {
    const held = loadF16('heldout.f16');
    const per = grid * grid;
    const scores: number[] = [];
    const patches: number[] = [];
    for (let k = 0; k < held.length / (per * D); k++) {
      const d = knnMinDist(held.subarray(k * per * D, (k + 1) * per * D), per, bank, sq, D);
      scores.push(maxOf(scoreMap(d, grid, inputSize)));
      patches.push(...d);
    }
    scores.forEach((s, i) => expect(s).toBeCloseTo(m.heldScores[i], 3));
    expect(thresholdFrom(Math.max(...scores), m.margin)).toBeCloseTo(m.threshold, 3);
    expect(median(patches)).toBeCloseTo(m.heatFloor, 3);
  });
});

describe('float16', () => {
  it('round-trips representable values and rounds others', () => {
    const src = new Float32Array([0, 1, -2.5, 65504, 1e-8, 0.1, 3.14159]);
    const back = decodeF16(encodeF16(src));
    expect(Array.from(back.slice(0, 4))).toEqual([0, 1, -2.5, 65504]);
    expect(back[5]).toBeCloseTo(0.1, 3);
    expect(back[6]).toBeCloseTo(3.14159, 2);
  });
});
