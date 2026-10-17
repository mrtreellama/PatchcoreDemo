import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import * as ort from 'onnxruntime-web';
import { Engine, toChw } from '../../src/worker/engine';
import { knnMinDist, rowSqNorms } from '../../src/patchcore/knn';
import type { ModelManifest } from '../../src/worker/protocol';
import { loadF16, loadF32, loadU8, meta } from './fixtures';

const MODELS = join(__dirname, '..', '..', 'public', 'models');
const manifest: ModelManifest = JSON.parse(readFileSync(join(MODELS, 'models.json'), 'utf8'));
const entry = manifest.models.find((m) => m.key === meta.model.key)!;
const bytes = (f: string) => new Uint8Array(readFileSync(join(MODELS, f)));

describe('ONNX Runtime Web (WASM) parity', () => {
  const engine = new Engine(ort);
  beforeAll(async () => {
    await engine.init('wasm');
    const r = await engine.loadModel(entry, bytes(entry.file), bytes(manifest.knn.file));
    expect(r.shaOk).toBe(true);
  });

  for (const tag of ['teach0', 'foreign']) {
    it(`backbone features match the Python fixture (${tag})`, async () => {
      const chw = toChw(loadU8(`parity_${tag}.u8`), entry.inputSize, 3);
      const got = await engine.features(chw);
      const ref = loadF32(`parity_${tag}.f32`);
      expect(got.length).toBe(ref.length);
      let worst = 0;
      for (let i = 0; i < ref.length; i++) worst = Math.max(worst, Math.abs(got[i] - ref[i]));
      expect(worst).toBeLessThan(1e-3);
    });
  }

  it('knn.onnx (chunked) matches the TypeScript reference', async () => {
    const D = entry.D;
    const src = loadF16('teach_bank_src.f16');
    // 5,000 rows forces two chunks of the 4,096-row default.
    const rows = 5000;
    const bank = src.slice(0, Math.min(rows, src.length / D) * D);
    const sq = rowSqNorms(bank, D);
    const f = loadF16('test_foreign_patch_301.f16');
    const n = f.length / D;
    const ref = knnMinDist(f, n, bank, sq, D);
    const got = await engine.minDist(f, bank, sq);
    let worst = 0;
    for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(got[i] - ref[i]));
    expect(worst).toBeLessThan(1e-3);
  });
});
