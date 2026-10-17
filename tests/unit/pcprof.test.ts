import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { decodePcprof, encodePcprof } from '../../src/profiles/pcprof';
import { rowSqNorms } from '../../src/patchcore/knn';
import type { ProfileHeader } from '../../src/profiles/types';
import { Hysteresis } from '../../src/patchcore/hysteresis';

const header = (D: number): ProfileHeader => ({
  name: 'Test ✓ profile', notes: 'unicode é', created: '2026-10-17T00:00:00Z', appVersion: '0.1.0',
  backboneId: 'r18', modelKey: 'r18_224', inputSize: 224, modelSha256: 'abc', D, M: 0,
  bankDtype: 'float32', threshold: 1.5, margin: 1.15, heatFloor: 0.4,
  roi: { x: 0.1, y: 0.1, w: 0.8, h: 0.8 },
  camera: { label: 'cam', width: 1280, height: 720, settings: { exposureTime: 100, focusMode: 'manual' } },
  rig: 'bench', teachStats: { frames: 15, heldOut: 3, heldScores: [1, 1.2, 1.3], maxHeld: 1.3, teachPatches: 9408 },
  thumbnailBytes: 0,
});

describe('.pcprof', () => {
  const D = 6;
  const bank = Float32Array.from({ length: 5 * D }, (_, i) => Math.sin(i) * 3);
  const thumb = new Uint8Array([0xff, 0xd8, 1, 2, 3, 0xff, 0xd9]);

  it('round-trips float32 banks exactly', () => {
    const bytes = encodePcprof(header(D), bank, thumb, 'float32');
    const d = decodePcprof(bytes.buffer as ArrayBuffer);
    expect(Array.from(d.bank)).toEqual(Array.from(bank));
    expect(Array.from(d.sqnorm)).toEqual(Array.from(rowSqNorms(bank, D)));
    expect(Array.from(d.thumbnail)).toEqual(Array.from(thumb));
    expect(d.header.M).toBe(5);
    expect(d.header.name).toBe('Test ✓ profile');
    expect(d.header.camera.settings.exposureTime).toBe(100);
  });

  it('round-trips float16 banks with consistent norms', () => {
    const bytes = encodePcprof(header(D), bank, thumb, 'float16');
    const d = decodePcprof(bytes.buffer as ArrayBuffer);
    for (let i = 0; i < bank.length; i++) expect(d.bank[i]).toBeCloseTo(bank[i], 2);
    expect(Array.from(d.sqnorm)).toEqual(Array.from(rowSqNorms(d.bank, D)));
  });

  it('rejects bad magic', () => {
    expect(() => decodePcprof(new Uint8Array(16).buffer)).toThrow(/magic/);
  });

  const read = (...p: string[]) => {
    const b = readFileSync(join(__dirname, '..', ...p));
    return decodePcprof(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer);
  };

  it('reads a profile written by tools/pcprof.py', () => {
    const d = read('fixtures', 'python_written.pcprof');
    expect(d.header.name).toBe('Python fixture');
    expect(d.header.bankDtype).toBe('float32');
    expect(d.bank.length).toBe(d.header.M * d.header.D);
    expect(Array.from(d.sqnorm.slice(0, 20))).toEqual(Array.from(rowSqNorms(d.bank, d.header.D).slice(0, 20)));
    expect(d.thumbnail[0]).toBe(0xff);
    // Re-encoding yields a file Python can read the same way (same layout).
    const again = decodePcprof(encodePcprof(d.header, d.bank, d.thumbnail, 'float32').buffer as ArrayBuffer);
    expect(Array.from(again.bank.slice(0, 50))).toEqual(Array.from(d.bank.slice(0, 50)));
  });

  it('reads the bundled example profile', () => {
    const d = read('..', 'public', 'profiles', 'carbon-twill-demo.pcprof');
    expect(d.header.backboneId).toBe('r18');
    expect(d.bank.length).toBe(d.header.M * d.header.D);
    expect(d.header.threshold).toBeGreaterThan(0);
  });
});

describe('hysteresis', () => {
  it('needs 2 frames above to flag and 3 below to clear', () => {
    const h = new Hysteresis();
    const seq = [2, 0, 2, 2, 0, 2, 0, 0, 0, 2];
    const out = seq.map((s) => h.update(s, 1));
    expect(out).toEqual(['PASS', 'PASS', 'PASS', 'DEFECT', 'DEFECT', 'DEFECT', 'DEFECT', 'DEFECT', 'PASS', 'PASS']);
  });
});
