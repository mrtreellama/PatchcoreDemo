import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { decodeF16 } from '../../src/patchcore/f16';

export const FIX = join(__dirname, '..', 'fixtures');

export interface MathTest {
  name: string;
  minDist: number[];
  imageScore: number;
  argmax: number;
  smoothedSample: number[];
}

export interface Meta {
  model: { key: string; file: string; inputSize: number; grid: number; D: number; sha256: string };
  roi: { x: number; y: number; w: number; h: number };
  teach: string[];
  tests: { file: string; name: string; kind: string; imageScore: number; defect: boolean }[];
  math: {
    teachPatches: number;
    heldoutPatches: number;
    coresetSeed: number;
    coresetK: number;
    margin: number;
    heldScores: number[];
    threshold: number;
    heatFloor: number;
    tests: MathTest[];
  };
  fullPrecision: { threshold: number; heldScores: number[]; heatFloor: number };
}

export const meta: Meta = JSON.parse(readFileSync(join(FIX, 'meta.json'), 'utf8'));

function buf(name: string): ArrayBuffer {
  const b = readFileSync(join(FIX, name));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

export const loadF16 = (name: string) => decodeF16(new Uint16Array(buf(name)));
export const loadF32 = (name: string) => new Float32Array(buf(name));
export const loadI32 = (name: string) => new Int32Array(buf(name));
export const loadU8 = (name: string) => new Uint8Array(buf(name));
