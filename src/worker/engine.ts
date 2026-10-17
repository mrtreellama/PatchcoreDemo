/**
 * PatchCore inference engine: ORT sessions, teach buffer, scoring.
 * DOM-free so it runs both in the Web Worker and under Vitest in Node.
 */
import type * as OrtModule from 'onnxruntime-web';
import { coresetTarget, gatherRows, greedyCoreset } from '../patchcore/coreset';
import { knnMinDist, rowSqNorms, runningMin } from '../patchcore/knn';
import { maxOf, median, scoreMap } from '../patchcore/heatmap';
import { holdoutCount, thresholdFrom } from '../patchcore/threshold';
import type {
  EpActual, EpChoice, InitResult, LoadModelResult, ModelEntry, ProgressEvent, ScoreResult,
  StageTimings, TeachFinishArgs, TeachResult,
} from './protocol';

type Ort = typeof OrtModule;
type Session = OrtModule.InferenceSession;

export const KNN_CHUNK_ROWS = 4096;

const now = () => performance.now();

export async function sha256Hex(bytes: ArrayBuffer | Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', bytes as BufferSource);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

/** RGBA or RGB uint8 HWC -> float32 CHW in 0..1. */
export function toChw(pixels: Uint8Array | Uint8ClampedArray, size: number, channels: 3 | 4): Float32Array {
  const plane = size * size;
  const out = new Float32Array(3 * plane);
  for (let i = 0, p = 0; i < plane; i++, p += channels) {
    out[i] = pixels[p] / 255;
    out[plane + i] = pixels[p + 1] / 255;
    out[2 * plane + i] = pixels[p + 2] / 255;
  }
  return out;
}

export interface EngineOptions {
  /** Use the TypeScript kNN instead of knn.onnx (reference / fallback). */
  jsKnn?: boolean;
}

export class Engine {
  ep: EpActual = 'wasm';
  requestedEp: EpChoice = 'auto';
  fallbackReason?: string;
  entry?: ModelEntry;
  private backbone?: Session;
  private knn?: Session;
  private bank?: Float32Array;
  private bankSq?: Float32Array;
  private bankD = 0;
  private teachFrames = new Map<string, Float32Array>();
  onProgress?: (e: ProgressEvent) => void;

  constructor(private ort: Ort, private opts: EngineOptions = {}) {}

  async init(requested: EpChoice, ortBase?: string): Promise<InitResult> {
    const { ort } = this;
    this.requestedEp = requested;
    const isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
    const cores = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 2;
    const threads = isolated ? Math.min(8, Math.ceil(cores / 2)) : 1;
    ort.env.wasm.numThreads = threads;
    ort.env.wasm.simd = true;
    if (ortBase) ort.env.wasm.wasmPaths = ortBase;
    ort.env.logLevel = 'error';

    let adapterInfo: Record<string, string> | undefined;
    this.ep = 'wasm';
    this.fallbackReason = undefined;
    if (requested !== 'wasm') {
      const gpu = (globalThis.navigator as Navigator & { gpu?: GPU })?.gpu;
      if (!gpu) {
        this.fallbackReason = 'WebGPU not available on this device, using WASM';
      } else {
        try {
          const adapter = await gpu.requestAdapter();
          if (!adapter) {
            this.fallbackReason = 'WebGPU adapter not found, using WASM';
          } else {
            this.ep = 'webgpu';
            const info = adapter.info;
            if (info) {
              adapterInfo = {
                vendor: info.vendor, architecture: info.architecture,
                device: info.device, description: info.description,
              };
            }
          }
        } catch (e) {
          this.fallbackReason = `WebGPU failed to initialise (${errText(e)}), using WASM`;
        }
      }
    }
    if (!isolated && this.ep === 'wasm') {
      this.fallbackReason = [this.fallbackReason, 'cross-origin isolation inactive: single-threaded WASM']
        .filter(Boolean).join('; ');
    }
    return { ep: this.ep, requestedEp: requested, threads, isolated, fallbackReason: this.fallbackReason, adapterInfo };
  }

  private async createSession(bytes: Uint8Array): Promise<Session> {
    return this.ort.InferenceSession.create(bytes, {
      executionProviders: [this.ep],
      graphOptimizationLevel: 'all',
    });
  }

  async loadModel(entry: ModelEntry, modelBytes: Uint8Array, knnBytes: Uint8Array, fetchMs = 0): Promise<LoadModelResult> {
    const sha = await sha256Hex(modelBytes);
    const t0 = now();
    await this.backbone?.release();
    await this.knn?.release();
    this.backbone = this.knn = undefined;
    try {
      this.backbone = await this.createSession(modelBytes);
      this.knn = await this.createSession(knnBytes);
    } catch (e) {
      if (this.ep !== 'webgpu') throw e;
      // Some WebGPU adapters reject ops or limits at session creation time.
      this.fallbackReason = `WebGPU session failed (${errText(e)}), using WASM`;
      this.ep = 'wasm';
      this.backbone = await this.createSession(modelBytes);
      this.knn = await this.createSession(knnBytes);
    }
    this.entry = entry;
    if (this.bank && this.bankD !== entry.D) this.clearBank();
    return {
      key: entry.key, ep: this.ep, fallbackReason: this.fallbackReason, fetchMs,
      sessionMs: now() - t0, sha256: sha, shaOk: sha === entry.sha256,
    };
  }

  private requireModel(): { session: Session; entry: ModelEntry } {
    if (!this.backbone || !this.entry) throw new Error('No backbone loaded');
    return { session: this.backbone, entry: this.entry };
  }

  /** CHW float32 [3, S, S] -> patch features [N, D]. */
  async features(chw: Float32Array): Promise<Float32Array> {
    const { session, entry } = this.requireModel();
    const s = entry.inputSize;
    const input = new this.ort.Tensor('float32', chw, [1, 3, s, s]);
    const out = await session.run({ image: input });
    const t = out.features;
    const data = (await t.getData()) as Float32Array;
    input.dispose();
    t.dispose();
    return data;
  }

  setBank(bank: Float32Array, sqnorm: Float32Array, D: number): void {
    if (this.entry && D !== this.entry.D) throw new Error(`Bank D=${D} does not match backbone D=${this.entry.D}`);
    this.bank = bank;
    this.bankSq = sqnorm;
    this.bankD = D;
  }

  clearBank(): void {
    this.bank = this.bankSq = undefined;
    this.bankD = 0;
  }

  get hasBank(): boolean {
    return !!this.bank;
  }

  /** Nearest-bank-row distance per patch, in bank chunks to bound GPU memory. */
  async minDist(features: Float32Array, bank = this.bank, sq = this.bankSq): Promise<Float32Array> {
    if (!bank || !sq) throw new Error('No profile loaded');
    const D = this.entry?.D ?? this.bankD;
    const n = features.length / D;
    const m = sq.length;
    if (this.opts.jsKnn || !this.knn) return knnMinDist(features, n, bank, sq, D);
    const acc = new Float32Array(n).fill(Infinity);
    const fT = new this.ort.Tensor('float32', features, [n, D]);
    for (let r0 = 0; r0 < m; r0 += KNN_CHUNK_ROWS) {
      const r1 = Math.min(m, r0 + KNN_CHUNK_ROWS);
      const bT = new this.ort.Tensor('float32', bank.subarray(r0 * D, r1 * D), [r1 - r0, D]);
      const sT = new this.ort.Tensor('float32', sq.subarray(r0, r1), [r1 - r0]);
      const out = await this.knn.run({ features: fT, bank: bT, bank_sqnorm: sT });
      runningMin(acc, (await out.min_dist.getData()) as Float32Array);
      out.min_dist.dispose();
      bT.dispose();
      sT.dispose();
    }
    fT.dispose();
    return acc;
  }

  async score(chw: Float32Array, preprocessMs = 0): Promise<ScoreResult> {
    const { entry } = this.requireModel();
    const t0 = now();
    const f = await this.features(chw);
    const t1 = now();
    const d = await this.minDist(f);
    const t2 = now();
    const smap = scoreMap(d, entry.grid, entry.inputSize);
    const imageScore = maxOf(smap);
    const map = Float32Array.from(smap);
    const t3 = now();
    const timings: StageTimings = {
      preprocess: preprocessMs, backbone: t1 - t0, knn: t2 - t1, post: t3 - t2,
      total: preprocessMs + (t3 - t0),
    };
    return { map, size: entry.inputSize, imageScore, sig: sceneSignature(chw, entry.inputSize), timings };
  }

  // ---- Teaching -------------------------------------------------------------

  async teachAdd(frameId: string, chw: Float32Array): Promise<number> {
    this.teachFrames.set(frameId, await this.features(chw));
    return this.teachFrames.size;
  }

  teachRemove(frameId: string): number {
    this.teachFrames.delete(frameId);
    return this.teachFrames.size;
  }

  teachReset(): void {
    this.teachFrames.clear();
  }

  async teachFinish({ margin, base }: TeachFinishArgs): Promise<TeachResult> {
    const { entry } = this.requireModel();
    const frames = [...this.teachFrames.values()];
    const D = entry.D;
    const per = entry.grid * entry.grid;
    const minFrames = base ? 1 : 5;
    if (frames.length < minFrames) throw new Error(`Need at least ${minFrames} good frames (have ${frames.length})`);
    // Adding images to an existing profile keeps its calibration; every new frame goes into the bank.
    const nHold = base ? 0 : holdoutCount(frames.length);
    const bankFrames = frames.slice(0, frames.length - nHold);
    const held = frames.slice(frames.length - nHold);

    const baseRows = base ? base.bank.length / D : 0;
    const n = baseRows + bankFrames.length * per;
    const src = new Float32Array(n * D);
    if (base) src.set(base.bank, 0);
    bankFrames.forEach((f, i) => src.set(f, (baseRows + i * per) * D));

    const t0 = now();
    const k = coresetTarget(n);
    const idx = greedyCoreset(src, n, D, k, undefined, (done, total) =>
      this.onProgress?.({ type: 'progress', stage: 'coreset', done, total }));
    const bank = gatherRows(src, D, idx);
    const sqnorm = rowSqNorms(bank, D);
    const coresetMs = now() - t0;

    const heldScores: number[] = [];
    const heldPatches: Float32Array[] = [];
    for (let i = 0; i < held.length; i++) {
      const d = await this.minDist(held[i], bank, sqnorm);
      heldPatches.push(d);
      heldScores.push(maxOf(scoreMap(d, entry.grid, entry.inputSize)));
      this.onProgress?.({ type: 'progress', stage: 'calibrate', done: i + 1, total: held.length });
    }
    const all = new Float32Array(heldPatches.length * per);
    heldPatches.forEach((p, i) => all.set(p, i * per));
    const maxHeld = base ? base.maxHeld : Math.max(...heldScores);
    const heatFloor = base ? base.heatFloor : median(all);
    this.setBank(bank, sqnorm, D);
    return {
      bank, sqnorm, D, M: k, maxHeld, heldScores, threshold: thresholdFrom(maxHeld, margin),
      heatFloor, frames: frames.length, heldOut: nHold,
      teachPatches: bankFrames.length * per, coresetMs,
    };
  }
}

/** 8x8 block means of luminance (0..1) from a CHW image. */
export function sceneSignature(chw: Float32Array, size: number): Float32Array {
  const g = 8;
  const out = new Float32Array(g * g);
  const counts = new Float32Array(g * g);
  const plane = size * size;
  for (let y = 0; y < size; y += 2) {
    const row = Math.floor((y * g) / size) * g;
    for (let x = 0; x < size; x += 2) {
      const i = y * size + x;
      const k = row + Math.floor((x * g) / size);
      out[k] += 0.299 * chw[i] + 0.587 * chw[plane + i] + 0.114 * chw[2 * plane + i];
      counts[k]++;
    }
  }
  for (let k = 0; k < out.length; k++) out[k] /= counts[k] || 1;
  return out;
}

export function errText(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}
