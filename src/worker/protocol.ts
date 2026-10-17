/** Typed message protocol between the UI thread and the inference worker. */

export type EpChoice = 'auto' | 'webgpu' | 'wasm';
export type EpActual = 'webgpu' | 'wasm';

export interface ModelEntry {
  key: string;
  id: string;
  name: string;
  file: string;
  inputSize: number;
  grid: number;
  D: number;
  dtype: 'fp32' | 'fp16';
  bytes: number;
  sha256: string;
  exportVersion: number;
}

export interface ModelManifest {
  exportVersion: number;
  models: ModelEntry[];
  knn: { file: string; sha256: string };
}

export interface InitResult {
  ep: EpActual;
  requestedEp: EpChoice;
  threads: number;
  isolated: boolean;
  fallbackReason?: string;
  adapterInfo?: Record<string, string>;
}

export interface LoadModelResult {
  key: string;
  ep: EpActual;
  fallbackReason?: string;
  fetchMs: number;
  sessionMs: number;
  sha256: string;
  shaOk: boolean;
}

export interface StageTimings {
  preprocess: number;
  backbone: number;
  knn: number;
  post: number;
  total: number;
}

export interface ScoreResult {
  /** Smoothed anomaly map at the backbone input size, row-major. */
  map: Float32Array;
  size: number;
  imageScore: number;
  /** 8x8 mean-luminance thumbnail of the input, for cheap scene-change detection. */
  sig: Float32Array;
  timings: StageTimings;
}

export interface TeachFinishArgs {
  margin: number;
  /** Existing bank rows to merge with the new captures ("add good images"). */
  base?: { bank: Float32Array; maxHeld: number; heatFloor: number };
}

export interface TeachResult {
  bank: Float32Array;
  sqnorm: Float32Array;
  D: number;
  M: number;
  maxHeld: number;
  heldScores: number[];
  threshold: number;
  heatFloor: number;
  frames: number;
  heldOut: number;
  teachPatches: number;
  coresetMs: number;
}

export type Request =
  | { type: 'init'; ep: EpChoice; ortBase: string }
  | { type: 'loadModel'; entry: ModelEntry; url: string; knnUrl: string }
  | { type: 'setBank'; bank: Float32Array; sqnorm: Float32Array; D: number }
  | { type: 'clearBank' }
  | { type: 'score'; bitmap: ImageBitmap }
  | { type: 'teachAdd'; frameId: string; bitmap: ImageBitmap }
  | { type: 'teachRemove'; frameId: string }
  | { type: 'teachReset' }
  | { type: 'teachFinish'; args: TeachFinishArgs }
  | { type: 'features'; bitmap: ImageBitmap };

export type Envelope = Request & { id: number };

export type Response =
  | { id: number; ok: true; result: unknown }
  | { id: number; ok: false; error: string };

export interface ProgressEvent {
  type: 'progress';
  stage: 'coreset' | 'calibrate';
  done: number;
  total: number;
}
