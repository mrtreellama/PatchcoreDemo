import type {
  EpChoice, InitResult, LoadModelResult, ModelEntry, ProgressEvent, Request, Response, ScoreResult,
  TeachFinishArgs, TeachResult,
} from './protocol';

type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void };

/** Promise-based RPC wrapper around the inference worker. */
export class InferenceClient {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  onProgress?: (e: ProgressEvent) => void;

  constructor() {
    this.worker = new Worker(new URL('./inference.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (ev: MessageEvent<Response | ProgressEvent>) => {
      const data = ev.data;
      if ('type' in data && data.type === 'progress') {
        this.onProgress?.(data);
        return;
      }
      const r = data as Response;
      const p = this.pending.get(r.id);
      if (!p) return;
      this.pending.delete(r.id);
      if (r.ok) p.resolve(r.result);
      else p.reject(new Error(r.error));
    };
    this.worker.onerror = (ev) => {
      const err = new Error(ev.message || 'Inference worker crashed');
      for (const p of this.pending.values()) p.reject(err);
      this.pending.clear();
    };
  }

  private call<T>(req: Request, transfer: Transferable[] = []): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
      this.worker.postMessage({ ...req, id }, transfer);
    });
  }

  init(ep: EpChoice, ortBase: string) {
    return this.call<InitResult>({ type: 'init', ep, ortBase });
  }

  loadModel(entry: ModelEntry, url: string, knnUrl: string) {
    return this.call<LoadModelResult>({ type: 'loadModel', entry, url, knnUrl });
  }

  /** Copies the bank so the caller keeps its own arrays. */
  setBank(bank: Float32Array, sqnorm: Float32Array, D: number) {
    const b = bank.slice();
    const s = sqnorm.slice();
    return this.call<boolean>({ type: 'setBank', bank: b, sqnorm: s, D }, [b.buffer, s.buffer]);
  }

  clearBank() {
    return this.call<boolean>({ type: 'clearBank' });
  }

  score(bitmap: ImageBitmap) {
    return this.call<ScoreResult>({ type: 'score', bitmap }, [bitmap]);
  }

  features(bitmap: ImageBitmap) {
    return this.call<Float32Array>({ type: 'features', bitmap }, [bitmap]);
  }

  teachAdd(frameId: string, bitmap: ImageBitmap) {
    return this.call<number>({ type: 'teachAdd', frameId, bitmap }, [bitmap]);
  }

  teachRemove(frameId: string) {
    return this.call<number>({ type: 'teachRemove', frameId });
  }

  teachReset() {
    return this.call<boolean>({ type: 'teachReset' });
  }

  teachFinish(args: TeachFinishArgs) {
    const transfer = args.base ? [args.base.bank.buffer] : [];
    return this.call<TeachResult>({ type: 'teachFinish', args }, transfer);
  }

  terminate() {
    this.worker.terminate();
    for (const p of this.pending.values()) p.reject(new Error('Worker terminated'));
    this.pending.clear();
  }
}
