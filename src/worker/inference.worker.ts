/// <reference lib="webworker" />
import * as ort from 'onnxruntime-web/webgpu';
import { Engine, errText, toChw } from './engine';
import type { Envelope, Response } from './protocol';

declare const self: DedicatedWorkerGlobalScope;

const engine = new Engine(ort);
engine.onProgress = (e) => self.postMessage(e);

let canvas: OffscreenCanvas | undefined;
let ctx: OffscreenCanvasRenderingContext2D | undefined;

/** Bitmap (already cropped + resized on the UI thread) -> CHW float32. */
function bitmapToChw(bitmap: ImageBitmap): { chw: Float32Array; ms: number } {
  const t0 = performance.now();
  const size = engine.entry?.inputSize ?? bitmap.width;
  if (!canvas || canvas.width !== size) {
    canvas = new OffscreenCanvas(size, size);
    ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  }
  ctx!.drawImage(bitmap, 0, 0, size, size);
  bitmap.close();
  const px = ctx!.getImageData(0, 0, size, size).data;
  return { chw: toChw(px, size, 4), ms: performance.now() - t0 };
}

async function fetchBytes(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to download ${url} (HTTP ${res.status})`);
  return new Uint8Array(await res.arrayBuffer());
}

async function handle(msg: Envelope): Promise<{ result: unknown; transfer?: Transferable[] }> {
  switch (msg.type) {
    case 'init':
      return { result: await engine.init(msg.ep, msg.ortBase) };
    case 'loadModel': {
      const t0 = performance.now();
      const [m, k] = await Promise.all([fetchBytes(msg.url), fetchBytes(msg.knnUrl)]);
      return { result: await engine.loadModel(msg.entry, m, k, performance.now() - t0) };
    }
    case 'setBank':
      engine.setBank(msg.bank, msg.sqnorm, msg.D);
      return { result: true };
    case 'clearBank':
      engine.clearBank();
      return { result: true };
    case 'score': {
      const { chw, ms } = bitmapToChw(msg.bitmap);
      const r = await engine.score(chw, ms);
      return { result: r, transfer: [r.map.buffer, r.sig.buffer] };
    }
    case 'features': {
      const { chw } = bitmapToChw(msg.bitmap);
      const f = await engine.features(chw);
      return { result: f, transfer: [f.buffer] };
    }
    case 'teachAdd': {
      const { chw } = bitmapToChw(msg.bitmap);
      return { result: await engine.teachAdd(msg.frameId, chw) };
    }
    case 'teachRemove':
      return { result: engine.teachRemove(msg.frameId) };
    case 'teachReset':
      engine.teachReset();
      return { result: true };
    case 'teachFinish': {
      const r = await engine.teachFinish(msg.args);
      // Keep our own copy of the bank; send the UI a copy it can store.
      const bank = r.bank.slice();
      const sqnorm = r.sqnorm.slice();
      return { result: { ...r, bank, sqnorm }, transfer: [bank.buffer, sqnorm.buffer] };
    }
  }
}

// Messages are processed strictly in order so sessions are never run concurrently.
let queue: Promise<void> = Promise.resolve();
self.onmessage = (ev: MessageEvent<Envelope>) => {
  const msg = ev.data;
  queue = queue.then(async () => {
    let reply: Response;
    let transfer: Transferable[] = [];
    try {
      const r = await handle(msg);
      reply = { id: msg.id, ok: true, result: r.result };
      transfer = r.transfer ?? [];
    } catch (e) {
      reply = { id: msg.id, ok: false, error: errText(e) };
    }
    self.postMessage(reply, transfer);
  });
};
