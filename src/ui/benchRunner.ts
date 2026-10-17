import { InferenceClient } from '../worker/client';
import type { ModelEntry, ModelManifest } from '../worker/protocol';
import { mulberry32 } from '../patchcore/prng';
import { rowSqNorms } from '../patchcore/knn';
import { defaultRoi, roiPixels } from '../camera/roi';
import { medianOf, percentile } from '../util/stats';
import { deviceInfo, type DeviceInfo } from '../util/device';
import { APP_VERSION, BASE } from '../app';

export const BENCH_FRAMES = 50;
export const BENCH_BANK_ROWS = 2000;
const STAGES = ['preprocess', 'backbone', 'knn', 'post', 'endToEnd'] as const;
export type Stage = (typeof STAGES)[number];

export interface BenchConfig {
  modelKey: string;
  ep: 'webgpu' | 'wasm';
}

export interface BenchRow {
  modelKey: string;
  requestedEp: string;
  ep?: string;
  threads?: number;
  fetchMs?: number;
  sessionMs?: number;
  warmupMs?: number;
  frames: number;
  stats?: Record<Stage, { median: number; p95: number }>;
  fps?: number;
  error?: string;
}

export interface BenchReport {
  app: string;
  date: string;
  device: DeviceInfo;
  workload: { frames: number; bankRows: number; images: number; note: string };
  results: BenchRow[];
}

async function loadWorkload(): Promise<ImageBitmap[]> {
  const { images } = (await (await fetch(`${BASE}samples/samples.json`)).json()) as { images: { file: string }[] };
  return Promise.all(images.map(async (i) => createImageBitmap(await (await fetch(`${BASE}samples/${i.file}`)).blob())));
}

function syntheticBank(D: number): { bank: Float32Array; sq: Float32Array } {
  const rng = mulberry32(42);
  const bank = new Float32Array(BENCH_BANK_ROWS * D);
  for (let i = 0; i < bank.length; i++) bank[i] = rng() * 2;
  return { bank, sq: rowSqNorms(bank, D) };
}

async function runOne(entry: ModelEntry, ep: 'webgpu' | 'wasm', images: ImageBitmap[], manifest: ModelManifest,
  onStep: (msg: string) => void): Promise<BenchRow> {
  const row: BenchRow = { modelKey: entry.key, requestedEp: ep, frames: 0 };
  const client = new InferenceClient();
  try {
    const init = await client.init(ep, `${BASE}ort/`);
    row.threads = init.threads;
    if (ep === 'webgpu' && init.ep !== 'webgpu') throw new Error(init.fallbackReason ?? 'WebGPU unavailable');
    onStep(`${entry.key} · ${ep}: loading model`);
    const knn = manifest.knn;
    const lm = await client.loadModel(entry, `${BASE}models/${entry.file}?v=${entry.sha256.slice(0, 12)}`,
      `${BASE}models/${knn.file}?v=${knn.sha256.slice(0, 12)}`);
    if (ep === 'webgpu' && lm.ep !== 'webgpu') throw new Error(lm.fallbackReason ?? 'WebGPU session failed');
    row.ep = lm.ep;
    row.fetchMs = lm.fetchMs;
    row.sessionMs = lm.sessionMs;
    const { bank, sq } = syntheticBank(entry.D);
    await client.setBank(bank, sq, entry.D);

    const s = entry.inputSize;
    const crop = async (img: ImageBitmap) => {
      const { sx, sy, sw, sh } = roiPixels(defaultRoi(img.width, img.height), img.width, img.height);
      return createImageBitmap(img, sx, sy, sw, sh, { resizeWidth: s, resizeHeight: s, resizeQuality: 'high' });
    };
    onStep(`${entry.key} · ${ep}: warm-up`);
    const w0 = performance.now();
    await client.score(await crop(images[0]));
    row.warmupMs = performance.now() - w0;

    const samples: Record<Stage, number[]> = { preprocess: [], backbone: [], knn: [], post: [], endToEnd: [] };
    for (let i = 0; i < BENCH_FRAMES; i++) {
      const t0 = performance.now();
      const bmp = await crop(images[i % images.length]);
      const t1 = performance.now();
      const r = await client.score(bmp);
      const t2 = performance.now();
      samples.preprocess.push(t1 - t0 + r.timings.preprocess);
      samples.backbone.push(r.timings.backbone);
      samples.knn.push(r.timings.knn);
      samples.post.push(r.timings.post);
      samples.endToEnd.push(t2 - t0);
      row.frames++;
      if (i % 5 === 0) onStep(`${entry.key} · ${ep}: frame ${i + 1}/${BENCH_FRAMES}`);
    }
    row.stats = Object.fromEntries(STAGES.map((k) => [k, { median: medianOf(samples[k]), p95: percentile(samples[k], 95) }])) as BenchRow['stats'];
    row.fps = 1000 / row.stats!.endToEnd.median;
  } catch (e) {
    row.error = (e as Error).message;
  } finally {
    client.terminate();
  }
  return row;
}

export async function runBenchmark(manifest: ModelManifest, configs: BenchConfig[],
  onStep: (msg: string) => void, onRow: (row: BenchRow) => void): Promise<BenchReport> {
  onStep('Loading workload images');
  const images = await loadWorkload();
  const device = await deviceInfo();
  const results: BenchRow[] = [];
  for (const c of configs) {
    const entry = manifest.models.find((m) => m.key === c.modelKey);
    if (!entry) continue;
    const row = await runOne(entry, c.ep, images, manifest, onStep);
    results.push(row);
    onRow(row);
  }
  images.forEach((b) => b.close());
  onStep('Done');
  return {
    app: `PatchCore demo v${APP_VERSION}`,
    date: new Date().toISOString(),
    device,
    workload: {
      frames: BENCH_FRAMES, bankRows: BENCH_BANK_ROWS, images: images.length,
      note: 'Bundled sample images, default ROI, synthetic memory bank. Model load = fetch + session creation.',
    },
    results,
  };
}

export function toTsv(report: BenchReport): string {
  const head = ['device', 'model', 'ep', 'threads', 'fetch_ms', 'session_ms', 'warmup_ms',
    ...STAGES.flatMap((s) => [`${s}_median_ms`, `${s}_p95_ms`]), 'fps', 'error'];
  const dev = `${report.device.platform} ${report.device.logicalCores}c ${report.device.webgpuAdapter?.vendor ?? ''}`.trim();
  const f = (v?: number) => (v === undefined ? '' : v.toFixed(1));
  const rows = report.results.map((r) => [
    dev, r.modelKey, r.ep ?? r.requestedEp, r.threads ?? '', f(r.fetchMs), f(r.sessionMs), f(r.warmupMs),
    ...STAGES.flatMap((s) => [f(r.stats?.[s].median), f(r.stats?.[s].p95)]), f(r.fps), r.error ?? '',
  ].join('\t'));
  return [head.join('\t'), ...rows].join('\n');
}
