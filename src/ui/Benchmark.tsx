import { useState } from 'preact/hooks';
import * as app from '../app';
import { store } from '../app';
import { useStore } from './store';
import { BENCH_BANK_ROWS, BENCH_FRAMES, runBenchmark, toTsv, type BenchConfig, type BenchReport, type BenchRow } from './benchRunner';

const hasWebGpu = typeof navigator !== 'undefined' && 'gpu' in navigator;

export function Benchmark() {
  const manifest = useStore(store, (s) => s.manifest);
  const [selected, setSelected] = useState<Set<string>>(() => new Set([`r18_224|wasm`, ...(hasWebGpu ? ['r18_224|webgpu'] : [])]));
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState('');
  const [rows, setRows] = useState<BenchRow[]>([]);
  const [report, setReport] = useState<BenchReport>();
  const [copied, setCopied] = useState(false);

  if (!manifest) return <main class="bench"><p>Loading…</p></main>;
  const combos = manifest.models.flatMap((m) => (['webgpu', 'wasm'] as const).map((ep) => ({ key: `${m.key}|${ep}`, m, ep })));

  const toggle = (k: string) => {
    const n = new Set(selected);
    if (n.has(k)) n.delete(k);
    else n.add(k);
    setSelected(n);
  };

  const run = async () => {
    // Free the live pipeline's GPU/CPU while measuring.
    app.stopDetect();
    setRunning(true);
    setRows([]);
    setReport(undefined);
    const configs: BenchConfig[] = combos.filter((c) => selected.has(c.key)).map((c) => ({ modelKey: c.m.key, ep: c.ep }));
    try {
      const r = await runBenchmark(manifest, configs, setStep, (row) => setRows((x) => [...x, row]));
      setReport(r);
    } catch (e) {
      setStep(`Failed: ${(e as Error).message}`);
    } finally {
      setRunning(false);
    }
  };

  const download = () => {
    if (!report) return;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
    a.download = `patchcore-benchmark-${report.date.slice(0, 19).replace(/[:T]/g, '-')}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  };

  const copy = async () => {
    if (!report) return;
    await navigator.clipboard.writeText(toTsv(report));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const f = (v?: number) => (v === undefined ? '—' : v.toFixed(1));
  return (
    <main class="bench">
      <section class="panel">
        <h2>Benchmark</h2>
        <p class="small muted">
          For each selected combination: model load (download + session), first-inference warm-up, then {BENCH_FRAMES} timed
          frames over the bundled sample images against a {BENCH_BANK_ROWS.toLocaleString()}-row memory bank.
          Live detection is stopped while this runs.
        </p>
        <div class="combos">
          {combos.map((c) => (
            <label class="inline" key={c.key}>
              <input type="checkbox" checked={selected.has(c.key)} disabled={running || (c.ep === 'webgpu' && !hasWebGpu)}
                onChange={() => toggle(c.key)} data-testid={`bench-${c.key}`} />
              {c.m.name} · {c.ep === 'webgpu' ? 'WebGPU' : 'WASM'}
            </label>
          ))}
        </div>
        <div class="row">
          <button class="primary" disabled={running || !selected.size} onClick={run} data-testid="bench-run">{running ? 'Running…' : 'Run benchmark'}</button>
          <button disabled={!report} onClick={download} data-testid="bench-download">Download JSON</button>
          <button disabled={!report} onClick={copy}>{copied ? 'Copied' : 'Copy for spreadsheet'}</button>
          <span class="small muted" data-testid="bench-step">{step}</span>
        </div>
      </section>
      <section class="panel scroll-x">
        <table class="bench-table" data-testid="bench-table">
          <thead>
            <tr>
              <th>Model</th><th>Backend</th><th>Load ms</th><th>Warm-up ms</th>
              <th>Pre med/p95</th><th>Backbone med/p95</th><th>kNN med/p95</th><th>Post med/p95</th><th>E2E med/p95</th><th>FPS</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td>{r.modelKey}</td>
                <td>{r.ep ?? r.requestedEp}{r.ep === 'wasm' && r.threads ? ` ×${r.threads}` : ''}</td>
                {r.error ? <td colSpan={8} class="bad-text">{r.error}</td> : (
                  <>
                    <td>{f((r.fetchMs ?? 0) + (r.sessionMs ?? 0))}</td>
                    <td>{f(r.warmupMs)}</td>
                    {(['preprocess', 'backbone', 'knn', 'post', 'endToEnd'] as const).map((k) => (
                      <td key={k}>{f(r.stats?.[k].median)} / {f(r.stats?.[k].p95)}</td>
                    ))}
                    <td><b>{f(r.fps)}</b></td>
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
        {report && (
          <dl class="small device">
            <dt>User agent</dt><dd>{report.device.userAgent}</dd>
            <dt>Cores</dt><dd>{report.device.logicalCores}{report.device.deviceMemoryGb ? ` · ${report.device.deviceMemoryGb} GB` : ''}</dd>
            <dt>WebGPU adapter</dt><dd>{report.device.webgpuAdapter ? Object.values(report.device.webgpuAdapter).filter(Boolean).join(' · ') : 'none'}</dd>
            <dt>Cross-origin isolated</dt><dd>{String(report.device.crossOriginIsolated)}</dd>
          </dl>
        )}
      </section>
    </main>
  );
}
