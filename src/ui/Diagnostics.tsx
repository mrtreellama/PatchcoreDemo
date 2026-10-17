import { APP_VERSION, makeOffline, store } from '../app';
import { useStore } from './store';

export function Diagnostics() {
  const s = useStore(store, (st) => ({
    init: st.init, model: st.model, manifest: st.manifest, coi: st.coi, lastError: st.lastError,
    engine: st.engine, device: st.device, precache: st.precache,
  }));
  const backend = s.init ? `${s.init.ep.toUpperCase()} (requested ${s.init.requestedEp})` : '…';
  return (
    <details class="diagnostics" data-testid="diagnostics">
      <summary>
        Diagnostics <span class="muted">· {s.init?.ep?.toUpperCase() ?? '…'} · {s.engine}{s.lastError ? ' · error' : ''}</span>
      </summary>
      <dl>
        <dt>Backend</dt><dd data-testid="diag-backend">{backend}</dd>
        {s.init?.fallbackReason && (<><dt>Fallback</dt><dd>{s.init.fallbackReason}</dd></>)}
        <dt>WASM threads</dt><dd>{s.init ? `${s.init.threads} ${s.init.threads > 1 ? '(enabled)' : '(single-threaded)'}` : '…'}</dd>
        <dt>Cross-origin isolated</dt><dd>{String(s.device?.crossOriginIsolated ?? self.crossOriginIsolated)} ({s.coi})</dd>
        {s.init?.adapterInfo && (<><dt>WebGPU adapter</dt><dd>{Object.values(s.init.adapterInfo).filter(Boolean).join(' · ') || 'unknown'}</dd></>)}
        <dt>Loaded model</dt>
        <dd>
          {s.model ? `${s.model.key}: fetch ${s.model.fetchMs.toFixed(0)} ms, session ${s.model.sessionMs.toFixed(0)} ms, ` : '…'}
          {s.model && (s.model.shaOk ? 'hash verified' : <b class="bad-text">HASH MISMATCH</b>)}
        </dd>
        <dt>Model hashes</dt>
        <dd>
          <ul class="hashes">
            {s.manifest?.models.map((m) => (
              <li key={m.key}><code>{m.file}</code> {m.sha256.slice(0, 16)}… ({(m.bytes / 1e6).toFixed(1)} MB)</li>
            ))}
            {s.manifest && <li><code>{s.manifest.knn.file}</code> {s.manifest.knn.sha256.slice(0, 16)}…</li>}
          </ul>
        </dd>
        <dt>Last error</dt><dd data-testid="diag-error">{s.lastError ?? 'none'}</dd>
        <dt>App</dt><dd>v{APP_VERSION} · built {__BUILD_TIME__.slice(0, 16).replace('T', ' ')}</dd>
        <dt>Offline</dt>
        <dd>
          <button class="small" onClick={() => makeOffline()}>Download all models for offline use</button>{' '}
          <span class="muted">{s.precache}</span>
        </dd>
      </dl>
    </details>
  );
}
