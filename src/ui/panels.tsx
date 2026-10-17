import { useEffect, useRef, useState } from 'preact/hooks';
import * as app from '../app';
import { store } from '../app';
import { useStore } from './store';
import { ImageSource, CameraSource } from '../camera/source';
import { applyControl, listControls, type CameraControl } from '../camera/controls';
import { MAX_FRAMES, MIN_FRAMES } from '../patchcore/threshold';
import { maxHeldOf } from '../profiles/types';
import { getSetting, setSetting } from '../profiles/store';

function Panel({ title, children, testid }: { title: string; children: preact.ComponentChildren; testid?: string }) {
  return (
    <section class="panel" data-testid={testid}>
      <h2>{title}</h2>
      {children}
    </section>
  );
}

// ---- Source ---------------------------------------------------------------------

export function SourcePanel() {
  const s = useStore(store, (st) => ({
    kind: st.sourceKind, label: st.sourceLabel, cameras: st.cameras, cameraId: st.cameraId,
    resolution: st.resolution, tick: st.sourceTick, rig: st.rig, size: st.sourceSize, lock: st.lock,
  }));
  const imgInput = useRef<HTMLInputElement>(null);
  const vidInput = useRef<HTMLInputElement>(null);
  const src = app.getSource();
  const run = (p: Promise<unknown>) => p.catch(() => undefined);

  return (
    <Panel title="Source" testid="source-panel">
      <div class="row wrap">
        <button class={s.kind === 'camera' ? 'on' : ''} onClick={() => run(app.openCamera(s.cameraId))}>Camera</button>
        <button onClick={() => imgInput.current?.click()} class={s.kind === 'images' && !s.label.startsWith('Sample') ? 'on' : ''}>Image files…</button>
        <button onClick={() => vidInput.current?.click()} class={s.kind === 'video' ? 'on' : ''}>Video file…</button>
        <button onClick={() => run(app.openSamples())} class={s.label.startsWith('Sample') ? 'on' : ''}>Samples</button>
        <input ref={imgInput} type="file" accept="image/*" multiple hidden data-testid="image-input"
          onChange={(e) => { const f = (e.target as HTMLInputElement).files; if (f?.length) run(app.openImages([...f])); }} />
        <input ref={vidInput} type="file" accept="video/*" hidden
          onChange={(e) => { const f = (e.target as HTMLInputElement).files?.[0]; if (f) run(app.openVideo(f)); }} />
      </div>
      <p class="muted small" data-testid="source-label">{s.label}{s.size[0] ? ` · ${s.size[0]}×${s.size[1]}` : ''}</p>

      {s.kind === 'camera' && (
        <>
          <div class="row">
            <label class="grow">Camera
              <select value={s.cameraId} onChange={(e) => run(app.openCamera((e.target as HTMLSelectElement).value))}>
                {s.cameras.map((c, i) => <option key={c.deviceId} value={c.deviceId}>{c.label || `Camera ${i + 1}`}</option>)}
              </select>
            </label>
            <label>Resolution
              <select value={s.resolution} onChange={(e) => run(app.openCamera(s.cameraId, (e.target as HTMLSelectElement).value as app.Resolution))}>
                <option value="720p">1280×720</option>
                <option value="1080p">1920×1080</option>
              </select>
            </label>
          </div>
          {src instanceof CameraSource && src.track && <CameraControls track={src.track} key={s.tick} />}
          {s.lock && (
            <p class="small">
              {s.lock.locked.length > 0 && <span class="good-text">Locked: {s.lock.locked.join(', ')}. </span>}
              {s.lock.failed.length > 0 && <span class="warn-text">Not locked: {s.lock.failed.join('; ')}</span>}
            </p>
          )}
        </>
      )}

      {src instanceof ImageSource && (
        <div class="row">
          <button onClick={() => app.stepImage(-1)} aria-label="Previous image">◀</button>
          <span class="grow center small" data-testid="image-name">{src.index + 1} / {src.images.length} · {src.name}</span>
          <button onClick={() => app.stepImage(1)} aria-label="Next image" data-testid="next-image">▶</button>
        </div>
      )}

      <div class="row">
        <button class="small" onClick={() => app.resetRoi()}>Reset ROI</button>
        <label class="grow">Rig name
          <input value={s.rig} placeholder="e.g. Booth rig A" onChange={(e) => app.setRig((e.target as HTMLInputElement).value)} />
        </label>
      </div>
    </Panel>
  );
}

function CameraControls({ track }: { track: MediaStreamTrack }) {
  const [controls, setControls] = useState<CameraControl[]>(() => listControls(track));
  const refresh = () => setControls(listControls(track));
  const set = async (name: string, v: number | string | boolean) => {
    try {
      await applyControl(track, name, v);
    } catch (e) {
      app.notify(`Camera rejected ${name}: ${(e as Error).message}`, 'warn');
    }
    refresh();
  };
  if (!controls.length) return <p class="muted small">This camera exposes no adjustable controls in this browser.</p>;
  return (
    <details class="cam-controls">
      <summary>Camera controls</summary>
      {controls.map((c) => (
        <label key={c.name} class="ctl">
          <span>{c.label}</span>
          {c.kind === 'mode' && (
            <select value={c.value} onChange={(e) => set(c.name, (e.target as HTMLSelectElement).value)}>
              {c.options.map((o) => <option key={o}>{o}</option>)}
            </select>
          )}
          {c.kind === 'range' && (
            <input type="range" min={c.min} max={c.max} step={c.step} value={c.value}
              onChange={(e) => set(c.name, Number((e.target as HTMLInputElement).value))} />
          )}
          {c.kind === 'toggle' && (
            <input type="checkbox" checked={c.value} onChange={(e) => set(c.name, (e.target as HTMLInputElement).checked)} />
          )}
        </label>
      ))}
      <button onClick={async () => { await app.lockCamera(); refresh(); }}>Lock exposure, focus &amp; white balance</button>
    </details>
  );
}

// ---- Model ----------------------------------------------------------------------

export function ModelPanel() {
  const s = useStore(store, (st) => ({ manifest: st.manifest, key: st.modelKey, ep: st.ep, engine: st.engine, init: st.init }));
  const run = (p: Promise<unknown>) => p.catch(() => undefined);
  return (
    <Panel title="Model">
      <div class="row">
        <label class="grow">Backbone &amp; input size
          <select value={s.key} disabled={s.engine === 'loading'} data-testid="model-select"
            onChange={(e) => run(app.setModel((e.target as HTMLSelectElement).value))}>
            {s.manifest?.models.map((m) => (
              <option key={m.key} value={m.key}>{m.name} ({(m.bytes / 1e6).toFixed(0)} MB)</option>
            ))}
          </select>
        </label>
        <label>Execution
          <select value={s.ep} data-testid="ep-select" onChange={(e) => run(app.setEp((e.target as HTMLSelectElement).value as app.AppState['ep']))}>
            <option value="auto">Auto</option>
            <option value="webgpu">WebGPU</option>
            <option value="wasm">WASM</option>
          </select>
        </label>
      </div>
      <p class="small muted" data-testid="engine-status">
        {s.engine === 'ready' ? `Ready on ${s.init?.ep.toUpperCase()}${s.init?.ep === 'wasm' ? ` · ${s.init.threads} thread${s.init.threads > 1 ? 's' : ''}` : ''}` : s.engine === 'error' ? 'Engine error, see diagnostics' : 'Loading model…'}
      </p>
    </Panel>
  );
}

// ---- Profiles -------------------------------------------------------------------

export function ProfilePanel() {
  const s = useStore(store, (st) => ({ profiles: st.profiles, activeId: st.activeId, mode: st.mode, manifest: st.manifest }));
  const p = s.profiles.find((x) => x.id === s.activeId);
  const fileRef = useRef<HTMLInputElement>(null);
  const run = (pr: Promise<unknown>) => pr.catch(() => undefined);
  const [thumb, setThumb] = useState<string>();
  useEffect(() => {
    if (!p?.thumbnail) return setThumb(undefined);
    const u = URL.createObjectURL(p.thumbnail);
    setThumb(u);
    return () => URL.revokeObjectURL(u);
  }, [p?.thumbnail]);

  return (
    <Panel title="Profile" testid="profile-panel">
      <div class="row">
        <select class="grow" value={s.activeId ?? ''} data-testid="profile-select"
          onChange={(e) => { const v = (e.target as HTMLSelectElement).value; if (v) run(app.activateProfile(v)); }}>
          <option value="" disabled>{s.profiles.length ? 'Select a profile…' : 'No profiles yet'}</option>
          {s.profiles.map((x) => (
            <option key={x.id} value={x.id} disabled={!app.modelForProfile(x)}>
              {x.header.name}{x.bundled ? ' (example)' : ''} · {x.header.modelKey}{app.modelForProfile(x) ? '' : ' · backbone missing'}
            </option>
          ))}
        </select>
      </div>
      {p && (
        <div class="profile-meta">
          {thumb && <img src={thumb} alt="" class="pthumb" />}
          <div class="small">
            <div><b>{p.header.name}</b>{p.bundled && <span class="tag">read-only</span>}</div>
            <div class="muted">{p.header.backboneId} @ {p.header.inputSize} · {p.header.M.toLocaleString()} rows · {p.header.teachStats.frames} frames</div>
            <div class="muted">Camera: {p.header.camera.label || '—'}{p.header.rig ? ` · Rig: ${p.header.rig}` : ''}</div>
          </div>
        </div>
      )}
      {p && (
        <label class="ctl">
          <span>Threshold margin ×{p.header.margin.toFixed(2)} → {p.header.threshold.toFixed(3)}</span>
          <input type="range" min="0.8" max="2" step="0.01" value={p.header.margin} data-testid="margin"
            onInput={(e) => run(app.setMargin(Number((e.target as HTMLInputElement).value)))} />
          <span class="muted small">Highest held-out score {maxHeldOf(p.header).toFixed(3)}</span>
        </label>
      )}
      <div class="row wrap">
        <button class="primary" disabled={s.mode === 'teach'} onClick={() => run(app.beginTeach())} data-testid="new-profile">Teach new…</button>
        {p && !p.bundled && <button disabled={s.mode === 'teach'} onClick={() => run(app.beginTeach(p.id))}>Add good images</button>}
        {p && !p.bundled && <button onClick={() => { const n = prompt('Rename profile', p.header.name); if (n) run(app.renameProfile(p.id, n)); }}>Rename</button>}
        {p && <button onClick={() => run(app.duplicateProfile(p.id))}>Duplicate</button>}
        {p && <button onClick={() => run(app.exportProfile(p.id))} data-testid="export">Export</button>}
        <button onClick={() => fileRef.current?.click()}>Import…</button>
        {p && !p.bundled && <button class="danger" onClick={() => { if (confirm(`Delete "${p.header.name}"?`)) run(app.removeProfile(p.id)); }}>Delete</button>}
        <input ref={fileRef} type="file" accept=".pcprof" multiple hidden data-testid="import-input"
          onChange={(e) => { const f = (e.target as HTMLInputElement).files; if (f?.length) run(app.importProfiles([...f])); (e.target as HTMLInputElement).value = ''; }} />
      </div>
      <p class="muted small">Tip: drop .pcprof files anywhere on this page to import.</p>
    </Panel>
  );
}

// ---- Teach ----------------------------------------------------------------------

export function TeachPanel() {
  const s = useStore(store, (st) => ({ t: st.teach, mode: st.mode, key: st.modelKey, kind: st.sourceKind, profiles: st.profiles }));
  if (s.mode !== 'teach') return null;
  const t = s.t;
  const n = t.frames.length;
  const adding = t.addTo ? s.profiles.find((p) => p.id === t.addTo) : undefined;
  const min = adding ? 1 : MIN_FRAMES;
  return (
    <Panel title={adding ? `Add good images to "${adding.header.name}"` : 'Teach a material'} testid="teach-panel">
      {!adding && (
        <div class="row">
          <label class="grow">Name
            <input value={t.name} onInput={(e) => app.setTeachField({ name: (e.target as HTMLInputElement).value })} data-testid="teach-name" />
          </label>
          <label>Frames
            <input type="number" min={MIN_FRAMES} max={MAX_FRAMES} value={t.target} style={{ width: '4.5em' }}
              onInput={(e) => app.setTeachField({ target: Math.min(MAX_FRAMES, Math.max(MIN_FRAMES, Number((e.target as HTMLInputElement).value) || MIN_FRAMES)) })} />
          </label>
        </div>
      )}
      <p class="small muted">
        Show only <b>good</b> material. Move the sample a little between captures. Backbone: {s.key}.
        {!adding && ' The last 20% of frames calibrate the threshold.'}
      </p>
      <div class="row wrap">
        <button class="primary" disabled={t.busy || n >= MAX_FRAMES} onClick={() => app.captureFrame()} data-testid="capture">Capture</button>
        <label class="inline">
          <input type="checkbox" checked={t.auto} disabled={t.busy} onChange={(e) => app.setAutoCapture((e.target as HTMLInputElement).checked)} data-testid="auto-capture" />
          Auto every 0.5 s
        </label>
        {s.kind === 'images' && <button disabled={t.busy} onClick={() => app.captureAllImages()} data-testid="capture-all">Capture all images</button>}
      </div>
      <div class="progress" aria-label="Captured frames"><div style={{ width: `${Math.min(100, (n / t.target) * 100)}%` }} /></div>
      <p class="small" data-testid="teach-count">{n} / {t.target} frames{n < min ? ` (minimum ${min})` : ''}</p>
      <div class="thumbs">
        {t.frames.map((f) => (
          <div class="thumb" key={f.id}>
            <img src={f.thumb} alt="" />
            <button class="x" aria-label="Delete capture" onClick={() => app.removeFrame(f.id)} disabled={t.busy}>×</button>
          </div>
        ))}
      </div>
      {t.busy && (
        <div>
          <p class="small">{t.stage} {t.progress !== undefined ? `${Math.round(t.progress * 100)}%` : ''}</p>
          <div class="progress"><div style={{ width: `${(t.progress ?? 0) * 100}%` }} /></div>
        </div>
      )}
      <div class="row">
        <button class="primary" disabled={t.busy || n < min} onClick={() => app.finishTeach().catch(() => undefined)} data-testid="finish-teach">
          {adding ? 'Update profile' : 'Finish teaching'}
        </button>
        <button disabled={t.busy} onClick={() => app.cancelTeach()}>Cancel</button>
      </div>
    </Panel>
  );
}

// ---- Booth options ----------------------------------------------------------------

export function useBoothOptions() {
  const [opts, setOpts] = useState({ sideBySide: true, attract: true });
  useEffect(() => { getSetting('booth', opts).then(setOpts); }, []);
  const update = (patch: Partial<typeof opts>) => {
    const next = { ...opts, ...patch };
    setOpts(next);
    setSetting('booth', next);
  };
  return [opts, update] as const;
}

export function BoothPanel() {
  const [o, set] = useBoothOptions();
  return (
    <Panel title="Booth">
      <label class="inline"><input type="checkbox" checked={o.sideBySide} onChange={(e) => set({ sideBySide: (e.target as HTMLInputElement).checked })} /> Show good reference next to live view</label>
      <label class="inline"><input type="checkbox" checked={o.attract} onChange={(e) => set({ attract: (e.target as HTMLInputElement).checked })} /> Idle attract loop (after 60 s without scene change)</label>
      <div class="row"><a class="button primary" href="#/booth">Launch booth mode</a></div>
      <p class="muted small">Exit booth mode by holding the logo for 2 seconds, or press Ctrl+Shift+E.</p>
    </Panel>
  );
}
