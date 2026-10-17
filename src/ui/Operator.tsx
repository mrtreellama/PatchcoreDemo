import { useState } from 'preact/hooks';
import * as app from '../app';
import { store } from '../app';
import { useStore } from './store';
import { LiveView } from './LiveView';
import { ScoreChart } from './ScoreChart';
import { BoothPanel, ModelPanel, ProfilePanel, SourcePanel, TeachPanel } from './panels';

function LiveBar() {
  const s = useStore(store, (st) => ({
    result: st.result, mode: st.mode, frozen: st.frozen, fps: st.fps, opacity: st.opacity,
    active: st.profiles.find((p) => p.id === st.activeId),
  }));
  const r = s.mode === 'detect' ? s.result : undefined;
  return (
    <div class="livebar">
      <div class={`verdict ${r ? (r.verdict === 'DEFECT' ? 'bad' : 'good') : 'none'}`} data-testid="verdict">
        {s.mode === 'teach' ? 'TEACHING' : r ? r.verdict : s.active ? 'READY' : 'NO PROFILE'}
      </div>
      <div class="scores">
        <div class="small muted">{s.active ? s.active.header.name : 'No profile selected'}</div>
        <div data-testid="score">
          Score <b>{r ? r.score.toFixed(3) : '—'}</b> / threshold <b>{s.active ? s.active.header.threshold.toFixed(3) : '—'}</b>
          <span class="muted"> · {s.mode === 'detect' ? `${s.fps.toFixed(1)} fps` : 'stopped'}</span>
        </div>
        <ScoreChart />
      </div>
      <div class="live-actions">
        {s.mode === 'detect'
          ? <button onClick={() => app.stopDetect()} data-testid="stop">Stop</button>
          : <button class="primary" disabled={!s.active || s.mode === 'teach'} onClick={() => app.startDetect()} data-testid="start">Detect</button>}
        <button disabled={s.mode !== 'detect'} class={s.frozen ? 'on' : ''} onClick={() => app.toggleFreeze()} data-testid="freeze">
          {s.frozen ? 'Unfreeze' : 'Freeze'}
        </button>
        <label class="ctl compact">
          <span class="small">Heatmap opacity</span>
          <input type="range" min="0" max="1" step="0.05" value={s.opacity} onInput={(e) => app.setOpacity(Number((e.target as HTMLInputElement).value))} />
        </label>
      </div>
      {r && (
        <div class="timings small muted">
          pre {r.timings.preprocess.toFixed(0)} · backbone {r.timings.backbone.toFixed(0)} · kNN {r.timings.knn.toFixed(0)} · post {r.timings.post.toFixed(0)} ms
        </div>
      )}
    </div>
  );
}

export function Operator() {
  const [drag, setDrag] = useState(false);
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDrag(false);
    const files = [...(e.dataTransfer?.files ?? [])];
    const profiles = files.filter((f) => f.name.toLowerCase().endsWith('.pcprof'));
    const images = files.filter((f) => f.type.startsWith('image/'));
    if (profiles.length) app.importProfiles(profiles);
    else if (images.length) app.openImages(images).catch(() => undefined);
  };
  return (
    <main
      class={`operator ${drag ? 'dragging' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={onDrop}
    >
      <section class="live">
        <LiveView editable />
        <LiveBar />
      </section>
      <aside class="panels">
        <TeachPanel />
        <ProfilePanel />
        <SourcePanel />
        <ModelPanel />
        <BoothPanel />
      </aside>
    </main>
  );
}
