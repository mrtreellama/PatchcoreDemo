import { useEffect, useRef, useState } from 'preact/hooks';
import * as app from '../app';
import { BASE, store } from '../app';
import { useStore } from './store';
import { LiveView } from './LiveView';
import { useBoothOptions } from './panels';
import { shortDevice } from '../util/device';

const IDLE_MS = 60_000;
const LONG_PRESS_MS = 2000;

function exitBooth() {
  if (document.fullscreenElement) document.exitFullscreen().catch(() => undefined);
  location.hash = '#/operator';
}

/** Keeps the screen awake while booth mode is shown; re-acquires after tab switches. */
function useWakeLock(active: boolean) {
  useEffect(() => {
    if (!active || !('wakeLock' in navigator)) return;
    let lock: WakeLockSentinel | undefined;
    let cancelled = false;
    const acquire = async () => {
      try {
        if (document.visibilityState === 'visible' && !cancelled) lock = await navigator.wakeLock.request('screen');
      } catch {
        /* denied (battery saver etc.) */
      }
    };
    const onVis = () => { if (document.visibilityState === 'visible') acquire(); };
    acquire();
    document.addEventListener('visibilitychange', onVis);
    return () => {
      cancelled = true;
      document.removeEventListener('visibilitychange', onVis);
      lock?.release().catch(() => undefined);
    };
  }, [active]);
}

export function Booth() {
  const s = useStore(store, (st) => ({
    result: st.result, mode: st.mode, fps: st.fps, init: st.init, device: st.device,
    active: st.profiles.find((p) => p.id === st.activeId), snapshots: st.snapshots,
    lastChange: st.lastSceneChange, engine: st.engine,
  }));
  const [opts] = useBoothOptions();
  const [started, setStarted] = useState(!!document.fullscreenElement);
  const [attractIdx, setAttractIdx] = useState<number | null>(null);
  const [refThumb, setRefThumb] = useState<string>();
  const pressTimer = useRef<number | undefined>(undefined);
  useWakeLock(started);

  // Detection always runs in booth mode.
  useEffect(() => {
    if (s.active && s.mode === 'idle' && s.engine === 'ready') app.startDetect();
  }, [s.active?.id, s.mode, s.engine]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 'e') exitBooth();
    };
    const onFs = () => setStarted(!!document.fullscreenElement);
    window.addEventListener('keydown', onKey);
    document.addEventListener('fullscreenchange', onFs);
    return () => {
      window.removeEventListener('keydown', onKey);
      document.removeEventListener('fullscreenchange', onFs);
    };
  }, []);

  useEffect(() => {
    if (!s.active?.thumbnail) return setRefThumb(undefined);
    const u = URL.createObjectURL(s.active.thumbnail);
    setRefThumb(u);
    return () => URL.revokeObjectURL(u);
  }, [s.active?.thumbnail]);

  // Idle attract loop: cycle through recent example detections.
  useEffect(() => {
    if (!opts.attract) return setAttractIdx(null);
    const id = setInterval(() => {
      const st = store.state;
      const idle = performance.now() - st.lastSceneChange > IDLE_MS;
      if (idle && st.snapshots.length) setAttractIdx((i) => ((i ?? -1) + 1) % st.snapshots.length);
      else setAttractIdx(null);
    }, 4000);
    return () => clearInterval(id);
  }, [opts.attract]);

  const start = async () => {
    try {
      await document.documentElement.requestFullscreen({ navigationUI: 'hide' });
    } catch {
      /* fullscreen not permitted; keep going windowed */
    }
    setStarted(true);
  };

  const pressStart = () => { pressTimer.current = window.setTimeout(exitBooth, LONG_PRESS_MS); };
  const pressEnd = () => clearTimeout(pressTimer.current);

  const verdict = s.mode === 'detect' ? s.result?.verdict : undefined;
  const attract = attractIdx !== null ? s.snapshots[attractIdx] : undefined;
  return (
    <div class={`booth ${verdict === 'DEFECT' ? 'is-defect' : verdict === 'PASS' ? 'is-pass' : ''}`} data-testid="booth">
      <header class="booth-head">
        <img src={`${BASE}brand/accudyne-logo.svg`} alt="Accudyne Systems" class="booth-logo"
          onPointerDown={pressStart} onPointerUp={pressEnd} onPointerLeave={pressEnd} onContextMenu={(e) => e.preventDefault()}
          draggable={false} />
        <div class="booth-title">
          <div>Composite Defect Detection</div>
          <div class="booth-profile">{s.active?.header.name ?? 'No profile selected'}</div>
        </div>
      </header>
      <main class="booth-main">
        {opts.sideBySide && refThumb && (
          <figure class="booth-ref">
            <img src={refThumb} alt="Good reference" />
            <figcaption>Good reference</figcaption>
          </figure>
        )}
        <div class="booth-live">
          <LiveView maxHeight="62vh" showVerdict={false} />
          {attract && (
            <div class="attract">
              <img src={attract} alt="Example detection" />
              <span>Example detection · place a sample under the camera</span>
            </div>
          )}
        </div>
      </main>
      <div class={`banner ${verdict === 'DEFECT' ? 'bad' : verdict === 'PASS' ? 'good' : 'none'}`} data-testid="booth-verdict">
        {verdict ?? (s.active ? 'STARTING…' : 'SELECT A PROFILE')}
      </div>
      <footer class="booth-foot">
        <span>{shortDevice(s.device)}</span>
        <span>{s.init?.ep.toUpperCase() ?? '…'}{s.init?.ep === 'wasm' ? ` ×${s.init.threads}` : ''}</span>
        <span>{s.fps.toFixed(1)} fps</span>
        <span class="muted">PatchCore anomaly detection · runs entirely in this browser</span>
      </footer>
      {!started && (
        <button class="booth-start" onClick={start}>Tap to start full-screen booth display</button>
      )}
    </div>
  );
}
