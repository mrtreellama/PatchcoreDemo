import { useEffect, useRef } from 'preact/hooks';
import { getSource, setRoi, store } from '../app';
import { useStore } from './store';
import { drawOverlay } from './overlay';
import type { Roi } from '../profiles/types';

interface Props {
  editable?: boolean;
  /** CSS max-height for the stage, e.g. "70vh". */
  maxHeight?: string;
  showVerdict?: boolean;
}

export function LiveView({ editable = false, maxHeight = '68vh', showVerdict = true }: Props) {
  const s = useStore(store, (st) => ({
    tick: st.sourceTick, size: st.sourceSize, roi: st.roi, result: st.result,
    mode: st.mode, opacity: st.opacity, frozen: st.frozen,
  }));
  const mediaRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const heatRef = useRef<HTMLCanvasElement>(null);
  const [W, H] = s.size;

  // Mount the source element (video or canvas) owned by the controller.
  useEffect(() => {
    const el = getSource()?.el;
    const host = mediaRef.current;
    if (!el || !host) return;
    if (el.parentElement !== host) {
      host.replaceChildren(el);
      if (el instanceof HTMLVideoElement && el.paused) el.play().catch(() => undefined);
    }
  }, [s.tick]);

  useEffect(() => {
    const c = heatRef.current;
    if (!c) return;
    if (s.result && s.mode === 'detect') drawOverlay(c, s.result);
    else c.getContext('2d')?.clearRect(0, 0, c.width, c.height);
  }, [s.result, s.mode]);

  const onPointerDown = (e: PointerEvent, kind: 'move' | 'resize') => {
    if (!editable || !stageRef.current) return;
    e.preventDefault();
    e.stopPropagation();
    const rect = stageRef.current.getBoundingClientRect();
    const start = { x: e.clientX, y: e.clientY, roi: { ...store.state.roi } };
    const target = e.currentTarget as HTMLElement;
    target.setPointerCapture(e.pointerId);
    const move = (ev: PointerEvent) => {
      const dx = (ev.clientX - start.x) / rect.width;
      const dy = (ev.clientY - start.y) / rect.height;
      let roi: Roi;
      if (kind === 'move') {
        roi = { ...start.roi, x: start.roi.x + dx, y: start.roi.y + dy };
      } else {
        // Grow the square by the larger drag component, measured in screen pixels.
        const dpx = Math.max(ev.clientX - start.x, ev.clientY - start.y);
        const side = start.roi.w * rect.width + dpx;
        roi = { ...start.roi, w: side / rect.width, h: side / rect.height };
      }
      setRoi(roi);
    };
    const up = () => {
      target.removeEventListener('pointermove', move);
      target.removeEventListener('pointerup', up);
      target.removeEventListener('pointercancel', up);
    };
    target.addEventListener('pointermove', move);
    target.addEventListener('pointerup', up);
    target.addEventListener('pointercancel', up);
  };

  const r = s.roi;
  const verdict = s.mode === 'detect' ? s.result?.verdict : undefined;
  const aspect = W && H ? W / H : 16 / 9;
  return (
    <div class="stage-wrap">
      <div
        ref={stageRef}
        class="stage"
        style={{ aspectRatio: `${aspect}`, width: `min(100%, calc(${maxHeight} * ${aspect}))` }}
      >
        <div ref={mediaRef} class="media" />
        {!W && <div class="stage-empty">No source. Pick a camera, images or a video.</div>}
        <div
          class={`roi ${editable ? 'editable' : ''} ${verdict === 'DEFECT' ? 'defect' : verdict === 'PASS' ? 'pass' : ''}`}
          style={{ left: `${r.x * 100}%`, top: `${r.y * 100}%`, width: `${r.w * 100}%`, height: `${r.h * 100}%` }}
          onPointerDown={(e) => onPointerDown(e, 'move')}
          data-testid="roi"
        >
          <canvas ref={heatRef} class="heat" style={{ opacity: s.opacity }} />
          {editable && <div class="roi-handle" onPointerDown={(e) => onPointerDown(e, 'resize')} title="Resize" />}
          {showVerdict && verdict && (
            <span class={`chip ${verdict === 'DEFECT' ? 'bad' : 'good'}`} data-testid="roi-verdict">{verdict}</span>
          )}
          {s.frozen && <span class="chip frozen">FROZEN</span>}
        </div>
      </div>
    </div>
  );
}
