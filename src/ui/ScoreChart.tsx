import { useEffect, useRef } from 'preact/hooks';
import { store } from '../app';
import { useStore } from './store';

/** Live sparkline of image score against threshold. */
export function ScoreChart({ height = 72 }: { height?: number }) {
  const history = useStore(store, (s) => s.history);
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = devicePixelRatio || 1;
    const w = c.clientWidth;
    c.width = Math.round(w * dpr);
    c.height = Math.round(height * dpr);
    const ctx = c.getContext('2d')!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, height);
    const css = getComputedStyle(c);
    const line = css.getPropertyValue('--chart-line').trim() || '#7aa2ff';
    const thr = css.getPropertyValue('--defect').trim() || '#e5484d';
    const grid = css.getPropertyValue('--border').trim() || '#333';
    if (!history.length) return;
    const n = 150;
    const top = Math.max(...history.map((h) => Math.max(h.s, h.t * 1.5)));
    const y = (v: number) => height - 4 - (v / top) * (height - 8);
    const x = (i: number) => (i + n - history.length) * (w / (n - 1));
    ctx.strokeStyle = grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height - 0.5);
    ctx.lineTo(w, height - 0.5);
    ctx.stroke();
    // Threshold (dashed)
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = thr;
    ctx.beginPath();
    history.forEach((h, i) => (i ? ctx.lineTo(x(i), y(h.t)) : ctx.moveTo(x(i), y(h.t))));
    ctx.stroke();
    ctx.setLineDash([]);
    // Score
    ctx.strokeStyle = line;
    ctx.lineWidth = 2;
    ctx.beginPath();
    history.forEach((h, i) => (i ? ctx.lineTo(x(i), y(h.s)) : ctx.moveTo(x(i), y(h.s))));
    ctx.stroke();
    const last = history[history.length - 1];
    ctx.fillStyle = last.s > last.t ? thr : line;
    ctx.beginPath();
    ctx.arc(x(history.length - 1), y(last.s), 3, 0, Math.PI * 2);
    ctx.fill();
  }, [history, height]);

  return <canvas ref={ref} class="chart" style={{ height: `${height}px` }} aria-label="Image score vs threshold" />;
}
