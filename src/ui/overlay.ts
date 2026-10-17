import { renderOverlay } from '../patchcore/heatmap';
import type { FrameSource } from '../camera/source';
import { roiPixels } from '../camera/roi';
import type { Roi } from '../profiles/types';
import type { LiveResult } from '../app';

/** Draw the colour-mapped anomaly map into a canvas sized to the map. */
export function drawOverlay(canvas: HTMLCanvasElement, r: LiveResult): void {
  if (canvas.width !== r.size || canvas.height !== r.size) {
    canvas.width = r.size;
    canvas.height = r.size;
  }
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(r.size, r.size);
  renderOverlay(r.map, r.size, r.heatFloor, r.threshold, img.data);
  ctx.putImageData(img, 0, 0);
}

/** Frame crop + heatmap composited into a JPEG object URL (booth attract loop). */
export async function composeSnapshot(src: FrameSource, roi: Roi, r: LiveResult, out = 480): Promise<string | undefined> {
  try {
    const { sx, sy, sw, sh } = roiPixels(roi, src.width(), src.height());
    const c = document.createElement('canvas');
    c.width = c.height = out;
    const ctx = c.getContext('2d')!;
    ctx.drawImage(src.el, sx, sy, sw, sh, 0, 0, out, out);
    const heat = document.createElement('canvas');
    drawOverlay(heat, r);
    ctx.globalAlpha = 0.75;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(heat, 0, 0, out, out);
    const blob = await new Promise<Blob | null>((res) => c.toBlob(res, 'image/jpeg', 0.85));
    return blob ? URL.createObjectURL(blob) : undefined;
  } catch {
    return undefined;
  }
}
