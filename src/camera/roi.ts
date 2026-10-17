import type { Roi } from '../profiles/types';

/** Centered square, 80% of the shorter side, normalised to a W x H frame. */
export function defaultRoi(W: number, H: number): Roi {
  const side = 0.8 * Math.min(W, H);
  return { x: (W - side) / 2 / W, y: (H - side) / 2 / H, w: side / W, h: side / H };
}

/** Keep the ROI square in source pixels and inside the frame. */
export function fitRoi(roi: Roi, W: number, H: number): Roi {
  let side = Math.min(roi.w * W, roi.h * H, W, H);
  side = Math.max(side, Math.min(32, W, H));
  const cx = (roi.x + roi.w / 2) * W;
  const cy = (roi.y + roi.h / 2) * H;
  const x = Math.min(Math.max(cx - side / 2, 0), W - side);
  const y = Math.min(Math.max(cy - side / 2, 0), H - side);
  return { x: x / W, y: y / H, w: side / W, h: side / H };
}

export function roiPixels(roi: Roi, W: number, H: number) {
  const sx = Math.round(roi.x * W);
  const sy = Math.round(roi.y * H);
  const side = Math.max(1, Math.round(Math.min(roi.w * W, roi.h * H)));
  return { sx, sy, sw: Math.min(side, W - sx), sh: Math.min(side, H - sy) };
}
