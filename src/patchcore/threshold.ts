export const DEFAULT_MARGIN = 1.15;
export const MIN_FRAMES = 5;
export const MAX_FRAMES = 50;
export const DEFAULT_FRAMES = 15;

/** Last 20% of teach frames (at least 2) are held out for calibration. */
export function holdoutCount(frames: number): number {
  return Math.max(2, Math.round(frames * 0.2));
}

/** Threshold = highest held-out image score x margin. */
export function thresholdFrom(maxHeldScore: number, margin = DEFAULT_MARGIN): number {
  return maxHeldScore * margin;
}
