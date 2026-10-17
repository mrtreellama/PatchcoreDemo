export interface Roi {
  /** Normalised to the source frame: x, y, w, h in 0..1. Square in source pixels. */
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface CameraInfo {
  label: string;
  deviceId?: string;
  width: number;
  height: number;
  /** Locked MediaTrack settings (exposureTime, focusDistance, colorTemperature, ...). */
  settings: Record<string, number | string | boolean>;
}

export interface TeachStats {
  frames: number;
  heldOut: number;
  heldScores: number[];
  maxHeld?: number;
  teachPatches: number;
}

export interface ProfileHeader {
  name: string;
  notes: string;
  created: string;
  appVersion: string;
  backboneId: string;
  modelKey: string;
  inputSize: number;
  modelSha256: string;
  D: number;
  M: number;
  bankDtype: 'float32' | 'float16';
  threshold: number;
  margin: number;
  heatFloor: number;
  roi: Roi;
  camera: CameraInfo;
  rig: string;
  teachStats: TeachStats;
  thumbnailBytes: number;
}

export interface Profile {
  id: string;
  header: ProfileHeader;
  bank: Float32Array;
  sqnorm: Float32Array;
  thumbnail?: Blob;
  /** Bundled profiles are read-only; duplicate them to edit. */
  bundled?: boolean;
  updated: number;
}

export function maxHeldOf(h: ProfileHeader): number {
  const s = h.teachStats;
  return s.maxHeld ?? (s.heldScores.length ? Math.max(...s.heldScores) : h.threshold / h.margin);
}

export function newId(): string {
  return crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}
