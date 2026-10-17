/**
 * Image Capture constraints exposed by MediaStreamTrack.getCapabilities():
 * show only what this camera supports, and lock auto modes to manual so the
 * image stays consistent between teaching and detecting.
 */

type Caps = MediaTrackCapabilities & Record<string, unknown>;
type Settings = MediaTrackSettings & Record<string, unknown>;

export interface RangeControl {
  kind: 'range';
  name: string;
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
}

export interface ModeControl {
  kind: 'mode';
  name: string;
  label: string;
  options: string[];
  value: string;
}

export interface ToggleControl {
  kind: 'toggle';
  name: string;
  label: string;
  value: boolean;
}

export type CameraControl = RangeControl | ModeControl | ToggleControl;

const RANGES: [string, string][] = [
  ['exposureTime', 'Exposure time'],
  ['exposureCompensation', 'Exposure compensation'],
  ['focusDistance', 'Focus distance'],
  ['colorTemperature', 'Colour temperature (K)'],
  ['brightness', 'Brightness'],
  ['zoom', 'Zoom'],
];
const MODES: [string, string][] = [
  ['exposureMode', 'Exposure mode'],
  ['focusMode', 'Focus mode'],
  ['whiteBalanceMode', 'White balance'],
];

/** [mode constraint, value constraint, human name] for the Lock button. */
const LOCKABLE: [string, string, string][] = [
  ['exposureMode', 'exposureTime', 'Exposure'],
  ['focusMode', 'focusDistance', 'Focus'],
  ['whiteBalanceMode', 'colorTemperature', 'White balance'],
];

export const PERSISTED_SETTINGS = [
  'exposureMode', 'exposureTime', 'exposureCompensation', 'focusMode', 'focusDistance',
  'whiteBalanceMode', 'colorTemperature', 'brightness', 'zoom',
];

function caps(track: MediaStreamTrack): Caps {
  return (track.getCapabilities?.() ?? {}) as Caps;
}

export function listControls(track: MediaStreamTrack): CameraControl[] {
  const c = caps(track);
  const s = track.getSettings() as Settings;
  const out: CameraControl[] = [];
  for (const [name, label] of MODES) {
    const opts = c[name];
    if (Array.isArray(opts) && opts.length) out.push({ kind: 'mode', name, label, options: opts as string[], value: String(s[name] ?? opts[0]) });
  }
  for (const [name, label] of RANGES) {
    const r = c[name] as { min?: number; max?: number; step?: number } | undefined;
    if (r && typeof r.min === 'number' && typeof r.max === 'number' && r.max > r.min) {
      out.push({ kind: 'range', name, label, min: r.min, max: r.max, step: r.step || (r.max - r.min) / 100, value: Number(s[name] ?? r.min) });
    }
  }
  if (c.torch) out.push({ kind: 'toggle', name: 'torch', label: 'Torch', value: Boolean(s.torch) });
  return out;
}

export async function applyControl(track: MediaStreamTrack, name: string, value: number | string | boolean): Promise<void> {
  // Setting a manual value implies the matching manual mode.
  const extra: Record<string, unknown> = {};
  for (const [mode, val] of LOCKABLE) if (val === name) extra[mode] = 'manual';
  await track.applyConstraints({ advanced: [{ ...extra, [name]: value } as MediaTrackConstraintSet] });
}

export interface LockResult {
  locked: string[];
  failed: string[];
  settings: Record<string, number | string | boolean>;
}

/** Switch every supported auto control to manual at its current value. */
export async function lockCamera(track: MediaStreamTrack): Promise<LockResult> {
  const c = caps(track);
  const locked: string[] = [];
  const failed: string[] = [];
  for (const [mode, val, human] of LOCKABLE) {
    const modes = c[mode];
    if (!Array.isArray(modes) || !modes.includes('manual')) {
      failed.push(`${human}: not supported by this camera`);
      continue;
    }
    const s = track.getSettings() as Settings;
    const set: Record<string, unknown> = { [mode]: 'manual' };
    if (s[val] !== undefined) set[val] = s[val];
    try {
      await track.applyConstraints({ advanced: [set as MediaTrackConstraintSet] });
      const after = track.getSettings() as Settings;
      if (after[mode] === 'manual') locked.push(human);
      else failed.push(`${human}: camera ignored the request`);
    } catch (e) {
      failed.push(`${human}: ${(e as Error).message}`);
    }
  }
  return { locked, failed, settings: snapshotSettings(track) };
}

export function snapshotSettings(track: MediaStreamTrack): Record<string, number | string | boolean> {
  const s = track.getSettings() as Settings;
  const out: Record<string, number | string | boolean> = {};
  for (const k of PERSISTED_SETTINGS) {
    const v = s[k];
    if (typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean') out[k] = v;
  }
  return out;
}

/** Re-apply settings saved in a profile; returns the names that failed. */
export async function reapplySettings(track: MediaStreamTrack, settings: Record<string, number | string | boolean>): Promise<string[]> {
  const c = caps(track);
  const failed: string[] = [];
  // Modes first so manual values are accepted.
  const keys = Object.keys(settings).sort((a, b) => Number(!a.endsWith('Mode')) - Number(!b.endsWith('Mode')));
  for (const k of keys) {
    if (!(k in c)) continue;
    try {
      await track.applyConstraints({ advanced: [{ [k]: settings[k] } as MediaTrackConstraintSet] });
    } catch {
      failed.push(k);
    }
  }
  return failed;
}
