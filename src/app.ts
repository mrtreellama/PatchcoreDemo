/**
 * Application controller: owns the inference worker, the frame source, the
 * profile list and the detect / teach loops. UI components read `app.store`
 * and call the functions exported here.
 */
import { InferenceClient } from './worker/client';
import type {
  EpChoice, InitResult, LoadModelResult, ModelEntry, ModelManifest, ScoreResult, StageTimings,
} from './worker/protocol';
import { Store } from './ui/store';
import { Hysteresis, type Verdict } from './patchcore/hysteresis';
import { DEFAULT_FRAMES, DEFAULT_MARGIN, MAX_FRAMES, MIN_FRAMES, thresholdFrom } from './patchcore/threshold';
import { CameraSource, ImageSource, VideoSource, listCameras, type FrameSource, type SourceKind } from './camera/source';
import { defaultRoi, fitRoi } from './camera/roi';
import { lockCamera as lockTrack, reapplySettings, snapshotSettings, type LockResult } from './camera/controls';
import { deleteProfile, getSetting, listProfiles, putProfile, setSetting } from './profiles/store';
import { loadBundledProfiles } from './profiles/bundled';
import { decodePcprof, encodePcprof } from './profiles/pcprof';
import { maxHeldOf, newId, type Profile, type ProfileHeader, type Roi } from './profiles/types';
import { deviceInfo, type DeviceInfo } from './util/device';
import { FpsMeter, nextFrame, sleep } from './util/stats';
import { precache, waitForController, type CoiStatus } from './util/coi';
import { composeSnapshot } from './ui/overlay';

export const BASE = import.meta.env.BASE_URL;
export const APP_VERSION = __APP_VERSION__;
export const DEFAULT_MODEL = 'r18_224';

export type Mode = 'idle' | 'detect' | 'teach';
export type Resolution = '720p' | '1080p';

export interface TeachFrame {
  id: string;
  thumb: string;
}

export interface LiveResult {
  map: Float32Array;
  size: number;
  score: number;
  threshold: number;
  heatFloor: number;
  verdict: Verdict;
  raw: boolean;
  timings: StageTimings;
}

export interface Notice {
  text: string;
  kind: 'info' | 'warn' | 'error';
}

export interface AppState {
  manifest?: ModelManifest;
  device?: DeviceInfo;
  coi: CoiStatus;
  /** Startup finished (model loaded, source open, saved profile restored). */
  booted: boolean;
  ep: EpChoice;
  modelKey: string;
  init?: InitResult;
  model?: LoadModelResult;
  engine: 'starting' | 'loading' | 'ready' | 'error';
  lastError?: string;
  notice?: Notice;
  profiles: Profile[];
  activeId?: string;
  sourceKind?: SourceKind;
  sourceLabel: string;
  sourceSize: [number, number];
  sourceTick: number;
  cameras: MediaDeviceInfo[];
  cameraId?: string;
  resolution: Resolution;
  roi: Roi;
  mode: Mode;
  frozen: boolean;
  result?: LiveResult;
  history: { s: number; t: number }[];
  fps: number;
  opacity: number;
  teach: {
    frames: TeachFrame[];
    target: number;
    auto: boolean;
    busy: boolean;
    stage?: string;
    progress?: number;
    name: string;
    notes: string;
    addTo?: string;
  };
  rig: string;
  lock?: LockResult;
  snapshots: string[];
  lastSceneChange: number;
  precache?: string;
  /** Shell, current model and samples are in the service-worker cache. */
  offlineReady: boolean;
}

export const store = new Store<AppState>({
  coi: 'not-isolated',
  booted: false,
  ep: 'auto',
  modelKey: DEFAULT_MODEL,
  engine: 'starting',
  profiles: [],
  sourceLabel: 'No source',
  sourceSize: [0, 0],
  sourceTick: 0,
  cameras: [],
  resolution: '720p',
  roi: { x: 0.1, y: 0.1, w: 0.8, h: 0.8 },
  mode: 'idle',
  frozen: false,
  history: [],
  fps: 0,
  opacity: 0.75,
  teach: { frames: [], target: DEFAULT_FRAMES, auto: false, busy: false, name: '', notes: '' },
  rig: '',
  snapshots: [],
  offlineReady: false,
  lastSceneChange: performance.now(),
});

const S = () => store.state;
let client: InferenceClient | undefined;
let source: FrameSource | undefined;
const hysteresis = new Hysteresis();
const fpsMeter = new FpsMeter();
let loopToken = 0;
let lastSig: Float32Array | undefined;
let teachThumbForProfile: Blob | undefined;

export const getSource = () => source;
export const activeProfile = (): Profile | undefined => S().profiles.find((p) => p.id === S().activeId);
export const currentEntry = (): ModelEntry | undefined => S().manifest?.models.find((m) => m.key === S().modelKey);

export function notify(text: string, kind: Notice['kind'] = 'info') {
  store.set({ notice: { text, kind } });
  if (kind === 'error') store.set({ lastError: text });
}

export function clearNotice() {
  store.set({ notice: undefined });
}

function fail(e: unknown, prefix = ''): never {
  const msg = `${prefix}${e instanceof Error ? e.message : String(e)}`;
  notify(msg, 'error');
  throw e instanceof Error ? e : new Error(msg);
}

// ---- Startup ------------------------------------------------------------------

export async function start(coi: CoiStatus): Promise<void> {
  store.set({ coi });
  const [manifest, ep, modelKey, opacity, rig, resolution, cameraId] = await Promise.all([
    fetch(`${BASE}models/models.json`).then((r) => r.json() as Promise<ModelManifest>),
    getSetting<EpChoice>('ep', 'auto'),
    getSetting('modelKey', DEFAULT_MODEL),
    getSetting('opacity', 0.75),
    getSetting('rig', ''),
    getSetting<Resolution>('resolution', '720p'),
    getSetting<string | undefined>('cameraId', undefined),
  ]);
  const validKey = manifest.models.some((m) => m.key === modelKey) ? modelKey : DEFAULT_MODEL;
  store.set({ manifest, ep, modelKey: validKey, opacity, rig, resolution, cameraId });
  deviceInfo().then((device) => store.set({ device }));
  await refreshProfiles();
  await startEngine();

  const sourcePref = await getSetting<SourceKind | 'samples'>('source', 'samples');
  if (sourcePref === 'camera') {
    await openCamera(cameraId).catch(() => openSamples());
  } else {
    await openSamples();
  }
  listCameras().then((cameras) => store.set({ cameras }));

  const activeId = await getSetting<string | undefined>('activeId', undefined);
  const fallback = S().profiles.find((p) => p.bundled)?.id;
  const id = S().profiles.some((p) => p.id === activeId) ? activeId : fallback;
  if (id) await activateProfile(id).catch(() => undefined);
  store.set({ booted: true });
  void warmOfflineCache();
}

async function startEngine(): Promise<void> {
  loopToken++;
  client?.terminate();
  store.set({ engine: 'starting', init: undefined, model: undefined });
  client = new InferenceClient();
  client.onProgress = (e) => store.set((s) => ({
    teach: { ...s.teach, stage: e.stage === 'coreset' ? 'Selecting coreset' : 'Calibrating threshold', progress: e.done / e.total },
  }));
  try {
    const init = await client.init(S().ep, `${BASE}ort/`);
    store.set({ init });
    if (init.fallbackReason) notify(init.fallbackReason, 'warn');
    await loadModel(S().modelKey);
  } catch (e) {
    store.set({ engine: 'error' });
    fail(e, 'Inference engine failed to start: ');
  }
}

async function loadModel(key: string): Promise<void> {
  const entry = S().manifest?.models.find((m) => m.key === key);
  if (!entry || !client) throw new Error(`Unknown model ${key}`);
  store.set({ engine: 'loading', modelKey: key });
  const v = entry.sha256.slice(0, 12);
  const knn = S().manifest!.knn;
  const model = await client.loadModel(entry, `${BASE}models/${entry.file}?v=${v}`, `${BASE}models/${knn.file}?v=${knn.sha256.slice(0, 12)}`);
  store.set({ model, engine: 'ready', init: { ...S().init!, ep: model.ep, fallbackReason: model.fallbackReason } });
  if (!model.shaOk) notify(`Model file hash mismatch for ${entry.file}: the download may be corrupt or stale`, 'error');
  else if (model.fallbackReason && model.fallbackReason !== S().init?.fallbackReason) notify(model.fallbackReason, 'warn');
}

export async function setEp(ep: EpChoice) {
  const wasDetecting = S().mode === 'detect';
  store.set({ ep, mode: 'idle' });
  await setSetting('ep', ep);
  await startEngine();
  await rebindProfile(wasDetecting);
}

export async function setModel(key: string) {
  if (key === S().modelKey && S().engine === 'ready') return;
  const wasDetecting = S().mode === 'detect';
  loopToken++;
  store.set({ mode: 'idle' });
  await setSetting('modelKey', key);
  try {
    await loadModel(key);
  } catch (e) {
    fail(e, 'Could not load model: ');
  }
  await client?.teachReset();
  clearTeachFrames();
  await rebindProfile(wasDetecting);
}

/** After a model or EP change, re-send the bank if the active profile still fits. */
async function rebindProfile(resume: boolean) {
  const p = activeProfile();
  if (!p || !client) return;
  const entry = currentEntry();
  if (entry?.sha256 !== p.header.modelSha256) {
    store.set({ activeId: undefined, result: undefined });
    notify(`Profile "${p.header.name}" was taught with ${p.header.modelKey}; select it again to switch back`, 'warn');
    return;
  }
  await client.setBank(p.bank, p.sqnorm, p.header.D);
  if (resume) startDetect();
}

// ---- Sources ------------------------------------------------------------------

function setSource(src: FrameSource) {
  if (source && source !== src) source.stop();
  source = src;
  const W = src.width();
  const H = src.height();
  const p = activeProfile();
  const roi = p ? fitRoi(p.header.roi, W, H) : defaultRoi(W, H);
  store.set((s) => ({ sourceKind: src.kind, sourceLabel: src.label, sourceSize: [W, H], roi, sourceTick: s.sourceTick + 1 }));
}

export async function openCamera(deviceId?: string, resolution = S().resolution) {
  const cam = new CameraSource();
  const [width, height] = resolution === '1080p' ? [1920, 1080] : [1280, 720];
  try {
    await cam.open({ deviceId, width, height });
  } catch (e) {
    const name = (e as Error).name;
    if (!deviceId || (name !== 'OverconstrainedError' && name !== 'NotFoundError')) fail(e, 'Camera unavailable: ');
    // The remembered camera is gone or got a new id (re-plugged): take any camera.
    try {
      await cam.open({ width, height });
      notify('The previously used camera was not found; using the default camera', 'warn');
    } catch (e2) {
      fail(e2, 'Camera unavailable: ');
    }
  }
  const id = cam.track?.getSettings().deviceId ?? deviceId;
  setSource(cam);
  store.set({ cameraId: id, resolution, lock: undefined });
  await Promise.all([setSetting('source', 'camera'), setSetting('cameraId', id), setSetting('resolution', resolution)]);
  store.set({ cameras: await listCameras() });
  const p = activeProfile();
  if (p) await applyProfileCamera(p);
}

export async function openImages(files: File[]) {
  const src = await ImageSource.fromFiles(files).catch((e) => fail(e));
  src.onChange = () => store.set((s) => ({ sourceTick: s.sourceTick + 1, sourceSize: [src.width(), src.height()] }));
  setSource(src);
  await setSetting('source', 'images');
}

export async function openSamples() {
  const r = await fetch(`${BASE}samples/samples.json`);
  const { images } = (await r.json()) as { images: { file: string; role: string }[] };
  // Test images first so detection has something interesting to show.
  const ordered = [...images.filter((i) => i.role === 'test'), ...images.filter((i) => i.role !== 'test')];
  const src = await ImageSource.fromUrls(ordered.map((i) => `${BASE}samples/${i.file}`));
  src.label = 'Sample images (carbon twill)';
  src.onChange = () => store.set((s) => ({ sourceTick: s.sourceTick + 1 }));
  setSource(src);
  await setSetting('source', 'samples');
}

export async function openVideo(file: File) {
  const src = new VideoSource(file);
  await src.open().catch((e) => fail(e, 'Could not play video: '));
  setSource(src);
}

export function stepImage(delta: number) {
  if (source instanceof ImageSource) {
    source.show(source.index + delta);
    hysteresis.reset();
  }
}

export function setRoi(roi: Roi) {
  const [W, H] = S().sourceSize;
  store.set({ roi: fitRoi(roi, W || 1, H || 1) });
}

export function resetRoi() {
  const [W, H] = S().sourceSize;
  store.set({ roi: defaultRoi(W || 1, H || 1) });
}

export async function lockCamera() {
  if (!(source instanceof CameraSource) || !source.track) return notify('Lock needs a live camera', 'warn');
  const lock = await lockTrack(source.track);
  store.set({ lock });
  if (lock.failed.length) notify(`Could not lock: ${lock.failed.join('; ')}`, 'warn');
  else notify(`Locked ${lock.locked.join(', ')}`);
}

async function applyProfileCamera(p: Profile) {
  if (!(source instanceof CameraSource) || !source.track) return;
  const cam = p.header.camera;
  if (cam.label && cam.label !== source.label) {
    notify(`Profile was taught on "${cam.label}", current camera is "${source.label}". Results are only valid under the camera and lighting it was taught with.`, 'warn');
    return;
  }
  if (Object.keys(cam.settings).length) {
    const failed = await reapplySettings(source.track, cam.settings);
    if (failed.length) notify(`Could not re-apply camera settings: ${failed.join(', ')}`, 'warn');
  }
}

export function setRig(rig: string) {
  store.set({ rig });
  setSetting('rig', rig);
}

export function setOpacity(opacity: number) {
  store.set({ opacity });
  setSetting('opacity', opacity);
}

// ---- Detection loop ------------------------------------------------------------

export function startDetect() {
  if (S().mode === 'teach') return;
  if (!activeProfile()) return notify('Select or teach a profile first', 'warn');
  hysteresis.reset();
  fpsMeter.reset();
  store.set({ mode: 'detect', frozen: false, history: [] });
  const token = ++loopToken;
  void detectLoop(token);
}

export function stopDetect() {
  loopToken++;
  store.set({ mode: 'idle', frozen: false });
}

export function toggleFreeze() {
  store.set((s) => ({ frozen: !s.frozen }));
}

async function detectLoop(token: number) {
  while (token === loopToken) {
    const p = activeProfile();
    if (S().frozen || !source?.ready() || !client || !p || S().engine !== 'ready') {
      await sleep(50);
      continue;
    }
    try {
      const t0 = performance.now();
      const bmp = await source.grab(S().roi, p.header.inputSize);
      const grabMs = performance.now() - t0;
      const r = await client.score(bmp);
      if (token !== loopToken || S().frozen) continue;
      r.timings.preprocess += grabMs;
      r.timings.total += grabMs;
      onResult(r, activeProfile() ?? p);
    } catch (e) {
      if (token === loopToken) notify(`Detection error: ${(e as Error).message}`, 'error');
      await sleep(500);
    }
    await nextFrame();
  }
}

function onResult(r: ScoreResult, p: Profile) {
  const threshold = p.header.threshold;
  const prev = hysteresis.state;
  const verdict = hysteresis.update(r.imageScore, threshold);
  const fps = fpsMeter.tick();
  const now = performance.now();
  let lastSceneChange = S().lastSceneChange;
  if (lastSig) {
    let diff = 0;
    for (let i = 0; i < r.sig.length; i++) diff += Math.abs(r.sig[i] - lastSig[i]);
    if (diff / r.sig.length > 0.02) lastSceneChange = now;
  }
  lastSig = r.sig;
  const result: LiveResult = {
    map: r.map, size: r.size, score: r.imageScore, threshold, heatFloor: p.header.heatFloor,
    verdict, raw: r.imageScore > threshold, timings: r.timings,
  };
  store.set((s) => ({
    result, fps, lastSceneChange,
    history: [...s.history.slice(-149), { s: r.imageScore, t: threshold }],
  }));
  if (verdict === 'DEFECT' && prev === 'PASS' && source) {
    composeSnapshot(source, S().roi, result).then((url) => {
      if (!url) return;
      store.set((s) => {
        const snaps = [...s.snapshots, url];
        while (snaps.length > 8) URL.revokeObjectURL(snaps.shift()!);
        return { snapshots: snaps };
      });
    });
  }
}

// ---- Teaching -------------------------------------------------------------------

function clearTeachFrames() {
  S().teach.frames.forEach((f) => URL.revokeObjectURL(f.thumb));
  teachThumbForProfile = undefined;
  store.set((s) => ({ teach: { ...s.teach, frames: [], busy: false, stage: undefined, progress: undefined, auto: false } }));
}

export async function beginTeach(addTo?: string) {
  stopDetect();
  await client?.teachReset();
  clearTeachFrames();
  const base = addTo ? S().profiles.find((p) => p.id === addTo) : undefined;
  if (base) {
    if (base.bundled) return notify('Bundled profiles are read-only: duplicate it first', 'warn');
    if (base.header.modelKey !== S().modelKey) await setModel(base.header.modelKey);
  }
  const name = base?.header.name ?? nextProfileName();
  store.set((s) => ({ mode: 'teach', result: undefined, teach: { ...s.teach, addTo, name, notes: base?.header.notes ?? '' } }));
}

function nextProfileName() {
  const names = new Set(S().profiles.map((p) => p.header.name));
  for (let i = 1; ; i++) if (!names.has(`Material ${i}`)) return `Material ${i}`;
}

export function cancelTeach() {
  void client?.teachReset();
  clearTeachFrames();
  store.set((s) => ({ mode: 'idle', teach: { ...s.teach, addTo: undefined } }));
}

export function setTeachField(patch: Partial<AppState['teach']>) {
  store.set((s) => ({ teach: { ...s.teach, ...patch } }));
}

let capturing: Promise<void> = Promise.resolve();

/** Captures are serialised so quick clicks never grab the same frame twice. */
export function captureFrame(): Promise<void> {
  capturing = capturing.then(captureOne, captureOne);
  return capturing;
}

async function captureOne(): Promise<void> {
  const t = S().teach;
  if (!source?.ready() || !client || S().mode !== 'teach') return;
  if (t.frames.length >= MAX_FRAMES) return notify(`Maximum ${MAX_FRAMES} frames`, 'warn');
  const entry = currentEntry();
  if (!entry) return;
  const id = newId();
  const roi = S().roi;
  const [bmp, thumbBmp] = await Promise.all([source.grab(roi, entry.inputSize), source.grab(roi, 128)]);
  // Image-file sources step forward so repeated captures cover the set.
  if (source instanceof ImageSource) source.next();
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  c.getContext('2d')!.drawImage(thumbBmp, 0, 0);
  thumbBmp.close();
  const blob = await new Promise<Blob>((r) => c.toBlob((b) => r(b!), 'image/jpeg', 0.85));
  teachThumbForProfile ??= blob;
  store.set((s) => ({ teach: { ...s.teach, frames: [...s.teach.frames, { id, thumb: URL.createObjectURL(blob) }] } }));
  try {
    await client.teachAdd(id, bmp);
  } catch (e) {
    removeFrame(id);
    fail(e, 'Feature extraction failed: ');
  }
}

export function removeFrame(id: string) {
  const f = S().teach.frames.find((x) => x.id === id);
  if (f) URL.revokeObjectURL(f.thumb);
  store.set((s) => ({ teach: { ...s.teach, frames: s.teach.frames.filter((x) => x.id !== id) } }));
  void client?.teachRemove(id);
}

let autoTimer: number | undefined;
export function setAutoCapture(on: boolean) {
  setTeachField({ auto: on });
  clearInterval(autoTimer);
  if (!on) return;
  let busy = false;
  autoTimer = window.setInterval(async () => {
    const t = S().teach;
    if (!t.auto || S().mode !== 'teach' || t.frames.length >= t.target) {
      clearInterval(autoTimer);
      setTeachField({ auto: false });
      return;
    }
    if (busy) return;
    busy = true;
    await captureFrame().catch(() => undefined);
    busy = false;
  }, 500);
}

export async function captureAllImages() {
  if (!(source instanceof ImageSource)) return;
  source.show(0);
  const n = Math.min(source.images.length, MAX_FRAMES);
  for (let i = 0; i < n; i++) await captureFrame();
}

export async function finishTeach(): Promise<void> {
  const t = S().teach;
  const entry = currentEntry();
  if (!client || !entry) return;
  const base = t.addTo ? S().profiles.find((p) => p.id === t.addTo) : undefined;
  const min = base ? 1 : MIN_FRAMES;
  if (t.frames.length < min) return notify(`Capture at least ${min} good frames`, 'warn');
  setAutoCapture(false);
  setTeachField({ busy: true, stage: 'Selecting coreset', progress: 0 });
  try {
    const margin = base?.header.margin ?? DEFAULT_MARGIN;
    const r = await client.teachFinish({
      margin,
      base: base && { bank: base.bank.slice(), maxHeld: maxHeldOf(base.header), heatFloor: base.header.heatFloor },
    });
    const cam = source instanceof CameraSource ? source : undefined;
    const [W, H] = S().sourceSize;
    const header: ProfileHeader = {
      name: t.name.trim() || nextProfileName(),
      notes: t.notes,
      created: base?.header.created ?? new Date().toISOString(),
      appVersion: APP_VERSION,
      backboneId: entry.id,
      modelKey: entry.key,
      inputSize: entry.inputSize,
      modelSha256: entry.sha256,
      D: r.D,
      M: r.M,
      bankDtype: 'float32',
      threshold: r.threshold,
      margin,
      heatFloor: r.heatFloor,
      roi: S().roi,
      camera: {
        label: source?.label ?? '',
        deviceId: cam?.track?.getSettings().deviceId,
        width: W, height: H,
        settings: cam?.track ? snapshotSettings(cam.track) : {},
      },
      rig: S().rig,
      teachStats: base
        ? { ...base.header.teachStats, frames: base.header.teachStats.frames + r.frames, teachPatches: base.header.teachStats.teachPatches + r.teachPatches }
        : { frames: r.frames, heldOut: r.heldOut, heldScores: r.heldScores, maxHeld: r.maxHeld, teachPatches: r.teachPatches },
      thumbnailBytes: 0,
    };
    const profile: Profile = {
      id: base?.id ?? newId(),
      header,
      bank: r.bank,
      sqnorm: r.sqnorm,
      thumbnail: base?.thumbnail ?? teachThumbForProfile,
      updated: Date.now(),
    };
    await putProfile(profile);
    clearTeachFrames();
    await refreshProfiles();
    store.set((s) => ({ teach: { ...s.teach, addTo: undefined }, activeId: profile.id, mode: 'idle' }));
    await setSetting('activeId', profile.id);
    notify(`${base ? 'Updated' : 'Taught'} "${header.name}": ${r.M.toLocaleString()} bank rows, threshold ${r.threshold.toFixed(3)} (coreset ${(r.coresetMs / 1000).toFixed(1)} s)`);
    hysteresis.reset();
    startDetect();
  } catch (e) {
    setTeachField({ busy: false, stage: undefined });
    fail(e, 'Teaching failed: ');
  }
}

// ---- Profiles ---------------------------------------------------------------------

export async function refreshProfiles() {
  const [local, bundled] = await Promise.all([listProfiles().catch(() => []), loadBundledProfiles().catch(() => [])]);
  store.set({ profiles: [...local, ...bundled] });
}

export function modelForProfile(p: Profile): ModelEntry | undefined {
  return S().manifest?.models.find((m) => m.sha256 === p.header.modelSha256);
}

export async function activateProfile(id: string): Promise<void> {
  const p = S().profiles.find((x) => x.id === id);
  if (!p || !client) return;
  const entry = modelForProfile(p);
  if (!entry) {
    return fail(new Error(`"${p.header.name}" needs backbone ${p.header.backboneId} @ ${p.header.inputSize} (sha ${p.header.modelSha256.slice(0, 12)}…), which this build does not provide`));
  }
  const t0 = performance.now();
  const teaching = S().mode === 'teach';
  if (!teaching) {
    loopToken++;
    store.set({ mode: 'idle', result: undefined });
  }
  if (entry.key !== S().modelKey || S().engine !== 'ready') {
    await setSetting('modelKey', entry.key);
    await loadModel(entry.key);
  }
  await client.setBank(p.bank, p.sqnorm, p.header.D);
  const [W, H] = S().sourceSize;
  store.set({ activeId: id, roi: W ? fitRoi(p.header.roi, W, H) : p.header.roi });
  await setSetting('activeId', id);
  if (p.header.rig && S().rig && p.header.rig !== S().rig) {
    notify(`Profile was taught on rig "${p.header.rig}", this station is "${S().rig}"`, 'warn');
  }
  await applyProfileCamera(p);
  console.info(`Profile switch took ${(performance.now() - t0).toFixed(0)} ms`);
  startDetect();
}

export async function setMargin(margin: number) {
  const p = activeProfile();
  if (!p) return;
  const header = { ...p.header, margin, threshold: thresholdFrom(maxHeldOf(p.header), margin) };
  const updated = { ...p, header };
  store.set((s) => ({ profiles: s.profiles.map((x) => (x.id === p.id ? updated : x)) }));
  hysteresis.reset();
  if (!p.bundled) await putProfile(updated);
}

export async function renameProfile(id: string, name: string) {
  const p = S().profiles.find((x) => x.id === id);
  if (!p || p.bundled || !name.trim()) return;
  await putProfile({ ...p, header: { ...p.header, name: name.trim() } });
  await refreshProfiles();
}

export async function duplicateProfile(id: string) {
  const p = S().profiles.find((x) => x.id === id);
  if (!p) return;
  const copy: Profile = {
    ...p, id: newId(), bundled: false,
    header: { ...p.header, name: `${p.header.name} (copy)` },
    bank: p.bank.slice(), sqnorm: p.sqnorm.slice(),
  };
  await putProfile(copy);
  await refreshProfiles();
  await activateProfile(copy.id);
}

export async function removeProfile(id: string) {
  const p = S().profiles.find((x) => x.id === id);
  if (!p || p.bundled) return;
  await deleteProfile(id);
  if (S().activeId === id) {
    stopDetect();
    store.set({ activeId: undefined, result: undefined });
    await client?.clearBank();
  }
  await refreshProfiles();
}

export async function exportProfile(id: string) {
  const p = S().profiles.find((x) => x.id === id);
  if (!p) return;
  const thumb = p.thumbnail ? new Uint8Array(await p.thumbnail.arrayBuffer()) : new Uint8Array(0);
  const bytes = encodePcprof(p.header, p.bank, thumb, 'float16');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([bytes as BlobPart], { type: 'application/octet-stream' }));
  a.download = `${p.header.name.replace(/[^\w.-]+/g, '_')}.pcprof`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

export async function importProfiles(files: File[]) {
  let last: string | undefined;
  for (const f of files) {
    try {
      const d = decodePcprof(await f.arrayBuffer());
      const p: Profile = {
        id: newId(), header: d.header, bank: d.bank, sqnorm: d.sqnorm,
        thumbnail: d.thumbnail.length ? new Blob([d.thumbnail as BlobPart], { type: 'image/jpeg' }) : undefined,
        updated: Date.now(),
      };
      if (!modelForProfile(p)) {
        notify(`Imported "${d.header.name}", but it needs backbone ${d.header.backboneId} @ ${d.header.inputSize}, which this build does not provide`, 'warn');
      }
      await putProfile(p);
      last = p.id;
    } catch (e) {
      notify(`Could not import ${f.name}: ${(e as Error).message}`, 'error');
    }
  }
  await refreshProfiles();
  if (last && modelForProfile(S().profiles.find((p) => p.id === last)!)) await activateProfile(last);
}

// ---- Offline ----------------------------------------------------------------------

const modelUrl = (e: { file: string; sha256: string }) => `${BASE}models/${e.file}?v=${e.sha256.slice(0, 12)}`;

/** Everything the app needs to start and detect without a network connection. */
async function offlineUrls(allModels: boolean): Promise<string[]> {
  const m = S().manifest!;
  const samples = (await (await fetch(`${BASE}samples/samples.json`)).json()) as { images: { file: string }[] };
  const bundled = (await (await fetch(`${BASE}profiles/profiles.json`)).json()) as { profiles: { file: string }[] };
  const models = allModels ? m.models : m.models.filter((e) => e.key === S().modelKey);
  // App shell, hashed bundles and the worker script, as already loaded by this page.
  const loaded = performance.getEntriesByType('resource')
    .map((e) => e.name)
    .filter((u) => u.startsWith(location.origin + BASE) && /\/assets\//.test(u));
  return [...new Set([
    BASE,
    ...loaded,
    `${BASE}ort/ort-wasm-simd-threaded.asyncify.mjs`,
    `${BASE}ort/ort-wasm-simd-threaded.asyncify.wasm`,
    `${BASE}models/models.json`,
    `${BASE}samples/samples.json`,
    `${BASE}profiles/profiles.json`,
    `${BASE}brand/accudyne-logo.svg`,
    `${BASE}brand/favicon.svg`,
    ...bundled.profiles.map((p) => `${BASE}profiles/${p.file}`),
    ...samples.images.map((i) => `${BASE}samples/${i.file}`),
    ...models.map(modelUrl),
    modelUrl(m.knn),
  ])];
}

/** After startup, quietly cache what this booth needs so it survives Wi-Fi drops. */
async function warmOfflineCache() {
  if (import.meta.env.DEV || !(await waitForController())) return;
  try {
    await precache(await offlineUrls(false));
    store.set({ offlineReady: true });
  } catch (e) {
    console.warn('Offline cache warm-up failed', e);
  }
}

export async function makeOffline() {
  if (!S().manifest) return;
  store.set({ precache: 'Starting…' });
  try {
    const urls = await offlineUrls(true);
    await precache(urls, (d, t) => store.set({ precache: `Cached ${d} / ${t}` }));
    store.set({ precache: `All ${urls.length} files available offline` });
  } catch (e) {
    store.set({ precache: (e as Error).message });
  }
}

export { MIN_FRAMES, MAX_FRAMES };
