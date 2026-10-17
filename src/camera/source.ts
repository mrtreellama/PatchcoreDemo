import type { Roi } from '../profiles/types';
import { roiPixels } from './roi';

export type SourceKind = 'camera' | 'images' | 'video';

/** Anything the app can grab square, resized frames from. */
export interface FrameSource {
  kind: SourceKind;
  label: string;
  /** The element shown in the live view. */
  el: HTMLVideoElement | HTMLCanvasElement;
  width(): number;
  height(): number;
  ready(): boolean;
  /** ROI crop + resize to size x size; the bitmap is transferable to the worker. */
  grab(roi: Roi, size: number): Promise<ImageBitmap>;
  stop(): void;
}

const resizeOpts = (size: number): ImageBitmapOptions => ({
  resizeWidth: size, resizeHeight: size, resizeQuality: 'high',
});

function grabFrom(src: CanvasImageSource & ImageBitmapSource, W: number, H: number, roi: Roi, size: number) {
  const { sx, sy, sw, sh } = roiPixels(roi, W, H);
  return createImageBitmap(src, sx, sy, sw, sh, resizeOpts(size));
}

// ---- Camera ------------------------------------------------------------------

export interface CameraOptions {
  deviceId?: string;
  width: number;
  height: number;
}

export class CameraSource implements FrameSource {
  kind = 'camera' as const;
  el: HTMLVideoElement;
  label = 'Camera';
  stream?: MediaStream;

  constructor() {
    this.el = document.createElement('video');
    this.el.muted = true;
    this.el.playsInline = true;
    this.el.autoplay = true;
  }

  get track(): MediaStreamTrack | undefined {
    return this.stream?.getVideoTracks()[0];
  }

  async open(o: CameraOptions): Promise<void> {
    this.stop();
    const video: MediaTrackConstraints = { width: { ideal: o.width }, height: { ideal: o.height } };
    if (o.deviceId) video.deviceId = { exact: o.deviceId };
    else video.facingMode = { ideal: 'environment' };
    this.stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
    this.el.srcObject = this.stream;
    this.label = this.track?.label || 'Camera';
    await this.el.play().catch(() => undefined);
    if (this.el.readyState < 2) {
      await new Promise<void>((r) => this.el.addEventListener('loadeddata', () => r(), { once: true }));
    }
  }

  width() { return this.el.videoWidth; }
  height() { return this.el.videoHeight; }
  ready() { return this.el.readyState >= 2 && this.el.videoWidth > 0; }
  grab(roi: Roi, size: number) { return grabFrom(this.el, this.width(), this.height(), roi, size); }

  stop() {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = undefined;
    this.el.srcObject = null;
  }
}

// ---- Image files ---------------------------------------------------------------

export class ImageSource implements FrameSource {
  kind = 'images' as const;
  el: HTMLCanvasElement;
  label: string;
  index = 0;
  private ctx: CanvasRenderingContext2D;
  onChange?: () => void;

  private constructor(public images: ImageBitmap[], public names: string[]) {
    this.el = document.createElement('canvas');
    this.ctx = this.el.getContext('2d')!;
    this.label = `${images.length} image file${images.length === 1 ? '' : 's'}`;
    this.show(0);
  }

  static async fromFiles(files: File[]): Promise<ImageSource> {
    const imgs = files.filter((f) => f.type.startsWith('image/')).sort((a, b) => a.name.localeCompare(b.name));
    if (!imgs.length) throw new Error('No image files selected');
    const bitmaps = await Promise.all(imgs.map((f) => createImageBitmap(f)));
    return new ImageSource(bitmaps, imgs.map((f) => f.name));
  }

  static async fromUrls(urls: string[]): Promise<ImageSource> {
    const bitmaps = await Promise.all(urls.map(async (u) => createImageBitmap(await (await fetch(u)).blob())));
    return new ImageSource(bitmaps, urls.map((u) => u.split('/').pop() ?? u));
  }

  get current(): ImageBitmap {
    return this.images[this.index];
  }

  get name(): string {
    return this.names[this.index];
  }

  show(i: number) {
    this.index = ((i % this.images.length) + this.images.length) % this.images.length;
    const b = this.current;
    if (this.el.width !== b.width || this.el.height !== b.height) {
      this.el.width = b.width;
      this.el.height = b.height;
    }
    this.ctx.drawImage(b, 0, 0);
    this.onChange?.();
  }

  next() { this.show(this.index + 1); }
  prev() { this.show(this.index - 1); }
  width() { return this.current.width; }
  height() { return this.current.height; }
  ready() { return true; }
  grab(roi: Roi, size: number) { return grabFrom(this.current, this.width(), this.height(), roi, size); }

  stop() {
    this.images.forEach((b) => b.close());
    this.images = [];
  }
}

// ---- Video file ------------------------------------------------------------------

export class VideoSource implements FrameSource {
  kind = 'video' as const;
  el: HTMLVideoElement;
  label: string;
  private url: string;

  constructor(file: File) {
    this.url = URL.createObjectURL(file);
    this.label = file.name;
    this.el = document.createElement('video');
    this.el.muted = true;
    this.el.loop = true;
    this.el.playsInline = true;
    this.el.src = this.url;
  }

  async open(): Promise<void> {
    await this.el.play();
    if (this.el.readyState < 2) {
      await new Promise<void>((r) => this.el.addEventListener('loadeddata', () => r(), { once: true }));
    }
  }

  width() { return this.el.videoWidth; }
  height() { return this.el.videoHeight; }
  ready() { return this.el.readyState >= 2; }
  grab(roi: Roi, size: number) { return grabFrom(this.el, this.width(), this.height(), roi, size); }

  stop() {
    this.el.pause();
    this.el.removeAttribute('src');
    URL.revokeObjectURL(this.url);
  }
}

export async function listCameras(): Promise<MediaDeviceInfo[]> {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  const all = await navigator.mediaDevices.enumerateDevices();
  return all.filter((d) => d.kind === 'videoinput');
}
