import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { chromium, expect, type BrowserContext, type Page } from '@playwright/test';

export const FIX = join(import.meta.dirname, '..', 'fixtures');
export const SCENES = join(FIX, 'scenes');
export const meta = JSON.parse(readFileSync(join(FIX, 'meta.json'), 'utf8')) as {
  teach: string[];
  tests: { file: string; name: string; kind: string; defect: boolean }[];
};

const WORK = mkdtempSync(join(tmpdir(), 'patchcore-e2e-'));

/** Motion-JPEG file for Chrome's fake camera: concatenated JPEG frames, looped. */
export function mjpeg(name: string, files: string[], repeat = 1): string {
  const dir = join(WORK, 'video');
  mkdirSync(dir, { recursive: true });
  const frames = files.flatMap((f) => Array(repeat).fill(readFileSync(join(SCENES, f))));
  const out = join(dir, `${name}.mjpeg`);
  writeFileSync(out, Buffer.concat(frames));
  return out;
}

export function newProfileDir(): string {
  return mkdtempSync(join(WORK, 'profile-'));
}

/** A persistent Chromium profile (IndexedDB survives restarts) with a fake camera. */
export async function launchWithCamera(userDataDir: string, video: string): Promise<BrowserContext> {
  return chromium.launchPersistentContext(userDataDir, {
    headless: true,
    args: [
      '--use-fake-device-for-media-stream',
      '--use-fake-ui-for-media-stream',
      `--use-file-for-fake-video-capture=${video}`,
    ],
    permissions: ['camera'],
  });
}

export function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push(e.message));
  return errors;
}

export async function waitReady(page: Page) {
  await page.waitForFunction(() => (window as any).patchcore?.store.state.booted && (window as any).patchcore.store.state.engine === 'ready', null, { timeout: 60_000 });
}

/** Number of scored frames since detection (re)started. */
export async function waitForFrames(page: Page, n: number) {
  await page.waitForFunction((k) => (window as any).patchcore.store.state.history.length >= k, n, { timeout: 60_000 });
}

export async function verdict(page: Page): Promise<string> {
  return (await page.getByTestId('verdict').textContent())?.trim() ?? '';
}

export { expect };
