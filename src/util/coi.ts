/**
 * Registers the service worker (public/sw.js), which adds COOP/COEP headers
 * on GitHub Pages and caches assets for offline use. On the very first visit
 * the page is not yet controlled, so we reload once after the worker takes
 * control; after that `crossOriginIsolated` is true and WASM can use threads.
 */
const BASE = import.meta.env.BASE_URL;
const RELOAD_KEY = 'coi-reloaded';

export type CoiStatus = 'isolated' | 'reloading' | 'no-sw' | 'not-isolated';

export async function ensureIsolation(): Promise<CoiStatus> {
  if (!('serviceWorker' in navigator)) return self.crossOriginIsolated ? 'isolated' : 'no-sw';
  // Dev and preview servers send the headers themselves.
  if (import.meta.env.DEV) return self.crossOriginIsolated ? 'isolated' : 'not-isolated';
  try {
    await navigator.serviceWorker.register(`${BASE}sw.js`, { scope: BASE });
  } catch (e) {
    console.warn('Service worker registration failed', e);
    return self.crossOriginIsolated ? 'isolated' : 'no-sw';
  }
  if (self.crossOriginIsolated) {
    sessionStorage.removeItem(RELOAD_KEY);
    return 'isolated';
  }
  // Avoid reload loops when the browser will not isolate (e.g. some in-app browsers).
  if (sessionStorage.getItem(RELOAD_KEY)) return 'not-isolated';
  sessionStorage.setItem(RELOAD_KEY, '1');
  if (navigator.serviceWorker.controller) {
    location.reload();
    return 'reloading';
  }
  await new Promise<void>((resolve) => {
    navigator.serviceWorker.addEventListener('controllerchange', () => resolve(), { once: true });
    setTimeout(resolve, 4000);
  });
  if (navigator.serviceWorker.controller) {
    location.reload();
    return 'reloading';
  }
  return 'not-isolated';
}

/** Resolves once a service worker controls this page (or after `timeoutMs`). */
export async function waitForController(timeoutMs = 10_000): Promise<boolean> {
  if (!('serviceWorker' in navigator)) return false;
  if (navigator.serviceWorker.controller) return true;
  await new Promise<void>((resolve) => {
    navigator.serviceWorker.addEventListener('controllerchange', () => resolve(), { once: true });
    setTimeout(resolve, timeoutMs);
  });
  return !!navigator.serviceWorker.controller;
}

/** Asks the service worker to cache every listed URL now (for offline booths). */
export function precache(urls: string[], onProgress?: (done: number, total: number) => void): Promise<void> {
  const ctl = navigator.serviceWorker?.controller;
  if (!ctl) return Promise.reject(new Error('Service worker not active (offline caching unavailable in dev mode)'));
  return new Promise((resolve) => {
    const onMsg = (ev: MessageEvent) => {
      if (ev.data?.type !== 'precache-progress') return;
      onProgress?.(ev.data.done, ev.data.total);
      if (ev.data.done >= ev.data.total) {
        navigator.serviceWorker.removeEventListener('message', onMsg);
        resolve();
      }
    };
    navigator.serviceWorker.addEventListener('message', onMsg);
    ctl.postMessage({ type: 'precache', urls });
  });
}
