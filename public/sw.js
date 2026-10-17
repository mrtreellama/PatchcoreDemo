/* PatchCore demo service worker.
 *
 * 1. Cross-origin isolation shim (coi-serviceworker pattern): GitHub Pages
 *    cannot set COOP/COEP headers, so every same-origin response passing
 *    through here gets them added. That enables SharedArrayBuffer and
 *    therefore multithreaded WASM.
 * 2. Offline cache: heavy, versioned assets (models, ORT runtime, samples,
 *    hashed build assets) are cache-first; the page and manifests are
 *    network-first with a cache fallback, so the booth keeps running when the
 *    venue Wi-Fi drops.
 */
const CACHE = 'patchcore-v1';
const CACHE_FIRST = /\/(models|ort|samples|profiles|brand|assets)\/[^?]+$/;
const NETWORK_FIRST_EXCEPTIONS = /\/(models\.json|profiles\.json|samples\.json)$/;

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) if (key !== CACHE) await caches.delete(key);
    await self.clients.claim();
  })());
});

self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'precache') {
    event.waitUntil(precache(event.data.urls, event.source));
  }
});

async function precache(urls, client) {
  const cache = await caches.open(CACHE);
  let done = 0;
  for (const url of urls) {
    try {
      if (!(await cache.match(url, { ignoreVary: true }))) {
        const res = await fetch(url, { cache: 'no-cache' });
        if (res.ok) await cache.put(url, res);
      }
    } catch (e) { /* reported as missing below */ }
    done++;
    client && client.postMessage({ type: 'precache-progress', done, total: urls.length });
  }
}

function withIsolation(response) {
  if (!response || response.status === 0 || response.type === 'opaque') return response;
  const headers = new Headers(response.headers);
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
  headers.set('Cross-Origin-Resource-Policy', 'same-origin');
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

async function cacheFirst(request) {
  const cache = await caches.open(CACHE);
  // ignoreVary: servers send Vary: Origin, and module requests carry an Origin header precache fetches lack.
  const hit = await cache.match(request, { ignoreVary: true });
  if (hit) return hit;
  const res = await fetch(request);
  if (res.ok) cache.put(request, res.clone());
  return res;
}

async function networkFirst(request) {
  const cache = await caches.open(CACHE);
  try {
    const res = await fetch(request);
    if (res.ok) cache.put(request, res.clone());
    return res;
  } catch (err) {
    const hit = await cache.match(request, { ignoreSearch: request.mode === 'navigate', ignoreVary: true });
    if (hit) return hit;
    throw err;
  }
}

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;
  if (req.headers.has('range')) return;
  const useCache = CACHE_FIRST.test(url.pathname) && !NETWORK_FIRST_EXCEPTIONS.test(url.pathname);
  event.respondWith((useCache ? cacheFirst(req) : networkFirst(req)).then(withIsolation));
});
