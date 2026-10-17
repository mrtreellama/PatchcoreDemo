/// <reference types="vitest/config" />
import { defaultClientConditions, defineConfig } from 'vite';
import preact from '@preact/preset-vite';

// GitHub Pages serves the site from /<repo>/. Override with BASE=/ for other hosts.
const base = process.env.BASE ?? '/PatchcoreDemo/';

// Dev and preview servers can set the isolation headers directly; on Pages the
// service worker (public/sw.js) adds them instead.
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  base,
  plugins: [preact()],
  define: {
    __APP_VERSION__: JSON.stringify(process.env.npm_package_version ?? '0.0.0'),
    __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
  },
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
  // Load ORT's .wasm/.mjs from public/ort (ort.env.wasm.wasmPaths) instead of bundling a second copy.
  resolve: { conditions: ['onnxruntime-web-use-extern-wasm', ...defaultClientConditions] },
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: { target: 'es2022', chunkSizeWarningLimit: 2000 },
  test: {
    include: ['tests/unit/**/*.test.ts'],
    environment: 'node',
    testTimeout: 120_000,
  },
});
