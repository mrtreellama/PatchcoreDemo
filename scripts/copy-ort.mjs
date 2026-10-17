// Copies the ONNX Runtime Web runtime used by `onnxruntime-web/webgpu` (the
// asyncify build, which serves both the WebGPU and WASM execution providers)
// into public/ort so it loads from the site itself, not a CDN, and is cached by
// the service worker for offline use.
import { cpSync, mkdirSync, readdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';

const src = 'node_modules/onnxruntime-web/dist';
const dst = 'public/ort';
rmSync(dst, { recursive: true, force: true });
mkdirSync(dst, { recursive: true });
const files = readdirSync(src).filter((f) => /^ort-wasm-simd-threaded\.asyncify\.(wasm|mjs)$/.test(f));
for (const f of files) cpSync(join(src, f), join(dst, f));
console.log(`copied ${files.length} ORT runtime files to ${dst}`);
