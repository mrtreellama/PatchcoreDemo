# Decisions

Choices made where `SPEC.md` is silent or where building it turned up a constraint. Newest first within each section.

## Models

- **`wrn50` ships as a mixed-precision file only.** Truncated after layer 3, `wide_resnet50_2` is 102.5 MB in FP32, which is over GitHub's push limit and Pages' file limit, so the FP32 file is not kept. Storing *all* weights as FP16 gave 1.14e-3 max error at 320 px (just over the spec's 1e-3). Storing only the layer-3 weights as FP16 (the bulk of the parameters) and keeping the stem, layers 1–2 and the projection in FP32 gives ≤ 2.2e-4 at 61 MB. Compute stays FP32: the FP16 initializers are `Cast` back in-graph, because the WASM backend has no FP16 kernels. `tools/reference.py` asserts parity for every shipped file.
- **Projection seed.** The fixed random projection for `wrn50` (1536 → 512) is a Gaussian matrix from `torch.Generator().manual_seed(1234)`, scaled by 1/√512.
- **Legacy ONNX exporter.** Exports use `torch.onnx.export(..., dynamo=False, opset_version=17)`, which produces single-file static-shape graphs.
- **Model cache busting.** Model URLs carry `?v=<first 12 hex of sha256>`, so the service worker's cache-first strategy picks up re-exported models.

## PatchCore math

- **Shared PRNG.** The coreset's random projection uses Mulberry32 with ±1 (Rademacher) entries, implemented identically in Python and TypeScript. TypeScript selects exactly the same coreset indices as the Python fixture.
- **Coreset start point.** The first coreset row is `floor(rng() * N)` from the same seeded stream, drawn after the projection matrix.
- **Rounding.** Coreset size and holdout count use round-half-up (`Math.round`), mirrored in Python as `floor(x + 0.5)` (Python's `round` uses banker's rounding).
- **Scoring resolution.** The anomaly map is upsampled and blurred (σ = 4, truncate 4, `reflect` borders like SciPy) at the backbone input size, and the image score is the maximum of that map. The overlay canvas is then scaled to the ROI's display size by the browser. Scores are therefore independent of screen size.
- **Heat floor.** This is the median of the raw (unsmoothed) held-out patch distances.
- **"Add good images".** New captures all go into the bank, which is recomputed from the existing bank rows plus the new patches. The profile keeps its original calibration: the threshold, highest held-out score and heat floor are unchanged. The original teach features are not stored, so the old bank rows stand in for them.
- **kNN.** The `knn.onnx` graph runs on the active backend in 4,096-row bank chunks, with a running minimum. The TypeScript twin in `src/patchcore/knn.ts` is the reference and fallback, and parity with it is unit-tested.

## Bundled profile and fixtures

- **The bundled profile is taught by the app itself.** `tools/bake_profile.mjs` runs the production build in headless Chromium and teaches the profile through the normal UI code path. A profile taught by the Python reference scored about 0.1 higher in the browser: Chrome's `createImageBitmap` resampling and JPEG decoder differ slightly from PIL's (≈1.1 grey levels of mean absolute error on the weave), and the threshold margin is tight. A profile is only valid under the pipeline it was taught with, just as the spec says for cameras and lighting.
- **The bundled profile uses 30 frames, not the 15-frame default.** With 6 held-out frames instead of 3, the threshold estimate is steadier. Example detections on all 16 sample images: good ≤ 1.05, threshold 1.19, defects ≥ 1.74.
- **Integer ROI crop.** The ROI crop is in integer pixels (`roiPixels`), and the Python reference uses the same box.
- **Procedural data.** The 2×2 twill texture has small random rotation and offset. The planted defects are a dropped tow, a foreign patch and a wrinkle (a warp displacement with a crease highlight).
- **`.pcprof` fixture from Python.** Python writes `tests/fixtures/python_written.pcprof` so the TypeScript reader is tested against an independent writer.

## `.pcprof` format

- **Header length.** `L` is the unpadded JSON byte length. The JSON is padded with spaces to 4-byte alignment, and the bank starts at `12 + align4(L)`.
- **Extra header fields.** The header adds `modelKey`, `thumbnailBytes` and `teachStats.maxHeld`, which re-thresholding needs.
- **Export dtype.** Exports use a float16 bank, which halves the size. The squared norms are computed from the float16-rounded bank, so the file is self-consistent. Banks in IndexedDB stay float32.

## App behaviour

- **First visit opens the samples.** On a first visit, the source is the bundled sample images and the bundled profile is active, so the demo works before any camera permission. The last source, camera, backbone, backend and profile are remembered in IndexedDB.
- **Missing camera fallback.** If the remembered camera ID is gone (for example, a webcam was re-plugged), the app falls back to the default camera with a warning. It no longer silently switches to the samples.
- **ROI edits.** Editing the ROI affects the session only. A profile's ROI is set when it is taught and re-applied whenever that profile is activated.
- **Teaching blocks detection.** While teach mode is active, detection cannot start, and startup cannot re-activate a profile into detect mode.
- **Serialized captures.** Captures are serialized, so quick clicks never grab the same frame twice. Image-file sources advance to the next image after each capture.
- **Switching backends.** Changing the execution provider restarts the inference worker, because ORT's WASM thread count is fixed per worker.
- **WASM threads.** With cross-origin isolation, the thread count is `min(8, ceil(cores / 2))`. Without it, WASM is single-threaded and the Diagnostics panel says so.
- **One service worker.** `public/sw.js` does both COOP/COEP isolation and offline caching. On first load the page reloads once to become isolated. Once the worker controls the page, the app caches the shell, ORT runtime, current model, samples and bundled profiles. "Download all models for offline use" in Diagnostics adds every backbone. Cache lookups use `ignoreVary`, because the server sends `Vary: Origin` and module script requests carry an `Origin` header that precache fetches lack.
- **ORT runtime from the site.** ORT's `onnxruntime-web/webgpu` build uses the asyncify WASM runtime. That runtime is copied to `public/ort/` and loaded from the site via `ort.env.wasm.wasmPaths`, with the `onnxruntime-web-use-extern-wasm` resolve condition so Vite doesn't bundle a second 27 MB copy.
- **Idle detection.** Idle uses an 8×8 luminance signature computed in the worker. The scene counts as changed when the mean absolute difference exceeds 0.02. The attract loop cycles through up to 8 snapshots of defect detections from the current session.
- **Booth exit.** Booth mode exits with a 2-second long-press on the logo or Ctrl+Shift+E. Esc only leaves fullscreen.
- **Benchmark workload.** The benchmark uses a synthetic, seeded 2,000-row memory bank, so kNN cost is the same for every configuration. Each combination runs in a fresh worker, and live detection is stopped during the run.
- **Test hook.** `window.patchcore` exposes the app controller for Playwright and console debugging.

## Branding

- **Logo.** The repository came with a vector logo, so it ships as `public/brand/accudyne-logo.svg` instead of the web PNG (the spec prefers vector when available). The wordmark is white with a navy gear, so both top bars are dark.
- **Favicon.** `public/brand/favicon.svg` is the gear mark cropped from the logo, on a white rounded square.

## Testing

- **End-to-end teaching uses the image-file source.** Chrome's fake camera loops the frames of the video file, so a fake-camera teach set of a few stills puts duplicate frames in both the bank and the held-out set, which collapses calibration. Detection after a browser restart does run on the fake camera (MJPEG made from each fixture image). This checks camera capture, settings restore and IndexedDB persistence.
- **Production build in e2e.** The e2e tests run against `vite preview` of the production build, so the service worker, base path and extern ORT runtime are all covered.
