# PatchCore Web Demo

Live, in-browser anomaly detection for composite materials, built as a trade-show demo for Accudyne Systems. You teach it a material from a handful of good samples in a few seconds. It then shows a live heatmap and a PASS / DEFECT verdict for each camera frame. Everything runs in the browser: there is no server, no install and no cloud inference.

**Live site:** https://mrtreellama.github.io/PatchcoreDemo/ (after the first deploy)

It works in Chrome and Edge on laptops, mini PCs, Chromebooks and Android phones. Inference uses ONNX Runtime Web on WebGPU, with automatic fallback to WASM.

## Using it

1. **Operator** (default view):
   - Pick a source: **Camera** (any webcam; a USB camera on a fixed rig is the main target), **Image files**, **Video file** or the bundled **Samples**.
   - Drag the square ROI over the area to inspect.
   - With a camera, open *Camera controls* and press **Lock** so exposure, focus and white balance stay fixed.
2. **Teach new…**:
   - Name the material and capture 5–50 good frames (default 15), moving the sample slightly between captures. Use the Capture button or *Auto every 0.5 s*.
   - Press **Finish teaching**. The last 20% of frames set the threshold.
3. **Detect**:
   - The heatmap, score chart and verdict update live.
   - **Freeze** holds a result on screen. The margin slider re-thresholds without reteaching.
   - **Add good images** fixes false alarms found on the show floor.
4. **Profiles** are stored in the browser and survive restarts. Use **Export** / **Import**, or drop a `.pcprof` file on the page, to move them between devices.
5. **Booth** (`#/booth`):
   - A full-screen visitor display that keeps the screen awake, with a large verdict banner, a good reference image, and an optional idle attract loop.
   - To exit, hold the logo for 2 s or press **Ctrl+Shift+E**.
6. **Benchmark** (`#/benchmark`): times model load, warm-up and 50 frames for each backbone, input size and backend. Download the results as JSON or copy them for a spreadsheet.

**Diagnostics** (bottom of the page) shows:

- the active backend, thread count and cross-origin isolation status
- model hashes and the last error
- a button that caches every model for offline use

## Models

| Backbone | Input sizes | Feature D | File size |
| --- | --- | --- | --- |
| ResNet-18 (`r18`, default) | 224, 320 | 384 | 11 MB |
| Wide ResNet-50-2 (`wrn50`) | 224, 320 | 512 (projected) | 61 MB (layer 3 stored as FP16) |

Each backbone is a single ONNX graph. It takes an RGB image in the range 0–1 and outputs PatchCore patch features. ImageNet normalization, layer 2 + 3 neighbourhood pooling, upsampling, concatenation and the projection all happen inside the graph. A separate `knn.onnx` computes nearest-neighbour distances to the memory bank.

## Development

Requirements: Node 20 or newer, and Python 3.11 for the model tooling.

```bash
npm install
```

```bash
npm run dev
```

```bash
npm test
```

```bash
npm run build
```

```bash
npx playwright install chromium
```

```bash
npx playwright test
```

- `npm run dev` starts a dev server at `http://localhost:5173/PatchcoreDemo/`.
- `npm test` runs the unit tests: the TypeScript math and ONNX Runtime Web features against the Python fixtures, the `.pcprof` round trip, and hysteresis.
- `npm run build` writes the production build to `dist/`.
- `npx playwright install chromium` is needed once, before the first end-to-end run.
- `npx playwright test` runs the end-to-end tests: teaching, restart, detection on Chrome's fake camera, offline mode, benchmark and booth mode.

### Model tooling (`tools/`)

```bash
py -3.11 -m venv tools/.venv
```

```bash
tools/.venv/Scripts/python -m pip install -r tools/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

```bash
tools/.venv/Scripts/python tools/export_backbone.py --only r18 --only wrn50 --only knn
```

```bash
tools/.venv/Scripts/python tools/reference.py
```

```bash
npm run build && node tools/bake_profile.mjs
```

- `export_backbone.py` writes the ONNX files and `models.json` to `public/models/`.
- `reference.py` checks ONNX/PyTorch parity and writes `tests/fixtures/` and `public/samples/`.
- `bake_profile.mjs` re-teaches the bundled example profile in Chromium.

On macOS or Linux, use `tools/.venv/bin/python` in place of `tools/.venv/Scripts/python`.

The test images are procedurally generated carbon-twill textures with planted defects (dropped tow, foreign patch, wrinkle), so no third-party dataset is involved.

## Deployment

Every push to `main` runs the unit and end-to-end tests in GitHub Actions and then deploys `dist/` to GitHub Pages (`.github/workflows/pages.yml`). Before the first deploy, set the repository's **Settings → Pages → Source** to **GitHub Actions**.

GitHub Pages can't set HTTP headers. To enable multithreaded WASM, `public/sw.js` adds the COOP/COEP headers itself. The page reloads once on the very first visit so the headers take effect. The same service worker caches the app, so the booth keeps working if the venue Wi-Fi drops.

See [`SPEC.md`](SPEC.md) for the specification and [`DECISIONS.md`](DECISIONS.md) for the choices made while building it.
