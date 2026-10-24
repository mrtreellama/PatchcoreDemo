// Teaches the bundled example profile with the app itself, in headless Chromium,
// so its threshold is calibrated on the browser's crop/resize/JPEG pipeline
// (a profile taught by the Python reference scores ~0.1 higher in the browser).
//
// Usage: npm run build && node tools/bake_profile.mjs
// Needs: tools/.cache/profile_scenes/*.jpg from tools/reference.py
import { spawn, spawnSync } from 'node:child_process';
import { readdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { chromium } from '@playwright/test';

const PORT = 4179;
const ROOT = resolve(import.meta.dirname, '..');
const SCENES = join(ROOT, 'tools', '.cache', 'profile_scenes');
const OUT = join(ROOT, 'public', 'profiles');
const FILE = 'carbon-twill-demo.pcprof';
const NAME = 'Carbon twill (demo)';

const scenes = readdirSync(SCENES).filter((f) => f.endsWith('.jpg')).sort().map((f) => join(SCENES, f));
if (!scenes.length) throw new Error(`No scenes in ${SCENES}; run tools/reference.py first`);

const server = spawn('npx', ['vite', 'preview', '--port', String(PORT), '--strictPort'], { cwd: ROOT, shell: true, stdio: 'ignore' });
const url = `http://localhost:${PORT}/`;
try {
  for (let i = 0; ; i++) {
    try { if ((await fetch(url)).ok) break; } catch { /* not up yet */ }
    if (i > 60) throw new Error('preview server did not start');
    await new Promise((r) => setTimeout(r, 500));
  }
  const browser = await chromium.launch();
  const page = await browser.newPage();
  page.on('console', (m) => m.type() === 'error' && console.error('[page]', m.text()));
  await page.goto(url);
  await page.waitForFunction(() => window.patchcore?.store.state.booted && window.patchcore.store.state.engine === 'ready', null, { timeout: 60_000 });
  await page.evaluate(() => window.patchcore.setEp('wasm'));
  await page.setInputFiles('[data-testid=image-input]', scenes);
  await page.waitForFunction((n) => window.patchcore.getSource()?.images?.length === n, scenes.length);
  const result = await page.evaluate(async (name) => {
    const a = window.patchcore;
    a.getSource().label = 'Sample images (carbon twill)';
    a.setRig('procedural');
    await a.beginTeach();
    a.setTeachField({ name, notes: 'Procedural 2x2 carbon twill. Taught in the browser by tools/bake_profile.mjs.' });
    await a.captureAllImages();
    const st = a.store.state;
    if (st.teach.frames.length < 5) {
      throw new Error(`only ${st.teach.frames.length} frames captured (mode=${st.mode}, source=${st.sourceLabel}, notice=${st.notice?.text})`);
    }
    await a.finishTeach();
    const p = a.activeProfile();
    if (!p || p.bundled || p.header.name !== name) throw new Error(`teaching did not produce a profile: ${a.store.state.notice?.text}`);
    const h = p.header;
    return { id: a.activeProfile().id, M: h.M, threshold: h.threshold, held: h.teachStats.heldScores };
  }, NAME);
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.evaluate((id) => window.patchcore.exportProfile(id), result.id),
  ]);
  await download.saveAs(join(OUT, FILE));
  writeFileSync(join(OUT, 'profiles.json'), JSON.stringify({ profiles: [{ file: FILE, name: NAME, modelKey: 'r18_224' }] }, null, 2) + '\n');
  console.log(`baked ${FILE}: M=${result.M} threshold=${result.threshold.toFixed(3)} held=${result.held.map((s) => s.toFixed(3)).join(', ')}`);
  await browser.close();
} finally {
  // shell: true wraps the server in a shell; kill the whole tree.
  if (process.platform === 'win32') spawnSync('taskkill', ['/pid', String(server.pid), '/T', '/F'], { stdio: 'ignore' });
  else server.kill();
}
