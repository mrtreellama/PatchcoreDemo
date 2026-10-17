import { join } from 'node:path';
import { test } from '@playwright/test';
import {
  collectErrors, expect, launchWithCamera, meta, mjpeg, newProfileDir, SCENES, verdict, waitForFrames, waitReady,
} from './helpers';

// One persistent browser profile shared by the steps below, so the taught
// profile must survive a browser restart (IndexedDB) to be used for detection.
const userDataDir = newProfileDir();

test.describe.serial('teach on a fake camera, then detect after restart', () => {
  test('teach from good fixture images (WASM), then switch to the camera', async ({ baseURL }) => {
    // Teaching uses the image-file source: a looping fake-camera video of a few
    // stills would put duplicate frames in both the bank and the held-out set.
    const ctx = await launchWithCamera(userDataDir, mjpeg('good', [meta.teach[0]]));
    const page = ctx.pages()[0] ?? (await ctx.newPage());
    const errors = collectErrors(page);
    await page.goto(baseURL!);
    await waitReady(page);

    await page.getByTestId('ep-select').selectOption('wasm');
    await expect(page.getByTestId('engine-status')).toContainText('WASM');
    await page.getByTestId('image-input').setInputFiles(meta.teach.map((f) => join(SCENES, f)));
    await expect(page.getByTestId('image-name')).toContainText(`1 / ${meta.teach.length}`);

    await page.getByTestId('new-profile').click();
    await page.getByTestId('teach-name').fill('E2E twill');
    await page.getByTestId('capture-all').click();
    await expect(page.getByTestId('teach-count')).toContainText(`${meta.teach.length} / 15`);
    await page.getByTestId('finish-teach').click();
    await expect(page.getByTestId('profile-select')).toContainText('E2E twill');
    await waitForFrames(page, 3);

    // Leave the camera selected so the next launches start on it.
    await page.getByRole('button', { name: 'Camera', exact: true }).click();
    await page.waitForFunction(() => (window as any).patchcore.store.state.sourceKind === 'camera');
    expect(errors).toEqual([]);
    await ctx.close();
  });

  for (const t of meta.tests) {
    test(`after restart: ${t.name} is ${t.defect ? 'DEFECT' : 'PASS'}`, async ({ baseURL }) => {
      const ctx = await launchWithCamera(userDataDir, mjpeg(t.name, [t.file]));
      const page = ctx.pages()[0] ?? (await ctx.newPage());
      const errors = collectErrors(page);
      await page.goto(baseURL!);
      await waitReady(page);
      // Camera source, WASM backend and the taught profile all come back from settings.
      await expect(page.getByTestId('profile-select')).toContainText('E2E twill');
      await expect(page.getByTestId('engine-status')).toContainText('WASM');
      await waitForFrames(page, 5);
      const score = await page.evaluate(() => (window as any).patchcore.store.state.result.score);
      console.log(`${t.name}: score ${score.toFixed(3)}`);
      expect(await verdict(page)).toBe(t.defect ? 'DEFECT' : 'PASS');
      expect(errors).toEqual([]);
      await ctx.close();
    });
  }
});
