import { readFileSync } from 'node:fs';
import { test } from '@playwright/test';
import { collectErrors, expect, waitForFrames, waitReady } from './helpers';

test('loads with the bundled profile and flags the first sample defect', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('./');
  await expect(page).toHaveTitle('Accudyne Systems · Composite Defect Detection Demo');
  await waitReady(page);
  await expect(page.getByTestId('profile-select')).toContainText('Carbon twill (demo)');
  // Samples open with the test images first; step to a planted defect.
  await page.evaluate(() => (window as any).patchcore.getSource().show(3));
  await waitForFrames(page, 4);
  await expect(page.getByTestId('verdict')).toHaveText('DEFECT');
  await page.getByTestId('diagnostics').locator('summary').click();
  await expect(page.getByTestId('diag-backend')).not.toBeEmpty();
  expect(errors).toEqual([]);
});

test('export and import a profile (.pcprof round trip)', async ({ page }) => {
  await page.goto('./');
  await waitReady(page);
  await expect(page.getByTestId('profile-select')).toContainText('Carbon twill (demo)');
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByTestId('export').click()]);
  const path = await download.path();
  expect(readFileSync(path!).subarray(0, 4).toString()).toBe('PCPF');
  await page.getByTestId('import-input').setInputFiles({
    name: 'roundtrip.pcprof', mimeType: 'application/octet-stream', buffer: readFileSync(path!),
  });
  // The import becomes the active (local, editable) profile.
  await expect(page.getByRole('button', { name: 'Delete' })).toBeVisible();
  const n = await page.evaluate(() => (window as any).patchcore.store.state.profiles.filter((p: any) => !p.bundled).length);
  expect(n).toBe(1);
});

test('benchmark produces a downloadable JSON report', async ({ page }) => {
  await page.goto('./#/benchmark');
  await waitReady(page);
  for (const box of await page.locator('input[type=checkbox][data-testid^="bench-"]').all()) {
    if (await box.isChecked()) await box.uncheck();
  }
  await page.getByTestId('bench-r18_224|wasm').check();
  await page.getByTestId('bench-run').click();
  await expect(page.getByTestId('bench-step')).toHaveText('Done', { timeout: 120_000 });
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByTestId('bench-download').click()]);
  const report = JSON.parse(readFileSync((await download.path())!, 'utf8'));
  expect(report.results[0].modelKey).toBe('r18_224');
  expect(report.results[0].frames).toBe(50);
  expect(report.results[0].stats.endToEnd.median).toBeGreaterThan(0);
  expect(report.device.logicalCores).toBeGreaterThan(0);
});

test('booth mode shows a verdict banner and exits with Ctrl+Shift+E', async ({ page }) => {
  await page.goto('./#/booth');
  await waitReady(page);
  await expect(page.getByTestId('booth-verdict')).toHaveText(/PASS|DEFECT/);
  await page.keyboard.press('Control+Shift+E');
  await expect(page).toHaveURL(/#\/operator$/);
});

test('keeps detecting offline after the first load', async ({ page, context }) => {
  await page.goto('./');
  await waitReady(page);
  await page.waitForFunction(() => (window as any).patchcore.store.state.offlineReady, null, { timeout: 60_000 });
  await context.setOffline(true);
  await page.reload();
  await waitReady(page);
  await expect(page.getByTestId('profile-select')).toContainText('Carbon twill (demo)');
  await page.evaluate(() => (window as any).patchcore.getSource().show(4));
  await waitForFrames(page, 4);
  await expect(page.getByTestId('verdict')).toHaveText('DEFECT');
  await context.setOffline(false);
});
