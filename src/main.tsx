import { render } from 'preact';
import './style.css';
import { App } from './ui/App';
import * as app from './app';
import { start, notify } from './app';
import { ensureIsolation } from './util/coi';

async function boot() {
  const coi = await ensureIsolation();
  if (coi === 'reloading') return;
  render(<App />, document.getElementById('app')!);
  try {
    await start(coi);
  } catch (e) {
    notify(`Startup failed: ${(e as Error).message}`, 'error');
    console.error(e);
  }
}

window.addEventListener('unhandledrejection', (e) => {
  // Surface async failures in the diagnostics panel instead of only the console.
  const msg = e.reason instanceof Error ? e.reason.message : String(e.reason);
  if (msg) notify(msg, 'error');
});

// Handle for automated tests and console debugging.
(window as unknown as { patchcore: typeof app }).patchcore = app;

boot();
