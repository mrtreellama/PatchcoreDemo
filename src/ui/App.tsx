import { useEffect, useState } from 'preact/hooks';
import { BASE, clearNotice, store } from '../app';
import { useStore } from './store';
import { Operator } from './Operator';
import { Booth } from './Booth';
import { Benchmark } from './Benchmark';
import { Diagnostics } from './Diagnostics';

type Route = 'operator' | 'booth' | 'benchmark';

function parseRoute(): Route {
  const h = location.hash.replace(/^#\/?/, '');
  return h === 'booth' || h === 'benchmark' ? h : 'operator';
}

function useRoute(): Route {
  const [r, setR] = useState(parseRoute);
  useEffect(() => {
    const on = () => setR(parseRoute());
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  return r;
}

function NoticeBar() {
  const n = useStore(store, (s) => s.notice);
  useEffect(() => {
    if (!n || n.kind === 'error') return;
    const t = setTimeout(clearNotice, n.kind === 'warn' ? 9000 : 5000);
    return () => clearTimeout(t);
  }, [n]);
  if (!n) return null;
  return (
    <div class={`notice ${n.kind}`} role={n.kind === 'error' ? 'alert' : 'status'} data-testid="notice">
      <span>{n.text}</span>
      <button class="x" onClick={clearNotice} aria-label="Dismiss">×</button>
    </div>
  );
}

export function App() {
  const route = useRoute();
  if (route === 'booth') {
    return (
      <>
        <Booth />
        <NoticeBar />
      </>
    );
  }
  return (
    <div class="shell">
      <header class="topbar">
        <a href="#/operator" class="brand">
          <img src={`${BASE}brand/accudyne-logo.svg`} alt="Accudyne Systems" />
          <span class="brand-title">Composite Defect Detection</span>
        </a>
        <nav>
          <a href="#/operator" class={route === 'operator' ? 'on' : ''}>Operator</a>
          <a href="#/booth">Booth</a>
          <a href="#/benchmark" class={route === 'benchmark' ? 'on' : ''}>Benchmark</a>
        </nav>
      </header>
      <NoticeBar />
      {route === 'benchmark' ? <Benchmark /> : <Operator />}
      <Diagnostics />
    </div>
  );
}
