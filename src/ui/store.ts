import { useEffect, useRef, useState } from 'preact/hooks';

/** Minimal observable store; components subscribe with a selector. */
export class Store<T extends object> {
  private listeners = new Set<() => void>();
  constructor(public state: T) {}

  get(): T {
    return this.state;
  }

  set(patch: Partial<T> | ((s: T) => Partial<T>)): void {
    const p = typeof patch === 'function' ? patch(this.state) : patch;
    this.state = { ...this.state, ...p };
    this.listeners.forEach((l) => l());
  }

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }
}

function shallowEqual(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (typeof a !== 'object' || typeof b !== 'object' || !a || !b || Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a);
  if (ka.length !== Object.keys(b).length) return false;
  return ka.every((k) => Object.is((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]));
}

export function useStore<T extends object, S>(store: Store<T>, selector: (s: T) => S): S {
  const [value, setValue] = useState(() => selector(store.state));
  const sel = useRef(selector);
  sel.current = selector;
  const last = useRef(value);
  useEffect(() => {
    const check = () => {
      const next = sel.current(store.state);
      if (!shallowEqual(next, last.current)) {
        last.current = next;
        setValue(() => next);
      }
    };
    check();
    return store.subscribe(check);
  }, [store]);
  return value;
}
