import { openDB, type DBSchema, type IDBPDatabase } from 'idb';
import type { Profile, ProfileHeader } from './types';

interface StoredProfile {
  id: string;
  header: ProfileHeader;
  bank: ArrayBuffer;
  sqnorm: ArrayBuffer;
  thumbnail: Blob | null;
  updated: number;
}

interface Schema extends DBSchema {
  profiles: { key: string; value: StoredProfile };
  settings: { key: string; value: unknown };
}

let dbp: Promise<IDBPDatabase<Schema>> | undefined;

function db() {
  dbp ??= openDB<Schema>('patchcore-demo', 1, {
    upgrade(d) {
      d.createObjectStore('profiles', { keyPath: 'id' });
      d.createObjectStore('settings');
    },
  });
  return dbp;
}

const toStored = (p: Profile): StoredProfile => ({
  id: p.id,
  header: p.header,
  bank: p.bank.slice().buffer,
  sqnorm: p.sqnorm.slice().buffer,
  thumbnail: p.thumbnail ?? null,
  updated: p.updated,
});

const fromStored = (s: StoredProfile): Profile => ({
  id: s.id,
  header: s.header,
  bank: new Float32Array(s.bank),
  sqnorm: new Float32Array(s.sqnorm),
  thumbnail: s.thumbnail ?? undefined,
  updated: s.updated,
});

export async function listProfiles(): Promise<Profile[]> {
  const all = await (await db()).getAll('profiles');
  return all.map(fromStored).sort((a, b) => a.header.name.localeCompare(b.header.name));
}

export async function getProfile(id: string): Promise<Profile | undefined> {
  const s = await (await db()).get('profiles', id);
  return s && fromStored(s);
}

export async function putProfile(p: Profile): Promise<void> {
  if (p.bundled) throw new Error('Bundled profiles are read-only; duplicate it first');
  await (await db()).put('profiles', toStored({ ...p, updated: Date.now() }));
}

export async function deleteProfile(id: string): Promise<void> {
  await (await db()).delete('profiles', id);
}

export async function getSetting<T>(key: string, fallback: T): Promise<T> {
  try {
    const v = await (await db()).get('settings', key);
    return v === undefined ? fallback : (v as T);
  } catch {
    return fallback;
  }
}

export async function setSetting(key: string, value: unknown): Promise<void> {
  await (await db()).put('settings', value, key);
}
