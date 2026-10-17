import { decodePcprof } from './pcprof';
import type { Profile } from './types';

const BASE = import.meta.env.BASE_URL;

interface BundledEntry {
  file: string;
  name: string;
  modelKey: string;
}

/** Read-only example profiles shipped in public/profiles. */
export async function loadBundledProfiles(): Promise<Profile[]> {
  const res = await fetch(`${BASE}profiles/profiles.json`);
  if (!res.ok) return [];
  const { profiles } = (await res.json()) as { profiles: BundledEntry[] };
  const out: Profile[] = [];
  for (const e of profiles) {
    try {
      const r = await fetch(`${BASE}profiles/${e.file}`);
      const d = decodePcprof(await r.arrayBuffer());
      out.push({
        id: `bundled:${e.file}`,
        header: d.header,
        bank: d.bank,
        sqnorm: d.sqnorm,
        thumbnail: d.thumbnail.length ? new Blob([d.thumbnail as BlobPart], { type: 'image/jpeg' }) : undefined,
        bundled: true,
        updated: 0,
      });
    } catch (err) {
      console.warn(`Skipping bundled profile ${e.file}:`, err);
    }
  }
  return out;
}
