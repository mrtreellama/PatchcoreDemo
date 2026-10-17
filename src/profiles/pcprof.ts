/**
 * .pcprof binary profile format (little-endian), mirrored by tools/pcprof.py.
 *
 *   0  magic "PCPF"          4 bytes
 *   4  format version (1)    uint32
 *   8  header length L       uint32 (unpadded JSON byte length)
 *  12  header                UTF-8 JSON, L bytes, space-padded to 4-byte alignment
 *   .. memory bank           float32 or float16 [M, D]
 *   .. squared norms         float32 [M]
 *   .. thumbnail             JPEG bytes, length in header.thumbnailBytes
 */
import { decodeF16, encodeF16 } from '../patchcore/f16';
import { rowSqNorms } from '../patchcore/knn';
import type { ProfileHeader } from './types';

export const MAGIC = 'PCPF';
export const FORMAT_VERSION = 1;

export interface DecodedProfile {
  header: ProfileHeader;
  bank: Float32Array;
  sqnorm: Float32Array;
  thumbnail: Uint8Array;
}

export function encodePcprof(
  header: ProfileHeader,
  bank: Float32Array,
  thumbnail: Uint8Array = new Uint8Array(0),
  bankDtype: 'float32' | 'float16' = 'float16',
): Uint8Array {
  const D = header.D;
  const M = bank.length / D;
  // Norms must describe the bank exactly as the reader will see it.
  let bankBytes: Uint8Array;
  let sq: Float32Array;
  if (bankDtype === 'float16') {
    const half = encodeF16(bank);
    bankBytes = new Uint8Array(half.buffer);
    sq = rowSqNorms(decodeF16(half), D);
  } else {
    bankBytes = new Uint8Array(bank.buffer, bank.byteOffset, bank.byteLength);
    sq = rowSqNorms(bank, D);
  }
  const h: ProfileHeader = { ...header, M, D, bankDtype, thumbnailBytes: thumbnail.length };
  const json = new TextEncoder().encode(JSON.stringify(h));
  const pad = (4 - (json.length % 4)) % 4;
  const total = 12 + json.length + pad + bankBytes.length + M * 4 + thumbnail.length;
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  out.set(new TextEncoder().encode(MAGIC), 0);
  dv.setUint32(4, FORMAT_VERSION, true);
  dv.setUint32(8, json.length, true);
  out.set(json, 12);
  out.fill(0x20, 12 + json.length, 12 + json.length + pad);
  let off = 12 + json.length + pad;
  out.set(bankBytes, off);
  off += bankBytes.length;
  out.set(new Uint8Array(sq.buffer), off);
  off += M * 4;
  out.set(thumbnail, off);
  return out;
}

export function decodePcprof(buf: ArrayBuffer): DecodedProfile {
  const u8 = new Uint8Array(buf);
  const dv = new DataView(buf);
  if (u8.length < 12 || new TextDecoder().decode(u8.subarray(0, 4)) !== MAGIC) {
    throw new Error('Not a .pcprof file (bad magic)');
  }
  const version = dv.getUint32(4, true);
  if (version !== FORMAT_VERSION) throw new Error(`Unsupported .pcprof version ${version}`);
  const L = dv.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(u8.subarray(12, 12 + L))) as ProfileHeader;
  let off = 12 + L + ((4 - (L % 4)) % 4);
  const { M, D } = header;
  const n = M * D;
  let bank: Float32Array;
  if (header.bankDtype === 'float16') {
    bank = decodeF16(new Uint16Array(buf.slice(off, off + n * 2)));
    off += n * 2;
  } else if (header.bankDtype === 'float32') {
    bank = new Float32Array(buf.slice(off, off + n * 4));
    off += n * 4;
  } else {
    throw new Error(`Unknown bank dtype ${String(header.bankDtype)}`);
  }
  const sqnorm = new Float32Array(buf.slice(off, off + M * 4));
  off += M * 4;
  const thumbnail = u8.slice(off, off + (header.thumbnailBytes ?? 0));
  if (bank.length !== n || sqnorm.length !== M) throw new Error('Truncated .pcprof file');
  return { header, bank, sqnorm, thumbnail };
}
