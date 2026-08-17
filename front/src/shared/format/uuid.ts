/**
 * UUIDv4 generation that works on a plain-HTTP LAN dashboard.
 *
 * `crypto.randomUUID()` is a **secure-context-only** API. The edge appliance is
 * reached over plain HTTP on the facility LAN by design (internal-only, never
 * publicly exposed), so on every real install `crypto.randomUUID` is
 * `undefined` and calling it throws. `crypto.getRandomValues()` carries no such
 * restriction, so it stays the entropy source in both contexts.
 *
 * Output matches the `contracts/edge_provisioning_validation.py::require_uuid`
 * contract: lowercase, version nibble in `[1-8]`, variant nibble in `[89ab]`.
 */

const HEX = '0123456789abcdef';

function randomBytes(count: number): Uint8Array {
  const bytes = new Uint8Array(count);
  const source = globalThis.crypto;
  if (source && typeof source.getRandomValues === 'function') {
    source.getRandomValues(bytes);
    return bytes;
  }
  // Last resort only: no Web Crypto at all. Keeps first-run enrollment usable
  // instead of throwing, and the value is a client-supplied installation ref
  // that the Hub re-validates, not a secret.
  for (let index = 0; index < count; index += 1) {
    bytes[index] = Math.floor(Math.random() * 256);
  }
  return bytes;
}

function hex(byte: number): string {
  return HEX[(byte >> 4) & 0x0f] + HEX[byte & 0x0f];
}

export function generateUuidV4(): string {
  const bytes = randomBytes(16);
  bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
  bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant 10xx
  const parts = Array.from(bytes, hex);
  return [
    parts.slice(0, 4).join(''),
    parts.slice(4, 6).join(''),
    parts.slice(6, 8).join(''),
    parts.slice(8, 10).join(''),
    parts.slice(10, 16).join(''),
  ].join('-');
}
