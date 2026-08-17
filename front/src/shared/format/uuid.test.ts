import { afterEach, describe, expect, it, vi } from 'vitest';
import { generateUuidV4 } from '@/shared/format/uuid';

// Mirrors contracts/edge_provisioning_validation.py::require_uuid
const CONTRACT_UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('generateUuidV4', () => {
  it('matches the edge provisioning UUID contract', () => {
    expect(generateUuidV4()).toMatch(CONTRACT_UUID);
  });

  it('works without crypto.randomUUID, which is secure-context only', () => {
    // A plain-HTTP LAN dashboard has getRandomValues but no randomUUID.
    const insecure = {
      getRandomValues: (bytes: Uint8Array) => {
        for (let index = 0; index < bytes.length; index += 1) bytes[index] = index * 7 + 1;
        return bytes;
      },
    };
    vi.stubGlobal('crypto', insecure);

    expect(() => generateUuidV4()).not.toThrow();
    expect(generateUuidV4()).toMatch(CONTRACT_UUID);
  });

  it('still produces a contract-valid id when Web Crypto is absent entirely', () => {
    vi.stubGlobal('crypto', undefined);

    expect(generateUuidV4()).toMatch(CONTRACT_UUID);
  });

  it('does not repeat across calls', () => {
    const values = new Set(Array.from({ length: 200 }, () => generateUuidV4()));
    expect(values.size).toBe(200);
  });
});
