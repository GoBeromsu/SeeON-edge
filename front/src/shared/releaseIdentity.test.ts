import { describe, expect, it } from 'vitest';
import {
  EDGE_DATABASE_FORMAT_IDENTITY,
  EDGE_DATABASE_SCHEMA_VERSION,
} from '@/shared/releaseIdentity';

describe('release identity', () => {
  it('advertises schema 18 for the baked dashboard', () => {
    expect(EDGE_DATABASE_SCHEMA_VERSION).toBe(18);
    expect(EDGE_DATABASE_FORMAT_IDENTITY).toBe('seeon-edge-v1');
  });
});
