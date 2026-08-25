import { describe, expect, it } from 'vitest';
import * as clientModule from '@/shared/api/client';

/**
 * Retirement guard for the analysis/derivative/legacy-label clients removed in Task 12/13.
 *
 * This lives in its own file so that the namespace is reached through exactly one top-level
 * request. `client.test.ts` already imports the module for its named exports; adding a second
 * static or dynamic request for the same module in one file gives the Vite SSR runner two
 * concurrent entries into a body that opens with top-level `await`, and the loser can observe a
 * half-built namespace (`TypeError: <export> is not a function`). One file, one request.
 */
describe('api client retirement', () => {

  it.each([
    ['fetchClipAnalysis'],
    ['controlClipDerivative'],
    ['requestClipDerivative'],
    ['setClipLabel'],
  ])('exports no retired %s client', (name) => {
    const exported = Object.keys(clientModule);
    // Positive control first: a bare `not.toContain` would also pass against an empty namespace,
    // so assert the module really is loaded before trusting the absence assertion.
    expect(exported).toContain('fetchClipArtifacts');
    expect(exported).toContain('deleteClip');
    expect(exported).not.toContain(name);
  });
});
