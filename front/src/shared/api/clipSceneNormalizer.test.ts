import { describe, expect, it } from 'vitest';
import goldenSceneIndex from '@/shared/api/scene-index.golden.json';
import { normalizeClipScene } from '@/shared/api/clipSceneNormalizer';

describe('normalizeClipScene', () => {
  it('accepts the Python writer golden fixture without losing scene details', () => {
    const scene = normalizeClipScene(goldenSceneIndex);

    expect(scene).not.toBeNull();
    expect(scene!.frames[0].ps[0].b).toEqual([1, 2, 300, 400]);
    expect(scene!.frames[0].lb[0].t).not.toBe('');
    expect(scene!.decision_provenance).toEqual([{
      m: 'fall.v1',
      effective_policy_id: 'c'.repeat(64),
      policy: 'fall-policy.v1',
      runtime_manifest_sha256: 'd'.repeat(64),
    }]);
  });
});
