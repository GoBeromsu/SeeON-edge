import { describe, expect, it } from 'vitest';
import { normalizeClipScene } from '@/shared/api/clipSceneNormalizer';
import { closestSceneFrameAtOrBefore } from '@/features/events/useClipScene';
import type { ClipSceneFrame } from '@/shared/api/types';

const frames = [0, 100, 220].map((t) => ({ t } as ClipSceneFrame));

describe('clip scene selection', () => {
  it('selects the latest frame at or before playback time and hides stale samples', () => {
    expect(closestSceneFrameAtOrBefore(frames, -1)).toBeNull();
    expect(closestSceneFrameAtOrBefore(frames, 100)?.t).toBe(100);
    expect(closestSceneFrameAtOrBefore(frames, 219)?.t).toBe(100);
    expect(closestSceneFrameAtOrBefore(frames, 221)?.t).toBe(220);
    expect(closestSceneFrameAtOrBefore(frames, 341)).toBeNull();
  });

  it('disables an unsupported future sidecar version', () => {
    expect(normalizeClipScene({
      camera_id: 'camera-1', clip_id: 'clip-1', coordinate_space: 'source-pixels', scene_index_schema_version: 2,
    })).toBeNull();
  });
});
