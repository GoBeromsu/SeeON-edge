import { describe, expect, it } from 'vitest';
import { findMatchingFrame } from '@/features/events/ClipOverlayCanvas';
import type { ClipAnalysisResult } from '@/shared/api/types';

const result: ClipAnalysisResult = {
  time_base: { numerator: 1, denominator: 1000 },
  frames: [
    { pts: 1000, status: 'available', boxes: [] },
    { pts: 2000, status: 'ambiguous_timestamp', boxes: [] },
  ],
  bed_geometries: [],
  image_width: 10,
  image_height: 10,
};

describe('findMatchingFrame', () => {
  it('matches only an exact same-frame timestamp', () => {
    expect(findMatchingFrame(result, 1)?.pts).toBe(1000);
    expect(findMatchingFrame(result, 1.001)).toBeNull();
  });

  it('rejects ambiguous frame matches', () => {
    expect(findMatchingFrame(result, 2)).toBeNull();
  });
});
