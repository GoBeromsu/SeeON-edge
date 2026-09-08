import { describe, expect, it } from 'vitest';
import { bedLabel, findMatchingFrame, personLabel } from '@/features/events/ClipOverlayCanvas';
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

describe('overlay labels', () => {
  it('uses the live-view wording: person confidence percent and bed index', () => {
    expect(personLabel(0.894)).toBe('사람 89%');
    expect(personLabel(1)).toBe('사람 100%');
    expect(bedLabel(0)).toBe('침대1');
    expect(bedLabel(4)).toBe('침대5');
  });
});
