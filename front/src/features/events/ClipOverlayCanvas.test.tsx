import { describe, expect, it } from 'vitest';
import { bedLabel, findMatchingFrame, labelOrigin, personLabel } from '@/features/events/ClipOverlayCanvas';
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
  it('formats the detector confidence as a rounded percent and beds as a 1-based index', () => {
    // Locale-independent contract: the numbers, not the prose.
    expect(personLabel(0.894)).toMatch(/\b89%/);
    expect(personLabel(1)).toMatch(/\b100%/);
    expect(bedLabel(0)).toMatch(/1$/);
    expect(bedLabel(4)).toMatch(/5$/);
  });

  it('keeps the label chip inside the drawable width and above the shape when there is room', () => {
    expect(labelOrigin(10, 40, 50, 300)).toEqual({ left: 10, top: 22 });
    expect(labelOrigin(290, 40, 50, 300)).toEqual({ left: 250, top: 22 });
    expect(labelOrigin(-5, 5, 50, 300)).toEqual({ left: 0, top: 5 });
  });
});
