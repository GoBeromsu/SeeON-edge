import { describe, expect, it } from 'vitest';
import { normalizeClipAnalysisStatus } from '@/shared/api/clipAnalysisNormalizer';

const available = {
  state: 'available',
  served_timing_identical: true,
  result: {
    source: 'clip_reanalysis',
    clip_id: 'clip-1',
    clip_sha256: 'a'.repeat(64),
    pose_model_sha256: 'b'.repeat(64),
    bed_model_sha256: 'c'.repeat(64),
    decoder_identity: 'pyav-16.1.0/hevc',
    analysis_profile_sha256: 'd'.repeat(64),
    time_base: { numerator: 1, denominator: 12000 },
    frames: [{ pts: 0, status: 'available', boxes: [{ x1: 1, y1: 2, x2: 3, y2: 4, confidence: 0.9 }] }],
    bed_geometries: [{ points: [[0, 0], [1, 0], [1, 1]], provenance_pts: 0 }],
    image_width: 1920,
    image_height: 1080,
  },
};

describe('normalizeClipAnalysisStatus', () => {
  it('accepts the complete available wire response', () => {
    expect(normalizeClipAnalysisStatus(available).state).toBe('available');
  });

  it.each([
    { ...available, unexpected: true },
    { ...available, served_timing_identical: undefined },
    { ...available, result: { ...available.result, frames: [{ ...available.result.frames[0], status: 'unknown' }] } },
  ])('rejects unknown or incomplete fields', (payload) => {
    expect(() => normalizeClipAnalysisStatus(payload)).toThrow();
  });
});
