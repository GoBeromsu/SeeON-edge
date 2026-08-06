import { describe, expect, it } from 'vitest';
import { assessRtspSubstreamGuidance } from '@/features/settings/rtspSubstreamGuidance';

describe('assessRtspSubstreamGuidance', () => {
  it('returns null for an empty or whitespace-only address', () => {
    expect(assessRtspSubstreamGuidance('')).toBeNull();
    expect(assessRtspSubstreamGuidance('   ')).toBeNull();
  });

  it('flags an IDIS main-stream URL (trackID=1)', () => {
    const guidance = assessRtspSubstreamGuidance('rtsp://admin:pw@192.0.2.10:554/trackID=1');
    expect(guidance?.level).toBe('main-stream');
    expect(guidance?.message).toContain('trackID=2');
  });

  it('is case-insensitive when matching trackID', () => {
    const guidance = assessRtspSubstreamGuidance('rtsp://admin:pw@192.0.2.10:554/TrackId=1');
    expect(guidance?.level).toBe('main-stream');
  });

  it('flags a URL with no trackID at all, with softer generic wording', () => {
    const guidance = assessRtspSubstreamGuidance('rtsp://admin:pw@192.0.2.10:554/stream');
    expect(guidance?.level).toBe('no-track-id');
    expect(guidance?.message).toContain('trackID=2');
  });

  it('does not flag an explicit substream selection (trackID=2)', () => {
    expect(assessRtspSubstreamGuidance('rtsp://admin:pw@192.0.2.10:554/trackID=2')).toBeNull();
  });

  it('does not flag an explicit non-main, non-default track (e.g. trackID=3)', () => {
    expect(assessRtspSubstreamGuidance('rtsp://admin:pw@192.0.2.10:554/trackID=3')).toBeNull();
  });
});
