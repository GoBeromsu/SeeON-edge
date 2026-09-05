import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ClipPlaybackModal } from '@/features/events/ClipPlaybackModal';
import { formatClipTimestamp } from '@/features/events/formatters';
import type { Clip } from '@/shared/api/types';

const activeRoots = new Set<ReturnType<typeof createRoot>>();

const baseClip: Clip = {
  id: 'clip-1',
  camera_id: 'cam-1',
  camera_label: '301호',
  event_type: 'fall',
  created_at: '2026-08-02T03:12:00Z',
  detected_at: null,
  truncation_reasons: [],
  video_path: '/api/v1/clips/clip-1/video',
  video_available: true,
  thumbnail_available: true,
  video_error: null,
};

function render(clip: Clip | null, open = true, onClose = vi.fn()) {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  activeRoots.add(root);
  act(() => root.render(
    <ClipPlaybackModal
      clip={clip}
      cameraLabel="301호"
      open={open}
      onClose={onClose}
      lookupStatus="success"
      onRetry={vi.fn()}
    />,
  ));
  return { host, root, onClose };
}

function dialog(): HTMLElement {
  return document.querySelector('[role="dialog"]') as HTMLElement;
}

beforeEach(() => {
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
});

afterEach(() => {
  act(() => activeRoots.forEach((root) => root.unmount()));
  activeRoots.clear();
  document.body.innerHTML = '';
  vi.restoreAllMocks();
});

describe('ClipPlaybackModal', () => {
  it('mounts exactly one native video with controls and attempts inline autoplay', async () => {
    render(baseClip);
    await act(async () => Promise.resolve());

    const videos = dialog().querySelectorAll('video');
    expect(videos).toHaveLength(1);
    expect(videos[0]?.controls).toBe(true);
    expect(videos[0]?.autoplay).toBe(true);
    expect(videos[0]?.playsInline).toBe(true);
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalledOnce();
  });

  it('plays the clean media URL and never requests a retired analysis or derivative route', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true, status: 200, json: async () => ({ clip_id: 'clip-1', clean: 'AVAILABLE', snapshot: null }),
    });
    vi.stubGlobal('fetch', fetchMock);

    render(baseClip);
    await act(async () => Promise.resolve());

    const requested = fetchMock.mock.calls.map(([input]) => String(input));
    expect(requested).toEqual(['/api/v1/clips/clip-1/artifacts']);
    expect(dialog().querySelector('video')?.getAttribute('src')).toBe('/api/v1/clips/clip-1/video');
    vi.unstubAllGlobals();
  });

  it('renders no retired annotated or derivative control when no sidecar is available', async () => {
    render(baseClip);
    await act(async () => Promise.resolve());

    expect(dialog().querySelector('[aria-label="증거 보기 선택"]')).toBeNull();
    expect(dialog().querySelector('[aria-label="파생 증거 제어"]')).toBeNull();
    expect(dialog().querySelector('[aria-label="적용 실행 증명"]')).toBeNull();
    expect(dialog().querySelector('input[type="checkbox"]')).toBeNull();
    expect(dialog().querySelectorAll('[aria-pressed]')).toHaveLength(0);
  });

  it('keeps native controls and explains how to continue when autoplay is rejected', async () => {
    vi.mocked(HTMLMediaElement.prototype.play).mockRejectedValueOnce(new DOMException('blocked', 'NotAllowedError'));

    render(baseClip);
    await act(async () => Promise.resolve());

    const video = dialog().querySelector('video');
    expect(video?.controls).toBe(true);
    expect(dialog().querySelector('[role="status"]')?.textContent).toContain('재생 버튼');
  });

  it('renders at the 720px design-spec width', () => {
    render(baseClip);
    expect(dialog().getAttribute('data-size')).toBe('xl');
  });

  it('omits the 크기 row when size_bytes is not present, never fabricating a value', () => {
    render(baseClip);
    expect(dialog().textContent).not.toContain('크기');
  });

  it('shows the 크기 row formatted in human-readable units when size_bytes is present', () => {
    render({ ...baseClip, size_bytes: 8_400_000 });
    expect(dialog().textContent).toContain('크기');
    expect(dialog().textContent).toContain('8.4 MB');
  });

  it('prefers the manifest duration_s for 길이 over video-metadata derivation', () => {
    render({ ...baseClip, duration_s: 12 });
    const rows = Array.from(dialog().querySelectorAll('dt'));
    const durationIndex = rows.findIndex((dt) => dt.textContent === '길이');
    const durationValue = rows[durationIndex]?.nextElementSibling?.textContent;
    expect(durationValue).toBe('0:12');
  });

  it('falls back to a dash for 길이 when duration_s is absent and the video is unavailable', () => {
    render({ ...baseClip, video_available: false, video_error: '저장된 영상을 사용할 수 없습니다.' });
    const rows = Array.from(dialog().querySelectorAll('dt'));
    const durationIndex = rows.findIndex((dt) => dt.textContent === '길이');
    expect(rows[durationIndex]?.nextElementSibling?.textContent).toBe('-');
  });

  it('shows the detection time, falling back to the clip start time for older manifests', () => {
    const detectedAt = '2026-08-02T03:12:00Z';
    const createdAt = '2026-08-02T03:11:30Z';
    render({ ...baseClip, created_at: createdAt, detected_at: detectedAt });
    const rows = Array.from((Array.from(document.querySelectorAll<HTMLElement>('[role="dialog"]')).at(-1) as HTMLElement).querySelectorAll('dt'));
    const timeIndex = rows.findIndex((dt) => dt.textContent === '시간');
    expect(rows[timeIndex]?.nextElementSibling?.textContent).toBe(formatClipTimestamp(detectedAt));

    render({ ...baseClip, created_at: createdAt, detected_at: null });
    const fallbackRows = Array.from((Array.from(document.querySelectorAll<HTMLElement>('[role="dialog"]')).at(-1) as HTMLElement).querySelectorAll('dt'));
    const fallbackTimeIndex = fallbackRows.findIndex((dt) => dt.textContent === '시간');
    expect(fallbackRows[fallbackTimeIndex]?.nextElementSibling?.textContent).toBe(formatClipTimestamp(createdAt));
  });
});
