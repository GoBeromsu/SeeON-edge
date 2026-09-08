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
const analysisResult = {
  source: 'clip_reanalysis',
  clip_id: 'clip-1',
  clip_sha256: 'a'.repeat(64),
  pose_model_sha256: 'b'.repeat(64),
  bed_model_sha256: 'c'.repeat(64),
  decoder_identity: 'decoder',
  analysis_profile_sha256: 'd'.repeat(64),
  time_base: { numerator: 1, denominator: 1000 },
  frames: [],
  bed_geometries: [],
  image_width: 640,
  image_height: 360,
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

  it('plays the clean media URL and requests clip analysis status', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true, status: 200, json: async () => ({ clip_id: 'clip-1', clean: 'AVAILABLE', snapshot: null }),
    });
    vi.stubGlobal('fetch', fetchMock);

    render(baseClip);
    await act(async () => Promise.resolve());

    const requested = fetchMock.mock.calls.map(([input]) => String(input));
    expect(requested).toEqual(expect.arrayContaining(['/api/v1/clips/clip-1/artifacts', '/api/v1/clips/clip-1/analysis']));
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

  it('unmounts a ready overlay after the video reports a media error', async () => {
    vi.stubGlobal('ResizeObserver', class {
      observe(): void {}
      disconnect(): void {}
    });
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(() => null);
    Object.defineProperty(HTMLVideoElement.prototype, 'requestVideoFrameCallback', {
      configurable: true,
      value: vi.fn(() => 1),
    });
    Object.defineProperty(HTMLVideoElement.prototype, 'cancelVideoFrameCallback', {
      configurable: true,
      value: vi.fn(),
    });
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/artifacts')) {
        return Promise.resolve({ ok: true, status: 200, json: async () => ({ clip_id: 'clip-1', clean: 'AVAILABLE', snapshot: null }) });
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({
          state: 'available',
          served_media_sha256: 'a'.repeat(64),
          served_timing_identical: true,
          result: analysisResult,
        }),
      });
    }));

    render(baseClip);
    await act(async () => Promise.resolve());
    await act(async () => Promise.resolve());

    const video = dialog().querySelector('video') as HTMLVideoElement;
    act(() => video.dispatchEvent(new Event('loadeddata')));
    expect(dialog().querySelector('canvas')).not.toBeNull();
    expect(dialog().querySelector('[aria-label="사람 오버레이"]')).not.toBeNull();
    expect(dialog().querySelector('[aria-label="침대 오버레이"]')).not.toBeNull();
    act(() => video.dispatchEvent(new Event('error')));
    expect(dialog().querySelector('canvas')).toBeNull();
    vi.unstubAllGlobals();
    delete (HTMLVideoElement.prototype as Partial<HTMLVideoElement>).requestVideoFrameCallback;
    delete (HTMLVideoElement.prototype as Partial<HTMLVideoElement>).cancelVideoFrameCallback;
  });

  it('shows the disabled analysis icon without an analysis trigger', async () => {
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/artifacts')) {
        return Promise.resolve({ ok: true, status: 200, json: async () => ({ clip_id: 'clip-1', clean: 'AVAILABLE', snapshot: null }) });
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({ state: 'unavailable', reason: 'analysis_disabled', served_media_sha256: 'a'.repeat(64) }),
      });
    }));

    render(baseClip);
    await act(async () => Promise.resolve());
    await act(async () => Promise.resolve());

    expect(dialog().querySelector('[title="오버레이 분석 비활성"]')).not.toBeNull();
    expect(dialog().querySelector('[aria-label="오버레이 분석"]')).toBeNull();
    vi.unstubAllGlobals();
  });

  it('shows queued analysis preparation with a cancel action and no trigger', async () => {
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/artifacts')) {
        return Promise.resolve({ ok: true, status: 200, json: async () => ({ clip_id: 'clip-1', clean: 'AVAILABLE', snapshot: null }) });
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({ state: 'queued', served_media_sha256: 'a'.repeat(64) }),
      });
    }));

    render(baseClip);
    await act(async () => Promise.resolve());
    await act(async () => Promise.resolve());

    expect(dialog().querySelector('[aria-label="오버레이 준비 중"]')).not.toBeNull();
    expect(dialog().querySelector('[aria-label="오버레이 분석 취소"]')).not.toBeNull();
    expect(dialog().querySelector('[aria-label="오버레이 분석"]')).toBeNull();
    vi.unstubAllGlobals();
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
