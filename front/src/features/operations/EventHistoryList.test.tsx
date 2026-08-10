import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { EventHistoryList } from '@/features/operations/EventHistoryList';

const activeRoots = new Set<ReturnType<typeof createRoot>>();

afterEach(() => {
  act(() => activeRoots.forEach((root) => root.unmount()));
  activeRoots.clear();
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
});

describe('EventHistoryList', () => {
  it('renders a lazy thumbnail without mounting card video', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      status: 200,
      json: async () => ({
        clips: [{
          clip_id: 'clip-1', camera_id: 'cam-1', event_ref: 'event-1', event_type: 'fall',
          started_at: '2026-08-02T03:12:00Z', duration_s: 12, codec: 'h264', path: null,
          video_available: true, thumbnail_available: true, video_error: null, finalized: true,
        }],
      }),
    })));
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    activeRoots.add(root);

    await act(async () => {
      root.render(<EventHistoryList cameraId="cam-1" cameraLabel="301호" onSelectClip={vi.fn()} />);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(host.querySelector('video')).toBeNull();
    const image = host.querySelector('img');
    expect(image?.getAttribute('src')).toBe('/api/v1/clips/clip-1/thumbnail');
    expect(image?.getAttribute('loading')).toBe('lazy');
    expect(image?.getAttribute('alt')).toBe('301호 낙상 이벤트 썸네일');
  });
});
