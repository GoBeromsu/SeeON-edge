import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';
import { ClipThumbnail } from '@/shared/ui/ClipThumbnail';
import type { Clip } from '@/shared/api/types';

const activeRoots = new Set<ReturnType<typeof createRoot>>();

const availableClip: Clip = {
  id: 'clip-1',
  camera_id: 'cam-1',
  camera_label: '301호',
  event_type: 'fall',
  created_at: '2026-08-02T03:12:00Z',
  video_path: '/api/v1/clips/clip-1/video',
  video_available: true,
  thumbnail_available: true,
  video_error: null,
  scene_available: false,
  scene_frame_count: null,
};

function renderThumbnail(clip: Clip) {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  activeRoots.add(root);

  const rerender = (nextClip: Clip): void => {
    act(() => root.render(<ClipThumbnail clip={nextClip} alt="301호 낙상 이벤트 썸네일" />));
  };

  rerender(clip);
  return { host, rerender };
}

afterEach(() => {
  act(() => activeRoots.forEach((root) => root.unmount()));
  activeRoots.clear();
  document.body.innerHTML = '';
});

describe('ClipThumbnail', () => {
  it('retries a transient thumbnail failure when fresh clip data arrives', () => {
    const { host, rerender } = renderThumbnail(availableClip);
    act(() => host.querySelector('img')?.dispatchEvent(new Event('error')));
    expect(host.querySelector('img')).toBeNull();

    rerender({ ...availableClip });

    expect(host.querySelector('img')?.getAttribute('src')).toBe('/api/v1/clips/clip-1/thumbnail');
  });

  it('recovers a failed thumbnail when the clip source changes', () => {
    const { host, rerender } = renderThumbnail(availableClip);
    act(() => host.querySelector('img')?.dispatchEvent(new Event('error')));

    rerender({
      ...availableClip,
      id: 'clip-2',
      video_path: '/api/v1/clips/clip-2/video',
    });

    expect(host.querySelector('img')?.getAttribute('src')).toBe('/api/v1/clips/clip-2/thumbnail');
  });

  it('loads a thumbnail when refreshed availability changes from missing to available', () => {
    const { host, rerender } = renderThumbnail({ ...availableClip, thumbnail_available: false });
    expect(host.querySelector('img')).toBeNull();

    rerender({ ...availableClip, thumbnail_available: true });

    expect(host.querySelector('img')?.getAttribute('loading')).toBe('lazy');
  });
});
