import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ClipCard } from '@/features/events/ClipCard';
import type { Clip } from '@/shared/api/types';

const availableClip: Clip = {
  id: 'clip-1',
  camera_id: 'cam-1',
  camera_label: '301호',
  event_type: 'fall',
  created_at: '2026-08-02T03:12:00Z',
  video_path: '/api/v1/clips/clip-1/video',
  video_available: true,
  video_error: null,
};

const unavailableClip: Clip = {
  ...availableClip,
  id: 'clip-2',
  video_available: false,
  video_error: '저장된 영상을 사용할 수 없습니다.',
};

afterEach(() => {
  document.body.innerHTML = '';
});

function render(clip: Clip, onSelect = vi.fn()): { host: HTMLDivElement; onSelect: typeof onSelect } {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<ClipCard clip={clip} cameraLabel={clip.camera_label} onSelect={onSelect} />));
  return { host, onSelect };
}

describe('ClipCard', () => {
  it('renders a metadata-only preview without mounting video when the clip is available', () => {
    const { host } = render(availableClip);

    expect(host.querySelector('video')).toBeNull();
    expect(host.querySelector('[data-testid="clip-thumbnail-available"]')).not.toBeNull();
    expect(host.querySelector('[data-testid="clip-thumbnail-unavailable"]')).toBeNull();
    expect(host.textContent).toContain('301호');
    expect(host.textContent).toContain('낙상');
  });

  it('shows the explicit unavailable state instead of a video element when video_available is false', () => {
    const { host } = render(unavailableClip);

    expect(host.querySelector('video')).toBeNull();
    const unavailable = host.querySelector('[data-testid="clip-thumbnail-unavailable"]');
    expect(unavailable).not.toBeNull();
    expect(unavailable?.textContent).toBe('저장된 영상을 사용할 수 없습니다.');
  });

  it('calls onSelect with the clip id when clicked', () => {
    const { host, onSelect } = render(availableClip);

    (host.querySelector('button') as HTMLButtonElement).click();

    expect(onSelect).toHaveBeenCalledWith('clip-1');
  });
});
