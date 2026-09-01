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
  thumbnail_available: true,
  video_error: null,
  scene_available: false,
  scene_frame_count: null,
};

const unavailableClip: Clip = {
  ...availableClip,
  id: 'clip-2',
  video_available: false,
  thumbnail_available: false,
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
  it('renders a lazy, contained thumbnail without mounting video when the thumbnail is available', () => {
    const { host } = render(availableClip);

    expect(host.querySelector('video')).toBeNull();
    const image = host.querySelector('img');
    expect(image?.getAttribute('src')).toBe('/api/v1/clips/clip-1/thumbnail');
    expect(image?.getAttribute('loading')).toBe('lazy');
    expect(image?.getAttribute('width')).toBe('640');
    expect(image?.getAttribute('height')).toBe('360');
    expect(image?.getAttribute('alt')).toBe('301호 낙상 이벤트 썸네일');
    expect(image?.classList.contains('object-contain')).toBe(true);
    expect(host.querySelector('[data-testid="clip-thumbnail-unavailable"]')).toBeNull();
    expect(host.textContent).toContain('301호');
    expect(host.textContent).toContain('낙상');
  });

  it('keeps an available clip selectable with the existing placeholder when no thumbnail exists', () => {
    const { host, onSelect } = render({ ...availableClip, thumbnail_available: false });

    expect(host.querySelector('img')).toBeNull();
    expect(host.querySelector('video')).toBeNull();
    expect(host.querySelector('[data-testid="clip-thumbnail-available"]')).not.toBeNull();

    (host.querySelector('button') as HTMLButtonElement).click();
    expect(onSelect).toHaveBeenCalledWith('clip-1');
  });

  it('switches a failed thumbnail image to the existing playable placeholder', () => {
    const { host } = render(availableClip);
    const image = host.querySelector('img') as HTMLImageElement;

    act(() => image.dispatchEvent(new Event('error')));

    expect(host.querySelector('img')).toBeNull();
    expect(host.querySelector('[data-testid="clip-thumbnail-available"]')).not.toBeNull();
  });

  it('shows the explicit unavailable state instead of a video element when video_available is false', () => {
    const { host } = render(unavailableClip);

    expect(host.querySelector('video')).toBeNull();
    expect(host.querySelector('img')).toBeNull();
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
