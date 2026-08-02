import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { CameraWallTile } from '@/features/operations/CameraWallTile';
import { SnapshotQueue } from '@/features/operations/SnapshotQueue';
import type { Camera } from '@/shared/api/client';

const onlineCamera: Camera = {
  id: 'cam-1',
  label: '101호',
  rtsp_url_masked: 'rtsp://redacted-camera/a',
  floor_name: '1층',
  status: 'online',
  created_at: null,
};

const offlineCamera: Camera = {
  ...onlineCamera,
  id: 'cam-2',
  label: '102호',
  status: 'offline',
};

function render(camera: Camera): { host: HTMLDivElement; root: Root } {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  const queue = new SnapshotQueue(() => '');
  act(() => root.render(<CameraWallTile camera={camera} snapshot={undefined} queue={queue} onSelect={vi.fn()} />));
  return { host, root };
}

afterEach(() => {
  document.body.innerHTML = '';
});

describe('CameraWallTile', () => {
  it('renders a gray bg-muted placeholder with "오프라인" text for an offline camera, not a black frame', () => {
    const { host } = render(offlineCamera);

    const placeholder = host.querySelector('.bg-muted');
    expect(placeholder).not.toBeNull();
    expect(placeholder?.textContent).toBe('오프라인');
    expect(host.querySelector('img')).toBeNull();
    expect(host.querySelector('.event-media-frame')).toBeNull();
  });

  it('shows the camera name and a red status dot in the bottom bar for an offline camera', () => {
    const { host } = render(offlineCamera);

    expect(host.textContent).toContain('102호');
    expect(host.querySelector('.bg-status-rejectedFg')).not.toBeNull();
  });

  it('does not render the offline placeholder for an online camera', () => {
    const { host } = render(onlineCamera);

    const placeholders = Array.from(host.querySelectorAll('*')).filter((el) => el.textContent === '오프라인');
    expect(placeholders.length).toBe(0);
  });
});
