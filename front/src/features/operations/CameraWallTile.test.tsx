import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CameraWallTile } from '@/features/operations/CameraWallTile';
import { SnapshotQueue, type SnapshotEntry } from '@/features/operations/SnapshotQueue';
import { getCameraStreamUrl, type Camera } from '@/shared/api/client';

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

const loadedSnapshot: SnapshotEntry = {
  id: 'cam-1',
  mediaId: 'cam-1',
  state: 'loaded',
  requestUrl: null,
  lastLoadedUrl: '/api/v1/streams/cam-1/snapshot?refresh=0',
  lastLoadedAt: Date.now(),
};

type IOEntry = { target: Element; isIntersecting: boolean };

class MockIntersectionObserver {
  callback: (entries: IOEntry[]) => void;
  elements = new Set<Element>();
  constructor(callback: (entries: IOEntry[]) => void) {
    this.callback = callback;
    ioInstances.push(this);
  }
  observe(el: Element): void { this.elements.add(el); }
  unobserve(el: Element): void { this.elements.delete(el); }
  disconnect(): void { this.elements.clear(); }
}

let ioInstances: MockIntersectionObserver[] = [];

function setIntersecting(target: Element, isIntersecting: boolean): void {
  for (const instance of ioInstances) {
    if (instance.elements.has(target)) {
      act(() => instance.callback([{ target, isIntersecting }]));
    }
  }
}

function render(camera: Camera, snapshot: SnapshotEntry | undefined = undefined): { host: HTMLDivElement; root: Root } {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  const queue = new SnapshotQueue(() => '');
  act(() => root.render(<CameraWallTile camera={camera} snapshot={snapshot} queue={queue} onSelect={vi.fn()} />));
  return { host, root };
}

beforeEach(() => {
  ioInstances = [];
  vi.stubGlobal('IntersectionObserver', MockIntersectionObserver);
});

afterEach(() => {
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
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

  it('does not mount the live MJPEG stream while the tile is offscreen, keeping the snapshot fallback', () => {
    const { host } = render(onlineCamera, loadedSnapshot);

    expect(host.querySelector('img[alt="101호 실시간 영상"]')).toBeNull();
    expect(host.querySelector('img[alt="101호 최근 영상"]')).not.toBeNull();
  });

  it('mounts the live MJPEG stream once the tile becomes visible, and reveals it once loaded', () => {
    const { host } = render(onlineCamera, loadedSnapshot);
    const target = host.querySelector('button') as Element;
    setIntersecting(target, true);

    const streamImg = host.querySelector('img[alt="101호 실시간 영상"]') as HTMLImageElement;
    expect(streamImg).not.toBeNull();
    expect(streamImg.getAttribute('src') ?? '').toContain(getCameraStreamUrl('cam-1'));
    expect(streamImg.className).toContain('opacity-0');

    act(() => streamImg.dispatchEvent(new Event('load')));

    const revealedImg = host.querySelector('img[alt="101호 실시간 영상"]') as HTMLImageElement;
    expect(revealedImg.className).not.toContain('opacity-0');
  });

  it('shows a "연결 끊김" badge and falls back to the snapshot while the stream is reconnecting after an error', () => {
    const { host } = render(onlineCamera, loadedSnapshot);
    const target = host.querySelector('button') as Element;
    setIntersecting(target, true);

    const streamImg = host.querySelector('img[alt="101호 실시간 영상"]') as HTMLImageElement;
    act(() => streamImg.dispatchEvent(new Event('error')));

    expect(host.querySelector('[role="status"]')?.textContent).toBe('연결 끊김');
    const stillStreamImg = host.querySelector('img[alt="101호 실시간 영상"]') as HTMLImageElement;
    expect(stillStreamImg.className).toContain('opacity-0');
  });

  it('shows a "연결 끊김" badge when the snapshot itself goes stale/error after a frame previously loaded', () => {
    const staleSnapshot: SnapshotEntry = { ...loadedSnapshot, state: 'stale' };
    const { host } = render(onlineCamera, staleSnapshot);

    expect(host.querySelector('[role="status"]')?.textContent).toBe('연결 끊김');
    expect(host.querySelector('img[alt="101호 최근 영상"]')).not.toBeNull();
  });

  it('does not show the "연결 끊김" badge for a healthy, loaded snapshot while offscreen', () => {
    const { host } = render(onlineCamera, loadedSnapshot);

    expect(host.querySelector('[role="status"]')).toBeNull();
  });
});
