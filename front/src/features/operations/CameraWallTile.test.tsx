import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CameraWallTile } from '@/features/operations/CameraWallTile';
import { SnapshotQueue, type SnapshotEntry } from '@/features/operations/SnapshotQueue';
import { type Camera } from '@/shared/api/client';

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

/** worker `_mjpeg_http.py:364-371` `_write_part` 의 와이어 포맷 그대로 파트 하나를 만든다. */
function encodePart(jpeg: Uint8Array<ArrayBuffer>, boundary = 'frame'): Uint8Array<ArrayBuffer> {
  const header = new TextEncoder().encode(`--${boundary}\r\nContent-Type: image/jpeg\r\nContent-Length: ${jpeg.length}\r\n\r\n`);
  const trailer = new TextEncoder().encode('\r\n');
  const out = new Uint8Array(header.length + jpeg.length + trailer.length);
  out.set(header, 0);
  out.set(jpeg, header.length);
  out.set(trailer, header.length + jpeg.length);
  return out;
}

type ReadResult = { done: boolean; value?: Uint8Array };

/** `push` 를 호출하기 전까지는 `read()` 가 계속 대기한다 -- 스톨을 흉내 내는 데 필요하다. */
function createControllableReader(): { reader: { read: () => Promise<ReadResult>; cancel: ReturnType<typeof vi.fn> }; push: (chunk: Uint8Array) => void } {
  const pendingResolvers: Array<(result: ReadResult) => void> = [];
  const queued: ReadResult[] = [];
  const reader = {
    read: (): Promise<ReadResult> => new Promise((resolve) => {
      const next = queued.shift();
      if (next) { resolve(next); return; }
      pendingResolvers.push(resolve);
    }),
    cancel: vi.fn(async () => undefined),
  };
  const deliver = (result: ReadResult): void => {
    const resolve = pendingResolvers.shift();
    if (resolve) resolve(result); else queued.push(result);
  };
  return { reader, push: (chunk) => deliver({ done: false, value: chunk }) };
}

function stubStreamingFetch(): ReturnType<typeof createControllableReader>[] {
  const streams: ReturnType<typeof createControllableReader>[] = [];
  vi.stubGlobal('fetch', vi.fn(() => {
    const controllable = createControllableReader();
    streams.push(controllable);
    return Promise.resolve({
      ok: true,
      status: 200,
      headers: { get: () => 'multipart/x-mixed-replace; boundary=frame' },
      body: { getReader: () => controllable.reader },
    });
  }));
  return streams;
}

async function flushMicrotasks(times = 3): Promise<void> {
  for (let i = 0; i < times; i += 1) {
    // eslint-disable-next-line no-await-in-loop
    await act(async () => { await Promise.resolve(); });
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
  vi.useFakeTimers();
  vi.stubGlobal('IntersectionObserver', MockIntersectionObserver);
  vi.stubGlobal('createImageBitmap', vi.fn(async () => ({ width: 4, height: 4, close: vi.fn() })));
});

afterEach(() => {
  document.body.innerHTML = '';
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('CameraWallTile', () => {
  it('renders a gray bg-muted placeholder with "오프라인" text for an offline camera, not a black frame', () => {
    const { host } = render(offlineCamera);

    const placeholder = host.querySelector('.bg-muted');
    expect(placeholder).not.toBeNull();
    expect(placeholder?.textContent).toBe('오프라인');
    expect(host.querySelector('canvas')).toBeNull();
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

    expect(host.querySelector('canvas[aria-label="101호 실시간 영상"]')).toBeNull();
    expect(host.querySelector('img[alt="101호 최근 영상"]')).not.toBeNull();
  });

  it('mounts the live MJPEG stream once the tile becomes visible, and reveals it once the first frame arrives', async () => {
    const streams = stubStreamingFetch();
    const { host } = render(onlineCamera, loadedSnapshot);
    const target = host.querySelector('button') as Element;
    setIntersecting(target, true);

    const streamCanvas = host.querySelector('canvas[aria-label="101호 실시간 영상"]') as HTMLCanvasElement;
    expect(streamCanvas).not.toBeNull();
    expect(streamCanvas.className).toContain('opacity-0');

    await flushMicrotasks();
    streams[0].push(encodePart(new Uint8Array([1, 2, 3])));
    await flushMicrotasks();

    const revealedCanvas = host.querySelector('canvas[aria-label="101호 실시간 영상"]') as HTMLCanvasElement;
    expect(revealedCanvas.className).not.toContain('opacity-0');
  });

  it('shows a "연결 끊김" badge and falls back to the snapshot once the stream stalls (no frame for 3s+) after being live', async () => {
    const streams = stubStreamingFetch();
    const { host } = render(onlineCamera, loadedSnapshot);
    const target = host.querySelector('button') as Element;
    setIntersecting(target, true);

    await flushMicrotasks();
    streams[0].push(encodePart(new Uint8Array([1, 2, 3])));
    await flushMicrotasks();
    expect(host.querySelector('[role="status"]')).toBeNull();

    await act(async () => { await vi.advanceTimersByTimeAsync(3_600); });

    expect(host.querySelector('[role="status"]')?.textContent).toBe('연결 끊김');
    const stillStreamCanvas = host.querySelector('canvas[aria-label="101호 실시간 영상"]') as HTMLCanvasElement;
    expect(stillStreamCanvas.className).toContain('opacity-0');
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
