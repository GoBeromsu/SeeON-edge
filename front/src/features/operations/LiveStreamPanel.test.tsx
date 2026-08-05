import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { LiveStreamPanel } from '@/features/operations/LiveStreamPanel';
import type { Camera, OverlayMode, RuntimeCameraDiagnostics } from '@/shared/api/client';

const onlineCamera: Camera = {
  id: 'cam-1',
  label: '101호',
  rtsp_url_masked: 'rtsp://redacted-camera/a',
  floor_name: '1층',
  status: 'online',
  created_at: null,
};

const offlineCamera: Camera = { ...onlineCamera, id: 'cam-2', label: '102호', status: 'offline' };

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

function render(
  camera: Camera,
  diagnostics: RuntimeCameraDiagnostics | undefined,
  onRetryConnection = vi.fn(),
  onManageConnection = vi.fn(),
  overlayMode: OverlayMode | null = null,
): { host: HTMLDivElement; root: Root; onRetryConnection: typeof onRetryConnection; onManageConnection: typeof onManageConnection } {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(
    <LiveStreamPanel
      camera={camera}
      diagnostics={diagnostics}
      overlayMode={overlayMode}
      onRetryConnection={onRetryConnection}
      onManageConnection={onManageConnection}
    />,
  ));
  return { host, root, onRetryConnection, onManageConnection };
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.stubGlobal('createImageBitmap', vi.fn(async () => ({ width: 4, height: 4, close: vi.fn() })));
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

describe('LiveStreamPanel', () => {
  it('shows a gray disconnected panel with both action buttons when the camera is offline', () => {
    const { host } = render(offlineCamera, undefined);

    expect(host.querySelector('.bg-muted')).not.toBeNull();
    expect(host.textContent).toContain('카메라에 연결할 수 없습니다');
    expect(host.textContent).toContain('탐지가 중단된 상태입니다');
    expect(host.querySelector('canvas')).toBeNull();

    const buttons = Array.from(host.querySelectorAll('button')).map((button) => button.textContent);
    expect(buttons).toContain('재연결 시도');
    expect(buttons).toContain('연결 관리');
  });

  it('calls onRetryConnection when 재연결 시도 is clicked', () => {
    const { host, onRetryConnection } = render(offlineCamera, undefined);

    const retryButton = Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '재연결 시도');
    act(() => retryButton?.click());

    expect(onRetryConnection).toHaveBeenCalledTimes(1);
  });

  it('calls onManageConnection when 연결 관리 is clicked', () => {
    const { host, onManageConnection } = render(offlineCamera, undefined);

    const manageButton = Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '연결 관리');
    act(() => manageButton?.click());

    expect(onManageConnection).toHaveBeenCalledTimes(1);
  });

  it('renders the live stream canvas and FPS diagnostics for an online camera, with no disconnected panel', () => {
    stubStreamingFetch();
    const diagnostics: RuntimeCameraDiagnostics = {
      camera_id: 'cam-1',
      decode: { requested: 'auto', selected: 'cpu', fallback_count: 0, last_reason: null, updated_at_sec: 1 },
      measured_fps: 12.4,
      latency: null,
      stale: false,
    };
    const { host } = render(onlineCamera, diagnostics);

    expect(host.querySelector('canvas')).not.toBeNull();
    expect(host.textContent).toContain('12.4 FPS');
    expect(host.textContent).not.toContain('카메라에 연결할 수 없습니다');
  });

  it('shows "FPS 측정 중" when no measurement has arrived yet (never measured, not stale)', () => {
    stubStreamingFetch();
    const diagnostics: RuntimeCameraDiagnostics = {
      camera_id: 'cam-1',
      decode: { requested: 'auto', selected: null, fallback_count: 0, last_reason: null, updated_at_sec: null },
      measured_fps: null,
      latency: null,
      stale: false,
    };
    const { host } = render(onlineCamera, diagnostics);

    expect(host.textContent).toContain('FPS 측정 중');
    expect(host.textContent).not.toContain('측정 중단');
  });

  it('shows a distinct "측정 중단" warning with the last known FPS once the worker goes stale (issue #160)', () => {
    stubStreamingFetch();
    const diagnostics: RuntimeCameraDiagnostics = {
      camera_id: 'cam-1',
      decode: { requested: 'auto', selected: 'opencv', fallback_count: 0, last_reason: null, updated_at_sec: 1 },
      measured_fps: 5.0,
      latency: null,
      stale: true,
    };
    const { host } = render(onlineCamera, diagnostics);

    // 마지막 값을 지우지 않고, 지금도 그 값이 유효한 것처럼 보여주지도 않는다.
    expect(host.textContent).toContain('측정 중단');
    expect(host.textContent).toContain('5.0');
    expect(host.textContent).not.toContain('5.0 FPS');
  });

  it('shows a bottom-left "라이브 · 온라인" badge when the overlay mode is not fall, matching the design handoff', () => {
    stubStreamingFetch();
    const { host } = render(onlineCamera, undefined);

    expect(host.textContent).toContain('라이브 · 온라인');
    expect(host.textContent).not.toContain('라이브 · 낙상 없음');
  });

  it('shows "라이브 · 온라인" for the bedexit overlay mode too (only fall gets its own label)', () => {
    stubStreamingFetch();
    const { host } = render(onlineCamera, undefined, vi.fn(), vi.fn(), 'bedexit');

    expect(host.textContent).toContain('라이브 · 온라인');
    expect(host.textContent).not.toContain('라이브 · 낙상 없음');
  });

  it('shows a bottom-left "라이브 · 낙상 없음" badge when the overlay mode is fall, matching design-handoff/Eldercare Prototype.dc.html:645', () => {
    stubStreamingFetch();
    const { host } = render(onlineCamera, undefined, vi.fn(), vi.fn(), 'fall');

    expect(host.textContent).toContain('라이브 · 낙상 없음');
    expect(host.textContent).not.toContain('라이브 · 온라인');
  });

  it('shows a "연결 끊김" corner badge once the stream has stalled for more than 3s, without dropping the canvas', async () => {
    stubStreamingFetch();
    const { host } = render(onlineCamera, undefined);

    await flushMicrotasks();
    // 3초 넘게 프레임이 없으면 재연결을 시도하고 그 사실을 배지로만 알린다 -- 검은 오버레이로
    // 캔버스를 덮지 않는다.
    await act(async () => { await vi.advanceTimersByTimeAsync(3_600); });

    expect(host.querySelector('[role="status"]')?.textContent).toContain('연결 끊김');
    expect(host.querySelector('canvas')).not.toBeNull();
  });

  it('clears the "연결 끊김" badge once the stream delivers a frame again', async () => {
    const streams = stubStreamingFetch();
    const { host } = render(onlineCamera, undefined);

    await flushMicrotasks();
    await act(async () => { await vi.advanceTimersByTimeAsync(3_600); });
    expect(host.textContent).toContain('연결 끊김');

    // 백오프 이후 재연결된 두 번째 스트림에 프레임을 흘려보낸다.
    await act(async () => { await vi.advanceTimersByTimeAsync(1_100); });
    await flushMicrotasks();
    streams[streams.length - 1].push(encodePart(new Uint8Array([1, 2, 3])));
    await flushMicrotasks();

    expect(host.textContent).not.toContain('연결 끊김');
  });
});
