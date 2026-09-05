import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { recognizeBedZone } from '@/shared/api/client';
import { HttpError } from '@/shared/api/http';
import { BedZoneRecognitionPanel } from '@/shared/ui/BedZoneRecognitionPanel';
import type { BedZone } from '@/shared/api/client';

vi.mock('@/shared/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return { ...actual, recognizeBedZone: vi.fn() };
});

const bedZone: BedZone = {
  polygon: [[0, 0], [100, 0], [100, 100], [0, 100]],
  image_width: 1920,
  image_height: 1080,
  recognized_at: '2026-08-01T00:00:00Z',
};

type ReadResult = { done: boolean; value?: Uint8Array };

/** worker `_mjpeg_http.py:364-371` `_write_part` 의 와이어 포맷은 신경 쓰지 않는다 -- 이 패널
 * 테스트는 프레임 내용이 아니라 스트림의 시작/정지만 검증하면 되므로, `read()` 가 언제까지나
 * 대기하는 컨트롤러블 리더만 있으면 충분하다. */
function createControllableReader(): {
  reader: { read: () => Promise<ReadResult>; cancel: ReturnType<typeof vi.fn> };
} {
  const pendingResolvers: Array<(result: ReadResult) => void> = [];
  const reader = {
    read: (): Promise<ReadResult> => new Promise((resolve) => { pendingResolvers.push(resolve); }),
    cancel: vi.fn(async () => undefined),
  };
  return { reader };
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

function render(zone: BedZone | null, onRecognized = vi.fn()): {
  host: HTMLDivElement;
  root: Root;
  onRecognized: typeof onRecognized;
  rerender: (cameraId: string, nextZone?: BedZone | null) => void;
} {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<BedZoneRecognitionPanel cameraId="cam-1" bedZone={zone} onRecognized={onRecognized} />));
  return {
    host,
    root,
    onRecognized,
    rerender: (cameraId, nextZone = zone) => {
      act(() => root.render(
        <BedZoneRecognitionPanel cameraId={cameraId} bedZone={nextZone} onRecognized={onRecognized} />,
      ));
    },
  };
}

function findButton(host: HTMLElement, label: string): HTMLButtonElement {
  const button = Array.from(host.querySelectorAll('button')).find((candidate) => candidate.textContent === label);
  if (!button) throw new Error(`missing button ${label}`);
  return button;
}

async function flushMicrotasks(times = 3): Promise<void> {
  for (let i = 0; i < times; i += 1) {
    await act(async () => { await Promise.resolve(); });
  }
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.mocked(recognizeBedZone).mockReset();
  vi.stubGlobal('createImageBitmap', vi.fn(async () => ({ width: 4, height: 4, close: vi.fn() })));
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

describe('BedZoneRecognitionPanel', () => {
  it('renders a live MJPEG canvas (not a static <img> snapshot) with the "▶ 인식 시작" trigger', () => {
    stubStreamingFetch();
    const { host, root } = render(null);

    expect(host.querySelector('canvas')).not.toBeNull();
    expect(host.querySelector('img')).toBeNull();
    expect(host.textContent).toContain('침대 영역 인식이 필요합니다.');
    expect(findButton(host, '▶ 인식 시작')).toBeTruthy();

    act(() => root.unmount());
  });

  it('recognizes exactly once per explicit click and allows another click after completion', async () => {
    stubStreamingFetch();
    vi.mocked(recognizeBedZone).mockResolvedValue(bedZone);
    const { host, root, onRecognized } = render(null);
    await flushMicrotasks();

    await act(async () => findButton(host, '▶ 인식 시작').click());
    expect(recognizeBedZone).toHaveBeenCalledTimes(1);
    expect(onRecognized).toHaveBeenCalledWith(bedZone);

    await act(async () => { await vi.advanceTimersByTimeAsync(10_000); });
    expect(recognizeBedZone).toHaveBeenCalledTimes(1);

    await act(async () => findButton(host, '▶ 인식 시작').click());
    expect(recognizeBedZone).toHaveBeenCalledTimes(2);
    expect(onRecognized).toHaveBeenCalledTimes(2);

    act(() => root.unmount());
  });

  it('keeps the persisted polygon overlay visible while recognition is pending', async () => {
    stubStreamingFetch();
    let resolveRecognition!: (zone: BedZone) => void;
    vi.mocked(recognizeBedZone).mockReturnValue(new Promise((resolve) => { resolveRecognition = resolve; }));
    const { host, root } = render(bedZone);
    await flushMicrotasks();

    act(() => findButton(host, '다시 인식').click());
    expect(host.querySelector('polygon')).not.toBeNull();
    expect(findButton(host, '인식 중...').disabled).toBe(true);

    await act(async () => resolveRecognition(bedZone));
    expect(host.querySelector('polygon')).not.toBeNull();
    expect(findButton(host, '다시 인식').disabled).toBe(false);

    act(() => root.unmount());
  });

  it('shows the "침대를 찾지 못했습니다" failure message on a 422 bed_not_found rejection', async () => {
    stubStreamingFetch();
    vi.mocked(recognizeBedZone).mockRejectedValue(
      new HttpError(422, { detail: { error_class: 'bed_not_found' } }),
    );
    const { host, root } = render(null);
    await flushMicrotasks();

    await act(async () => findButton(host, '▶ 인식 시작').click());

    expect(host.querySelector('[role="alert"]')?.textContent).toContain('침대를 찾지 못했습니다');
    expect(findButton(host, '▶ 인식 시작').disabled).toBe(false);

    act(() => root.unmount());
  });

  it('ignores a stale recognition response after the camera changes', async () => {
    stubStreamingFetch();
    let resolveRecognition!: (zone: BedZone) => void;
    vi.mocked(recognizeBedZone).mockReturnValue(new Promise((resolve) => { resolveRecognition = resolve; }));
    const { host, root, onRecognized, rerender } = render(null);
    await flushMicrotasks();

    act(() => findButton(host, '▶ 인식 시작').click());
    expect(findButton(host, '인식 중...').disabled).toBe(true);

    rerender('cam-2');
    expect(findButton(host, '▶ 인식 시작').disabled).toBe(false);
    await act(async () => resolveRecognition(bedZone));
    expect(onRecognized).not.toHaveBeenCalled();
    expect(host.querySelector('[role="alert"]')).toBeNull();

    act(() => root.unmount());
  });

  it('ignores a pending recognition response after unmount and stops the MJPEG stream', async () => {
    const streams = stubStreamingFetch();
    let resolveRecognition!: (zone: BedZone) => void;
    vi.mocked(recognizeBedZone).mockReturnValue(new Promise((resolve) => { resolveRecognition = resolve; }));
    const { host, root, onRecognized } = render(null);
    await flushMicrotasks();

    act(() => findButton(host, '▶ 인식 시작').click());
    expect(recognizeBedZone).toHaveBeenCalledTimes(1);
    expect(streams[0]?.reader.cancel).not.toHaveBeenCalled();

    act(() => root.unmount());
    await act(async () => resolveRecognition(bedZone));
    expect(onRecognized).not.toHaveBeenCalled();
    expect(streams[0]?.reader.cancel).toHaveBeenCalled();
  });
});
