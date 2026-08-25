import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { recognizeBedZone } from '@/shared/api/client';
import { HttpError } from '@/shared/api/http';
import { BedZoneRecognitionPanel } from '@/features/settings/BedZoneRecognitionPanel';
import type { BedZone } from '@/shared/api/client';

vi.mock('@/shared/api/client', async () => {
  const { withOverrides } = await vi.importActual<typeof import('@/test/moduleMock')>('@/test/moduleMock');
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return withOverrides(actual, { recognizeBedZone: vi.fn() });
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

function render(zone: BedZone | null, onRecognized = vi.fn()): { host: HTMLDivElement; root: Root; onRecognized: typeof onRecognized } {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<BedZoneRecognitionPanel cameraId="cam-1" bedZone={zone} onRecognized={onRecognized} />));
  return { host, root, onRecognized };
}

function findButton(host: HTMLElement, label: string): HTMLButtonElement {
  const button = Array.from(host.querySelectorAll('button')).find((candidate) => candidate.textContent === label);
  if (!button) throw new Error(`missing button ${label}`);
  return button;
}

async function flushMicrotasks(times = 3): Promise<void> {
  for (let i = 0; i < times; i += 1) {
    // eslint-disable-next-line no-await-in-loop
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

  it('toggles a recognition session: recognizes immediately, keeps re-recognizing every 2s, and stops on "■ 인식 중지"', async () => {
    stubStreamingFetch();
    vi.mocked(recognizeBedZone).mockResolvedValue(bedZone);
    const { host, root, onRecognized } = render(null);
    await flushMicrotasks();

    await act(async () => findButton(host, '▶ 인식 시작').click());
    expect(recognizeBedZone).toHaveBeenCalledTimes(1);
    expect(onRecognized).toHaveBeenCalledWith(bedZone);
    expect(findButton(host, '■ 인식 중지')).toBeTruthy();

    await act(async () => { await vi.advanceTimersByTimeAsync(2_000); });
    expect(recognizeBedZone).toHaveBeenCalledTimes(2);

    await act(async () => { await vi.advanceTimersByTimeAsync(2_000); });
    expect(recognizeBedZone).toHaveBeenCalledTimes(3);

    await act(async () => findButton(host, '■ 인식 중지').click());
    await act(async () => { await vi.advanceTimersByTimeAsync(6_000); });
    // 세션을 멈춘 뒤에는 시간이 지나도 더 이상 재인식을 호출하지 않는다.
    expect(recognizeBedZone).toHaveBeenCalledTimes(3);
    expect(findButton(host, '▶ 인식 시작')).toBeTruthy();

    act(() => root.unmount());
  });

  it('keeps the polygon overlay visible while a session is running instead of clearing it every frame', async () => {
    stubStreamingFetch();
    vi.mocked(recognizeBedZone).mockResolvedValue(bedZone);
    const { host, root } = render(bedZone);
    await flushMicrotasks();

    await act(async () => findButton(host, '▶ 인식 시작').click());
    expect(host.querySelector('polygon')).not.toBeNull();

    await act(async () => { await vi.advanceTimersByTimeAsync(2_000); });
    // 재인식 사이클이 한 번 더 지나도(=캔버스가 새 프레임으로 다시 그려져도) 오버레이 svg는
    // 캔버스와 별개 엘리먼트이므로 계속 남아 있어야 한다.
    expect(host.querySelector('polygon')).not.toBeNull();

    act(() => root.unmount());
  });

  it('shows the "침대를 찾지 못했습니다" failure message on a 422 bed_not_found rejection without ending the session', async () => {
    stubStreamingFetch();
    vi.mocked(recognizeBedZone).mockRejectedValue(
      new HttpError(422, { detail: { error_class: 'bed_not_found' } }),
    );
    const { host, root } = render(null);
    await flushMicrotasks();

    await act(async () => findButton(host, '▶ 인식 시작').click());

    expect(host.querySelector('[role="alert"]')?.textContent).toContain('침대를 찾지 못했습니다');
    expect(findButton(host, '■ 인식 중지')).toBeTruthy();

    act(() => root.unmount());
  });

  it('stops both the recognition timer and the MJPEG stream once the panel unmounts', async () => {
    const streams = stubStreamingFetch();
    vi.mocked(recognizeBedZone).mockResolvedValue(bedZone);
    const { host, root } = render(null);
    await flushMicrotasks();

    await act(async () => findButton(host, '▶ 인식 시작').click());
    expect(recognizeBedZone).toHaveBeenCalledTimes(1);
    expect(streams[0]?.reader.cancel).not.toHaveBeenCalled();

    act(() => root.unmount());

    // 언마운트 후에는 타이머가 더 이상 돌지 않는다 -- 재인식 호출이 늘지 않아야 한다.
    await act(async () => { await vi.advanceTimersByTimeAsync(10_000); });
    expect(recognizeBedZone).toHaveBeenCalledTimes(1);

    // useMjpegStream의 언마운트 정리로 스트림 리더도 취소된다.
    expect(streams[0]?.reader.cancel).toHaveBeenCalled();
  });
});
