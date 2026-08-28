import { createElement } from 'react';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { extractFrame, parseBoundary, useMjpegStream, type MjpegStream } from '@/shared/api/useMjpegStream';

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

/** 실제 fetch 스트림처럼, `push`/`end` 를 호출하기 전까지는 `read()` 가 계속 대기한다 --
 * 스톨(3초 넘게 프레임 없음)을 흉내 내려면 이 대기가 꼭 필요하다. */
function createControllableReader(): {
  reader: { read: () => Promise<ReadResult>; cancel: ReturnType<typeof vi.fn> };
  push: (chunk: Uint8Array) => void;
  end: () => void;
} {
  const pendingResolvers: Array<(result: ReadResult) => void> = [];
  const queued: ReadResult[] = [];
  const cancel = vi.fn(async () => undefined);

  const reader = {
    read: (): Promise<ReadResult> => new Promise((resolve) => {
      const next = queued.shift();
      if (next) { resolve(next); return; }
      pendingResolvers.push(resolve);
    }),
    cancel,
  };

  const deliver = (result: ReadResult): void => {
    const resolve = pendingResolvers.shift();
    if (resolve) resolve(result); else queued.push(result);
  };

  return {
    reader,
    push: (chunk) => deliver({ done: false, value: chunk }),
    end: () => deliver({ done: true }),
  };
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

let latestStream: MjpegStream | undefined;

function Harness({ url }: { url: string | null }): JSX.Element {
  const stream = useMjpegStream(url);
  latestStream = stream;
  return createElement('canvas', { ref: stream.canvasRef });
}

function mount(baseUrl: string | null): { root: Root; current: () => MjpegStream } {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(createElement(Harness, { url: baseUrl })));
  return { root, current: () => latestStream as MjpegStream };
}

async function flushMicrotasks(times = 3): Promise<void> {
  for (let i = 0; i < times; i += 1) {
    await act(async () => { await Promise.resolve(); });
  }
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

describe('useMjpegStream', () => {
  it('stays live and never reconnects while frames keep arriving within the stall threshold', async () => {
    const streams = stubStreamingFetch();
    const { current, root } = mount('/api/v1/streams/cam-1');

    await flushMicrotasks();
    streams[0].push(encodePart(new Uint8Array([1, 2, 3])));
    await flushMicrotasks();

    expect(current().status).toBe('live');

    // 하트비트가 1초마다 오므로 2.5초 동안 계속 프레임을 흘려보내면 재연결이 없어야 한다.
    for (let elapsed = 0; elapsed < 2_500; elapsed += 1_000) {
      await act(async () => { await vi.advanceTimersByTimeAsync(1_000); });
      streams[0].push(encodePart(new Uint8Array([1, 2, 3])));
      await flushMicrotasks();
    }

    expect(current().status).toBe('live');
    expect((globalThis.fetch as ReturnType<typeof vi.fn>)).toHaveBeenCalledTimes(1);
    act(() => root.unmount());
  });

  it('reconnects only after frames stop arriving for longer than the 3s stall threshold', async () => {
    const streams = stubStreamingFetch();
    const { current, root } = mount('/api/v1/streams/cam-1');

    await flushMicrotasks();
    streams[0].push(encodePart(new Uint8Array([1, 2, 3])));
    await flushMicrotasks();
    expect(current().status).toBe('live');

    // 3초 미만은 재연결하지 않는다.
    await act(async () => { await vi.advanceTimersByTimeAsync(2_900); });
    expect(current().status).toBe('live');
    expect((globalThis.fetch as ReturnType<typeof vi.fn>)).toHaveBeenCalledTimes(1);

    // 3초를 넘기면 그제서야 재연결한다.
    await act(async () => { await vi.advanceTimersByTimeAsync(700); });
    expect(current().status).toBe('stalled');

    // 백오프(1초) 이후 새 fetch 가 나간다.
    await act(async () => { await vi.advanceTimersByTimeAsync(1_100); });
    expect((globalThis.fetch as ReturnType<typeof vi.fn>)).toHaveBeenCalledTimes(2);

    act(() => root.unmount());
  });

  it('cancels the underlying reader when the stream is aborted (unmount)', async () => {
    const streams = stubStreamingFetch();
    const { root } = mount('/api/v1/streams/cam-1');

    await flushMicrotasks();
    streams[0].push(encodePart(new Uint8Array([1, 2, 3])));
    await flushMicrotasks();

    act(() => root.unmount());
    await flushMicrotasks();

    expect(streams[0].reader.cancel).toHaveBeenCalled();
  });

  it('cancels the underlying reader when baseUrl switches to null (offline / offscreen)', async () => {
    const streams = stubStreamingFetch();
    const { current, root } = mount('/api/v1/streams/cam-1');

    await flushMicrotasks();
    streams[0].push(encodePart(new Uint8Array([1, 2, 3])));
    await flushMicrotasks();
    expect(current().status).toBe('live');

    act(() => root.render(createElement(Harness, { url: null })));
    await flushMicrotasks();

    expect(streams[0].reader.cancel).toHaveBeenCalled();
    act(() => root.unmount());
  });
});

describe('extractFrame (multipart parsing)', () => {
  it('cuts the JPEG body using exactly Content-Length, even when the bytes inside it look like a boundary', () => {
    // 경계 문자열과 똑같은 바이트열(--frame)을 JPEG 바디 안에 일부러 심어서, 몸통 길이를
    // "다음 경계 스캔"이 아니라 Content-Length 로만 결정하는지 검증한다. JPEG 매직 바이트
    // (0xff 0xd8 ... 0xff 0xd9)는 문자열 이스케이프로 넣으면 UTF-8 다중 바이트로 인코딩돼
    // 버리므로 숫자 배열로 직접 넣는다.
    const trickyJpeg = new Uint8Array([
      0xff, 0xd8,
      ...new TextEncoder().encode('--frame\r\nnot-a-real-boundary'),
      0xff, 0xd9,
    ]);
    const part = encodePart(trickyJpeg);
    const boundaryBytes = new TextEncoder().encode('--frame');

    const result = extractFrame(part, boundaryBytes);

    expect(result).not.toBeNull();
    expect(result?.jpeg).toEqual(trickyJpeg);
    // 몸통 뒤에 남는 건 `_write_part` 가 프레임마다 붙이는 트레일링 "\r\n" 뿐이다.
    expect(result?.rest.length).toBe(2);
  });

  it('returns null when the body has not fully arrived yet (partial Content-Length)', () => {
    const jpeg = new Uint8Array(10).fill(7);
    const full = encodePart(jpeg);
    const partial = full.slice(0, full.length - 3);
    const boundaryBytes = new TextEncoder().encode('--frame');

    expect(extractFrame(partial, boundaryBytes)).toBeNull();
  });

  it('parses two consecutive parts back to back, handing back the remainder for the next call', () => {
    const first = new Uint8Array([1, 2, 3]);
    const second = new Uint8Array([4, 5, 6, 7]);
    const combined = new Uint8Array([...encodePart(first), ...encodePart(second)]);
    const boundaryBytes = new TextEncoder().encode('--frame');

    const firstResult = extractFrame(combined, boundaryBytes);
    expect(firstResult?.jpeg).toEqual(first);

    const secondResult = extractFrame(firstResult!.rest, boundaryBytes);
    expect(secondResult?.jpeg).toEqual(second);
    // 두 번째 파트 뒤에도 그 파트 자신의 트레일링 "\r\n" 만 남는다.
    expect(secondResult?.rest.length).toBe(2);
  });
});

describe('parseBoundary', () => {
  it('reads the boundary token from the Content-Type header', () => {
    expect(parseBoundary('multipart/x-mixed-replace; boundary=frame')).toBe('frame');
    expect(parseBoundary('multipart/x-mixed-replace; boundary="my-boundary"')).toBe('my-boundary');
  });

  it('falls back to "frame" when the header is missing or has no boundary', () => {
    expect(parseBoundary(null)).toBe('frame');
    expect(parseBoundary('multipart/x-mixed-replace')).toBe('frame');
  });
});
