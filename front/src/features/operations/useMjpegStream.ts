import { useCallback, useEffect, useRef, useState } from 'react';

export type MjpegStreamStatus = 'connecting' | 'live' | 'stalled';

export type MjpegStream = {
  /** `<canvas>` 에 그대로 꽂는 ref 콜백. 훅이 프레임마다 이 캔버스에 직접 그린다 -- 캔버스는
   * 마지막으로 그린 프레임을 계속 들고 있으므로, 재연결이 진행되는 동안에도 화면이 비지
   * 않는다 (이게 `<img>` 대신 `<canvas>` 를 쓰는 핵심 이유). */
  canvasRef: (node: HTMLCanvasElement | null) => void;
  status: MjpegStreamStatus;
};

/** worker 하트비트 주기(`_mjpeg_http.py` `HEARTBEAT_INTERVAL_SECONDS = 1.0`)의 3배. 정상
 * 상태라면 1초마다 최소 한 프레임(하트비트 포함)이 오므로, 이 시간 동안 프레임이 전혀
 * 없다는 건 연결이 소리 없이 멈췄다는 뜻이다 -- 그때만 재연결한다. */
const STALL_THRESHOLD_MS = 3_000;
const STALL_CHECK_INTERVAL_MS = 500;
const BASE_BACKOFF_MS = 1_000;
const MAX_BACKOFF_MS = 16_000;

// `fetch` 의 `ReadableStreamDefaultReader<Uint8Array<ArrayBuffer>>` 가 내려주는 청크와 타입을
// 맞추기 위해 우리 쪽 버퍼도 전부 `ArrayBuffer` 기반으로 고정한다 (TS 5.7+ 에서 Uint8Array 가
// 버퍼 종류에 대해 제네릭이 되면서, 이걸 안 맞추면 `SharedArrayBuffer` 가능성 때문에 대입이
// 막힌다 -- `new Blob([...])` 에 그대로 못 넘기는 것도 같은 이유).
type Bytes = Uint8Array<ArrayBuffer>;

const textDecoder = new TextDecoder();
const HEADER_TERMINATOR: Bytes = new Uint8Array([13, 10, 13, 10]); // "\r\n\r\n"

function indexOfBytes(haystack: Bytes, needle: Bytes, fromIndex = 0): number {
  const limit = haystack.length - needle.length;
  outer: for (let i = Math.max(fromIndex, 0); i <= limit; i += 1) {
    for (let j = 0; j < needle.length; j += 1) {
      if (haystack[i + j] !== needle[j]) continue outer;
    }
    return i;
  }
  return -1;
}

function concatBytes(a: Bytes, b: Bytes): Bytes {
  if (a.length === 0) return b;
  const merged: Bytes = new Uint8Array(a.length + b.length);
  merged.set(a, 0);
  merged.set(b, a.length);
  return merged;
}

/**
 * 버퍼에서 완성된 파트 하나를 뽑아낸다 (와이어 포맷은 worker `_mjpeg_http.py:364-371`
 * `_write_part` 참고: `--{boundary}\r\nContent-Type: ...\r\nContent-Length: n\r\n\r\n` 뒤에
 * n바이트의 JPEG, 그리고 `\r\n`). 경계 문자열은 파트 "시작" 위치를 찾는 데만 쓰고, 몸통
 * 길이는 반드시 `Content-Length` 헤더 값으로 정확히 잘라낸다 -- JPEG 바이너리 안에 우연히
 * 경계 문자열과 같은 바이트열이 섞여 있을 수 있으므로, "다음 경계를 스캔해서 몸통 끝을
 * 추정"하는 방식은 오작동할 수 있어 쓰지 않는다.
 *
 * 아직 완성되지 않은 파트(헤더나 바디가 덜 도착함)면 `null` 을 반환한다 -- 호출부는 버퍼를
 * 그대로 두고 다음 청크를 기다린다.
 */
export function extractFrame(
  buffer: Bytes,
  boundaryBytes: Bytes,
): { jpeg: Bytes; rest: Bytes } | null {
  const markerIndex = indexOfBytes(buffer, boundaryBytes);
  if (markerIndex === -1) return null;
  const headerEnd = indexOfBytes(buffer, HEADER_TERMINATOR, markerIndex);
  if (headerEnd === -1) return null;
  const headerText = textDecoder.decode(buffer.subarray(markerIndex, headerEnd));
  const lengthMatch = /Content-Length:\s*(\d+)/i.exec(headerText);
  if (!lengthMatch) {
    // 헤더가 깨졌다면 이 파트는 포기하고 다음 경계부터 다시 찾는다.
    return extractFrame(buffer.subarray(headerEnd + HEADER_TERMINATOR.length), boundaryBytes);
  }
  const contentLength = Number(lengthMatch[1]);
  const bodyStart = headerEnd + HEADER_TERMINATOR.length;
  const bodyEnd = bodyStart + contentLength;
  if (buffer.length < bodyEnd) return null;
  return { jpeg: buffer.slice(bodyStart, bodyEnd), rest: buffer.subarray(bodyEnd) };
}

/** 응답 헤더의 `Content-Type: multipart/x-mixed-replace; boundary=frame` 에서 경계 문자열을
 * 읽는다. 하드코딩하지 않는 이유: 워커가 언젠가 다른 경계 문자열을 쓰게 되어도 깨지지 않게.
 * 헤더가 없거나 파싱에 실패하면 현재 워커 구현값인 `frame` 으로 폴백한다. */
export function parseBoundary(contentType: string | null): string {
  const match = contentType ? /boundary=("?)([^";]+)\1/i.exec(contentType) : null;
  return match?.[2] ?? 'frame';
}

/**
 * MJPEG multipart 스트림을 `fetch` + `<canvas>` 로 직접 구동한다 (스톨 복구 issue #83 후속).
 *
 * 기존에는 `<img src="...">` 로 브라우저의 multipart/x-mixed-replace 디코더에 맡겼다. 문제는
 * `<img>` 가 "마지막 프레임이 언제 왔는지"를 알려주지 않아 스톨을 감지할 수 없다는 것 -- 그
 * 대체물이 "20초마다 무조건 재마운트"였고, 이는 `key` 로 `<img>` 를 통째로 갈아엎어 매번 새
 * 연결을 강제했다. 정상적으로 프레임이 흐르는 스트림까지 20초마다 끊었다는 뜻이다.
 *
 * 이 훅은 `fetch(url, { signal })` 로 응답 바디를 직접 읽어 파트를 파싱하고, 파트마다
 * `createImageBitmap` 으로 만든 비트맵을 캔버스에 `drawImage` 한다. 캔버스는 마지막으로
 * 그린 프레임을 계속 들고 있으므로 재연결 중에도 화면이 비지 않는다. "마지막 프레임 도착
 * 시각"을 추적해 `STALL_THRESHOLD_MS` 를 넘게 프레임이 없을 때만 재연결하고(worker 가
 * 1초마다 하트비트 프레임을 보내므로 정상 상태에서는 이 공백이 나오지 않는다), 재연결이
 * 반복 실패하면 지수 백오프(1초 -> 최대 `MAX_BACKOFF_MS`)를 적용한다.
 *
 * `baseUrl` 이 `null` 이면 스트리밍을 하지 않는다(오프라인 카메라, 화면 밖으로 스크롤된
 * 타일 등) -- `AbortController.abort()` 로 즉시 끊고, 호출부는 스냅샷 등으로 폴백해야 한다.
 * 매번 새 `fetch` 이므로 `<img>` 시절의 캐시버스팅 쿼리(`_r=`)는 더 이상 필요 없다.
 */
export function useMjpegStream(baseUrl: string | null): MjpegStream {
  const [status, setStatus] = useState<MjpegStreamStatus>('connecting');
  const canvasNodeRef = useRef<HTMLCanvasElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const readerRef = useRef<ReadableStreamDefaultReader<Bytes> | null>(null);
  const backoffTimerRef = useRef<number | null>(null);
  const errorCountRef = useRef(0);
  const lastFrameAtRef = useRef(0);
  const generationRef = useRef(0);

  const canvasRef = useCallback((node: HTMLCanvasElement | null) => {
    canvasNodeRef.current = node;
  }, []);

  const clearBackoffTimer = (): void => {
    if (backoffTimerRef.current !== null) {
      window.clearTimeout(backoffTimerRef.current);
      backoffTimerRef.current = null;
    }
  };

  const drawBitmap = (bitmap: ImageBitmap): void => {
    const canvas = canvasNodeRef.current;
    if (!canvas) {
      bitmap.close();
      return;
    }
    // 크기가 그대로면 굳이 다시 대입하지 않는다 -- canvas.width/height 대입은 그 자체로
    // 캔버스를 지워버리므로, 불필요하게 건드리면 프레임 사이에 잠깐씩 빈 화면이 낀다.
    if (canvas.width !== bitmap.width) canvas.width = bitmap.width;
    if (canvas.height !== bitmap.height) canvas.height = bitmap.height;
    canvas.getContext('2d')?.drawImage(bitmap, 0, 0);
    bitmap.close();
  };

  function scheduleReconnect(url: string, generation: number): void {
    if (generationRef.current !== generation) return;
    setStatus('stalled');
    errorCountRef.current += 1;
    const backoff = Math.min(BASE_BACKOFF_MS * 2 ** (errorCountRef.current - 1), MAX_BACKOFF_MS);
    clearBackoffTimer();
    backoffTimerRef.current = window.setTimeout(() => {
      backoffTimerRef.current = null;
      if (generationRef.current !== generation) return;
      void connect(url, generation);
    }, backoff);
  }

  async function handleFrame(jpeg: Bytes, generation: number): Promise<void> {
    lastFrameAtRef.current = Date.now();
    if (generationRef.current !== generation) return;
    errorCountRef.current = 0;
    clearBackoffTimer();
    setStatus('live');
    try {
      const bitmap = await createImageBitmap(new Blob([jpeg], { type: 'image/jpeg' }));
      if (generationRef.current !== generation) {
        bitmap.close();
        return;
      }
      drawBitmap(bitmap);
    } catch {
      // 손상된 프레임 하나는 무시한다 -- 대개 1초 이내에 오는 다음 프레임(하트비트 포함)이
      // 대체한다.
    }
  }

  async function connect(url: string, generation: number): Promise<void> {
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const response = await fetch(url, { signal: controller.signal, cache: 'no-store' });
      if (generationRef.current !== generation) return;
      if (!response.ok || !response.body) {
        throw new Error(`stream responded with ${response.status}`);
      }
      const boundaryBytes = new TextEncoder().encode(`--${parseBoundary(response.headers.get('Content-Type'))}`);
      const reader = response.body.getReader();
      readerRef.current = reader;
      lastFrameAtRef.current = Date.now();
      let buffer = new Uint8Array(0);

      for (;;) {
        const { done, value } = await reader.read();
        if (generationRef.current !== generation) {
          void reader.cancel().catch(() => {});
          return;
        }
        if (done) break;
        buffer = concatBytes(buffer, value);
        let frame = extractFrame(buffer, boundaryBytes);
        while (frame) {
          buffer = frame.rest;
          await handleFrame(frame.jpeg, generation);
          if (generationRef.current !== generation) return;
          frame = extractFrame(buffer, boundaryBytes);
        }
      }
      throw new Error('stream ended');
    } catch (error) {
      if (generationRef.current !== generation) return;
      if (error instanceof DOMException && error.name === 'AbortError') return;
      scheduleReconnect(url, generation);
    }
  }

  // 새 타겟(카메라 전환) 또는 baseUrl === null(오프라인 / 화면 밖) 이면 이전 연결을 즉시
  // 끊고 새로 시작한다. generation 카운터로 abort 이후 뒤늦게 도착하는 프레임/에러가 상태를
  // 건드리지 못하게 막는다.
  useEffect(() => {
    generationRef.current += 1;
    const generation = generationRef.current;
    errorCountRef.current = 0;
    clearBackoffTimer();

    if (!baseUrl) {
      abortRef.current?.abort();
      abortRef.current = null;
      setStatus('connecting');
      return undefined;
    }

    setStatus('connecting');
    lastFrameAtRef.current = Date.now();
    void connect(baseUrl, generation);

    // 조건 없이 20초마다 재마운트하던 기존 타이머를 대신한다: 실제로 프레임이 끊긴
    // 경우에만 재연결하므로, 정상 스트림은 절대 강제로 끊기지 않는다.
    const stallCheck = window.setInterval(() => {
      if (generationRef.current !== generation) return;
      if (backoffTimerRef.current !== null) return; // 이미 재연결 대기 중
      if (Date.now() - lastFrameAtRef.current > STALL_THRESHOLD_MS) {
        abortRef.current?.abort();
        scheduleReconnect(baseUrl, generation);
      }
    }, STALL_CHECK_INTERVAL_MS);

    return () => {
      generationRef.current += 1;
      window.clearInterval(stallCheck);
      clearBackoffTimer();
      abortRef.current?.abort();
      abortRef.current = null;
      void readerRef.current?.cancel().catch(() => {});
      readerRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseUrl]);

  return { canvasRef, status };
}
