import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { fetchCameras, fetchClips, fetchSystem } from '@/shared/api/client';
import { usePollingResource, type PollingResource } from '@/shared/api/usePollingResource';

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function mount<T>(
  loader: (signal: AbortSignal) => Promise<T>,
  enabled = true,
  intervalMs = 1_000,
  accept?: (incoming: T, current: T) => boolean,
) {
  let current!: PollingResource<T>;
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  function Harness(): JSX.Element {
    current = usePollingResource(loader, { enabled, intervalMs, accept });
    return <span>{current.status}</span>;
  }
  act(() => root.render(<Harness />));
  return { get current() { return current; }, root, host };
}

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

describe('usePollingResource', () => {
  const validCamera = {
    id: 'cam-1', label: '301호', rtsp_url_masked: 'rtsp://***', mapping_pending: false, status: 'online',
    decode_backend: null, created_at: null, floor_name: null,
  };
  const validClip = {
    clip_id: 'clip-1', camera_id: 'cam-1', event_ref: 'event-1', event_type: 'fall',
    started_at: '2026-07-21T00:00:00Z', duration_s: 4, codec: '', path: null,
    video_available: false, video_error: null, finalized: true,
  };
  const loadCameras = (signal: AbortSignal): Promise<unknown> => fetchCameras(signal);
  const loadClips = (signal: AbortSignal): Promise<unknown> => fetchClips(undefined, signal);
  const malformedNestedCases: Array<[string, (signal: AbortSignal) => Promise<unknown>, unknown]> = [
    ['cameras', loadCameras, { registry_version: 1, cameras: [{ ...validCamera, status: 'stale' }] }],
    ['clips', loadClips, { clips: [{ ...validClip, duration_s: -1 }] }],
  ];
  const laterMalformedNestedCases: Array<[string, (signal: AbortSignal) => Promise<unknown>, unknown, unknown]> = [
    ['cameras', loadCameras, { registry_version: 1, cameras: [validCamera] }, { registry_version: 2, cameras: [{ ...validCamera, label: '' }] }],
    ['clips', loadClips, { clips: [validClip] }, { clips: [{ ...validClip, video_available: 'yes' }] }],
  ];
  const validSystem = {
    updated_at: '2026-07-21T00:00:00.000Z',
    backend: { configured: true, reachable: true, last_ok_at: null },
    version: '2026.07.21',
    image_digests: { ml_api: null, ml_worker: null },
    storage: { clip_store: { total_bytes: null, used_bytes: null, used_pct: null } },
  };

  it('reports a malformed first system success response as an error without fabricated data', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ backend: { reachable: true } }),
    }));
    const view = mount(fetchSystem);

    await act(async () => { await Promise.resolve(); });

    expect(view.current).toMatchObject({ status: 'error', data: null, lastSuccessAt: null, refreshing: false });
    expect(view.current.error).toEqual(expect.objectContaining({ message: 'Invalid system response' }));
    act(() => view.root.unmount());
  });

  it('retains the last-good system snapshot when a later 2xx payload is malformed', async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => validSystem })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ backend: { configured: false } }) });
    vi.stubGlobal('fetch', fetchMock);
    const view = mount(fetchSystem);
    await act(async () => { await Promise.resolve(); });
    expect(view.current).toMatchObject({ status: 'success', data: validSystem });

    await act(async () => { await vi.advanceTimersByTimeAsync(1_000); });

    expect(view.current).toMatchObject({ status: 'error', data: validSystem, refreshing: false });
    expect(view.current.error).toEqual(expect.objectContaining({ message: 'Invalid system response' }));
    act(() => view.root.unmount());
  });

  it.each(malformedNestedCases)('reports a malformed nested %s entry on first load without fabricated empty data', async (_name, loader, payload) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => payload }));
    const view = mount(loader);

    await act(async () => { await Promise.resolve(); });

    expect(view.current).toMatchObject({ status: 'error', data: null, lastSuccessAt: null, refreshing: false });
    act(() => view.root.unmount());
  });

  it.each(laterMalformedNestedCases)('retains last-good %s data when a later success envelope has a malformed nested entry', async (_name, loader, good, malformed) => {
    vi.useFakeTimers();
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => good })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => malformed });
    vi.stubGlobal('fetch', fetchMock);
    const view = mount(loader);
    await act(async () => { await Promise.resolve(); });
    const lastGood = view.current.data;

    await act(async () => { await vi.advanceTimersByTimeAsync(1_000); });

    expect(view.current).toMatchObject({ status: 'error', data: lastGood, refreshing: false });
    act(() => view.root.unmount());
  });

  it('reports an initial loader rejection as an error without inventing empty data', async () => {
    const error = new Error('Invalid cameras response');
    const view = mount<string[]>(vi.fn().mockRejectedValue(error));

    await act(async () => { await Promise.resolve(); });

    expect(view.current).toMatchObject({ status: 'error', data: null, error, lastSuccessAt: null, refreshing: false });
    act(() => view.root.unmount());
  });

  it('does not overlap polls and ignores a stale completion after retry', async () => {
    vi.useFakeTimers();
    const first = deferred<string>();
    const second = deferred<string>();
    const loader = vi.fn().mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const view = mount(loader);
    expect(loader).toHaveBeenCalledTimes(1);

    await act(async () => { await vi.advanceTimersByTimeAsync(5_000); });
    expect(loader).toHaveBeenCalledTimes(1);

    act(() => view.current.retry());
    expect(loader).toHaveBeenCalledTimes(2);
    await act(async () => { second.resolve('new'); await second.promise; });
    await act(async () => { first.resolve('old'); await first.promise; });
    expect(view.current.data).toBe('new');
    act(() => view.root.unmount());
  });

  it('keeps an authoritative mutation response when the follow-up refresh fails', async () => {
    const first = deferred<string>();
    const loader = vi.fn().mockReturnValueOnce(first.promise).mockRejectedValueOnce(new Error('offline'));
    const view = mount(loader);

    act(() => view.current.replace('saved-v2'));
    act(() => view.current.retry());
    await act(async () => { await Promise.resolve(); });
    await act(async () => { first.resolve('stale-v1'); await first.promise; });

    expect(view.current).toMatchObject({ status: 'error', data: 'saved-v2' });
    act(() => view.root.unmount());
  });

  it('rejects a successful refresh older than an authoritative versioned response', async () => {
    const stale = deferred<{ version: number }>();
    const loader = vi.fn().mockResolvedValueOnce({ version: 1 }).mockReturnValueOnce(stale.promise);
    const view = mount<{ version: number }>(
      loader,
      true,
      1_000,
      (incoming, current) => incoming.version >= current.version,
    );
    await act(async () => { await Promise.resolve(); });

    act(() => view.current.replace({ version: 2 }));
    act(() => view.current.retry());
    await act(async () => { stale.resolve({ version: 1 }); await stale.promise; });

    expect(view.current).toMatchObject({ status: 'success', data: { version: 2 } });
    act(() => view.root.unmount());
  });

  it('retains last-good data and timestamp through rejection then retries', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-21T00:00:00Z'));
    const second = deferred<string>();
    const loader = vi.fn().mockResolvedValueOnce('good').mockRejectedValueOnce(new Error('Invalid cameras response')).mockReturnValueOnce(second.promise);
    const view = mount(loader);
    await act(async () => { await Promise.resolve(); });
    expect(view.current).toMatchObject({ status: 'success', data: 'good', lastSuccessAt: new Date('2026-07-21T00:00:00Z').getTime() });

    await act(async () => { await vi.advanceTimersByTimeAsync(1_000); });
    expect(view.current).toMatchObject({ status: 'error', data: 'good', lastSuccessAt: new Date('2026-07-21T00:00:00Z').getTime() });
    act(() => view.current.retry());
    await act(async () => { second.resolve('recovered'); await second.promise; });
    expect(view.current).toMatchObject({ status: 'success', data: 'recovered', error: null });
    act(() => view.root.unmount());
  });

  it('aborts on cleanup and stays idle while disabled', () => {
    const signals: AbortSignal[] = [];
    const loader = vi.fn((signal: AbortSignal) => {
      signals.push(signal);
      return new Promise<string>(() => undefined);
    });
    const disabled = mount(loader, false);
    expect(disabled.current.status).toBe('idle');
    expect(loader).not.toHaveBeenCalled();
    act(() => disabled.root.unmount());

    const active = mount(loader);
    expect(signals[0]?.aborted).toBe(false);
    act(() => active.root.unmount());
    expect(signals[0]?.aborted).toBe(true);
  });
});
