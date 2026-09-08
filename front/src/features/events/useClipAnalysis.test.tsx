import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { useClipAnalysis } from '@/features/events/useClipAnalysis';

let latest: ReturnType<typeof useClipAnalysis> | null = null;
const roots = new Set<ReturnType<typeof createRoot>>();

function Harness(): JSX.Element {
  latest = useClipAnalysis('clip-1', true);
  return <button type="button" onClick={latest.trigger}>trigger</button>;
}

afterEach(() => {
  act(() => roots.forEach((root) => root.unmount()));
  roots.clear();
  latest = null;
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('useClipAnalysis', () => {
  it('polls a queued analysis every two seconds', async () => {
    vi.useFakeTimers();
    const queued = { state: 'queued', served_media_sha256: 'e'.repeat(64) };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => queued })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => queued });
    vi.stubGlobal('fetch', fetchMock);
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    roots.add(root);
    await act(async () => { root.render(<Harness />); });
    await act(async () => { await Promise.resolve(); });
    expect(latest?.status.state).toBe('queued');
    await act(async () => { await vi.advanceTimersByTimeAsync(2_000); });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('polls a running analysis every two seconds and stops at available', async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ state: 'running', served_media_sha256: 'e'.repeat(64) }) })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ state: 'available', served_media_sha256: 'e'.repeat(64), served_timing_identical: true, result: { source: 'clip_reanalysis', clip_id: 'clip-1', clip_sha256: 'a'.repeat(64), pose_model_sha256: 'b'.repeat(64), bed_model_sha256: 'c'.repeat(64), decoder_identity: 'pyav-16.1.0/hevc', analysis_profile_sha256: 'd'.repeat(64), time_base: { numerator: 1, denominator: 1 }, frames: [], bed_geometries: [], image_width: 1, image_height: 1 } }) });
    vi.stubGlobal('fetch', fetchMock);
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    roots.add(root);
    await act(async () => { root.render(<Harness />); });
    await act(async () => { await Promise.resolve(); });
    expect(latest?.status.state).toBe('running');
    await act(async () => { await vi.advanceTimersByTimeAsync(2_000); });
    expect(latest?.status.state).toBe('available');
    await act(async () => { await vi.advanceTimersByTimeAsync(4_000); });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('starts polling after a 202 trigger status envelope', async () => {
    vi.useFakeTimers();
    const running = { state: 'running', served_media_sha256: 'a'.repeat(64) };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ state: 'idle', served_media_sha256: 'a'.repeat(64) }) })
      .mockResolvedValueOnce({ ok: true, status: 202, json: async () => running })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => running });
    vi.stubGlobal('fetch', fetchMock);
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    roots.add(root);
    await act(async () => { root.render(<Harness />); });
    await act(async () => { await Promise.resolve(); });
    await act(async () => { host.querySelector('button')?.click(); await Promise.resolve(); });
    expect(latest?.status.state).toBe('running');
    await act(async () => { await vi.advanceTimersByTimeAsync(2_000); });
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/v1/clips/clip-1/analysis', expect.objectContaining({ method: 'POST' }));
  });

});
