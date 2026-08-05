import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ProcessingStatusCard } from '@/features/settings/ProcessingStatusCard';
import type { StatusSnapshot } from '@/shared/api/client';
import type { PollingResource } from '@/shared/api/usePollingResource';

const status: StatusSnapshot = {
  cameras: {},
  stale_after_sec: 30,
  runtime: {
    cameras: {
      'cam-1': {
        camera_id: 'cam-1',
        decode: { requested: 'auto', selected: 'opencv (CPU)', fallback_count: 0, last_reason: null, updated_at_sec: null },
        measured_fps: 24.5,
        latency: { first_attempt_samples: 10, max_sec: 1.84, since_sec: 60 },
        stale: false,
      },
    },
    worker: { alive: true, pid: 123, started_at_sec: 100 },
    device: { backend: 'MPS', available: true, device_name: 'Apple M2', captured_at_sec: 100 },
    clip_recorder: {
      available: true,
      dropped_frames: 0,
      dropped_events: 0,
      failed_writes: 0,
      finalized_clips: 4,
      video_unavailable_clips: 0,
      active_clips: 1,
      encoder: 'libx264',
    },
    stale_after_sec: 30,
  },
};

function makeResource(overrides: Partial<PollingResource<StatusSnapshot>> = {}): PollingResource<StatusSnapshot> {
  return {
    status: 'success',
    data: status,
    error: null,
    lastSuccessAt: Date.now(),
    refreshing: false,
    retry: vi.fn(),
    ...overrides,
  };
}

function render(resource: PollingResource<StatusSnapshot> = makeResource()) {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<ProcessingStatusCard resource={resource} />));
  return { host, root, resource };
}

afterEach(() => {
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

describe('ProcessingStatusCard', () => {
  it('shows a loading message before the first successful fetch', () => {
    const { host, root } = render(makeResource({ status: 'loading', data: null }));
    expect(host.textContent).toContain('불러오는 중입니다');
    act(() => root.unmount());
  });

  it('shows a retry affordance when the initial fetch fails', () => {
    const { host, root, resource } = render(makeResource({ status: 'error', data: null }));
    const retryButton = Array.from(host.querySelectorAll('button')).find((btn) => btn.textContent === '다시 시도');
    act(() => retryButton?.click());
    expect(resource.retry).toHaveBeenCalled();
    act(() => root.unmount());
  });

  it('renders device, decode, encode, and latency without hardcoding CUDA-only wording', () => {
    const { host, root } = render();
    expect(host.textContent).toContain('Apple M2 · MPS');
    expect(host.textContent).toContain('opencv (CPU)');
    expect(host.textContent).toContain('libx264');
    expect(host.textContent).toContain('최대 1.84초');
    expect(host.textContent).not.toMatch(/CUDA|NVML/);
    act(() => root.unmount());
  });

  it('shows 정상 when the worker is alive', () => {
    const { host, root } = render();
    expect(host.textContent).toContain('정상');
    act(() => root.unmount());
  });

  it('shows 중단됨 when the worker has died', () => {
    const dead: StatusSnapshot = { ...status, runtime: { ...status.runtime, worker: { alive: false, pid: null, started_at_sec: null } } };
    const { host, root } = render(makeResource({ data: dead }));
    expect(host.textContent).toContain('중단됨');
    act(() => root.unmount());
  });

  it('falls back to 확인 중 for every field when runtime diagnostics are empty', () => {
    const empty: StatusSnapshot = {
      cameras: {},
      stale_after_sec: null,
      runtime: { cameras: {}, worker: null, device: null, clip_recorder: null, stale_after_sec: null },
    };
    const { host, root } = render(makeResource({ data: empty }));
    const text = host.textContent ?? '';
    expect((text.match(/확인 중/g) ?? []).length).toBeGreaterThanOrEqual(4);
    act(() => root.unmount());
  });
});
