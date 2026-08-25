import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CameraWall } from '@/features/operations/CameraWall';
import { useStatusResource } from '@/shared/api/usePollingResource';
import type { Camera, RuntimeDetectionDiagnostics, StatusSnapshot } from '@/shared/api/client';
import type { PollingResource } from '@/shared/api/usePollingResource';

// Narrowest seam: the wall's own status subscription. The camera list, filtering, snapshot queue
// and tiles all stay real so this exercises the actual wall, not a stand-in.
vi.mock('@/shared/api/usePollingResource', async () => {
  const actual = await vi.importActual<typeof import('@/shared/api/usePollingResource')>('@/shared/api/usePollingResource');
  return withOverrides(actual, { useStatusResource: vi.fn() });
});

const cameraA: Camera = {
  id: 'cam-1',
  label: '101호',
  rtsp_url_masked: 'rtsp://redacted-camera/a',
  floor_name: '1층',
  status: 'online',
  created_at: null,
};

const cameraB: Camera = { ...cameraA, id: 'cam-2', label: '201호', floor_name: '2층' };

function detection(overrides: Partial<RuntimeDetectionDiagnostics> = {}): RuntimeDetectionDiagnostics {
  return {
    state: 'healthy',
    reason: null,
    recent_success_rate: 1,
    last_completed_at_sec: 5,
    evaluation_window_sec: 120,
    timeout_sec: 120,
    ...overrides,
  };
}

function snapshotWith(detections: Record<string, RuntimeDetectionDiagnostics>): StatusSnapshot {
  const cameras: StatusSnapshot['runtime']['cameras'] = {};
  Object.entries(detections).forEach(([cameraId, value]) => {
    cameras[cameraId] = {
      camera_id: cameraId,
      decode: { requested: 'auto', selected: 'opencv', fallback_count: 0, last_reason: null, updated_at_sec: null },
      measured_fps: 12,
      latency: null,
      stale: false,
      detection: value,
    };
  });
  return {
    cameras: {},
    stale_after_sec: 30,
    runtime: {
      cameras,
      worker: { alive: true, pid: 1, started_at_sec: 1 },
      device: { backend: 'cpu', available: true, device_name: 'CPU', captured_at_sec: 1 },
      clip_export_applied: { enabled: false, version: 0, freshness: 'fresh' },
      clip_recorder: {
        available: true,
        dropped_frames: 0,
        dropped_events: 0,
        failed_writes: 0,
        finalized_clips: 0,
        video_unavailable_clips: 0,
        active_clips: 0,
        encoder: 'libx264',
      },
      stale_after_sec: 30,
    },
  };
}

function statusResource(data: StatusSnapshot | null): PollingResource<StatusSnapshot> {
  return {
    status: data ? 'success' : 'loading',
    data,
    error: null,
    lastSuccessAt: data ? Date.now() : null,
    refreshing: false,
    retry: vi.fn(),
    replace: vi.fn(),
  };
}

type WallOptions = {
  cameras?: Camera[];
  floor?: string | undefined;
  status?: PollingResource<StatusSnapshot>['status'];
};

function renderWall(
  snapshot: StatusSnapshot | null,
  { cameras = [cameraA, cameraB], floor = undefined }: WallOptions = {},
): { host: HTMLDivElement; root: Root; rerender: (next: StatusSnapshot | null) => void } {
  vi.mocked(useStatusResource).mockReturnValue(statusResource(snapshot));
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  const paint = (): void => {
    act(() => root.render(
      <CameraWall
        status="success"
        cameras={cameras}
        floor={floor}
        onFloorChange={vi.fn()}
        onSelectCamera={vi.fn()}
        onRetry={vi.fn()}
      />,
    ));
  };
  paint();
  return {
    host,
    root,
    rerender: (next) => {
      vi.mocked(useStatusResource).mockReturnValue(statusResource(next));
      paint();
    },
  };
}

function tileBadge(host: HTMLElement, cameraId: string): HTMLElement | null {
  return host.querySelector(`[data-camera-id="${cameraId}"] [data-testid="tile-detection"]`);
}

function summary(host: HTMLElement): HTMLElement | null {
  return host.querySelector('[data-testid="wall-detection-summary"]');
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn());
  vi.mocked(useStatusResource).mockReturnValue(statusResource(null));
});

afterEach(() => {
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe('기준선 — 감지 상태를 붙이기 전의 카메라 월 동작', () => {
  it('보이는 카메라마다 타일을 하나씩 그리고 라이브 스트림은 열지 않는다', () => {
    const { host } = renderWall(null);

    expect(host.querySelectorAll('[data-camera-id]').length).toBe(2);
    expect(host.querySelector('canvas')).toBeNull();
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it('층 필터가 걸리면 해당 층 카메라만 남는다', () => {
    const { host } = renderWall(null, { floor: '2층' });

    expect(host.querySelectorAll('[data-camera-id]').length).toBe(1);
    expect(host.querySelector('[data-camera-id="cam-2"]')).not.toBeNull();
  });

  it('온라인/오프라인 집계 뱃지는 그대로 유지된다', () => {
    const { host } = renderWall(null, { cameras: [cameraA, { ...cameraB, status: 'offline' }] });

    expect(host.textContent).toContain('온라인');
    expect(host.textContent).toContain('오프라인');
  });
});

describe('카메라 월 — 타일 감지 뱃지', () => {
  it('로컬 카메라 id로 진단을 찾아 붙인다', () => {
    const { host } = renderWall(snapshotWith({
      'cam-1': detection({ state: 'healthy' }),
      'cam-2': detection({ state: 'blind', reason: 'pose_not_completing' }),
    }));

    expect(tileBadge(host, 'cam-1')?.dataset.detectionState).toBe('healthy');
    expect(tileBadge(host, 'cam-2')?.dataset.detectionState).toBe('blind');
  });

  it.each([
    ['healthy'],
    ['starting'],
    ['unknown'],
    ['disabled'],
    ['blind'],
  ] as const)('%s 상태는 읽을 수 있는 글자를 가진 뱃지로 보인다', (state) => {
    const { host } = renderWall(snapshotWith({
      'cam-1': detection({ state, reason: state === 'blind' ? 'no_completed_cycles' : null }),
    }));

    const badge = tileBadge(host, 'cam-1');
    expect(badge?.dataset.detectionState).toBe(state);
    // 색만으로 뜻을 전하지 않는다: 읽을 수 있는 글자 + 형태 신호가 항상 함께 있다.
    expect((badge?.textContent ?? '').trim().length).toBeGreaterThan(0);
    expect(badge?.querySelector('[aria-hidden="true"]')).not.toBeNull();
  });

  it('감지 중단 타일은 다른 상태와 구분되는 경고 형태를 갖는다', () => {
    const { host } = renderWall(snapshotWith({
      'cam-1': detection({ state: 'blind', reason: 'pose_not_completing' }),
      'cam-2': detection({ state: 'healthy' }),
    }));

    expect(tileBadge(host, 'cam-1')?.querySelector('svg')).not.toBeNull();
    expect(tileBadge(host, 'cam-2')?.querySelector('svg')).toBeNull();
  });

  it('진단이 없는 카메라는 확인 불가로 남고 정상으로 지어내지 않는다', () => {
    const { host } = renderWall(snapshotWith({ 'cam-1': detection({ state: 'healthy' }) }));

    expect(tileBadge(host, 'cam-2')?.dataset.detectionState).toBe('unknown');
  });

  it('연결된 카메라도 감지 경보를 동시에 띄운다', () => {
    const { host } = renderWall(snapshotWith({
      'cam-1': detection({ state: 'blind', reason: 'decision_not_completing' }),
    }));

    const tile = host.querySelector('[data-camera-id="cam-1"]');
    expect(tile?.querySelector('img[alt="101호 최근 영상"], .event-media-frame')).not.toBeNull();
    expect(tileBadge(host, 'cam-1')?.dataset.detectionState).toBe('blind');
  });

  it('스냅샷/미디어 표시는 감지 뱃지가 붙어도 그대로다', () => {
    const { host } = renderWall(snapshotWith({ 'cam-1': detection({ state: 'blind', reason: 'pose_not_completing' }) }));

    expect(host.querySelector('canvas')).toBeNull();
    expect(globalThis.fetch).not.toHaveBeenCalled();
    expect(host.querySelectorAll('[data-camera-id]').length).toBe(2);
  });
});

describe('카메라 월 — 지속형 경보 요약', () => {
  it('보이는 카메라 중 하나라도 감지 중단이면 alert 요약을 하나 띄운다', () => {
    const { host } = renderWall(snapshotWith({
      'cam-1': detection({ state: 'blind', reason: 'pose_not_completing' }),
      'cam-2': detection({ state: 'healthy' }),
    }));

    const alerts = host.querySelectorAll('[role="alert"]');
    expect(alerts.length).toBe(1);
    expect(summary(host)?.getAttribute('role')).toBe('alert');
    expect((summary(host)?.textContent ?? '').trim().length).toBeGreaterThan(0);
  });

  it('여러 대가 중단이어도 요약은 하나만 남는다', () => {
    const { host } = renderWall(snapshotWith({
      'cam-1': detection({ state: 'blind', reason: 'pose_not_completing' }),
      'cam-2': detection({ state: 'blind', reason: 'no_completed_cycles' }),
    }));

    expect(host.querySelectorAll('[role="alert"]').length).toBe(1);
    expect(summary(host)?.dataset.blindCount).toBe('2');
  });

  it('감지 중단 카메라가 층 필터로 빠지면 요약도 사라진다', () => {
    const { host } = renderWall(
      snapshotWith({
        'cam-1': detection({ state: 'blind', reason: 'pose_not_completing' }),
        'cam-2': detection({ state: 'healthy' }),
      }),
      { floor: '2층' },
    );

    expect(summary(host)).toBeNull();
    expect(host.querySelectorAll('[role="alert"]').length).toBe(0);
  });

  it('모두 정상이면 요약이 없다', () => {
    const { host } = renderWall(snapshotWith({
      'cam-1': detection({ state: 'healthy' }),
      'cam-2': detection({ state: 'healthy' }),
    }));

    expect(summary(host)).toBeNull();
    expect(host.querySelectorAll('[role="alert"]').length).toBe(0);
  });

  it('경보는 다음 스냅샷에서도 유지된다', () => {
    const blind = snapshotWith({
      'cam-1': detection({ state: 'blind', reason: 'decision_not_completing' }),
      'cam-2': detection({ state: 'healthy' }),
    });
    const { host, rerender } = renderWall(blind);
    expect(summary(host)?.getAttribute('role')).toBe('alert');

    rerender(snapshotWith({
      'cam-1': detection({ state: 'blind', reason: 'decision_not_completing' }),
      'cam-2': detection({ state: 'healthy' }),
    }));

    expect(summary(host)?.getAttribute('role')).toBe('alert');
    expect(tileBadge(host, 'cam-1')?.dataset.detectionState).toBe('blind');
  });

  it('정상으로 회복한 첫 스냅샷에서 요약과 타일 경보가 함께 사라진다', () => {
    const { host, rerender } = renderWall(snapshotWith({
      'cam-1': detection({ state: 'blind', reason: 'no_completed_cycles' }),
      'cam-2': detection({ state: 'healthy' }),
    }));
    expect(summary(host)).not.toBeNull();

    rerender(snapshotWith({
      'cam-1': detection({ state: 'healthy' }),
      'cam-2': detection({ state: 'healthy' }),
    }));

    expect(summary(host)).toBeNull();
    expect(host.querySelectorAll('[role="alert"]').length).toBe(0);
    expect(tileBadge(host, 'cam-1')?.dataset.detectionState).toBe('healthy');
  });

  it('매핑 대기와 감지 중단은 월에서도 각자 남는다', () => {
    const { host } = renderWall(
      snapshotWith({ 'cam-1': detection({ state: 'blind', reason: 'pose_not_completing' }) }),
      { cameras: [{ ...cameraA, mapping_pending: true, backend_camera_id: null }] },
    );

    expect(tileBadge(host, 'cam-1')?.dataset.detectionState).toBe('blind');
    expect(summary(host)?.getAttribute('role')).toBe('alert');
    // 매핑 상태는 월 타일이 소유하지 않는다 — 호실 상세 카드가 계속 소유한다.
    expect(host.querySelector('[data-testid="cloud-mapping"]')).toBeNull();
  });

  it('상태 응답이 아직 없으면 타일은 확인 불가, 경보는 없다', () => {
    const { host } = renderWall(null);

    expect(tileBadge(host, 'cam-1')?.dataset.detectionState).toBe('unknown');
    expect(summary(host)).toBeNull();
  });
});
