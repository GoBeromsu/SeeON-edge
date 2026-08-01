import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { StatusSnapshot, SystemSnapshot } from '@/shared/api/client';
import type { PollingResource } from '@/shared/api/usePollingResource';
import { SystemPanel } from '@/shared/ui/SystemPanels';

const retrySystem = vi.fn();
const retryStatus = vi.fn();

const system: SystemSnapshot = {
  updated_at: '2026-07-20T01:59:59.000Z',
  version: 'edge-1.2.3',
  image_digests: { ml_api: 'sha256:api', ml_worker: 'sha256:worker' },
  backend: { configured: true, reachable: true, last_ok_at: '2026-07-07T00:00:00.000Z' },
  storage: { clip_store: { total_bytes: 2_000, used_bytes: 500, used_pct: 25 } },
};

const status: StatusSnapshot = {
  cameras: {
    camera1: {
      camera_id: 'camera-1', facility_id: 'facility-a', status: 'online',
      last_heartbeat_at: 1_783_382_400, age_sec: 2, config_version: 7,
    },
  },
  stale_after_sec: 30,
  runtime: {
    stale_after_sec: 15,
    facilities: {
      facilityA: {
        facility_id: 'facility-a', generation: 4, seq: 19,
        received_at: 1_783_382_400, stale: false,
        cameras: [{ camera_id: 'camera-1', measured_fps: 29.7, decode: { requested: 'auto', selected: 'nvdec', fallback_count: 0, last_reason: null, updated_at_sec: 1_783_382_399 } }],
        clip_recorder: { available: true, dropped_frames: 1, dropped_events: 2, failed_writes: 3, finalized_clips: 12, video_unavailable_clips: 4, active_clips: 1, encoder: 'h264' },
        gpu: { nvml_available: true, cuda_context_ok: true, driver_version: '555.42', device_name: 'NVIDIA Test GPU', captured_at_sec: 1_783_382_300 },
        worker: { alive: true, pid: 1234, started_at_sec: 1_783_382_000 },
        latency: { first_attempt_samples: 7, max_sec: 0.42, since_sec: 1_783_382_000 },
      },
    },
  },
};

function resource<T>(data: T | null, options: Partial<PollingResource<T>> = {}): PollingResource<T> {
  return {
    status: data === null ? 'loading' : 'success', data, error: null,
    lastSuccessAt: data === null ? null : 1_783_382_400_000, refreshing: false,
    retry: vi.fn(), ...options,
  };
}

function renderPanel(
  systemResource: PollingResource<SystemSnapshot>,
  statusResource: PollingResource<StatusSnapshot>,
): { host: HTMLDivElement; root: Root } {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<SystemPanel systemResource={systemResource} statusResource={statusResource} />));
  return { host, root };
}

afterEach(() => {
  document.body.innerHTML = '';
  vi.clearAllMocks();
});

describe('SystemPanel', () => {
  it('renders an honest complete snapshot and keeps technical identifiers collapsed', () => {
    const { host, root } = renderPanel(resource(system), resource(status));

    // 정상 상태는 침묵해야 한다 — "확인 완료" 문구도, 다른 안심시키는 대체 문구도 없다.
    expect(host.textContent).not.toContain('시스템 상태 확인 완료');
    expect(host.querySelector('[aria-live="polite"]')).toBeNull();
    expect(host.textContent).toContain('시스템 기준 시각');
    expect(host.textContent).toContain('시스템 정보 · 마지막 확인');
    expect(host.textContent).toContain('런타임 상태 · 마지막 확인');
    expect(host.textContent).toContain(`시스템 기준 시각 ${new Date(system.updated_at!).toLocaleString('ko-KR')}`);
    expect(host.textContent).toContain('설정됨');
    expect(host.textContent).toMatch(/\d+(초|분|시간|일) 전 동기화/);
    expect(host.textContent).toContain('500 B / 2.0 KB');
    expect(host.textContent).toContain('25.0%');
    expect(host.textContent).toContain('카메라 1대');
    expect(host.textContent).toContain('활성 클립 1개');
    expect(host.textContent).toContain('완료 클립 12개');
    expect(host.textContent).toContain('GPU 및 워커 복구 상태');
    expect(host.textContent).toContain('NVML 사용 가능');
    expect(host.textContent).toContain('555.42');
    expect(host.textContent).toContain('NVIDIA Test GPU');
    expect(host.textContent).toContain('생성 가능');
    expect(host.textContent).toContain('워커 실행 중');
    expect(host.textContent).toContain('실제 디코드 nvdec');
    expect(host.textContent).toContain('실측 29.7 FPS');
    expect(host.textContent).toContain('최초 전송 지연 표본 7건');
    expect(host.textContent).toContain('최대 지연 0.42초');
    expect(host.querySelector('details:not([open]) > summary')?.textContent).toBe('기술 정보');
    expect(host.textContent).toContain('edge-1.2.3');
    expect(host.textContent).toContain('sha256:api');
    expect(host.textContent).toContain('설정 버전 7');
    expect(host.textContent).toContain('세대 4');
    expect(host.textContent).toContain('순번 19');
    act(() => root.unmount());
  });
  it.each([
    [5, '5초 이내'],
    [5.01, '5초 초과'],
    [null, '정보 없음'],
  ])('renders the five-second latency gate for max %s', (maxSec, expected) => {
    const gatedStatus: StatusSnapshot = {
      ...status,
      runtime: {
        ...status.runtime,
        facilities: {
          facilityA: {
            ...status.runtime.facilities.facilityA,
            latency: maxSec === null ? null : { first_attempt_samples: 1, max_sec: maxSec, since_sec: 1 },
          },
        },
      },
    };
    const { host, root } = renderPanel(resource(system), resource(gatedStatus));

    expect(host.textContent).toContain(expected);
    act(() => root.unmount());
  });

  it('renders absent runtime diagnostics as information missing for legacy workers', () => {
    const legacyStatus: StatusSnapshot = {
      ...status,
      runtime: {
        ...status.runtime,
        facilities: {
          facilityA: {
            ...status.runtime.facilities.facilityA,
            cameras: [{ ...status.runtime.facilities.facilityA.cameras[0], measured_fps: null }],
            gpu: null,
            worker: null,
            latency: null,
          },
        },
      },
    };
    const { host, root } = renderPanel(resource(system), resource(legacyStatus));
    const recovery = host.querySelector('[aria-label="복구 상태"]');

    expect(recovery?.textContent).toContain('정보 없음');
    expect(recovery?.textContent).toContain('실측 정보 없음');
    act(() => root.unmount());
  });

  it('marks unavailable NVML as a visible failure', () => {
    const failedGpuStatus: StatusSnapshot = {
      ...status,
      runtime: {
        ...status.runtime,
        facilities: {
          facilityA: {
            ...status.runtime.facilities.facilityA,
            gpu: { ...status.runtime.facilities.facilityA.gpu!, nvml_available: false },
          },
        },
      },
    };
    const { host, root } = renderPanel(resource(system), resource(failedGpuStatus));
    const nvml = Array.from(host.querySelectorAll('span')).find((element) => element.textContent?.includes('NVML 사용 불가'));

    expect(nvml?.textContent).toContain('NVML 사용 불가');
    expect(nvml?.className).toContain('text-status-danger');
    // 장애 진단은 클릭 없이 읽혀야 한다 — 접힌 <details>는 오직 "기술 정보" 식별자용 1개뿐이어야 한다.
    expect(nvml?.closest('details')).toBeNull();
    expect(host.querySelectorAll('details')).toHaveLength(1);
    expect(host.querySelector('details')?.querySelector('summary')?.textContent).toBe('기술 정보');
    act(() => root.unmount());
  });

  it.each([null, 'unknown'])('renders an unavailable %s version as information missing', (version) => {
    const { host, root } = renderPanel(resource({ ...system, version }), resource(status));
    const versionValue = Array.from(host.querySelectorAll('dt'))
      .find((term) => term.textContent === '버전')
      ?.nextElementSibling;

    expect(versionValue?.textContent).toBe('정보 없음');
    expect(host.textContent).not.toContain('unknown');
    act(() => root.unmount());
  });

  it('shows initial loading without inventing health or storage values', () => {
    const { host, root } = renderPanel(resource<SystemSnapshot>(null), resource<StatusSnapshot>(null));
    expect(host.textContent).toContain('시스템 상태를 불러오는 중');
    expect(host.querySelector('[role="meter"]')).toBeNull();
    expect(host.textContent).not.toContain('시스템 상태 확인 완료');
    act(() => root.unmount());
  });

  it('shows success-empty runtime as empty, not failed', () => {
    const emptyStatus: StatusSnapshot = { cameras: {}, stale_after_sec: null, runtime: { facilities: {}, stale_after_sec: null } };
    const { host, root } = renderPanel(resource(system), resource(emptyStatus));
    expect(host.textContent).toContain('시설 처리 정보가 없습니다');
    expect(host.textContent).not.toContain('런타임 상태를 불러오지 못했습니다');
    act(() => root.unmount());
  });

  it('shows a partial state while preserving the successful endpoint', () => {
    const failedStatus = resource<StatusSnapshot>(null, { status: 'error', error: new Error('offline'), retry: retryStatus });
    const { host, root } = renderPanel(resource(system), failedStatus);
    expect(host.textContent).toContain('일부 상태만 확인됨');
    expect(host.textContent).toMatch(/\d+(초|분|시간|일) 전 동기화/);
    expect(host.textContent).toContain('런타임 상태를 불러오지 못했습니다');
    act(() => Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '다시 시도')?.click());
    expect(retryStatus).toHaveBeenCalledOnce();
    act(() => root.unmount());
  });

  it('keeps a still-loading status endpoint distinct from failure', () => {
    const { host, root } = renderPanel(resource(system), resource<StatusSnapshot>(null));
    expect(host.textContent).toContain('일부 상태를 불러오는 중');
    expect(host.textContent).toContain('런타임 상태를 불러오는 중');
    expect(host.textContent).not.toContain('런타임 상태를 불러오지 못했습니다');
    act(() => root.unmount());
  });

  it('keeps a still-loading system endpoint distinct from failure', () => {
    const { host, root } = renderPanel(resource<SystemSnapshot>(null), resource(status));
    expect(host.textContent).toContain('일부 상태를 불러오는 중');
    expect(host.textContent).toContain('시스템 정보를 불러오는 중');
    expect(host.textContent).not.toContain('백엔드와 저장 공간 상태를 불러오지 못했습니다');
    act(() => root.unmount());
  });

  it('shows typed runtime data when only the system request fails', () => {
    const failedSystem = resource<SystemSnapshot>(null, { status: 'error', error: new Error('offline'), retry: retrySystem });
    const { host, root } = renderPanel(failedSystem, resource(status));
    expect(host.textContent).toContain('일부 상태만 확인됨');
    expect(host.textContent).toContain('카메라 1대');
    expect(host.textContent).toContain('백엔드와 저장 공간 상태를 불러오지 못했습니다');
    act(() => root.unmount());
  });

  it('shows total failure and retries both failed resources', () => {
    const failedSystem = resource<SystemSnapshot>(null, { status: 'error', error: new Error('offline'), retry: retrySystem });
    const failedStatus = resource<StatusSnapshot>(null, { status: 'error', error: new Error('offline'), retry: retryStatus });
    const { host, root } = renderPanel(failedSystem, failedStatus);
    expect(host.textContent).toContain('시스템 상태를 불러오지 못했습니다');
    act(() => Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '다시 시도')?.click());
    expect(retrySystem).toHaveBeenCalledOnce();
    expect(retryStatus).toHaveBeenCalledOnce();
    expect(host.querySelector('.brand-action')?.textContent).toBe('다시 시도');
    act(() => root.unmount());
  });

  it('retains stale data after refresh failure and clears the warning on recovery', () => {
    const failedSystem = resource(system, { status: 'error', error: new Error('offline'), retry: retrySystem });
    const failedStatus = resource(status, { status: 'error', error: new Error('offline'), retry: retryStatus });
    const { host, root } = renderPanel(failedSystem, failedStatus);
    expect(host.textContent).toContain('마지막 성공 상태를 표시합니다');
    expect(host.textContent).toContain('시스템 정보 갱신 지연 · 마지막 확인');
    expect(host.textContent).toContain('런타임 상태 갱신 지연 · 마지막 확인');
    expect(host.textContent).toContain('카메라 1대');

    act(() => root.render(<SystemPanel systemResource={resource(system)} statusResource={resource(status)} />));
    // 회복 후에도 정상 상태는 침묵한다 — 경고 문구만 사라지고 안심 문구가 새로 나타나지 않는다.
    expect(host.textContent).not.toContain('시스템 상태 확인 완료');
    expect(host.querySelector('[aria-live="polite"]')).toBeNull();
    expect(host.textContent).not.toContain('마지막 성공 상태를 표시합니다');
    act(() => root.unmount());
  });

  it('reports unavailable storage, stale/unknown facilities, and invalid times honestly', () => {
    const unavailableSystem: SystemSnapshot = {
      ...system,
      backend: { configured: false, reachable: null, last_ok_at: 'not-a-time' },
      storage: { clip_store: { total_bytes: null, used_bytes: null, used_pct: null } },
    };
    const uncertainStatus: StatusSnapshot = {
      ...status,
      runtime: {
        ...status.runtime,
        facilities: {
          stale: { ...status.runtime.facilities.facilityA, facility_id: 'stale-facility', stale: true, received_at: Number.NaN },
          unknown: { ...status.runtime.facilities.facilityA, facility_id: 'unknown-facility', stale: null, received_at: null },
        },
      },
    };
    const { host, root } = renderPanel(resource(unavailableSystem), resource(uncertainStatus));
    expect(host.textContent).toContain('설정되지 않음');
    expect(host.textContent).toContain('저장 공간 정보 없음');
    expect(host.textContent).toContain('연결 지연');
    expect(host.textContent).toContain('신선도 정보 없음');
    expect(host.textContent).toContain('마지막 확인 정보 없음');
    expect(host.querySelector('[role="meter"]')).toBeNull();
    act(() => root.unmount());
  });

  it('keeps independently available storage amounts and percentages honest', () => {
    const amountOnly: SystemSnapshot = { ...system, storage: { clip_store: { total_bytes: 2_000, used_bytes: 500, used_pct: null } } };
    const percentageOnly: SystemSnapshot = { ...system, storage: { clip_store: { total_bytes: null, used_bytes: null, used_pct: 25 } } };
    const { host, root } = renderPanel(resource(amountOnly), resource(status));
    expect(host.textContent).toContain('500 B / 2.0 KB');
    expect(host.textContent).toContain('사용률 정보 없음');
    expect(host.querySelector('[role="meter"]')).toBeNull();

    act(() => root.render(<SystemPanel systemResource={resource(percentageOnly)} statusResource={resource(status)} />));
    expect(host.textContent).toContain('25.0%');
    expect(host.querySelector('[role="meter"]')).not.toBeNull();
    expect(host.textContent).not.toContain('500 B / 2.0 KB');
    act(() => root.unmount());
  });

  it('does not render forbidden implementation, event, history, or placeholder copy', () => {
    const { host, root } = renderPanel(resource(system), resource(status));
    expect(host.textContent).not.toMatch(/\/api\/v1|\/system|\/status|raw JSON|업데이트 이력|롤백 이력|알림 센터|ops 이벤트|스토리지 게이지|연결되면 이 영역/);
    expect(host.querySelector('pre')).toBeNull();
    act(() => root.unmount());
  });

  describe('backend connection: elapsed time, never a bare boolean', () => {
    it('shows elapsed time since last success when reachable, not a "연결됨" boolean', () => {
      const recentSystem: SystemSnapshot = { ...system, backend: { ...system.backend, reachable: true, last_ok_at: new Date(Date.now() - 12_000).toISOString() } };
      const { host, root } = renderPanel(resource(recentSystem), resource(status));
      expect(host.textContent).toMatch(/\d+초 전 동기화/);
      expect(host.textContent).not.toContain('연결됨');
      expect(host.textContent).not.toContain('연결 실패');
      act(() => root.unmount());
    });

    it('shows elapsed time since last success when unreachable, phrased as an ongoing failure', () => {
      const failingSystem: SystemSnapshot = { ...system, backend: { ...system.backend, reachable: false, last_ok_at: new Date(Date.now() - 4 * 60_000).toISOString() } };
      const { host, root } = renderPanel(resource(failingSystem), resource(status));
      expect(host.textContent).toMatch(/4분째 실패/);
      expect(host.textContent).not.toContain('연결 실패');
      act(() => root.unmount());
    });

    it('reports missing sync history honestly instead of a boolean when never synced', () => {
      const neverSyncedSystem: SystemSnapshot = { ...system, backend: { ...system.backend, reachable: null, last_ok_at: null } };
      const { host, root } = renderPanel(resource(neverSyncedSystem), resource(status));
      expect(host.textContent).toContain('동기화 이력 없음');
      act(() => root.unmount());
    });
  });

  it('surfaces per-camera staleness (age) without hiding it behind a click', () => {
    const staleStatus: StatusSnapshot = {
      ...status,
      cameras: {
        ...status.cameras,
        camera2: { camera_id: 'camera-2', facility_id: 'facility-a', status: 'stale', last_heartbeat_at: 1, age_sec: 42, config_version: 1 },
        camera3: { camera_id: 'camera-3', facility_id: 'facility-a', status: 'never_seen', last_heartbeat_at: null, age_sec: null, config_version: null },
      },
    };
    const { host, root } = renderPanel(resource(system), resource(staleStatus));
    expect(host.textContent).toContain('camera-2');
    expect(host.textContent).toContain('마지막 신호 42초 전');
    expect(host.textContent).toContain('camera-3');
    expect(host.textContent).toContain('연결 이력 없음');
    expect(host.querySelector('details ul')).toBeNull();
    act(() => root.unmount());
  });
});
