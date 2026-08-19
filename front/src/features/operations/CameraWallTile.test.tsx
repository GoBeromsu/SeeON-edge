import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CameraWallTile } from '@/features/operations/CameraWallTile';
import { SnapshotQueue, type SnapshotEntry } from '@/features/operations/SnapshotQueue';
import { type Camera, type RuntimeCameraDiagnostics, type RuntimeDetectionDiagnostics } from '@/shared/api/client';

const onlineCamera: Camera = {
  id: 'cam-1',
  label: '101호',
  rtsp_url_masked: 'rtsp://redacted-camera/a',
  floor_name: '1층',
  status: 'online',
  created_at: null,
};

const offlineCamera: Camera = {
  ...onlineCamera,
  id: 'cam-2',
  label: '102호',
  status: 'offline',
};

const loadedSnapshot: SnapshotEntry = {
  id: 'cam-1',
  mediaId: 'cam-1',
  state: 'loaded',
  requestUrl: null,
  lastLoadedUrl: '/api/v1/streams/cam-1/snapshot?refresh=0',
  lastLoadedAt: Date.now(),
};

function diagnosticsWith(detection: RuntimeDetectionDiagnostics | undefined): RuntimeCameraDiagnostics {
  return {
    camera_id: 'cam-1',
    decode: { requested: 'auto', selected: 'opencv', fallback_count: 0, last_reason: null, updated_at_sec: null },
    measured_fps: 12,
    latency: null,
    stale: false,
    detection,
  };
}

function detection(overrides: Partial<RuntimeDetectionDiagnostics> = {}): RuntimeDetectionDiagnostics {
  return {
    state: 'healthy',
    reason: null,
    recent_success_rate: 1,
    last_completed_at_sec: 4,
    evaluation_window_sec: 120,
    timeout_sec: 120,
    ...overrides,
  };
}

function render(
  camera: Camera,
  snapshot: SnapshotEntry | undefined = undefined,
  diagnostics: RuntimeCameraDiagnostics | undefined = undefined,
): { host: HTMLDivElement; root: Root; queue: SnapshotQueue } {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  const queue = new SnapshotQueue(() => '');
  act(() => root.render(
    <CameraWallTile camera={camera} snapshot={snapshot} diagnostics={diagnostics} queue={queue} onSelect={vi.fn()} />,
  ));
  return { host, root, queue };
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn());
});

afterEach(() => {
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
});

describe('CameraWallTile', () => {
  it('renders a gray bg-muted placeholder with "오프라인" text for an offline camera, not a black frame', () => {
    const { host } = render(offlineCamera);

    const placeholder = host.querySelector('.bg-muted');
    expect(placeholder).not.toBeNull();
    expect(placeholder?.textContent).toBe('오프라인');
    expect(host.querySelector('canvas')).toBeNull();
    expect(host.querySelector('.event-media-frame')).toBeNull();
  });

  it('shows the camera name and a red status dot in the bottom bar for an offline camera', () => {
    const { host } = render(offlineCamera);

    expect(host.textContent).toContain('102호');
    expect(host.querySelector('.bg-status-rejectedFg')).not.toBeNull();
  });

  it('does not render the offline placeholder for an online camera', () => {
    const { host } = render(onlineCamera);

    const placeholders = Array.from(host.querySelectorAll('*')).filter((el) => el.textContent === '오프라인');
    expect(placeholders.length).toBe(0);
  });

  it('uses the loaded snapshot without opening an MJPEG stream for an online wall tile', () => {
    // Given an online tile with one loaded snapshot.
    const { host } = render(onlineCamera, loadedSnapshot);

    // When the wall tile renders in the active wall.
    const tile = host.querySelector('button');

    // Then it remains an accessible snapshot tile and never starts live media.
    expect(tile?.getAttribute('aria-label')).toBe('101호 열기');
    expect(host.querySelector('canvas[aria-label="101호 실시간 영상"]')).toBeNull();
    expect(host.querySelector('img[alt="101호 최근 영상"]')).not.toBeNull();
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it('renders the loading state before the one snapshot settles', () => {
    // Given an online camera before a snapshot has loaded.
    const loading = render(onlineCamera);

    // When its snapshot state is absent.
    // Then the stable media frame exposes the loading copy.
    expect(loading.host.textContent).toContain('불러오는 중…');
  });

  it('renders the unavailable state when the one snapshot fails', () => {
    // Given an online camera after its one snapshot request fails.
    const failedSnapshot: SnapshotEntry = { ...loadedSnapshot, state: 'error', lastLoadedUrl: null };
    const unavailable = render(onlineCamera, failedSnapshot);

    // Then the tile exposes the unavailable copy without a live fallback.
    expect(unavailable.host.textContent).toContain('영상을 불러올 수 없습니다');
    expect(unavailable.host.querySelector('canvas')).toBeNull();
  });

  it('shows a "연결 끊김" badge when the snapshot itself goes stale/error after a frame previously loaded', () => {
    const staleSnapshot: SnapshotEntry = { ...loadedSnapshot, state: 'stale' };
    const { host } = render(onlineCamera, staleSnapshot);

    expect(host.querySelector('[role="status"]')?.textContent).toBe('연결 끊김');
    expect(host.querySelector('img[alt="101호 최근 영상"]')).not.toBeNull();
  });

  it('does not show the "연결 끊김" badge for a healthy, loaded snapshot', () => {
    const { host } = render(onlineCamera, loadedSnapshot);

    expect(host.querySelector('[role="status"]')).toBeNull();
  });

  it('resolves the single hidden snapshot request through the queue', () => {
    // Given a tile whose one-shot snapshot request is active.
    const pendingSnapshot: SnapshotEntry = {
      ...loadedSnapshot,
      state: 'loading',
      requestUrl: '/api/v1/streams/cam-1/snapshot?refresh=0',
      lastLoadedUrl: null,
      lastLoadedAt: null,
    };
    const { host, queue } = render(onlineCamera, pendingSnapshot);
    const resolve = vi.spyOn(queue, 'resolve');
    const requestImage = host.querySelector('img[aria-hidden="true"]');

    // When the browser finishes the snapshot image request.
    act(() => requestImage?.dispatchEvent(new Event('load')));

    // Then exactly that request identity is settled.
    expect(resolve).toHaveBeenCalledWith(
      'cam-1',
      'cam-1',
      'loaded',
      '/api/v1/streams/cam-1/snapshot?refresh=0',
    );
  });
});

describe('CameraWallTile — 감지 상태 뱃지', () => {
  function badge(host: HTMLElement): HTMLElement | null {
    return host.querySelector('[data-testid="tile-detection"]');
  }

  it('온라인 타일에서 감지 중단을 연결 상태와 함께 보여준다', () => {
    const { host } = render(onlineCamera, loadedSnapshot, diagnosticsWith(detection({ state: 'blind', reason: 'pose_not_completing' })));

    expect(badge(host)?.dataset.detectionState).toBe('blind');
    expect(badge(host)?.dataset.detectionReason).toBe('pose_not_completing');
    // 스냅샷 미디어는 그대로 남는다.
    expect(host.querySelector('img[alt="101호 최근 영상"]')).not.toBeNull();
  });

  it('오프라인 타일에서도 감지 상태를 따로 남긴다', () => {
    const { host } = render(offlineCamera, undefined, diagnosticsWith(detection({ state: 'disabled' })));

    expect(host.textContent).toContain('오프라인');
    expect(badge(host)?.dataset.detectionState).toBe('disabled');
  });

  it('진단이 없으면 확인 불가로 남고 정상으로 지어내지 않는다', () => {
    const { host } = render(onlineCamera, loadedSnapshot, undefined);

    expect(badge(host)?.dataset.detectionState).toBe('unknown');
  });

  it('타일 뱃지 자체는 alert 역할을 쓰지 않는다 — 경보 요약은 월이 하나만 소유한다', () => {
    const { host } = render(onlineCamera, loadedSnapshot, diagnosticsWith(detection({ state: 'blind', reason: 'no_completed_cycles' })));

    expect(badge(host)?.getAttribute('role')).toBeNull();
    expect(host.querySelector('[role="alert"]')).toBeNull();
  });

  it('감지 뱃지가 붙어도 스냅샷 요청 동작은 그대로다', () => {
    const pending: SnapshotEntry = {
      ...loadedSnapshot,
      state: 'loading',
      requestUrl: '/api/v1/streams/cam-1/snapshot?refresh=0',
      lastLoadedUrl: null,
      lastLoadedAt: null,
    };
    const { host, queue } = render(onlineCamera, pending, diagnosticsWith(detection({ state: 'blind', reason: 'pose_not_completing' })));
    const resolve = vi.spyOn(queue, 'resolve');

    act(() => host.querySelector('img[aria-hidden="true"]')?.dispatchEvent(new Event('load')));

    expect(resolve).toHaveBeenCalledWith('cam-1', 'cam-1', 'loaded', '/api/v1/streams/cam-1/snapshot?refresh=0');
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });
});
