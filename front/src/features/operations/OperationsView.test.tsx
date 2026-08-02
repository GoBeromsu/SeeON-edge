import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Camera, StatusSnapshot } from '@/shared/api/client';
import type { DashboardLocation } from '@/app/dashboardLocation';
import { OperationsView } from '@/features/operations/OperationsView';

const cleanups: Array<() => void> = [];
afterEach(() => {
  while (cleanups.length) cleanups.pop()?.();
  vi.unstubAllGlobals();
});
// jsdom doesn't implement scrollTo; the component calls it whenever a focus->wall
// transition restores scroll position, so stub it globally to keep test output clean.
let scrollToSpy: ReturnType<typeof vi.spyOn>;
beforeEach(() => { scrollToSpy = vi.spyOn(window, 'scrollTo').mockImplementation(() => undefined); });

function camera(id: string, backend = id): Camera {
  return { id, backend_camera_id: backend, label: `카메라 ${id}`, floor_name: '1층', space_id: 'room-1', space_name: '101호', rtsp_url_masked: '', status: 'unknown', created_at: null };
}

function renderView(options: { cameras?: Camera[]; location?: DashboardLocation; status?: StatusSnapshot | null } = {}) {
  const host = document.createElement('div');
  host.id = 'main-content';
  document.body.append(host);
  const root: Root = createRoot(host);
  const navigate = vi.fn();
  const replace = vi.fn();
  const cameras = options.cameras ?? [camera('local', 'worker/cam')];
  const location = options.location ?? { page: 'operations', wallPage: '1' };
  act(() => root.render(<OperationsView active cameras={cameras} camerasState="success" status={options.status ?? null} statusState="success" location={location} onNavigate={navigate} onReplace={replace} />));
  cleanups.push(() => { act(() => root.unmount()); host.remove(); });
  return { host, navigate, replace, rerender: (next: DashboardLocation, nextCameras = cameras) => act(() => root.render(<OperationsView active cameras={nextCameras} camerasState="success" status={options.status ?? null} statusState="success" location={next} onNavigate={navigate} onReplace={replace} />)) };
}

describe('OperationsView', () => {
  it('renders honest empty/loading/error states and retry actions', () => {
    const host = document.createElement('div'); document.body.append(host); const root = createRoot(host); const retry = vi.fn();
    act(() => root.render(<OperationsView active cameras={[]} camerasState="loading" status={null} statusState="loading" location={{ page: 'operations', wallPage: '1' }} onNavigate={vi.fn()} onReplace={vi.fn()} onRetryCameras={retry} />));
    expect(host.textContent).toContain('카메라 불러오는 중');
    expect(host.querySelector('[data-camera-refresh-status]')).toBeNull();
    act(() => root.render(<OperationsView active cameras={[]} camerasState="error" status={null} statusState="error" location={{ page: 'operations', wallPage: '1' }} onNavigate={vi.fn()} onReplace={vi.fn()} onRetryCameras={retry} />));
    expect(host.textContent).toContain('카메라 목록을 불러올 수 없음');
    expect(host.querySelector('[data-camera-refresh-status]')).toBeNull();
    act(() => Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '다시 시도')?.click());
    expect(retry).toHaveBeenCalled();
    act(() => root.render(<OperationsView active cameras={[]} camerasState="success" status={null} statusState="success" location={{ page: 'operations', wallPage: '1' }} onNavigate={vi.fn()} onReplace={vi.fn()} />));
    expect(host.textContent).toContain('등록된 카메라가 없습니다');
    act(() => root.unmount()); host.remove();
  });

  it('offers a semantic camera-management action from the successful empty state', () => {
    const { host, navigate } = renderView({ cameras: [] });
    const action = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find((button) => button.textContent === '카메라 등록하기');

    expect(action?.classList.contains('brand-action')).toBe(true);
    act(() => action?.click());
    expect(navigate).toHaveBeenCalledWith({
      page: 'cameras', floor: null, room: null, camera: null,
      event: null, clip: null, wallPage: null,
    });
  });

  it('shows retained camera data with its real last-success time and clears freshness copy after recovery', () => {
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    const lastSuccessAt = Date.parse('2026-07-21T00:15:00Z');
    const common = {
      active: true,
      status: null,
      statusState: 'success' as const,
      location: { page: 'operations', wallPage: '1' } as DashboardLocation,
      onNavigate: vi.fn(),
      onReplace: vi.fn(),
    };

    act(() => root.render(<OperationsView {...common} cameras={[camera('local')]} camerasState="error" camerasLastSuccessAt={lastSuccessAt} />));
    expect(host.textContent).toContain('마지막 카메라 목록을 표시합니다.');
    expect(host.textContent).toContain(`마지막 확인 ${new Date(lastSuccessAt).toLocaleString('ko-KR')}`);

    act(() => root.render(<OperationsView {...common} cameras={[camera('local')]} camerasState="success" camerasLastSuccessAt={Date.parse('2026-07-21T00:20:00Z')} />));
    expect(host.textContent).not.toContain('마지막 카메라 목록을 표시합니다.');
    expect(host.textContent).not.toContain('마지막 확인');

    act(() => root.render(<OperationsView {...common} cameras={[]} camerasState="error" camerasLastSuccessAt={null} />));
    expect(host.textContent).toContain('카메라 목록을 불러올 수 없음');
    expect(host.textContent).not.toContain('마지막 확인');
    act(() => root.unmount());
    host.remove();
  });

  it('syncs filters/page and navigates directly to the clicked camera without stream URLs in wall', () => {
    const cameras = Array.from({ length: 13 }, (_, index) => ({ ...camera(index === 0 ? 'local' : String(index), index === 0 ? 'worker/cam' : String(index)), label: index === 0 ? '카메라 local' : `카메라 ${String(index).padStart(2, '0')}`, floor_name: index === 12 ? '2층' : '1층' }));
    const { host, navigate } = renderView({ cameras });
    act(() => Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '다음')?.click());
    expect(navigate).toHaveBeenCalledWith({ wallPage: '2' });
    const selects = host.querySelectorAll('select');
    act(() => { (selects[0] as HTMLSelectElement).value = '1층'; selects[0].dispatchEvent(new Event('change', { bubbles: true })); });
    expect(navigate).toHaveBeenCalledWith(expect.objectContaining({ floor: '1층', wallPage: '1' }));
    const card = host.querySelector<HTMLButtonElement>('[aria-label="카메라 local 열기"]');
    expect(card?.hasAttribute('aria-pressed')).toBe(false);
    act(() => card?.click());
    expect(navigate).toHaveBeenCalledWith({ camera: 'local' });
    expect(host.querySelector('[data-stream]')).toBeNull();
    expect(host.innerHTML).not.toContain('/streams/worker%2Fcam?');
  });

  it('uses one responsive camera grid per floor across distinct rooms', () => {
    const cameras = [
      { ...camera('c'), space_id: 'room-c', space_name: '다실', label: '다 카메라' },
      { ...camera('a'), space_id: 'room-a', space_name: '가실', label: '가 카메라' },
      { ...camera('b'), space_id: 'room-b', space_name: '나실', label: '나 카메라' },
    ];
    const { host } = renderView({ cameras });
    const floor = host.querySelector('section[aria-labelledby="floor-1층"]');
    const floorGrid = Array.from(floor?.children ?? []).find((element) => element.tagName === 'DIV' && element.classList.contains('grid'));

    expect(floorGrid?.classList.contains('md:grid-cols-2')).toBe(true);
    expect(floorGrid?.classList.contains('xl:grid-cols-3')).toBe(true);
    expect(floorGrid?.querySelectorAll(':scope > article')).toHaveLength(3);
    expect(Array.from(floorGrid?.querySelectorAll(':scope > article > h3') ?? []).map((heading) => heading.textContent)).toEqual(['가실', '나실', '다실']);
    expect(Array.from(floorGrid?.querySelectorAll(':scope > article button[aria-label]') ?? []).map((card) => card.getAttribute('aria-label'))).toEqual(['가 카메라 열기', '나 카메라 열기', '다 카메라 열기']);
  });

  it('navigates directly from a wall card click to the focused live view in a single call (AC-1)', () => {
    const { host, navigate, rerender } = renderView();
    expect(host.querySelector('img')?.getAttribute('src')).toContain('/streams/worker%2Fcam/snapshot');
    act(() => host.querySelector<HTMLButtonElement>('[aria-label="카메라 local 열기"]')?.click());
    expect(navigate).toHaveBeenCalledTimes(1);
    expect(navigate).toHaveBeenCalledWith({ camera: 'local' });
    rerender({ page: 'operations', camera: 'local', wallPage: '1' });
    expect(host.querySelectorAll('[data-stream]')).toHaveLength(1);
    expect(host.querySelector('[data-stream]')?.getAttribute('src')).toContain('/streams/worker%2Fcam');
  });

  it('unmounts the single focus stream on back, page change, and camera switch and reports focus failure', () => {
    const cameras = [camera('a', 'worker-a'), camera('b', 'worker-b')];
    const { host, rerender } = renderView({ cameras, location: { page: 'operations', camera: 'a', wallPage: '1' } });
    const first = host.querySelector('[data-stream]');
    expect(first).not.toBeNull();
    act(() => first?.dispatchEvent(new Event('error', { bubbles: true })));
    expect(host.textContent).toContain('라이브 영상을 불러올 수 없음');
    rerender({ page: 'operations', camera: 'b', wallPage: '1' });
    expect(first?.isConnected).toBe(false);
    expect(host.querySelectorAll('[data-stream]')).toHaveLength(1);
    rerender({ page: 'operations', wallPage: '1' });
    expect(host.querySelector('[data-stream]')).toBeNull();
    rerender({ page: 'events' });
    expect(host.querySelector('[data-stream]')).toBeNull();
  });

  it('retries a failed focus stream only when its resolved media identity changes', () => {
    const location: DashboardLocation = { page: 'operations', camera: 'local', wallPage: '1' };
    const { host, rerender } = renderView({ cameras: [camera('local', 'worker-old')], location });
    const failedStream = host.querySelector('[data-stream]');

    act(() => failedStream?.dispatchEvent(new Event('error', { bubbles: true })));
    expect(host.querySelector('[data-stream]')).toBeNull();
    expect(host.textContent).toContain('라이브 영상을 불러올 수 없음');

    rerender(location, [camera('local', 'worker-old')]);
    expect(host.querySelector('[data-stream]')).toBeNull();

    rerender(location, [camera('local', 'worker-new')]);
    const recoveredStream = host.querySelector('[data-stream]');
    expect(host.querySelectorAll('[data-stream]')).toHaveLength(1);
    expect(recoveredStream?.getAttribute('src')).toContain('/streams/worker-new');
    act(() => recoveredStream?.dispatchEvent(new Event('load', { bubbles: true })));
    expect(host.querySelector('[data-stream]')).toBe(recoveredStream);
    expect(host.textContent).not.toContain('라이브 영상을 불러올 수 없음');
  });

  it('reports the focused stream lifecycle independently of the heartbeat/snapshot status strip and resets it for a new media identity', () => {
    const location: DashboardLocation = { page: 'operations', camera: 'local', wallPage: '1' };
    const { host, rerender } = renderView({ cameras: [camera('local', 'worker-old')], location });
    const oldStream = host.querySelector('[data-stream]');
    const liveStatus = host.querySelector('[role="status"]');
    expect(host.querySelectorAll('[role="status"]')).toHaveLength(2);
    expect(liveStatus?.getAttribute('aria-live')).toBe('polite');
    expect(liveStatus?.getAttribute('aria-atomic')).toBe('true');
    expect(liveStatus?.textContent).toBe('라이브 영상 불러오는 중');

    act(() => oldStream?.dispatchEvent(new Event('load', { bubbles: true })));
    expect(host.querySelector('[role="status"]')).toBe(liveStatus);
    expect(liveStatus?.textContent).toBe('라이브 영상 연결됨');

    rerender(location, [camera('local', 'worker-new')]);
    const newStream = host.querySelector('[data-stream]');
    const newLiveStatus = host.querySelector('[role="status"]');
    expect(newStream).not.toBe(oldStream);
    expect(newLiveStatus).not.toBe(liveStatus);
    expect(host.querySelectorAll('[role="status"]')).toHaveLength(2);
    expect(newLiveStatus?.textContent).toBe('라이브 영상 불러오는 중');

    act(() => newStream?.dispatchEvent(new Event('error', { bubbles: true })));
    expect(host.querySelectorAll('[role="status"]')).toHaveLength(2);
    expect(host.querySelector('[role="status"]')).toBe(newLiveStatus);
    expect(newLiveStatus?.textContent).toBe('라이브 영상을 불러올 수 없음');
    expect(host.textContent?.match(/라이브 영상을 불러올 수 없음/g)).toHaveLength(1);

    rerender(location, [camera('local', 'worker-recovered')]);
    const recoveredLiveStatus = host.querySelector('[role="status"]');
    expect(recoveredLiveStatus).not.toBe(newLiveStatus);
    expect(recoveredLiveStatus?.textContent).toBe('라이브 영상 불러오는 중');
    act(() => host.querySelector('[data-stream]')?.dispatchEvent(new Event('load', { bubbles: true })));
    expect(host.querySelector('[role="status"]')).toBe(recoveredLiveStatus);
    expect(recoveredLiveStatus?.textContent).toBe('라이브 영상 연결됨');
  });

  it('does not show an old snapshot when its late load arrives after a media identity change', () => {
    const location: DashboardLocation = { page: 'operations', wallPage: '1' };
    const { host, rerender } = renderView({ cameras: [camera('local', 'worker-old')], location });
    const oldRequest = host.querySelector<HTMLImageElement>('[data-snapshot-preload]');

    rerender(location, [camera('local', 'worker-new')]);
    const newRequest = host.querySelector<HTMLImageElement>('[data-snapshot-preload]');
    expect(newRequest?.getAttribute('src')).toContain('/streams/worker-new/snapshot');
    act(() => newRequest?.dispatchEvent(new Event('error', { bubbles: true })));
    act(() => oldRequest?.dispatchEvent(new Event('load', { bubbles: true })));

    expect(host.textContent).toContain('영상을 불러올 수 없음');
    expect(host.querySelector('img:not([data-snapshot-preload])')).toBeNull();
    expect(host.innerHTML).not.toContain('/streams/worker-old/snapshot');
  });

  it('shows snapshot failure separately from liveness and never paints a loaded-success state early', () => {
    const heartbeat = { camera_id: 'local', facility_id: null, status: 'online' as const, last_heartbeat_at: null, age_sec: 1, config_version: null };
    const status: StatusSnapshot = { cameras: { local: heartbeat }, stale_after_sec: null, runtime: { facilities: {}, stale_after_sec: null } };
    const { host } = renderView({ status });
    expect(host.textContent).toContain('연결됨');
    expect(host.textContent).toContain('영상 불러오는 중');
    const image = host.querySelector('img');
    act(() => image?.dispatchEvent(new Event('error', { bubbles: true })));
    expect(host.textContent).toContain('영상을 불러올 수 없음');
    expect(host.textContent).toContain('연결됨');
  });

  it('uses semantic action and media-overlay classes instead of raw color aliases', () => {
    const { host, rerender } = renderView({ location: { page: 'operations', wallPage: '1' } });

    expect(host.querySelector('.media-status-overlay')?.textContent).toContain('영상 불러오는 중');
    rerender({ page: 'operations', camera: 'local', wallPage: '1' });
    expect(host.querySelector('.brand-action')?.textContent).toBe('이 카메라 클립 보기');
    expect(host.querySelector('.media-status-overlay')?.textContent).toContain('라이브 영상 불러오는 중');
  });

  it('keeps the last-good snapshot timestamp after a successful image load', () => {
    const { host } = renderView();
    const candidate = host.querySelector<HTMLImageElement>('[data-snapshot-preload]');
    act(() => candidate?.dispatchEvent(new Event('load', { bubbles: true })));
    expect(host.textContent).toContain('영상 확인됨');
    expect(host.textContent).toContain('마지막 영상');
    expect(host.querySelector('img:not([data-snapshot-preload])')).toBe(candidate);
  });

  it('keeps the last-good frame visible until a refreshed snapshot loads', () => {
    vi.useFakeTimers();
    const { host } = renderView();
    const firstRequest = host.querySelector('img')?.getAttribute('src');
    act(() => host.querySelector('img')?.dispatchEvent(new Event('load', { bubbles: true })));
    const lastGood = host.querySelector<HTMLImageElement>('img:not([data-snapshot-preload])');

    act(() => vi.advanceTimersByTime(6_000));

    expect(host.querySelector('img:not([data-snapshot-preload])')).toBe(lastGood);
    expect(lastGood?.getAttribute('src')).toBe(firstRequest);
    const failedRefresh = host.querySelector<HTMLImageElement>('[data-snapshot-preload]');
    expect(failedRefresh?.getAttribute('src')).not.toBe(firstRequest);
    act(() => host.querySelector('[data-snapshot-preload]')?.dispatchEvent(new Event('error', { bubbles: true })));
    expect(host.querySelector('img:not([data-snapshot-preload])')).toBe(lastGood);
    expect(failedRefresh?.isConnected).toBe(false);

    act(() => vi.advanceTimersByTime(6_000));
    const successfulRefresh = host.querySelector<HTMLImageElement>('[data-snapshot-preload]');
    act(() => successfulRefresh?.dispatchEvent(new Event('load', { bubbles: true })));
    expect(host.querySelector('img:not([data-snapshot-preload])')).toBe(successfulRefresh);
    expect(lastGood?.isConnected).toBe(false);
  });

  it('labels stale and never-seen liveness without changing the image lifecycle', () => {
    const cameras = [camera('stale'), camera('never')];
    const status: StatusSnapshot = {
      cameras: {
        stale: { camera_id: 'stale', facility_id: null, status: 'stale', last_heartbeat_at: null, age_sec: 90, config_version: null },
        never: { camera_id: 'never', facility_id: null, status: 'never_seen', last_heartbeat_at: null, age_sec: null, config_version: null },
      },
      stale_after_sec: null,
      runtime: { facilities: {}, stale_after_sec: null },
    };
    const { host } = renderView({ cameras, status });
    expect(host.textContent).toContain('연결 지연');
    expect(host.textContent).toContain('연결 이력 없음');
    expect(host.textContent?.match(/영상 불러오는 중/g)).toHaveLength(2);
  });

  it.each([375, 768, 1024, 1440])('never renders an inspector or bottom sheet at %ipx on wall or focus, and has no "집중 보기" button anywhere (AC-2/AC-5)', (width) => {
    const previousWidth = window.innerWidth;
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: width });
    cleanups.push(() => Object.defineProperty(window, 'innerWidth', { configurable: true, value: previousWidth }));

    const wall = renderView({ location: { page: 'operations', wallPage: '1' } });
    expect(wall.host.querySelector('[role="complementary"]')).toBeNull();
    expect(document.body.querySelector('[role="dialog"][aria-modal="true"]')).toBeNull();
    expect(Array.from(wall.host.querySelectorAll('button')).some((button) => button.textContent === '집중 보기')).toBe(false);

    const focus = renderView({ location: { page: 'operations', camera: 'local', wallPage: '1' } });
    expect(focus.host.querySelector('[role="complementary"]')).toBeNull();
    expect(document.body.querySelector('[role="dialog"][aria-modal="true"]')).toBeNull();
    expect(Array.from(focus.host.querySelectorAll('button')).some((button) => button.textContent === '집중 보기')).toBe(false);
  });

  it('renders the focused view as page content with a back control, heading, and clip/settings actions carrying correct navigation payloads', () => {
    const { host, navigate } = renderView({ location: { page: 'operations', camera: 'local', wallPage: '1' } });
    expect(host.querySelector('h1')?.textContent).toBe('카메라 local · 1층');
    expect(Array.from(host.querySelectorAll('button')).some((button) => button.textContent === '← 관제')).toBe(true);

    const clipsButton = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find((button) => button.textContent === '이 카메라 클립 보기');
    act(() => clipsButton?.click());
    expect(navigate).toHaveBeenLastCalledWith({ page: 'events', floor: null, room: null, camera: 'local', event: null, clip: null, wallPage: null });

    const settingsButton = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find((button) => button.textContent === '카메라 설정');
    act(() => settingsButton?.click());
    expect(navigate).toHaveBeenLastCalledWith({ page: 'cameras', floor: null, room: null, camera: null, event: null, clip: null, wallPage: null });

    const back = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find((button) => button.textContent === '← 관제');
    act(() => back?.click());
    expect(navigate).toHaveBeenLastCalledWith({ camera: null });
  });

  it('keeps the focus-view tab order visual (back -> clip -> settings), excluding the programmatic-only heading from the tab ring (AC-14)', () => {
    const { host } = renderView({ location: { page: 'operations', camera: 'local', wallPage: '1' } });
    const section = host.querySelector('section[aria-labelledby="focused-camera-title"]');
    const heading = section?.querySelector('h1');
    expect(heading?.tabIndex).toBe(-1);

    const focusable = Array.from(section?.querySelectorAll<HTMLElement>('button, [href], input, select, textarea, [tabindex]') ?? [])
      .filter((element) => element.tabIndex !== -1);
    expect(focusable.map((element) => element.textContent)).toEqual(['← 관제', '이 카메라 클립 보기', '카메라 설정', '계정 설정', '자세 표시 켜기']);
  });

  it('gives every focus-view action and the wall card a 44px-tall touch target via the min-h-11 convention (AC-15, height only)', () => {
    const { host, rerender } = renderView({ location: { page: 'operations', wallPage: '1' } });
    const card = host.querySelector<HTMLButtonElement>('[aria-label="카메라 local 열기"]');
    expect(card?.classList.contains('min-h-11')).toBe(true);

    rerender({ page: 'operations', camera: 'local', wallPage: '1' });
    for (const label of ['← 관제', '이 카메라 클립 보기', '카메라 설정', '계정 설정', '자세 표시 켜기']) {
      const button = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find((entry) => entry.textContent === label);
      expect(button?.classList.contains('min-h-11')).toBe(true);
    }

    const cameras = Array.from({ length: 13 }, (_, index) => camera(String(index)));
    const paged = renderView({ cameras, location: { page: 'operations', wallPage: '1' } });
    for (const label of ['이전', '다음']) {
      const button = Array.from(paged.host.querySelectorAll<HTMLButtonElement>('button')).find((entry) => entry.textContent === label);
      expect(button?.classList.contains('min-h-11')).toBe(true);
    }
  });

  it('loads the per-camera pose-overlay state and toggles it through the streams pose endpoint (#40)', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ show_pose: false }) })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ show_pose: true }) });
    vi.stubGlobal('fetch', fetchMock);

    const { host } = renderView({ location: { page: 'operations', camera: 'local', wallPage: '1' } });
    await act(async () => {
      await Promise.resolve();
    });

    const findPoseButton = () => Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find((button) => button.textContent?.startsWith('자세 표시'));
    expect(findPoseButton()?.textContent).toBe('자세 표시 켜기');
    expect(findPoseButton()?.disabled).toBe(false);
    expect(fetchMock).toHaveBeenNthCalledWith(1, expect.stringContaining('/streams/worker%2Fcam/pose'), expect.objectContaining({ credentials: 'same-origin' }));
    expect(fetchMock.mock.calls[0]?.[1]?.method).toBeUndefined();

    await act(async () => {
      findPoseButton()?.click();
      await Promise.resolve();
    });

    expect(fetchMock).toHaveBeenNthCalledWith(2, expect.stringContaining('/streams/worker%2Fcam/pose'), expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ show_pose: true }),
    }));
    expect(findPoseButton()?.textContent).toBe('자세 표시 끄기');
    expect(findPoseButton()?.getAttribute('aria-pressed')).toBe('true');
  });

  it('mounts no snapshot polling while focused, honoring the exactly-one-stream media budget', () => {
    const { host, rerender } = renderView({ location: { page: 'operations', camera: 'local', wallPage: '1' } });
    expect(host.querySelector('[data-snapshot-preload]')).toBeNull();
    expect(host.querySelectorAll('[data-stream]')).toHaveLength(1);
    rerender({ page: 'operations', wallPage: '1' });
    expect(host.querySelector('[data-stream]')).toBeNull();
  });

  it('moves focus to the focused-view heading on entering focus, and restores focus to the clicked card on returning to the wall (AC-11/AC-12)', () => {
    const { host, rerender } = renderView({ location: { page: 'operations', wallPage: '1' } });
    const card = host.querySelector<HTMLButtonElement>('[aria-label="카메라 local 열기"]');
    card?.focus();
    act(() => card?.click());
    rerender({ page: 'operations', camera: 'local', wallPage: '1' });

    expect(document.activeElement).toBe(host.querySelector('#focused-camera-title'));

    const back = Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '← 관제');
    act(() => back?.click());
    rerender({ page: 'operations', wallPage: '1' });

    expect(document.activeElement).toBe(host.querySelector('[aria-label="카메라 local 열기"]'));
  });

  it('focuses the focused-view heading immediately for a direct camera deep link, and falls back to the wall heading if that camera is gone on return (AC-11/AC-12)', () => {
    const { host, rerender } = renderView({ cameras: [camera('a'), camera('b')], location: { page: 'operations', camera: 'a', wallPage: '1' } });
    expect(document.activeElement).toBe(host.querySelector('#focused-camera-title'));

    rerender({ page: 'operations', wallPage: '1' }, [camera('b')]);
    expect(document.activeElement).toBe(host.querySelector('#operations-title'));
  });

  it('remembers and restores wall scroll position across a focus round-trip (AC-9)', () => {
    const { host, rerender } = renderView({ location: { page: 'operations', wallPage: '1' } });
    Object.defineProperty(window, 'scrollY', { configurable: true, value: 240 });
    cleanups.push(() => Object.defineProperty(window, 'scrollY', { configurable: true, value: 0 }));

    const card = host.querySelector<HTMLButtonElement>('[aria-label="카메라 local 열기"]');
    act(() => card?.click());
    rerender({ page: 'operations', camera: 'local', wallPage: '1' });

    Object.defineProperty(window, 'scrollY', { configurable: true, value: 0 });
    const back = Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '← 관제');
    act(() => back?.click());
    rerender({ page: 'operations', wallPage: '1' });

    expect(scrollToSpy).toHaveBeenCalledWith(0, 240);
  });

  it('never applies transition/transform/animate classes to the wall or focus root across the transition (AC-16)', () => {
    const { host, rerender } = renderView({ location: { page: 'operations', wallPage: '1' } });
    const wallRoot = host.querySelector('section[aria-labelledby="operations-title"]');
    expect(wallRoot?.className ?? '').not.toMatch(/transition|animate|transform/);

    rerender({ page: 'operations', camera: 'local', wallPage: '1' });
    const focusRoot = host.querySelector('section[aria-labelledby="focused-camera-title"]');
    expect(focusRoot?.className ?? '').not.toMatch(/transition|animate|transform/);
  });

  it('announces selection invalidation and corrects page without selecting another camera (AC-10/AC-13)', () => {
    const { host, replace } = renderView({ cameras: Array.from({ length: 13 }, (_, index) => camera(String(index))), location: { page: 'operations', camera: 'gone', wallPage: '9' } });
    expect(replace).toHaveBeenCalledWith({ camera: null, wallPage: '2' });
    expect(host.querySelector('[aria-live="polite"]')?.textContent).toContain('선택한 카메라');
    expect(host.querySelector('[aria-pressed]')).toBeNull();
  });
});
