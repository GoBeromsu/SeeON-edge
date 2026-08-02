import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App, { Dashboard } from '@/app/App';
import { deleteCamera, fetchCameras, fetchClips, fetchConnection, fetchDashboardSession, fetchStatus, fetchSystem, loginDashboard, logoutDashboard, type ConnectionView, type StatusSnapshot, type SystemSnapshot } from '@/shared/api/client';
import { HttpError } from '@/shared/api/http';

const defaultInnerWidth = window.innerWidth;

vi.mock('@/shared/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return {
    ...actual,
    deleteCamera: vi.fn(),
    fetchCameras: vi.fn(),
    fetchClips: vi.fn(),
    fetchConnection: vi.fn(),
    fetchDashboardSession: vi.fn(),
    fetchStatus: vi.fn(),
    fetchSystem: vi.fn(),
    loginDashboard: vi.fn(),
    logoutDashboard: vi.fn(),
  };
});

const camera = {
  id: 'local-a', backend_camera_id: 'worker-a', label: '동쪽 카메라', rtsp_url_masked: 'rtsp://***',
  floor_name: '2층', space_id: 'room-a', space_name: '공용실', status: 'online' as const, created_at: null,
};
const clip = {
  id: 'clip-a', camera_id: 'worker-a', camera_label: '동쪽 카메라', event_type: 'fall', created_at: '2026-07-20T01:00:00Z',
  label: null, reviewState: 'unknown' as const, video_path: '/clips/clip-a/video', video_available: true, video_error: null,
};
const status: StatusSnapshot = { cameras: {}, stale_after_sec: 30, runtime: { facilities: {}, stale_after_sec: 30 } };
const system: SystemSnapshot = { version: '1', image_digests: { ml_api: null, ml_worker: null }, backend: { configured: true, reachable: true, last_ok_at: null }, storage: {} };
const connection: ConnectionView = {
  events_url: null, config_url: null, facility_id: null, facility_token_set: false, facility_token_masked: null,
  configured: false, reachable: null, last_ok_at: null, updated_at: null,
};

function mockResources(): void {
  vi.mocked(deleteCamera).mockResolvedValue();
  vi.mocked(fetchCameras).mockResolvedValue({ registry_version: 1, cameras: [camera] });
  vi.mocked(fetchClips).mockResolvedValue([clip]);
  vi.mocked(fetchConnection).mockResolvedValue(connection);
  vi.mocked(fetchDashboardSession).mockRejectedValue(new HttpError(401));
  vi.mocked(fetchStatus).mockResolvedValue(status);
  vi.mocked(fetchSystem).mockResolvedValue(system);
  vi.mocked(loginDashboard).mockResolvedValue();
  vi.mocked(logoutDashboard).mockResolvedValue();
}

async function renderDashboard(element: JSX.Element = <Dashboard />) {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  await act(async () => root.render(element));
  return { host, root };
}

async function renderAuthorizedApp(): Promise<{ host: HTMLDivElement; root: ReturnType<typeof createRoot> }> {
  vi.mocked(fetchDashboardSession).mockResolvedValue();
  return renderDashboard(<App />);
}

function clickDestination(host: HTMLElement, label: string): void {
  const button = Array.from(host.querySelectorAll<HTMLButtonElement>('.desktop-rail nav button')).find((entry) => entry.textContent?.includes(label));
  act(() => button?.click());
}

async function submitLogin(host: HTMLElement): Promise<void> {
  const inputs = host.querySelectorAll('input');
  await act(async () => {
    for (const [input, value] of [[inputs[0], 'operator'], [inputs[1], 'correct-password']] as const) {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set?.call(input, value);
      input.dispatchEvent(new Event('input', { bubbles: true }));
    }
    host.querySelector('form')?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
  });
}

async function clickLogout(host: HTMLElement): Promise<void> {
  await act(async () => {
    Array.from(host.querySelectorAll<HTMLButtonElement>('header button')).find((button) => button.textContent === '로그아웃')?.click();
  });
}

function readCssWithLocalImports(path: string): string {
  const source = readFileSync(path, 'utf8');
  return source.replace(/@import\s+["'](\.[^"']+)["']\s*;/g, (_statement, importPath: string) => (
    readCssWithLocalImports(resolve(dirname(path), importPath))
  ));
}

beforeEach(() => {
  mockResources();
  window.history.replaceState(null, '', '/');
  // jsdom doesn't implement scrollTo; a wall->focus->wall round trip now restores
  // scroll position, so stub it globally to keep test output clean.
  vi.spyOn(window, 'scrollTo').mockImplementation(() => undefined);
});

afterEach(() => {
  document.body.innerHTML = '';
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  Object.defineProperty(window, 'innerWidth', { configurable: true, value: defaultInnerWidth });
});

describe('Dashboard integration', () => {
  it('canonicalizes to the operations wall and exposes exactly four destinations with one mounted page', async () => {
    const { host, root } = await renderDashboard();
    expect(window.location.search).toBe('?page=operations&wallPage=1');
    expect(new Set(Array.from(host.querySelectorAll('nav button')).map((button) => button.textContent))).toHaveLength(4);
    expect(host.querySelectorAll('[aria-current="page"]')).toHaveLength(2);
    expect(host.querySelectorAll('main h1')).toHaveLength(1);
    expect(host.querySelector('main h1')?.textContent).toBe('관제');
    expect(host.textContent).not.toMatch(/홈|방별 보기|탐지 설정|다크 모드|실시간 알림/);
    act(() => root.unmount());
  });

  it('pushes the successful empty-state action to the canonical camera-management URL', async () => {
    vi.mocked(fetchCameras).mockResolvedValue({ registry_version: 1, cameras: [] });
    const push = vi.spyOn(window.history, 'pushState');
    const { host, root } = await renderDashboard();
    push.mockClear();

    const action = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find((button) => button.textContent === '카메라 등록하기');
    act(() => action?.click());

    expect(window.location.search).toBe('?page=cameras');
    expect(push).toHaveBeenCalledTimes(1);
    expect(host.querySelector('main h1')?.textContent).toBe('카메라 관리');
    act(() => root.unmount());
  });

  it('mounts each page exclusively and stops off-page polling', async () => {
    vi.useFakeTimers();
    const { host, root } = await renderDashboard();
    expect(fetchCameras).toHaveBeenCalledTimes(1);
    expect(fetchStatus).toHaveBeenCalledTimes(1);
    expect(fetchClips).not.toHaveBeenCalled();
    expect(fetchSystem).not.toHaveBeenCalled();

    clickDestination(host, '이벤트 기록');
    await act(async () => {});
    expect(host.querySelector('main h1')?.textContent).toBe('이벤트 기록');
    expect(fetchCameras).toHaveBeenCalledTimes(2);
    expect(fetchClips).toHaveBeenCalledTimes(1);
    await act(async () => vi.advanceTimersByTimeAsync(8_000));
    expect(fetchStatus).toHaveBeenCalledTimes(1);
    expect(fetchClips).toHaveBeenCalledTimes(2);

    clickDestination(host, '카메라 관리');
    await act(async () => {});
    expect(host.querySelector('main h1')?.textContent).toBe('카메라 관리');
    expect(host.querySelector('[data-testid="event-history-list"]')).toBeNull();

    clickDestination(host, '시스템');
    await act(async () => {});
    expect(host.querySelector('main h1')?.textContent).toBe('시스템');
    expect(fetchSystem).toHaveBeenCalledTimes(1);
    act(() => root.unmount());
  });

  it('restores focus to the camera-management heading after deleting the invoking card', async () => {
    window.history.replaceState(null, '', '/?page=cameras');
    const { host, root } = await renderDashboard();
    const deleteButton = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find((button) => button.textContent === '삭제');
    act(() => {
      deleteButton?.focus();
      deleteButton?.click();
    });

    await act(async () => {
      Array.from(document.querySelectorAll<HTMLButtonElement>('[role="dialog"] button')).find((button) => button.textContent === '삭제')?.click();
    });

    const heading = host.querySelector('main h1');
    expect(deleteCamera).toHaveBeenCalledWith('local-a');
    expect(heading?.getAttribute('tabindex')).toBe('-1');
    expect(document.activeElement).toBe(heading);
    act(() => root.unmount());
  });

  it('navigates from a camera card to its scoped clips view', async () => {
    window.history.replaceState(null, '', '/?page=cameras');
    const { host, root } = await renderDashboard();

    const viewClips = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find((button) => button.textContent === '클립 보기');
    act(() => viewClips?.click());

    expect(window.location.search).toBe('?page=events&camera=local-a');
    expect(host.querySelector('main h1')?.textContent).toBe('이벤트 기록');
    act(() => root.unmount());
  });

  it('restores direct links and popstate without pushing history', async () => {
    window.history.replaceState(null, '', '/?page=events&floor=2%EC%B8%B5&camera=local-a&event=fall&clip=clip-a');
    const push = vi.spyOn(window.history, 'pushState');
    const { host, root } = await renderDashboard();
    expect(host.querySelector('main h1')?.textContent).toBe('이벤트 기록');
    expect(document.body.querySelector('video')?.getAttribute('src')).toBe('/clips/clip-a/video');

    window.history.replaceState(null, '', '/?page=system&junk=1');
    await act(async () => window.dispatchEvent(new PopStateEvent('popstate')));
    expect(host.querySelector('main h1')?.textContent).toBe('시스템');
    expect(window.location.search).toBe('?page=system');
    expect(push).not.toHaveBeenCalled();
    act(() => root.unmount());
  });

  it('drops a legacy mode= deep link and clears an invalid camera/floor selection back to the wall', async () => {
    window.history.replaceState(null, '', '/?page=operations&floor=missing&camera=removed&mode=focus&wallPage=99');
    const { root } = await renderDashboard();
    expect(window.location.search).toBe('?page=operations&wallPage=1');
    expect(document.body.querySelector('[data-stream]')).toBeNull();
    act(() => root.unmount());
  });

  it('replaces a focused camera with the wall when a successful refresh removes it', async () => {
    vi.useFakeTimers();
    vi.mocked(fetchCameras)
      .mockResolvedValueOnce({ registry_version: 1, cameras: [camera] })
      .mockResolvedValueOnce({ registry_version: 2, cameras: [] });
    window.history.replaceState(null, '', '/?page=operations&camera=local-a&wallPage=1');
    const { root } = await renderDashboard();
    expect(document.body.querySelector('[data-stream]')).not.toBeNull();
    await act(async () => vi.advanceTimersByTimeAsync(5_000));
    expect(window.location.search).toBe('?page=operations&wallPage=1');
    expect(document.body.querySelector('[data-stream]')).toBeNull();
    act(() => root.unmount());
  });

  it('shows the camera resource last-success time during retained refresh failure and removes it on recovery', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-21T00:15:00Z'));
    const retainedRegistry = { registry_version: 1, cameras: [camera] };
    vi.mocked(fetchCameras)
      .mockResolvedValueOnce(retainedRegistry)
      .mockRejectedValueOnce(new Error('camera refresh failed'))
      .mockResolvedValueOnce(retainedRegistry);
    const { host, root } = await renderDashboard();

    vi.setSystemTime(new Date('2026-07-21T00:20:00Z'));
    await act(async () => vi.advanceTimersByTimeAsync(5_000));
    expect(host.textContent).toContain('마지막 카메라 목록을 표시합니다.');
    expect(host.textContent).toContain(`마지막 확인 ${new Date('2026-07-21T00:15:00Z').toLocaleString('ko-KR')}`);

    await act(async () => vi.advanceTimersByTimeAsync(5_000));
    expect(host.textContent).not.toContain('마지막 카메라 목록을 표시합니다.');
    expect(host.textContent).not.toContain('마지막 확인');
    act(() => root.unmount());
  });

  it('keeps an empty last-good registry truthful through refresh failure and recovery', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-21T00:15:00Z'));
    const emptyRegistry = { registry_version: 1, cameras: [] };
    vi.mocked(fetchCameras)
      .mockResolvedValueOnce(emptyRegistry)
      .mockRejectedValueOnce(new Error('camera refresh failed'))
      .mockResolvedValueOnce({ registry_version: 2, cameras: [] });
    const { host, root } = await renderDashboard();
    const addCameraAction = (): HTMLButtonElement | undefined => Array.from(host.querySelectorAll<HTMLButtonElement>('button'))
      .find((button) => button.textContent === '카메라 등록하기');

    expect(addCameraAction()).not.toBeUndefined();
    vi.setSystemTime(new Date('2026-07-21T00:20:00Z'));
    await act(async () => vi.advanceTimersByTimeAsync(5_000));
    expect(host.textContent).toContain('마지막 확인 당시 등록된 카메라가 없습니다.');
    expect(host.textContent).toContain(`마지막 확인 ${new Date('2026-07-21T00:15:00Z').toLocaleString('ko-KR')}`);
    expect(host.textContent).not.toContain('카메라 목록을 불러올 수 없음');
    expect(addCameraAction()).not.toBeUndefined();

    vi.setSystemTime(new Date('2026-07-21T00:25:00Z'));
    await act(async () => vi.advanceTimersByTimeAsync(5_000));
    expect(host.textContent).not.toContain('마지막 확인 당시 등록된 카메라가 없습니다.');
    expect(host.textContent).not.toContain('마지막 확인');
    expect(host.textContent).toContain('등록된 카메라가 없습니다.');
    expect(addCameraAction()).not.toBeUndefined();
    act(() => root.unmount());
  });

  it.each([
    { label: 'empty', cameras: [] },
    { label: 'populated', cameras: [camera] },
  ])('keeps $label operations content and focus while a camera refresh is pending', async ({ cameras }) => {
    vi.useFakeTimers();
    let resolveRefresh: ((value: { registry_version: number; cameras: typeof cameras }) => void) | undefined;
    vi.mocked(fetchCameras)
      .mockResolvedValueOnce({ registry_version: 1, cameras })
      .mockImplementationOnce(() => new Promise((resolve) => { resolveRefresh = resolve; }));
    const { host, root } = await renderDashboard();
    const focusTarget = cameras.length === 0
      ? Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find((button) => button.textContent === '카메라 등록하기')
      : host.querySelector<HTMLButtonElement>('[aria-label="동쪽 카메라 열기"]');
    focusTarget?.focus();

    await act(async () => vi.advanceTimersByTimeAsync(5_000));
    const refreshStatus = host.querySelector('[data-camera-refresh-status]');
    expect(host.querySelectorAll('[data-camera-refresh-status]')).toHaveLength(1);
    expect(refreshStatus?.getAttribute('role')).toBe('status');
    expect(refreshStatus?.getAttribute('aria-live')).toBe('polite');
    expect(refreshStatus?.textContent).toBe('카메라 목록을 새로 확인하고 있습니다.');
    expect(document.activeElement).toBe(focusTarget);
    if (cameras.length === 0) expect(host.textContent).toContain('등록된 카메라가 없습니다.');
    else expect(host.querySelector('[aria-label="동쪽 카메라 열기"]')).not.toBeNull();

    await act(async () => resolveRefresh?.({ registry_version: 2, cameras }));
    expect(host.querySelector('[data-camera-refresh-status]')).toBeNull();
    expect(document.activeElement).toBe(focusTarget);
    act(() => root.unmount());
  });

  it('defers operations popstate DTO validation during an error and replaces once after recovery', async () => {
    vi.useFakeTimers();
    const retainedRegistry = { registry_version: 1, cameras: [camera] };
    vi.mocked(fetchCameras)
      .mockResolvedValueOnce(retainedRegistry)
      .mockRejectedValueOnce(new Error('camera refresh failed'))
      .mockResolvedValueOnce(retainedRegistry);
    window.history.replaceState(null, '', '/?page=operations&camera=local-a&wallPage=1');
    const { root } = await renderDashboard();
    await act(async () => vi.advanceTimersByTimeAsync(5_000));

    const historyLength = window.history.length;
    const push = vi.spyOn(window.history, 'pushState');
    const replace = vi.spyOn(window.history, 'replaceState');
    window.history.replaceState(null, '', '/?page=operations&floor=missing&camera=removed&wallPage=99');
    replace.mockClear();
    await act(async () => window.dispatchEvent(new PopStateEvent('popstate')));

    expect(window.location.search).toBe('?page=operations&floor=missing&camera=removed&wallPage=99');
    expect(replace).not.toHaveBeenCalled();
    expect(push).not.toHaveBeenCalled();
    expect(window.history.length).toBe(historyLength);

    await act(async () => vi.advanceTimersByTimeAsync(5_000));
    expect(window.location.search).toBe('?page=operations&wallPage=1');
    expect(replace).toHaveBeenCalledTimes(1);
    expect(push).not.toHaveBeenCalled();
    expect(window.history.length).toBe(historyLength);
    act(() => root.unmount());
  });

  it('defers event popstate DTO validation during a clip error and replaces once after recovery', async () => {
    vi.useFakeTimers();
    const retainedClips = [clip];
    vi.mocked(fetchClips)
      .mockResolvedValueOnce(retainedClips)
      .mockRejectedValueOnce(new Error('clip refresh failed'))
      .mockResolvedValueOnce(retainedClips);
    window.history.replaceState(null, '', '/?page=events&floor=2%EC%B8%B5&camera=local-a&event=fall&clip=clip-a');
    const { root } = await renderDashboard();
    await act(async () => vi.advanceTimersByTimeAsync(8_000));

    const historyLength = window.history.length;
    const push = vi.spyOn(window.history, 'pushState');
    const replace = vi.spyOn(window.history, 'replaceState');
    window.history.replaceState(null, '', '/?page=events&floor=missing&camera=removed&event=bed_exit&clip=missing');
    replace.mockClear();
    await act(async () => window.dispatchEvent(new PopStateEvent('popstate')));

    expect(window.location.search).toBe('?page=events&floor=missing&camera=removed&event=bed_exit&clip=missing');
    expect(replace).not.toHaveBeenCalled();
    expect(push).not.toHaveBeenCalled();
    expect(window.history.length).toBe(historyLength);

    await act(async () => vi.advanceTimersByTimeAsync(8_000));
    expect(window.location.search).toBe('?page=events');
    expect(replace).toHaveBeenCalledTimes(1);
    expect(push).not.toHaveBeenCalled();
    expect(window.history.length).toBe(historyLength);
    act(() => root.unmount());
  });

  it('resumes event URL validation after popstate restoration across later successful polls', async () => {
    vi.useFakeTimers();
    const laterClip = { ...clip, id: 'clip-b', video_path: '/clips/clip-b/video' };
    vi.mocked(fetchClips)
      .mockResolvedValueOnce([clip, laterClip])
      .mockResolvedValueOnce([laterClip])
      .mockResolvedValueOnce([]);
    window.history.replaceState(null, '', '/?page=events');
    const { host, root } = await renderDashboard();

    const historyLength = window.history.length;
    const push = vi.spyOn(window.history, 'pushState');
    const replace = vi.spyOn(window.history, 'replaceState');
    window.history.replaceState(null, '', '/?page=events&floor=2%EC%B8%B5&camera=local-a&event=fall&clip=clip-a&junk=1');
    replace.mockClear();
    await act(async () => window.dispatchEvent(new PopStateEvent('popstate')));
    expect(window.location.search).toBe('?page=events&floor=2%EC%B8%B5&camera=local-a&event=fall&clip=clip-a');
    expect(document.body.querySelector('video')?.getAttribute('src')).toBe('/clips/clip-a/video');

    await act(async () => vi.advanceTimersByTimeAsync(8_000));
    expect(window.location.search).toBe('?page=events&floor=2%EC%B8%B5&camera=local-a&event=fall');
    expect(document.body.querySelector('video')).toBeNull();
    expect(host.querySelector<HTMLSelectElement>('[aria-label="이벤트 필터"]')?.value).toBe('fall');

    await act(async () => vi.advanceTimersByTimeAsync(8_000));
    expect(window.location.search).toBe('?page=events&floor=2%EC%B8%B5&camera=local-a');
    expect(host.querySelector<HTMLSelectElement>('[aria-label="이벤트 필터"]')?.value).toBe('');
    expect(replace).toHaveBeenCalledTimes(3);
    expect(push).not.toHaveBeenCalled();
    expect(window.history.length).toBe(historyLength);
    act(() => root.unmount());
  });

  it('renders the focused view directly with no inspector or bottom sheet at any width, without changing URL selection', async () => {
    window.history.replaceState(null, '', '/?page=operations&camera=local-a&wallPage=1');
    const { host, root } = await renderDashboard();

    expect(host.querySelector('[role="complementary"]')).toBeNull();
    expect(document.body.querySelector('[role="dialog"][aria-modal="true"]')).toBeNull();
    expect(host.querySelector('#focused-camera-title')?.textContent).toBe('동쪽 카메라 · 2층');
    expect(host.querySelectorAll('[data-stream]')).toHaveLength(1);
    expect(window.location.search).toContain('camera=local-a');
    act(() => root.unmount());
  });

  it.each([375, 768])('navigates directly into the focused view and back at %ipx, moving focus exactly per the new screen-transition contract', async (width) => {
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: width });
    const { host, root } = await renderDashboard();
    const card = host.querySelector<HTMLButtonElement>('[aria-label="동쪽 카메라 열기"]');
    act(() => {
      card?.focus();
      card?.click();
    });

    expect(window.location.search).toBe('?page=operations&camera=local-a&wallPage=1');
    expect(document.activeElement).toBe(host.querySelector('#focused-camera-title'));
    expect(document.body.querySelector('[role="dialog"]')).toBeNull();
    expect(document.body.querySelector('[role="complementary"]')).toBeNull();

    const back = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find((button) => button.textContent === '← 관제');
    act(() => back?.click());

    expect(window.location.search).toBe('?page=operations&wallPage=1');
    expect(document.body.querySelector('[role="dialog"]')).toBeNull();
    expect(document.activeElement).toBe(host.querySelector('[aria-label="동쪽 카메라 열기"]'));
    act(() => root.unmount());
  });

  it('resets location on logout and starts the default wall after login again', async () => {
    window.history.replaceState(null, '', '/?page=system');
    const { host, root } = await renderDashboard(<App />);
    await submitLogin(host);
    expect(loginDashboard).toHaveBeenCalledWith('operator', 'correct-password');
    expect(host.querySelector('main h1')?.textContent).toBe('시스템');
    await clickLogout(host);
    expect(logoutDashboard).toHaveBeenCalledTimes(1);
    expect(window.location.search).toBe('?page=operations&wallPage=1');
    expect(host.querySelector('#login-title')?.textContent).toBe('로그인');
    await submitLogin(host);
    expect(host.querySelector('main h1')?.textContent).toBe('관제');
    act(() => root.unmount());
  });

  it('keeps the dashboard authorized when logout fails', async () => {
    window.history.replaceState(null, '', '/?page=system');
    vi.mocked(logoutDashboard).mockRejectedValueOnce(new Error('logout failed'));
    const { host, root } = await renderAuthorizedApp();
    expect(host.querySelector('main h1')?.textContent).toBe('시스템');
    await clickLogout(host);
    expect(logoutDashboard).toHaveBeenCalledTimes(1);
    expect(host.querySelector('main h1')?.textContent).toBe('시스템');
    expect(host.textContent).toContain('로그아웃하지 못했습니다. 다시 시도해 주세요.');
    expect(host.querySelector('#login-title')).toBeNull();
    expect(window.location.search).toBe('?page=system');
    act(() => root.unmount());
  });
});

describe('responsive shell contract', () => {
  it('keeps fixed navigation clear, controls touchable, operational numbers tabular, and reduced motion static', () => {
    const css = readCssWithLocalImports('src/styles.css');
    expect(css).toMatch(/padding-bottom:\s*calc\(72px \+ env\(safe-area-inset-bottom\)\)/);
    expect(css).toMatch(/min-height:\s*44px/);
    expect(css).toMatch(/font-variant-numeric:\s*tabular-nums/);
    expect(css).toMatch(/@media \(prefers-reduced-motion: reduce\)/);
    expect(css).toMatch(/overflow-x:\s*hidden/);
    expect(css).toMatch(/@media \(min-width: 1024px\)/);
    const app = readFileSync('src/app/App.tsx', 'utf8');
    expect(app).not.toMatch(/screen === ['"](?:home|rooms|settings)['"]/);
    expect(app).not.toMatch(/DetectionSettingsForm|useDarkMode/);
  });
});
