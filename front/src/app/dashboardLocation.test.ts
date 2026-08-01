import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  DashboardLocationController,
  canonicalizeDashboardLocation,
  type DashboardLocationData,
} from '@/app/dashboardLocation';

const cameras = [
  { id: 'cam-A', backend_camera_id: 'worker-A', floor_name: '한 글', space_id: '방/1' },
  { id: 'cam-B', backend_camera_id: null, floor_name: '2층', space_id: 'room-2' },
];
const clips = [
  { id: 'clip-1', camera_id: 'cam-A', event_type: 'fall' },
  { id: 'orphan', camera_id: 'missing', event_type: 'bed_exit' },
];
const successData: DashboardLocationData = {
  cameras: { status: 'success', data: cameras },
  clips: { status: 'success', data: clips },
};

afterEach(() => {
  window.history.replaceState(null, '', '/');
  vi.restoreAllMocks();
});

describe('dashboard location syntax', () => {
  it('inserts operations defaults and serializes exact canonical order', () => {
    expect(canonicalizeDashboardLocation('').search).toBe('?page=operations&wallPage=1');
    const result = canonicalizeDashboardLocation('?wallPage=9007199254740991&clip=c&event=fall&camera=cam-A&room=%EB%B0%A9%2F1&floor=%ED%95%9C+%EA%B8%80&page=events');
    expect(result.search).toBe('?page=events&floor=%ED%95%9C+%EA%B8%80&room=%EB%B0%A9%2F1&camera=cam-A&event=fall&clip=c');
    expect(result.location.floor).toBe('한 글');
    expect(result.location.room).toBe('방/1');
  });

  it('uses the last duplicate, drops unknown/empty/inapplicable keys (including a legacy mode=), and preserves exact Unicode', () => {
    const result = canonicalizeDashboardLocation('?page=events&page=operations&floor=&floor=%ED%95%9C+%EA%B8%80&room=%EB%B0%A9%2F1&camera=Cam-A&event=fall&clip=c&wat=no&mode=focus&wallPage=01');
    expect(result.search).toBe('?page=operations&floor=%ED%95%9C+%EA%B8%80&room=%EB%B0%A9%2F1&camera=Cam-A&wallPage=1');
    expect(result.location.camera).toBe('Cam-A');
    expect(result.location).not.toHaveProperty('mode');
  });

  it.each(['+1', '-1', '0', '01', '9007199254740992', '1.0'])('rejects noncanonical wallPage %s without precision loss', (wallPage) => {
    expect(canonicalizeDashboardLocation(`?page=operations&wallPage=${encodeURIComponent(wallPage)}`).location.wallPage).toBe('1');
  });

  it('accepts the safe integer upper bound and every page value', () => {
    expect(canonicalizeDashboardLocation('?page=operations&wallPage=9007199254740991').location).toMatchObject({ page: 'operations', wallPage: '9007199254740991' });
    expect(canonicalizeDashboardLocation('?page=events&floor=x&room=y&camera=z&event=e&clip=c&wallPage=2').search).toBe('?page=events&floor=x&room=y&camera=z&event=e&clip=c');
    for (const page of ['cameras', 'system'] as const) expect(canonicalizeDashboardLocation(`?page=${page}&floor=x&room=y&camera=z&event=e&clip=c&wallPage=2`).search).toBe(`?page=${page}`);
  });
});

describe('dashboard location DTO validation', () => {
  it.each(['idle', 'loading', 'error'] as const)('retains DTO-backed values and wall page while cameras are %s', (status) => {
    const result = canonicalizeDashboardLocation('?page=operations&floor=x&room=y&camera=z&wallPage=50', { cameras: { status } });
    expect(result.search).toBe('?page=operations&floor=x&room=y&camera=z&wallPage=50');
  });

  it('invalidates in order after success and clamps wall pages; an invalidated camera simply falls back to the wall (no mode key involved)', () => {
    expect(canonicalizeDashboardLocation('?page=operations&floor=missing&room=room-2&camera=cam-B&wallPage=50', successData).search)
      .toBe('?page=operations&camera=cam-B&wallPage=1');
    expect(canonicalizeDashboardLocation('?page=operations&floor=missing&room=room-2&camera=missing&wallPage=50', successData).search)
      .toBe('?page=operations&wallPage=1');
    expect(canonicalizeDashboardLocation('?page=operations&floor=2%EC%B8%B5&room=room-2&camera=cam-B&wallPage=9007199254740991', successData).search)
      .toBe('?page=operations&floor=2%EC%B8%B5&room=room-2&camera=cam-B&wallPage=1');
  });

  it('waits for both event resources and retains only compatible orphan clips', () => {
    const query = '?page=events&floor=x&room=y&camera=z&event=bed_exit&clip=orphan';
    expect(canonicalizeDashboardLocation(query, { cameras: successData.cameras, clips: { status: 'error' } }).search).toBe(query);
    expect(canonicalizeDashboardLocation(query, successData).search).toBe('?page=events&event=bed_exit&clip=orphan');
    expect(canonicalizeDashboardLocation('?page=events&event=bed_exit&clip=orphan', successData).search).toBe('?page=events&event=bed_exit&clip=orphan');
    expect(canonicalizeDashboardLocation('?page=events&floor=2%EC%B8%B5&event=bed_exit&clip=orphan', successData).search).toBe('?page=events&floor=2%EC%B8%B5');
  });
});

describe('DashboardLocationController history', () => {
  it('replaces canonicalization, pushes deliberate changes, and never rewrites identical queries', () => {
    window.history.replaceState(null, '', '/?unknown=1');
    const push = vi.spyOn(window.history, 'pushState');
    const replace = vi.spyOn(window.history, 'replaceState');
    const controller = new DashboardLocationController(window);
    controller.start();
    expect(replace).toHaveBeenCalledTimes(1);
    controller.navigate({ page: 'events' });
    expect(push).toHaveBeenCalledTimes(1);
    controller.canonicalize();
    expect(replace).toHaveBeenCalledTimes(1);
    controller.dispose();
  });

  it('drops a legacy mode= deep link as an unrecognized key via replaceState, leaving history.length unchanged (AC-7)', () => {
    window.history.replaceState(null, '', '/?page=operations&camera=cam-A&mode=focus&wallPage=1');
    const length = window.history.length;
    const push = vi.spyOn(window.history, 'pushState');
    const replace = vi.spyOn(window.history, 'replaceState');
    const controller = new DashboardLocationController(window);
    const location = controller.start();
    expect(location).not.toHaveProperty('mode');
    expect(window.location.search).toBe('?page=operations&camera=cam-A&wallPage=1');
    expect(push).not.toHaveBeenCalled();
    expect(replace).toHaveBeenCalledTimes(1);
    expect(window.history.length).toBe(length);
    controller.dispose();
  });

  it('pushes exactly one history entry for a wall-to-camera navigation, and restoring the prior URL preserves wallPage/floor/room (AC-8/AC-9)', () => {
    window.history.replaceState(null, '', '/?page=operations&floor=2%EC%B8%B5&room=room-2&wallPage=3');
    const previousSearch = window.location.search;
    const startLength = window.history.length;
    const push = vi.spyOn(window.history, 'pushState');
    const controller = new DashboardLocationController(window);
    controller.start();

    controller.navigate({ camera: 'cam-B' });
    expect(push).toHaveBeenCalledTimes(1);
    expect(window.history.length).toBe(startLength + 1);
    expect(window.location.search).toBe('?page=operations&floor=2%EC%B8%B5&room=room-2&camera=cam-B&wallPage=3');

    // Simulate the browser restoring the prior history entry (back navigation).
    window.history.replaceState(null, '', previousSearch);
    window.dispatchEvent(new PopStateEvent('popstate'));
    expect(window.location.search).toBe(previousSearch);
    controller.dispose();
  });

  it('restores with zero pushes and at most one immediate plus one later replacement', () => {
    window.history.replaceState(null, '', '/?page=events&page=operations&camera=missing&mode=focus&wallPage=2&bad=1');
    const length = window.history.length;
    const push = vi.spyOn(window.history, 'pushState');
    const replace = vi.spyOn(window.history, 'replaceState');
    const controller = new DashboardLocationController(window);
    controller.start();
    window.history.replaceState(null, '', '/?page=events&page=operations&camera=missing&mode=focus&wallPage=2&bad=1');
    replace.mockClear();
    window.dispatchEvent(new PopStateEvent('popstate'));
    expect(push).not.toHaveBeenCalled();
    expect(replace).toHaveBeenCalledTimes(1);
    expect(window.location.search).toBe('?page=operations&camera=missing&wallPage=2');
    controller.canonicalize(successData);
    controller.canonicalize(successData);
    expect(replace).toHaveBeenCalledTimes(2);
    expect(window.location.search).toBe('?page=operations&wallPage=1');
    expect(window.history.length).toBe(length);
    controller.dispose();
  });

  it('resumes ordinary data replacements after the first restoration validation', () => {
    const camerasFor = (count: number) => Array.from({ length: count }, (_, index) => ({
      id: `cam-${index}`,
      backend_camera_id: null,
      floor_name: '2층',
      space_id: 'room-2',
    }));
    window.history.replaceState(null, '', '/?page=operations&wallPage=1');
    const length = window.history.length;
    const push = vi.spyOn(window.history, 'pushState');
    const replace = vi.spyOn(window.history, 'replaceState');
    const controller = new DashboardLocationController(window);
    controller.start({ cameras: { status: 'success', data: camerasFor(25) } });

    window.history.replaceState(null, '', '/?page=operations&wallPage=99&junk=1');
    replace.mockClear();
    window.dispatchEvent(new PopStateEvent('popstate'));
    controller.canonicalize({ cameras: { status: 'success', data: camerasFor(25) } });
    expect(window.location.search).toBe('?page=operations&wallPage=3');

    controller.canonicalize({ cameras: { status: 'success', data: camerasFor(13) } });
    expect(window.location.search).toBe('?page=operations&wallPage=2');
    controller.canonicalize({ cameras: { status: 'success', data: camerasFor(1) } });
    expect(window.location.search).toBe('?page=operations&wallPage=1');
    expect(replace).toHaveBeenCalledTimes(4);
    expect(push).not.toHaveBeenCalled();
    expect(window.history.length).toBe(length);
    controller.dispose();
  });
});
