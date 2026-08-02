import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  DASHBOARD_PAGES,
  DASHBOARD_QUERY_KEYS,
  DashboardLocationController,
  canonicalizeDashboardLocation,
  serializeDashboardLocation,
} from './dashboardLocation';

function resetLocation(search = '') {
  window.history.replaceState(null, '', `/${search}`);
}

afterEach(() => {
  resetLocation();
});

describe('DASHBOARD_PAGES / DASHBOARD_QUERY_KEYS', () => {
  it('exposes the 3-page information architecture', () => {
    expect(DASHBOARD_PAGES).toEqual(['operations', 'events', 'settings']);
  });

  it('exposes only the surviving query keys (no room/space/wallPage)', () => {
    expect(DASHBOARD_QUERY_KEYS).toEqual(['page', 'floor', 'camera', 'event', 'clip']);
  });
});

describe('canonicalizeDashboardLocation - syntax', () => {
  it('defaults to the operations page when search is empty', () => {
    const result = canonicalizeDashboardLocation('');
    expect(result.location).toEqual({ page: 'operations' });
    expect(result.search).toBe('?page=operations');
    expect(result.changed).toBe(true);
  });

  it('defaults to operations for an unknown page value', () => {
    const result = canonicalizeDashboardLocation('?page=cameras');
    expect(result.location.page).toBe('operations');
  });

  it('keeps floor/camera on the operations page', () => {
    const result = canonicalizeDashboardLocation('?page=operations&floor=2F&camera=cam-1');
    expect(result.location).toEqual({ page: 'operations', floor: '2F', camera: 'cam-1' });
    expect(result.changed).toBe(false);
  });

  it('strips floor/camera when the page is events', () => {
    const result = canonicalizeDashboardLocation('?page=events&floor=2F&camera=cam-1');
    expect(result.location).toEqual({ page: 'events' });
    expect(result.changed).toBe(true);
  });

  it('keeps event/clip on the events page', () => {
    const result = canonicalizeDashboardLocation('?page=events&event=fall&clip=clip-1');
    expect(result.location).toEqual({ page: 'events', event: 'fall', clip: 'clip-1' });
    expect(result.changed).toBe(false);
  });

  it('strips event/clip when the page is operations', () => {
    const result = canonicalizeDashboardLocation('?page=operations&event=fall&clip=clip-1');
    expect(result.location).toEqual({ page: 'operations' });
    expect(result.changed).toBe(true);
  });

  it('carries no extra keys for the settings page', () => {
    const result = canonicalizeDashboardLocation('?page=settings&floor=2F&event=fall');
    expect(result.location).toEqual({ page: 'settings' });
    expect(result.changed).toBe(true);
  });

  it('takes the last value when a key repeats', () => {
    const result = canonicalizeDashboardLocation('?page=operations&floor=1F&floor=2F');
    expect(result.location.floor).toBe('2F');
  });
});

describe('canonicalizeDashboardLocation - operations data validation', () => {
  const cameras = {
    status: 'success' as const,
    data: [
      { id: 'cam-1', floor_name: '2F' },
      { id: 'cam-2', floor_name: '3F' },
    ],
  };

  it('drops an unknown floor', () => {
    const result = canonicalizeDashboardLocation('?page=operations&floor=9F', { cameras });
    expect(result.location).toEqual({ page: 'operations' });
    expect(result.changed).toBe(true);
  });

  it('drops an unknown camera id', () => {
    const result = canonicalizeDashboardLocation('?page=operations&camera=cam-missing', { cameras });
    expect(result.location).toEqual({ page: 'operations' });
  });

  it('drops a camera that does not belong to the selected floor', () => {
    const result = canonicalizeDashboardLocation('?page=operations&floor=2F&camera=cam-2', { cameras });
    expect(result.location).toEqual({ page: 'operations', floor: '2F' });
  });

  it('keeps a camera that matches the selected floor', () => {
    const result = canonicalizeDashboardLocation('?page=operations&floor=2F&camera=cam-1', { cameras });
    expect(result.location).toEqual({ page: 'operations', floor: '2F', camera: 'cam-1' });
    expect(result.changed).toBe(false);
  });

  it('does not validate against cameras while the resource has not succeeded', () => {
    const result = canonicalizeDashboardLocation('?page=operations&floor=9F', { cameras: { status: 'loading' } });
    expect(result.location).toEqual({ page: 'operations', floor: '9F' });
    expect(result.changed).toBe(false);
  });
});

describe('canonicalizeDashboardLocation - events data validation', () => {
  const clips = {
    status: 'success' as const,
    data: [
      { id: 'clip-1', event_type: 'fall' },
      { id: 'clip-2', event_type: 'bedexit' },
    ],
  };

  it('drops an unknown event type', () => {
    const result = canonicalizeDashboardLocation('?page=events&event=unknown', { clips });
    expect(result.location).toEqual({ page: 'events' });
  });

  it('drops an unknown clip id', () => {
    const result = canonicalizeDashboardLocation('?page=events&clip=clip-missing', { clips });
    expect(result.location).toEqual({ page: 'events' });
  });

  it('drops a clip whose event type mismatches the selected event filter', () => {
    const result = canonicalizeDashboardLocation('?page=events&event=fall&clip=clip-2', { clips });
    expect(result.location).toEqual({ page: 'events', event: 'fall' });
  });

  it('keeps a clip that matches the selected event filter', () => {
    const result = canonicalizeDashboardLocation('?page=events&event=fall&clip=clip-1', { clips });
    expect(result.location).toEqual({ page: 'events', event: 'fall', clip: 'clip-1' });
    expect(result.changed).toBe(false);
  });
});

describe('serializeDashboardLocation', () => {
  it('serializes keys in canonical order', () => {
    const search = serializeDashboardLocation({ page: 'events', clip: 'clip-1', event: 'fall' });
    expect(search).toBe('?page=events&event=fall&clip=clip-1');
  });

  it('omits undefined keys', () => {
    expect(serializeDashboardLocation({ page: 'settings' })).toBe('?page=settings');
  });
});

describe('DashboardLocationController', () => {
  it('canonicalizes the current URL on start and reports it via onChange', () => {
    resetLocation('?page=cameras');
    const onChange = vi.fn();
    const controller = new DashboardLocationController(window, onChange);
    const location = controller.start();
    expect(location).toEqual({ page: 'operations' });
    expect(window.location.search).toBe('?page=operations');
    expect(onChange).toHaveBeenCalledWith({ page: 'operations' });
    controller.dispose();
  });

  it('navigate pushes a new history entry and merges into the current location', () => {
    resetLocation('?page=operations');
    const onChange = vi.fn();
    const controller = new DashboardLocationController(window, onChange);
    controller.start();

    const pushSpy = vi.spyOn(window.history, 'pushState');
    const next = controller.navigate({ floor: '2F' });
    expect(next).toEqual({ page: 'operations', floor: '2F' });
    expect(window.location.search).toBe('?page=operations&floor=2F');
    expect(pushSpy).toHaveBeenCalled();

    controller.navigate({ page: 'events', floor: null });
    expect(window.location.search).toBe('?page=events');

    pushSpy.mockRestore();
    controller.dispose();
  });

  it('reset returns to the default operations location via replaceState', () => {
    resetLocation('?page=events&event=fall');
    const onChange = vi.fn();
    const controller = new DashboardLocationController(window, onChange);
    controller.start();

    const replaceSpy = vi.spyOn(window.history, 'replaceState');
    const location = controller.reset();
    expect(location).toEqual({ page: 'operations' });
    expect(window.location.search).toBe('?page=operations');
    expect(replaceSpy).toHaveBeenCalled();
    replaceSpy.mockRestore();
    controller.dispose();
  });

  it('re-canonicalizes on popstate', () => {
    resetLocation('?page=operations');
    const onChange = vi.fn();
    const controller = new DashboardLocationController(window, onChange);
    controller.start();

    window.history.pushState(null, '', '/?page=events&event=fall');
    window.dispatchEvent(new PopStateEvent('popstate'));

    expect(onChange).toHaveBeenLastCalledWith({ page: 'events', event: 'fall' });
    controller.dispose();
  });

  it('dispose removes the popstate listener', () => {
    resetLocation('?page=operations');
    const onChange = vi.fn();
    const controller = new DashboardLocationController(window, onChange);
    controller.start();
    controller.dispose();

    onChange.mockClear();
    window.history.pushState(null, '', '/?page=events');
    window.dispatchEvent(new PopStateEvent('popstate'));
    expect(onChange).not.toHaveBeenCalled();
  });
});
