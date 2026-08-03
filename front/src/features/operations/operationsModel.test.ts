import { describe, expect, it } from 'vitest';
import {
  countCamerasByLiveness,
  eventTypeLabel,
  filterCamerasByFloor,
  formatClipDateTime,
  isCameraOnline,
  listFloors,
  sortCameras,
} from '@/features/operations/operationsModel';
import type { Camera } from '@/shared/api/client';

function camera(overrides: Partial<Camera> & Pick<Camera, 'id' | 'label'>): Camera {
  return {
    rtsp_url_masked: 'rtsp://***',
    floor_name: null,
    status: 'online',
    created_at: null,
    ...overrides,
  };
}

describe('sortCameras', () => {
  it('sorts by floor then label using locale-aware numeric comparison', () => {
    const cameras = [
      camera({ id: 'c', label: '10호', floor_name: '2층' }),
      camera({ id: 'b', label: '2호', floor_name: '2층' }),
      camera({ id: 'a', label: '101호', floor_name: '1층' }),
    ];
    expect(sortCameras(cameras).map((c) => c.id)).toEqual(['a', 'b', 'c']);
  });

  it('sorts cameras with no floor after cameras with a floor', () => {
    const cameras = [
      camera({ id: 'a', label: '무층', floor_name: null }),
      camera({ id: 'b', label: '1호', floor_name: '1층' }),
    ];
    expect(sortCameras(cameras).map((c) => c.id)).toEqual(['b', 'a']);
  });
});

describe('listFloors', () => {
  it('returns unique, sorted floor names, ignoring cameras with no floor', () => {
    const cameras = [
      camera({ id: 'a', label: 'a', floor_name: '2층' }),
      camera({ id: 'b', label: 'b', floor_name: '1층' }),
      camera({ id: 'c', label: 'c', floor_name: '1층' }),
      camera({ id: 'd', label: 'd', floor_name: null }),
    ];
    expect(listFloors(cameras)).toEqual(['1층', '2층']);
  });
});

describe('filterCamerasByFloor', () => {
  it('returns all cameras, sorted, when floor is undefined', () => {
    const cameras = [
      camera({ id: 'b', label: 'b', floor_name: '2층' }),
      camera({ id: 'a', label: 'a', floor_name: '1층' }),
    ];
    expect(filterCamerasByFloor(cameras, undefined).map((c) => c.id)).toEqual(['a', 'b']);
  });

  it('returns only cameras on the given floor', () => {
    const cameras = [
      camera({ id: 'a', label: 'a', floor_name: '1층' }),
      camera({ id: 'b', label: 'b', floor_name: '2층' }),
    ];
    expect(filterCamerasByFloor(cameras, '2층').map((c) => c.id)).toEqual(['b']);
  });
});

describe('isCameraOnline / countCamerasByLiveness', () => {
  it('treats only "online" status as online', () => {
    expect(isCameraOnline({ status: 'online' })).toBe(true);
    expect(isCameraOnline({ status: 'offline' })).toBe(false);
    expect(isCameraOnline({ status: 'starting' })).toBe(false);
    expect(isCameraOnline({ status: 'unknown' })).toBe(false);
  });

  it('counts online vs offline cameras', () => {
    const cameras = [
      camera({ id: 'a', label: 'a', status: 'online' }),
      camera({ id: 'b', label: 'b', status: 'offline' }),
      camera({ id: 'c', label: 'c', status: 'starting' }),
    ];
    expect(countCamerasByLiveness(cameras)).toEqual({ online: 1, offline: 2 });
  });
});

describe('eventTypeLabel', () => {
  it('translates known event types to Korean labels', () => {
    expect(eventTypeLabel('bed-exit')).toBe('침대 이탈');
    expect(eventTypeLabel('fall')).toBe('낙상');
  });

  it('falls back to the raw type for unknown values', () => {
    expect(eventTypeLabel('mystery')).toBe('mystery');
  });
});

describe('formatClipDateTime', () => {
  it('formats a valid ISO timestamp', () => {
    expect(formatClipDateTime('2026-08-02T03:12:00Z')).not.toBe('시간 정보 없음');
  });

  it('returns a fallback message for null or invalid timestamps', () => {
    expect(formatClipDateTime(null)).toBe('시간 정보 없음');
    expect(formatClipDateTime('not-a-date')).toBe('시간 정보 없음');
  });
});
