import { describe, expect, it } from 'vitest';
import type { RosterSyncResult } from '@/shared/api/types';
import { rosterSyncResultMessage } from '@/features/camera-management/rosterSyncMessage';

function result(overrides: Partial<RosterSyncResult>): RosterSyncResult {
  return { status: 'synced', error_class: null, detail: null, last_ok_at: null, next_retry_at: null, camera_count: 0, ...overrides };
}

describe('rosterSyncResultMessage', () => {
  it.each([
    [result({ status: 'synced', camera_count: 3 }), '카메라 동기화 완료 · 3대'],
    [result({ status: 'pending', camera_count: 2 }), '카메라 동기화가 대기 중입니다 · 2대'],
    [result({ status: 'disabled' }), '카메라 동기화가 비활성화되어 있습니다.'],
    [result({ status: 'failed', detail: '백엔드에 연결할 수 없습니다. 주소와 네트워크 연결을 확인하세요.' }), '백엔드에 연결할 수 없습니다. 주소와 네트워크 연결을 확인하세요.'],
    [result({ status: 'failed', detail: null }), '카메라 동기화에 실패했습니다.'],
  ])('maps %j to %s', (input, expected) => {
    expect(rosterSyncResultMessage(input)).toBe(expected);
  });
});
