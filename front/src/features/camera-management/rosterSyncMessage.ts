import type { RosterSyncResult } from '@/shared/api/types';

/** Maps a fresh POST /connection/sync-cameras result to the Korean summary shown after the technician triggers a manual sync. */
export function rosterSyncResultMessage(result: RosterSyncResult): string {
  switch (result.status) {
    case 'synced':
      return `카메라 동기화 완료 · ${result.camera_count}대`;
    case 'pending':
      return `카메라 동기화가 대기 중입니다 · ${result.camera_count}대`;
    case 'disabled':
      return '카메라 동기화가 비활성화되어 있습니다.';
    case 'failed':
      return result.detail ?? '카메라 동기화에 실패했습니다.';
    default:
      return '카메라 동기화 상태를 확인할 수 없습니다.';
  }
}
