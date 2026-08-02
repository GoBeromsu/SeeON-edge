import { describe, expect, it } from 'vitest';
import { getBackendStatus, getCameraStatusMeta, getCameraSyncMeta, getConnectionStatus } from '@/shared/ui/StatusBadge';
import type { CameraSyncStatus, ConnectionView, SystemSnapshot } from '@/shared/api/client';

describe('status badge mapping', () => {
  it.each([
    ['online', '온라인', 'bg-status-stableBg text-status-stable ring-status-stable'],
    ['offline', '오프라인', 'bg-status-dangerBg text-status-danger ring-status-danger'],
    ['starting', '시작 중', 'bg-surface2 text-ink-soft ring-border'],
    ['unknown', '확인 중', 'bg-surface2 text-ink-soft ring-border'],
  ] as const)('maps %s camera status to localized copy and semantic tokens', (status, label, className) => {
    expect(getCameraStatusMeta(status)).toEqual({ label, className });
  });

  it.each([
    [null, '백엔드 확인 중', 'bg-surface2 text-ink-soft ring-border'],
    [{ configured: false, reachable: null }, '백엔드 미설정', 'bg-status-cautionBg text-status-caution ring-status-caution'],
    [{ configured: true, reachable: true }, '백엔드 연결됨', 'bg-status-stableBg text-status-stable ring-status-stable'],
    [{ configured: true, reachable: false }, '백엔드 연결 실패', 'bg-status-dangerBg text-status-danger ring-status-danger'],
    [{ configured: true, reachable: null }, '백엔드 대기 중', 'bg-surface2 text-ink-soft ring-border'],
  ] as const)('maps backend state %# to localized copy and semantic tokens', (backend, label, className) => {
    const system: SystemSnapshot | null = backend === null ? null : {
      version: 'test',
      backend: {
        ...backend,
        last_ok_at: null,
      },
    };

    expect(getBackendStatus(system)).toEqual({ label, className });
  });

  it.each([
    [null, '백엔드 확인 중', 'bg-surface2 text-ink-soft ring-border'],
    [{ configured: false, reachable: null }, '백엔드 미설정', 'bg-status-cautionBg text-status-caution ring-status-caution'],
    [{ configured: true, reachable: true }, '백엔드 연결됨', 'bg-status-stableBg text-status-stable ring-status-stable'],
    [{ configured: true, reachable: false }, '백엔드 연결 실패', 'bg-status-dangerBg text-status-danger ring-status-danger'],
    [{ configured: true, reachable: null }, '백엔드 대기 중', 'bg-surface2 text-ink-soft ring-border'],
  ] as const)('maps connection state %# to localized copy and semantic tokens', (backend, label, className) => {
    const connection: ConnectionView | null = backend === null ? null : {
      events_url: null,
      config_url: null,
      facility_id: null,
      facility_token_set: false,
      facility_token_masked: null,
      ...backend,
      last_ok_at: null,
      updated_at: null,
    };

    expect(getConnectionStatus(connection)).toEqual({ label, className });
  });

  it.each([
    ['synced', '동기화됨', 'bg-status-stableBg text-status-stable ring-status-stable'],
    ['pending', '동기화 대기', 'bg-status-cautionBg text-status-caution ring-status-caution'],
    ['failed', '동기화 실패', 'bg-status-dangerBg text-status-danger ring-status-danger'],
    ['disabled', '동기화 비활성', 'bg-surface2 text-ink-soft ring-border'],
  ] as const)('maps camera sync status %s to localized copy and semantic tokens', (status, label, className) => {
    expect(getCameraSyncMeta(status)).toEqual({ label, className });
  });

  it('never exposes one-off raw palette utilities', () => {
    const systemStates: Array<SystemSnapshot | null> = [
      null,
      { version: 'test', backend: { configured: false, reachable: null, last_ok_at: null } },
      { version: 'test', backend: { configured: true, reachable: true, last_ok_at: null } },
      { version: 'test', backend: { configured: true, reachable: false, last_ok_at: null } },
      { version: 'test', backend: { configured: true, reachable: null, last_ok_at: null } },
    ];
    const connectionStates: Array<ConnectionView | null> = [
      null,
      { events_url: null, config_url: null, facility_id: null, facility_token_set: false, facility_token_masked: null, configured: false, reachable: null, last_ok_at: null, updated_at: null },
      { events_url: null, config_url: null, facility_id: null, facility_token_set: false, facility_token_masked: null, configured: true, reachable: true, last_ok_at: null, updated_at: null },
      { events_url: null, config_url: null, facility_id: null, facility_token_set: false, facility_token_masked: null, configured: true, reachable: false, last_ok_at: null, updated_at: null },
      { events_url: null, config_url: null, facility_id: null, facility_token_set: false, facility_token_masked: null, configured: true, reachable: null, last_ok_at: null, updated_at: null },
    ];
    const syncStatuses: CameraSyncStatus[] = ['synced', 'pending', 'failed', 'disabled'];
    const classNames = [
      ...(['online', 'offline', 'starting', 'unknown'] as const).map((status) => getCameraStatusMeta(status).className),
      ...systemStates.map((system) => getBackendStatus(system).className),
      ...connectionStates.map((connection) => getConnectionStatus(connection).className),
      ...syncStatuses.map((status) => getCameraSyncMeta(status).className),
    ];

    expect(classNames.join(' ')).not.toMatch(/(?:bg|text|ring)-(?:emerald|amber|violet|slate|rose)-/);
  });
});
