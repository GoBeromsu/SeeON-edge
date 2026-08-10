import { describe, expect, it } from 'vitest';
import { getBackendStatus, getCameraStatusMeta, getConnectionStatus } from '@/shared/ui/StatusBadge';
import type { ConnectionView, SystemSnapshot } from '@/shared/api/client';

const connectionBase = {
  events_url: null, config_url: null, facility_code: null, client_installation_ref: null,
  facility_id: null, edge_installation_id: null, enrollment_generation: null,
  facility_token_set: false, facility_token_masked: null, enrolled: false,
  last_ok_at: null, updated_at: null,
};

describe('status badge mapping', () => {
  it.each([
    ['online', '온라인', 'approved'],
    ['offline', '오프라인', 'rejected'],
    ['starting', '시작 중', 'pending'],
    ['unknown', '확인 중', 'closed'],
  ] as const)('maps %s camera status to localized copy and the %s semantic token', (status, label, variant) => {
    const meta = getCameraStatusMeta(status);
    expect(meta.label).toBe(label);
    expect(meta.variant).toBe(variant);
    expect(meta.className).toContain(`bg-status-${variant}Bg`);
    expect(meta.className).toContain(`text-status-${variant}Fg`);
  });

  it.each([
    [null, '백엔드 확인 중'],
    [{ configured: false, reachable: null }, '백엔드 미설정'],
    [{ configured: true, reachable: true }, '백엔드 연결됨'],
    [{ configured: true, reachable: false }, '백엔드 연결 실패'],
    [{ configured: true, reachable: null }, '백엔드 대기 중'],
  ] as const)('maps backend state %# to localized copy', (backend, label) => {
    const system: SystemSnapshot | null = backend === null ? null : {
      version: 'test',
      backend: {
        ...backend,
        last_ok_at: null,
      },
    };

    expect(getBackendStatus(system).label).toBe(label);
  });

  it.each([
    [null, '확인 중'],
    [{ configured: false, reachable: null }, '미설정'],
    [{ configured: true, reachable: true }, '정상'],
    [{ configured: true, reachable: false }, '연결 실패'],
    [{ configured: true, reachable: null }, '확인 중'],
  ] as const)('maps connection state %# to the 서버 연결 카드 wording', (backend, label) => {
    const connection: ConnectionView | null = backend === null ? null : {
      ...connectionBase,
      ...backend,
      last_ok_at: null,
      updated_at: null,
    };

    expect(getConnectionStatus(connection).label).toBe(label);
  });

  it('never exposes retired or raw one-off palette utilities', () => {
    const systemStates: Array<SystemSnapshot | null> = [
      null,
      { version: 'test', backend: { configured: false, reachable: null, last_ok_at: null } },
      { version: 'test', backend: { configured: true, reachable: true, last_ok_at: null } },
    ];
    const connectionStates: Array<ConnectionView | null> = [
      null,
      { ...connectionBase, configured: false, reachable: null },
      { ...connectionBase, configured: true, reachable: true },
    ];
    const classNames = [
      ...(['online', 'offline', 'starting', 'unknown'] as const).map((status) => getCameraStatusMeta(status).className),
      ...systemStates.map((system) => getBackendStatus(system).className),
      ...connectionStates.map((connection) => getConnectionStatus(connection).className),
    ].join(' ');

    expect(classNames).not.toMatch(/(?:bg|text|ring)-(?:emerald|amber|violet|slate|rose|surface2?|ink)(?:-|Bg|Fg)?/);
    expect(classNames).not.toMatch(/\b(?:bg-black|bg-white|text-black|text-white)\b/);
  });
});
