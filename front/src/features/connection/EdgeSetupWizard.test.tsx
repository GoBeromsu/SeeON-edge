import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fetchTopology, fetchTopologyPreview, type CameraTopology, type TopologyPreview } from '@/shared/api/topologyClient';
import type { CameraRegistry, ConnectionView } from '@/shared/api/client';
import type { PollingResource } from '@/shared/api/usePollingResource';
import { EdgeSetupWizard } from '@/features/connection/EdgeSetupWizard';

vi.mock('@/shared/api/topologyClient', async () => {
  const actual = await vi.importActual<typeof import('@/shared/api/topologyClient')>('@/shared/api/topologyClient');
  return {
    ...actual,
    fetchTopology: vi.fn(),
    fetchTopologyPreview: vi.fn(),
    syncTopology: vi.fn(),
    confirmTopologyPreview: vi.fn(),
  };
});

const notEnrolledConnection: ConnectionView = {
  events_url: null,
  config_url: null,
  facility_code: null,
  client_installation_ref: null,
  facility_id: null,
  edge_installation_id: null,
  enrollment_generation: null,
  facility_token_set: false,
  facility_token_masked: null,
  enrolled: false,
  configured: false,
  reachable: null,
  last_ok_at: null,
  updated_at: null,
};

const enrolledConnection: ConnectionView = {
  ...notEnrolledConnection,
  events_url: 'https://api.eldercare.example/api/v1/events',
  facility_code: 'NH-7H2K9M4QXP',
  client_installation_ref: 'aa83ea3f-6e5f-4f45-a401-fb36c38835b6',
  facility_id: 'facility-42',
  edge_installation_id: 'c72bd9a7-3e04-47ba-a8cd-a56e54f98152',
  enrollment_generation: 1,
  facility_token_set: true,
  facility_token_masked: '****ab12',
  enrolled: true,
  configured: true,
  reachable: true,
  last_ok_at: '2026-08-01T00:00:00Z',
  updated_at: '2026-08-01T00:00:00Z',
};

const readyTopology: CameraTopology = {
  registry_version: 3,
  dirty_registry_version: null,
  readiness_error: null,
  unmapped_camera_ids: [],
  floors: [{
    edge_ref: 'floor-1', name: '1층', order_index: 1,
    rooms: [{ edge_ref: 'room-1', name: '101호', room_type: 'ROOM', capacity: 1, legacy_canonical_space_id: null, cameras: [{ edge_ref: 'cam-0', label: 'cam-0' }] }],
  }],
};

const emptyTopology: CameraTopology = { registry_version: 1, dirty_registry_version: null, readiness_error: null, unmapped_camera_ids: [], floors: [] };

const pendingPreview: TopologyPreview = {
  confirmation_id: '0197f671-3a31-7a6c-a6e4-83ed412de81b',
  digest: 'a'.repeat(64),
  expires_at: '2099-01-01T00:00:00.000Z',
  snapshot_id: '0197f671-3a31-7a6c-a6e4-83ed412de81c',
  client_revision: 1,
  server_revision: 2,
  cameras: 1,
  rooms: 0,
  floors: 0,
  confirmed: false,
};

function connectionResource(data: ConnectionView): PollingResource<ConnectionView> {
  return { status: 'success', data, error: null, lastSuccessAt: Date.now(), refreshing: false, retry: vi.fn() };
}

function camerasResource(data: CameraRegistry): PollingResource<CameraRegistry> {
  return { status: 'success', data, error: null, lastSuccessAt: Date.now(), refreshing: false, retry: vi.fn() };
}

async function flush(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  });
}

async function renderWizard(connection: ConnectionView, cameras: CameraRegistry): Promise<{ readonly host: HTMLDivElement; readonly root: Root }> {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => { root.render(<EdgeSetupWizard connectionResource={connectionResource(connection)} camerasResource={camerasResource(cameras)} />); });
  await flush();
  return { host, root };
}

function step(host: ParentNode, index: 1 | 2 | 3): Element {
  const el = host.querySelector(`[aria-labelledby="wizard-step-${index}-title"]`);
  if (!el) throw new Error(`missing step ${index}`);
  return el;
}

/** The step header's "완료" completion badge is a span whose *entire* text is exactly "완료" --
 * matched exactly (not as a substring) so unrelated copy like "매핑이 완료되었습니다." can't false-positive. */
function hasCompleteBadge(container: Element): boolean {
  return Array.from(container.querySelectorAll('span')).some((span) => span.textContent === '완료');
}

beforeEach(() => {
  vi.mocked(fetchTopology).mockResolvedValue(readyTopology);
  vi.mocked(fetchTopologyPreview).mockResolvedValue({ preview: null });
});

afterEach(() => {
  document.body.innerHTML = '';
  vi.clearAllMocks();
});

describe('EdgeSetupWizard', () => {
  it('locks steps 2 and 3 with a stated reason until device enrollment (step 1) has actually succeeded', async () => {
    const { host } = await renderWizard(notEnrolledConnection, { registry_version: 1, cameras: [] });

    expect(step(host, 2).textContent).toContain('1단계 장치 연결을 먼저 완료하세요.');
    expect(step(host, 2).querySelector('button')).toBeNull();
    expect(step(host, 3).textContent).toContain('1단계 장치 연결을 먼저 완료하세요.');
    // Step 1 itself is never locked -- its body (ConnectionSettingsPanel) renders normally.
    expect(step(host, 1).textContent).toContain('현장에서는 시설 코드와 토큰만 입력합니다.');
  });

  it('unlocks step 2 once step 1 is enrolled, and treats zero registered cameras as a real failure state, not success', async () => {
    const { host } = await renderWizard(enrolledConnection, { registry_version: 1, cameras: [] });

    expect(step(host, 2).textContent).not.toContain('1단계 장치 연결을 먼저 완료하세요.');
    expect(step(host, 2).querySelector('[role="alert"]')?.textContent).toContain('등록된 카메라가 없습니다');
    expect(hasCompleteBadge(step(host, 2))).toBe(false);
    // Step 3 stays locked because step 2 (zero cameras) has not completed.
    expect(step(host, 3).textContent).toContain('2단계 카메라 확인을 먼저 완료하세요.');
  });

  it('derives step completion purely from server state on reload: an already-enrolled, already-synced device resumes unlocked on step 3', async () => {
    const { host } = await renderWizard(enrolledConnection, { registry_version: 1, cameras: [{ id: 'cam-0', label: 'cam-0', rtsp_url_masked: 'x', floor_name: null, status: 'online', created_at: null }] });

    expect(step(host, 3).textContent).not.toContain('2단계 카메라 확인을 먼저 완료하세요.');
    expect(step(host, 3).getAttribute('aria-current')).toBe('step');
    expect(step(host, 1).getAttribute('aria-current')).toBeNull();
  });

  it('never auto-advances through an unconfirmed pending preview -- step 3 stays incomplete and requires the explicit confirm action', async () => {
    vi.mocked(fetchTopologyPreview).mockResolvedValue({ preview: pendingPreview });
    const { host } = await renderWizard(enrolledConnection, { registry_version: 1, cameras: [{ id: 'cam-0', label: 'cam-0', rtsp_url_masked: 'x', floor_name: null, status: 'online', created_at: null }] });

    expect(step(host, 3).textContent).toContain('반영 전 확인이 필요합니다');
    expect(hasCompleteBadge(step(host, 3))).toBe(false);
  });
});
