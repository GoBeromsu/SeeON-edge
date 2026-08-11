import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { updateCamera } from '@/shared/api/client';
import {
  confirmTopologyPreview,
  createTopologyFloor,
  fetchTopology,
  fetchTopologyPreview,
  syncTopology,
} from '@/shared/api/topologyClient';
import { TopologyEditor } from '@/features/cameras/TopologyEditor';
import type { CameraRegistry } from '@/shared/api/client';
import type { PollingResource } from '@/shared/api/usePollingResource';

vi.mock('@/shared/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return { ...actual, updateCamera: vi.fn() };
});
vi.mock('@/shared/api/topologyClient', () => ({
  confirmTopologyPreview: vi.fn(),
  createTopologyFloor: vi.fn(),
  createTopologyRoom: vi.fn(),
  fetchTopology: vi.fn(),
  fetchTopologyPreview: vi.fn(),
  syncTopology: vi.fn(),
  updateTopologyFloor: vi.fn(),
  updateTopologyRoom: vi.fn(),
  deleteTopologyFloor: vi.fn(),
  deleteTopologyRoom: vi.fn(),
}));

const topology = {
  registry_version: 7,
  dirty_registry_version: 7,
  readiness_error: 'LEGACY_MAPPING_REQUIRED' as const,
  unmapped_camera_ids: ['local-legacy'],
  floors: [{
    edge_ref: 'floor-1', name: '1층', order_index: 1,
    rooms: [{ edge_ref: 'room-101', name: '101호', room_type: 'ROOM' as const, capacity: 1, legacy_canonical_space_id: null, cameras: [] }],
  }],
};
const preview = {
  confirmation_id: '0197f671-3a31-7a6c-a6e4-83ed412de81b',
  digest: 'a'.repeat(64),
  expires_at: '2099-01-01T00:00:00.000Z',
  snapshot_id: '0197f671-3a31-7a6c-a6e4-83ed412de81c',
  client_revision: 4,
  server_revision: 9,
  cameras: 2,
  rooms: 1,
  floors: 0,
  confirmed: false,
};
const cameras: CameraRegistry = {
  registry_version: 7,
  cameras: [{
    id: 'local-legacy', label: '복도 카메라', rtsp_url_masked: 'RTSP URL 비공개', floor_name: null,
    status: 'offline', created_at: null,
  }],
};

function cameraResource(): PollingResource<CameraRegistry> {
  return { status: 'success', data: cameras, error: null, lastSuccessAt: Date.now(), refreshing: false, retry: vi.fn() };
}

async function renderEditor(): Promise<{ readonly host: HTMLDivElement; readonly root: Root }> {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  await act(async () => { root.render(<TopologyEditor camerasResource={cameraResource()} />); });
  return { host, root };
}

function setValue(host: ParentNode, selector: string, value: string): void {
  const control = host.querySelector(selector);
  if (!(control instanceof HTMLInputElement) && !(control instanceof HTMLSelectElement)) throw new Error(`missing ${selector}`);
  act(() => {
    const prototype = control instanceof HTMLInputElement ? window.HTMLInputElement.prototype : window.HTMLSelectElement.prototype;
    Object.getOwnPropertyDescriptor(prototype, 'value')?.set?.call(control, value);
    control.dispatchEvent(new Event('change', { bubbles: true }));
    control.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

function button(scope: ParentNode, label: string): HTMLButtonElement {
  const match = Array.from(scope.querySelectorAll('button')).find((candidate) => candidate.textContent?.trim() === label);
  if (!match) throw new Error(`missing button ${label}`);
  return match;
}

beforeEach(() => {
  vi.mocked(fetchTopology).mockResolvedValue(topology);
  vi.mocked(fetchTopologyPreview).mockResolvedValue({ preview });
  vi.mocked(createTopologyFloor).mockResolvedValue({ edge_ref: 'floor-2', name: '2층', order_index: 2 });
  vi.mocked(updateCamera).mockResolvedValue({ ...cameras.cameras[0], edge_ref: 'camera-hall', room_edge_ref: 'room-101' });
  vi.mocked(syncTopology).mockResolvedValue({ status: 'pending', error_class: null, detail: null, last_ok_at: null, next_retry_at: null, camera_count: 1 });
  vi.mocked(confirmTopologyPreview).mockResolvedValue({ snapshot_id: preview.snapshot_id, client_revision: 4, server_revision: 10 });
});

afterEach(() => {
  document.body.innerHTML = '';
  vi.clearAllMocks();
});

describe('TopologyEditor', () => {
  it('shows current and dirty registry versions, readiness, revisions, and redacted omission counts', async () => {
    const { host } = await renderEditor();

    expect(host.textContent).toContain('현재 7');
    expect(host.textContent).toContain('동기화 대기 7');
    expect(host.textContent).toContain('매핑 필요');
    expect(host.textContent).toContain('클라이언트 4');
    expect(host.textContent).toContain('서버 9');
    expect(host.textContent).toContain('카메라 2대');
    expect(host.textContent).not.toContain(preview.digest);
    expect(host.textContent).not.toContain(preview.confirmation_id);
  });

  it('creates a floor through the shared adapter and keeps the form explicit', async () => {
    const { host } = await renderEditor();
    setValue(host, 'input[name="floor_edge_ref"]', 'floor-2');
    setValue(host, 'input[name="floor_name"]', '2층');
    setValue(host, 'input[name="floor_order"]', '2');

    await act(async () => button(host, '층 추가').click());

    expect(createTopologyFloor).toHaveBeenCalledWith({ edge_ref: 'floor-2', name: '2층', order_index: 2 });
  });

  it('pairs an existing camera only with explicit camera and room refs', async () => {
    const resource = cameraResource();
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    await act(async () => { root.render(<TopologyEditor camerasResource={resource} />); });
    setValue(host, 'input[name="camera_ref_local-legacy"]', 'camera-hall');
    setValue(host, 'select[name="camera_room_local-legacy"]', 'room-101');

    await act(async () => button(host, '매핑 저장').click());

    expect(updateCamera).toHaveBeenCalledWith('local-legacy', { edge_ref: 'camera-hall', room_edge_ref: 'room-101' });
    expect(resource.retry).toHaveBeenCalled();
  });

  it('rechecks the exact preview before one explicit confirmation and blocks double submit', async () => {
    let finish: (() => void) | undefined;
    vi.mocked(confirmTopologyPreview).mockReturnValue(new Promise((resolve) => {
      finish = () => resolve({ snapshot_id: preview.snapshot_id, client_revision: 4, server_revision: 10 });
    }));
    const { host } = await renderEditor();
    act(() => button(host, '변경 제외 확인').click());
    const dialog = document.querySelector('[role="dialog"]');
    if (!dialog) throw new Error('missing dialog');

    const confirm = button(dialog, '제외 확정');
    await act(async () => { confirm.click(); confirm.click(); await Promise.resolve(); await Promise.resolve(); });

    expect(fetchTopologyPreview).toHaveBeenCalledTimes(2);
    expect(confirmTopologyPreview).toHaveBeenCalledTimes(1);
    expect(confirmTopologyPreview).toHaveBeenCalledWith({
      confirmation_id: preview.confirmation_id,
      digest: preview.digest,
      client_revision: 4,
      server_revision: 9,
    });
    await act(async () => finish?.());
  });

  it('refuses confirmation when the durable preview changed after the dialog opened', async () => {
    const changed = { ...preview, server_revision: 10 };
    vi.mocked(fetchTopologyPreview).mockResolvedValueOnce({ preview }).mockResolvedValueOnce({ preview: changed });
    await renderEditor();
    act(() => button(document, '변경 제외 확인').click());

    await act(async () => button(document, '제외 확정').click());

    expect(confirmTopologyPreview).not.toHaveBeenCalled();
    expect(document.querySelector('[role="alert"]')?.textContent).toContain('변경되었습니다');
  });

  it('renders a retryable nonsecret conflict state from manual sync', async () => {
    vi.mocked(fetchTopology).mockResolvedValue({ ...topology, readiness_error: null, unmapped_camera_ids: [] });
    vi.mocked(syncTopology).mockResolvedValue({ status: 'failed', error_class: 'conflict', detail: null, last_ok_at: null, next_retry_at: null, camera_count: 1 });
    const { host } = await renderEditor();

    await act(async () => button(host, '지금 동기화').click());

    expect(host.querySelector('[role="alert"]')?.textContent).toContain('충돌');
  });
});
