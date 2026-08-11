import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { updateCamera } from '@/shared/api/client';
import { createTopologyFloor, fetchTopology, syncTopology } from '@/shared/api/topologyClient';
import type { CameraTopology } from '@/shared/api/topologyClient';
import type { CameraRegistry } from '@/shared/api/client';
import type { PollingResource } from '@/shared/api/usePollingResource';
import { CameraSyncStep } from '@/features/connection/CameraSyncStep';

vi.mock('@/shared/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return { ...actual, updateCamera: vi.fn() };
});
vi.mock('@/shared/api/topologyClient', async () => {
  const actual = await vi.importActual<typeof import('@/shared/api/topologyClient')>('@/shared/api/topologyClient');
  return { ...actual, createTopologyFloor: vi.fn(), fetchTopology: vi.fn(), syncTopology: vi.fn() };
});

const mappedTopology: CameraTopology = {
  registry_version: 7,
  dirty_registry_version: null,
  readiness_error: null,
  unmapped_camera_ids: [],
  floors: [{
    edge_ref: 'floor-1', name: '1층', order_index: 1,
    rooms: [{ edge_ref: 'room-101', name: '101호', room_type: 'ROOM', capacity: 1, legacy_canonical_space_id: null, cameras: [{ edge_ref: 'camera-hall', label: '복도 카메라' }] }],
  }],
};

const unmappedTopology: CameraTopology = {
  registry_version: 7,
  dirty_registry_version: 7,
  readiness_error: 'LEGACY_MAPPING_REQUIRED',
  unmapped_camera_ids: ['local-legacy'],
  floors: [{
    edge_ref: 'floor-1', name: '1층', order_index: 1,
    rooms: [{ edge_ref: 'room-101', name: '101호', room_type: 'ROOM', capacity: 1, legacy_canonical_space_id: null, cameras: [] }],
  }],
};

function cameras(list: CameraRegistry['cameras']): CameraRegistry {
  return { registry_version: 7, cameras: list };
}

function cameraResource(data: CameraRegistry): PollingResource<CameraRegistry> {
  return { status: 'success', data, error: null, lastSuccessAt: Date.now(), refreshing: false, retry: vi.fn() };
}

async function renderStep(topology: CameraTopology, camerasData: CameraRegistry, onChanged = () => Promise.resolve()): Promise<{ readonly host: HTMLDivElement; readonly root: Root }> {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  await act(async () => { root.render(<CameraSyncStep camerasResource={cameraResource(camerasData)} topology={topology} onChanged={onChanged} />); });
  return { host, root };
}

function button(scope: ParentNode, label: string): HTMLButtonElement {
  const match = Array.from(scope.querySelectorAll('button')).find((candidate) => candidate.textContent?.trim() === label);
  if (!match) throw new Error(`missing button ${label}`);
  return match;
}

function setValue(host: ParentNode, selector: string, value: string): void {
  const control = host.querySelector(selector);
  if (!(control instanceof HTMLInputElement)) throw new Error(`missing ${selector}`);
  act(() => {
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set?.call(control, value);
    control.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

beforeEach(() => {
  vi.mocked(fetchTopology).mockResolvedValue(mappedTopology);
  vi.mocked(createTopologyFloor).mockResolvedValue({ edge_ref: 'floor-2', name: '2층', order_index: 2 });
  vi.mocked(updateCamera).mockResolvedValue({ id: 'local-legacy', label: '복도 카메라', rtsp_url_masked: 'RTSP URL 비공개', floor_name: null, status: 'offline', created_at: null, edge_ref: 'camera-hall', room_edge_ref: 'room-101' });
  vi.mocked(syncTopology).mockResolvedValue({ status: 'synced', error_class: null, detail: null, last_ok_at: null, next_retry_at: null, camera_count: 1 });
});

afterEach(() => {
  document.body.innerHTML = '';
  vi.clearAllMocks();
});

describe('CameraSyncStep', () => {
  it('shows zero registered cameras as a real failure state, not a silent success', async () => {
    const { host } = await renderStep(mappedTopology, cameras([]));

    expect(host.querySelector('[role="alert"]')?.textContent).toContain('등록된 카메라가 없습니다');
    expect(button(host, '카메라 동기화').disabled).toBe(true);
  });

  it('disables sync while any camera is unmapped and surfaces the pending count', async () => {
    const { host } = await renderStep(unmappedTopology, cameras([{ id: 'local-legacy', label: '복도 카메라', rtsp_url_masked: 'x', floor_name: null, status: 'offline', created_at: null }]));

    expect(host.textContent).toContain('배치 필요 카메라 1대');
    expect(button(host, '카메라 동기화').disabled).toBe(true);
  });

  it('creates a floor through the shared adapter and reloads via onChanged', async () => {
    const onChanged = vi.fn().mockResolvedValue(undefined);
    const { host } = await renderStep(mappedTopology, cameras([{ id: 'cam-1', label: 'cam', rtsp_url_masked: 'x', floor_name: null, status: 'online', created_at: null }]), onChanged);
    setValue(host, 'input[name="floor_edge_ref"]', 'floor-2');
    setValue(host, 'input[name="floor_name"]', '2층');
    setValue(host, 'input[name="floor_order"]', '2');

    await act(async () => button(host, '층 추가').click());

    expect(createTopologyFloor).toHaveBeenCalledWith({ edge_ref: 'floor-2', name: '2층', order_index: 2 });
    expect(onChanged).toHaveBeenCalled();
  });

  it('pairs an unmapped camera through the shared adapter', async () => {
    const onChanged = vi.fn().mockResolvedValue(undefined);
    const { host } = await renderStep(unmappedTopology, cameras([{ id: 'local-legacy', label: '복도 카메라', rtsp_url_masked: 'x', floor_name: null, status: 'offline', created_at: null }]), onChanged);
    setValue(host, 'input[name="camera_ref_local-legacy"]', 'camera-hall');
    const select = host.querySelector('select[name="camera_room_local-legacy"]');
    if (!(select instanceof HTMLSelectElement)) throw new Error('missing select');
    act(() => {
      Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, 'value')?.set?.call(select, 'room-101');
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });

    await act(async () => button(host, '매핑 저장').click());

    expect(updateCamera).toHaveBeenCalledWith('local-legacy', { edge_ref: 'camera-hall', room_edge_ref: 'room-101' });
    expect(onChanged).toHaveBeenCalled();
  });

  it('reports a successful manual sync distinctly from a failed one', async () => {
    const { host } = await renderStep(mappedTopology, cameras([{ id: 'cam-1', label: 'cam', rtsp_url_masked: 'x', floor_name: null, status: 'online', created_at: null }]));

    await act(async () => button(host, '카메라 동기화').click());

    expect(host.querySelector('[role="status"]')?.textContent).toContain('동기화했습니다');
  });

  it.each([
    ['auth', '인증'],
    ['conflict', '서버 상태'],
    ['timeout', '네트워크'],
    ['unreachable', '네트워크'],
    ['unconfigured', '1단계'],
  ] as const)('classifies a %s sync failure distinctly', async (errorClass, expectedFragment) => {
    vi.mocked(syncTopology).mockResolvedValue({ status: 'failed', error_class: errorClass, detail: null, last_ok_at: null, next_retry_at: null, camera_count: 1 });
    const { host } = await renderStep(mappedTopology, cameras([{ id: 'cam-1', label: 'cam', rtsp_url_masked: 'x', floor_name: null, status: 'online', created_at: null }]));

    await act(async () => button(host, '카메라 동기화').click());

    expect(host.querySelector('[role="alert"]')?.textContent).toContain(expectedFragment);
  });

  it('reports a request-level failure (network never reached the server) distinctly from a classified backend failure', async () => {
    vi.mocked(syncTopology).mockRejectedValue(new Error('network down'));
    const { host } = await renderStep(mappedTopology, cameras([{ id: 'cam-1', label: 'cam', rtsp_url_masked: 'x', floor_name: null, status: 'online', created_at: null }]));

    await act(async () => button(host, '카메라 동기화').click());

    expect(host.querySelector('[role="alert"]')?.textContent).toContain('동기화 요청에 실패했습니다');
  });
});
