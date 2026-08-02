import { act } from 'react';
import { createRoot } from 'react-dom/client';
import type { ComponentProps } from 'react';
import { describe, expect, it, vi } from 'vitest';
import { CameraManagementPanel } from '@/features/camera-management/CameraManagementPanel';
import type { Camera } from '@/shared/api/client';

const camera: Camera = { id: 'cam-1', label: '301호 A', rtsp_url_masked: 'rtsp://***', space_id: '301', space_name: '301호', floor_name: '3층', backend_camera_id: null, status: 'online', created_at: null };

const baseProps: ComponentProps<typeof CameraManagementPanel> = {
  registry: { registry_version: 0, cameras: [] },
  cameraError: '카메라 목록을 불러오지 못했습니다.',
  lastSuccessAt: null,
  refreshing: false,
  onAddCamera: vi.fn(),
  onUpdateStarted: vi.fn(() => 1),
  onUpdated: vi.fn(),
  onDelete: vi.fn(),
  hasLoadedSuccessfully: false,
};

function renderPanel(overrides: Partial<ComponentProps<typeof CameraManagementPanel>> = {}): { host: HTMLDivElement; root: ReturnType<typeof createRoot> } {
  const host = document.createElement('div');
  const root = createRoot(host);
  act(() => root.render(<CameraManagementPanel {...baseProps} {...overrides} />));
  return { host, root };
}

describe('CameraManagementPanel freshness', () => {
  it('does not fabricate a last checked time on first-load failure', () => {
    const { host, root } = renderPanel();
    // Registry version is diagnostic info for an engineer audience — it must not hide behind a click.
    expect(host.querySelector('details')).toBeNull();
    expect(host.querySelector('summary')).toBeNull();
    expect(host.textContent).toContain('카메라 목록을 불러오지 못했습니다.');
    expect(host.textContent).not.toContain('마지막 확인');
    act(() => root.unmount());
  });

  it('shows the actual last successful poll time with a degraded error', () => {
    const lastSuccessAt = Date.parse('2026-07-21T03:04:05Z');
    const { host, root } = renderPanel({ lastSuccessAt });
    expect(host.textContent).toContain(`마지막 확인 ${new Date(lastSuccessAt).toLocaleString('ko-KR')}`);
    act(() => root.unmount());
  });

  it('keeps a successful empty state visible during one polite background refresh', () => {
    const { host, root } = renderPanel({
      cameraError: null,
      hasLoadedSuccessfully: true,
      refreshing: true,
    });
    const indicators = host.querySelectorAll('[role="status"]');
    expect(indicators).toHaveLength(1);
    expect(indicators[0]?.getAttribute('aria-live')).toBe('polite');
    expect(indicators[0]?.textContent).toContain('카메라 목록을 새로 확인하고 있습니다.');
    expect(host.textContent).toContain('등록된 카메라가 없습니다.');
    expect(host.textContent).not.toContain('카메라 목록을 불러오는 중');
    act(() => root.unmount());
  });

  it('keeps camera cards visible during one polite background refresh', () => {
    const { host, root } = renderPanel({
      registry: { registry_version: 1, cameras: [camera] },
      cameraError: null,
      hasLoadedSuccessfully: true,
      refreshing: true,
    });
    expect(host.querySelectorAll('[role="status"]')).toHaveLength(1);
    expect(host.textContent).toContain('카메라 목록을 새로 확인하고 있습니다.');
    expect(host.textContent).toContain('301호 A');
    act(() => root.unmount());
  });
});

describe('CameraManagementPanel roster sync', () => {
  it('omits the sync button when onSyncCameras is not provided', () => {
    const { host, root } = renderPanel();
    expect(Array.from(host.querySelectorAll('button')).some((button) => button.textContent === '카메라 동기화')).toBe(false);
    act(() => root.unmount());
  });

  it('renders a busy sync button and its result message', () => {
    const onSyncCameras = vi.fn();
    const { host, root } = renderPanel({
      cameraError: null,
      hasLoadedSuccessfully: true,
      onSyncCameras,
      syncBusy: true,
      syncMessage: '카메라 동기화 완료 · 2대',
    });
    const syncButton = Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find((button) => button.textContent === '동기화 중...');
    expect(syncButton?.disabled).toBe(true);
    act(() => syncButton?.click());
    expect(onSyncCameras).not.toHaveBeenCalled();
    expect(host.textContent).toContain('카메라 동기화 완료 · 2대');
    act(() => root.unmount());
  });
});

describe('CameraManagementPanel sort order', () => {
  it('sorts cards for fault triage: offline, then never-connected/unknown, then online', () => {
    const online = { ...camera, id: 'cam-online', label: '온라인 카메라', status: 'online' as const };
    const offline = { ...camera, id: 'cam-offline', label: '오프라인 카메라', status: 'offline' as const };
    const neverConnected = { ...camera, id: 'cam-never', label: '미연결 카메라', status: 'unknown' as const, never_connected: true };
    const { host, root } = renderPanel({
      registry: { registry_version: 1, cameras: [online, offline, neverConnected] },
      cameraError: null,
      hasLoadedSuccessfully: true,
    });

    const labels = Array.from(host.querySelectorAll('h3')).map((heading) => heading.textContent);
    expect(labels).toEqual(['오프라인 카메라', '미연결 카메라', '온라인 카메라']);
    act(() => root.unmount());
  });
});
