import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { LiveStreamPanel } from '@/features/operations/LiveStreamPanel';
import type { Camera, RuntimeCameraDiagnostics } from '@/shared/api/client';

const onlineCamera: Camera = {
  id: 'cam-1',
  label: '101호',
  rtsp_url_masked: 'rtsp://redacted-camera/a',
  floor_name: '1층',
  status: 'online',
  created_at: null,
};

const offlineCamera: Camera = { ...onlineCamera, id: 'cam-2', label: '102호', status: 'offline' };

function render(
  camera: Camera,
  diagnostics: RuntimeCameraDiagnostics | undefined,
  onRetryConnection = vi.fn(),
  onManageConnection = vi.fn(),
): { host: HTMLDivElement; root: Root; onRetryConnection: typeof onRetryConnection; onManageConnection: typeof onManageConnection } {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(
    <LiveStreamPanel camera={camera} diagnostics={diagnostics} onRetryConnection={onRetryConnection} onManageConnection={onManageConnection} />,
  ));
  return { host, root, onRetryConnection, onManageConnection };
}

afterEach(() => {
  document.body.innerHTML = '';
});

describe('LiveStreamPanel', () => {
  it('shows a gray disconnected panel with both action buttons when the camera is offline', () => {
    const { host } = render(offlineCamera, undefined);

    expect(host.querySelector('.bg-muted')).not.toBeNull();
    expect(host.textContent).toContain('카메라에 연결할 수 없습니다');
    expect(host.textContent).toContain('탐지가 중단된 상태입니다');
    expect(host.querySelector('img')).toBeNull();

    const buttons = Array.from(host.querySelectorAll('button')).map((button) => button.textContent);
    expect(buttons).toContain('재연결 시도');
    expect(buttons).toContain('연결 관리');
  });

  it('calls onRetryConnection when 재연결 시도 is clicked', () => {
    const { host, onRetryConnection } = render(offlineCamera, undefined);

    const retryButton = Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '재연결 시도');
    act(() => retryButton?.click());

    expect(onRetryConnection).toHaveBeenCalledTimes(1);
  });

  it('calls onManageConnection when 연결 관리 is clicked', () => {
    const { host, onManageConnection } = render(offlineCamera, undefined);

    const manageButton = Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '연결 관리');
    act(() => manageButton?.click());

    expect(onManageConnection).toHaveBeenCalledTimes(1);
  });

  it('renders the live stream image and FPS diagnostics for an online camera, with no disconnected panel', () => {
    const diagnostics: RuntimeCameraDiagnostics = {
      camera_id: 'cam-1',
      decode: { requested: 'auto', selected: 'cpu', fallback_count: 0, last_reason: null, updated_at_sec: 1 },
      measured_fps: 12.4,
      latency: null,
    };
    const { host } = render(onlineCamera, diagnostics);

    expect(host.querySelector('img')).not.toBeNull();
    expect(host.textContent).toContain('12.4 FPS');
    expect(host.textContent).not.toContain('카메라에 연결할 수 없습니다');
  });

  it('shows a bottom-left "라이브 · 온라인" badge for an online camera, matching the design handoff', () => {
    const { host } = render(onlineCamera, undefined);

    expect(host.textContent).toContain('라이브 · 온라인');
  });

  it('shows a "연결 끊김" overlay while the stream is reconnecting after an onError, without dropping the mounted img', () => {
    const { host } = render(onlineCamera, undefined);

    const img = host.querySelector('img') as HTMLImageElement;
    expect(img).not.toBeNull();
    act(() => img.dispatchEvent(new Event('error')));

    expect(host.querySelector('[role="status"]')?.textContent).toContain('연결 끊김');
    expect(host.querySelector('img')).not.toBeNull();
  });

  it('clears the "연결 끊김" overlay once the stream loads successfully again', () => {
    const { host } = render(onlineCamera, undefined);

    const img = host.querySelector('img') as HTMLImageElement;
    act(() => img.dispatchEvent(new Event('error')));
    expect(host.textContent).toContain('연결 끊김');

    act(() => img.dispatchEvent(new Event('load')));
    expect(host.textContent).not.toContain('연결 끊김');
  });
});
