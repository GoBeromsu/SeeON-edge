import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { CameraInfoCard } from '@/features/operations/CameraInfoCard';
import type { Camera } from '@/shared/api/client';

const camera: Camera = {
  id: 'cam-1',
  label: '101호',
  rtsp_url_masked: 'rtsp://redacted-camera/a',
  floor_name: '1층',
  status: 'online',
  created_at: null,
};

function render(target: Camera, onManageConnection = vi.fn()): { host: HTMLDivElement; root: Root; onManageConnection: typeof onManageConnection } {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<CameraInfoCard camera={target} onManageConnection={onManageConnection} />));
  return { host, root, onManageConnection };
}

afterEach(() => {
  document.body.innerHTML = '';
});

describe('CameraInfoCard', () => {
  it('uses the camera name as the card title instead of "카메라 정보"', () => {
    const { host } = render(camera);

    expect(host.querySelector('h2')?.textContent).toBe('101호');
    expect(host.textContent).not.toContain('카메라 정보');
  });

  it('shows floor and masked RTSP rows, but no 상태 row', () => {
    const { host } = render(camera);

    expect(host.textContent).toContain('층');
    expect(host.textContent).toContain('1층');
    expect(host.textContent).toContain('RTSP 주소');
    expect(host.textContent).toContain('rtsp://redacted-camera/a');
    expect(host.textContent).not.toContain('상태');
    expect(host.textContent).not.toContain('온라인');
  });

  it('falls back to 미지정 when the floor is unset', () => {
    const { host } = render({ ...camera, floor_name: null });
    expect(host.textContent).toContain('미지정');
  });

  it('calls onManageConnection when the header 연결 관리 button is clicked, instead of navigating to settings', () => {
    const { host, onManageConnection } = render(camera);

    const button = Array.from(host.querySelectorAll('button')).find((el) => el.textContent === '연결 관리');
    expect(button).toBeDefined();
    act(() => button?.click());

    expect(onManageConnection).toHaveBeenCalledTimes(1);
  });
});
