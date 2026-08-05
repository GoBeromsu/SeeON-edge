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

describe('I10 — 클라우드 연동 상태 표시', () => {
  it('매핑이 끝났으면 연동 완료로 보여준다', () => {
    const { host } = render({
      ...camera,
      mapping_pending: false,
      backend_camera_id: 'cam_sp_205',
    });

    expect(host.querySelector('[data-testid="cloud-mapping"]')?.textContent).toContain('연동 완료');
  });

  it('매핑이 진행 중이면 무엇을 해야 하는지 알려준다', () => {
    const { host } = render({ ...camera, mapping_pending: true, backend_camera_id: null });

    const text = host.querySelector('[data-testid="cloud-mapping"]')?.textContent ?? '';
    expect(text).toContain('연동 대기');
    expect(text).toContain('방을 지정하면 전송됩니다.');
  });

  it('backend_camera_id가 없으면 pending이 false여도 완료로 표시하지 않는다', () => {
    // 서버가 pending=false만 보내고 id를 못 준 상태를 "완료"로 읽으면
    // 기사님이 안 붙은 카메라를 두고 현장을 떠난다.
    const { host } = render({ ...camera, mapping_pending: false, backend_camera_id: null });

    expect(host.querySelector('[data-testid="cloud-mapping"]')?.textContent).toContain('연동 대기');
  });

  it('필드가 아예 없으면 보수적으로 연동 대기로 본다', () => {
    const { host } = render(camera);

    expect(host.querySelector('[data-testid="cloud-mapping"]')?.textContent).toContain('연동 대기');
  });
});

describe('연결 이력 — 미연결과 단절을 구분한다', () => {
  function historyText(host: HTMLElement): string {
    return host.querySelector('[data-testid="connection-history"]')?.textContent ?? '';
  }

  it('한 번도 연결된 적 없으면 무엇을 확인할지 알려준다', () => {
    const { host } = render({ ...camera, never_connected: true, last_ok_at: null });

    expect(historyText(host)).toContain('한 번도 연결된 적 없음');
    expect(historyText(host)).toContain('주소와 계정');
  });

  it('연결된 적이 있으면 마지막 연결 시각을 보여준다', () => {
    const { host } = render({
      ...camera,
      never_connected: false,
      last_ok_at: '2026-08-03T01:00:00Z',
    });

    expect(historyText(host)).toContain('마지막 연결');
    expect(historyText(host)).not.toContain('한 번도');
  });

  it('시각이 깨져 있으면 지어내지 않는다', () => {
    const { host } = render({ ...camera, never_connected: false, last_ok_at: 'not-a-date' });

    expect(historyText(host)).toContain('시각 불명');
  });

  it('정보가 없으면 연결 기록 없음으로 남긴다', () => {
    const { host } = render({ ...camera, never_connected: false, last_ok_at: null });

    expect(historyText(host)).toContain('연결 기록 없음');
  });
});
