import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { CameraInfoCard } from '@/features/operations/CameraInfoCard';
import type { Camera, RuntimeDetectionDiagnostics } from '@/shared/api/client';

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

function detection(overrides: Partial<RuntimeDetectionDiagnostics> = {}): RuntimeDetectionDiagnostics {
  return {
    state: 'healthy',
    reason: null,
    recent_success_rate: 1,
    last_completed_at_sec: 12,
    evaluation_window_sec: 120,
    timeout_sec: 120,
    ...overrides,
  };
}

/** 동일한 root를 유지하며 다음 스냅샷을 그려야 복구 동작을 검증할 수 있다. */
function renderWithDetection(
  target: Camera,
  diagnostics: RuntimeDetectionDiagnostics | undefined,
): { host: HTMLDivElement; rerender: (next: RuntimeDetectionDiagnostics | undefined) => void } {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  const paint = (next: RuntimeDetectionDiagnostics | undefined): void => {
    act(() => root.render(<CameraInfoCard camera={target} detection={next} onManageConnection={vi.fn()} />));
  };
  paint(diagnostics);
  return { host, rerender: paint };
}

function detectionCell(host: HTMLElement): HTMLElement | null {
  return host.querySelector('[data-testid="detection-status"]');
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

  it('shows floor and masked RTSP rows, but no 연결 상태 row', () => {
    // 연결 상태는 breadcrumb 뱃지가 소유한다(DESIGN.md 관제 §CameraInfoCard).
    // 감지 상태는 그와 별개의 사실이라 이 카드에 남는다.
    const { host } = render(camera);

    const labels = Array.from(host.querySelectorAll('dt')).map((el) => el.textContent);
    expect(host.textContent).toContain('층');
    expect(host.textContent).toContain('1층');
    expect(host.textContent).toContain('RTSP 주소');
    expect(host.textContent).toContain('rtsp://redacted-camera/a');
    expect(labels).not.toContain('상태');
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

describe('기준선 — 연결과 매핑은 서로 다른 줄에서 각자 표시된다', () => {
  // Todo 6이 감지 상태를 추가하기 전의 현재 동작을 고정한다. 감지 표시가 붙더라도
  // 연결 이력과 클라우드 연동은 서로를 가리거나 대체하지 않아야 한다.
  it('연결됨 + 매핑 대기 카메라는 두 상태를 동시에 각각 보여준다', () => {
    const { host } = render({
      ...camera,
      status: 'online',
      never_connected: false,
      last_ok_at: '2026-08-03T01:00:00Z',
      mapping_pending: true,
      backend_camera_id: null,
    });

    const mapping = host.querySelector('[data-testid="cloud-mapping"]');
    const history = host.querySelector('[data-testid="connection-history"]');
    expect(mapping).not.toBeNull();
    expect(history).not.toBeNull();
    expect(mapping?.contains(history as Node)).toBe(false);
    expect(history?.contains(mapping as Node)).toBe(false);
    expect(mapping?.textContent).toContain('연동 대기');
    expect(history?.textContent).toContain('마지막 연결');
  });

  it('매핑 완료 + 미연결 카메라도 두 상태를 각각 유지한다', () => {
    const { host } = render({
      ...camera,
      status: 'offline',
      never_connected: true,
      last_ok_at: null,
      mapping_pending: false,
      backend_camera_id: 'cam_sp_205',
    });

    expect(host.querySelector('[data-testid="cloud-mapping"]')?.textContent).toContain('연동 완료');
    expect(host.querySelector('[data-testid="connection-history"]')?.textContent).toContain('한 번도 연결된 적 없음');
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

describe('감지 상태 — 연결/매핑과 독립적으로 항상 보인다', () => {
  it('감지가 정상이면 이벤트가 하나도 없어도 차분한 정상 상태로 남는다', () => {
    const { host } = renderWithDetection(camera, detection({ state: 'healthy', last_completed_at_sec: 0 }));

    const cell = detectionCell(host);
    expect(cell?.dataset.detectionState).toBe('healthy');
    expect(cell?.getAttribute('role')).toBe('status');
    expect(host.querySelector('[role="alert"]')).toBeNull();
  });

  it('연결된 카메라라도 감지가 멀었으면 alert 역할로 경보를 띄운다', () => {
    const { host } = renderWithDetection(
      { ...camera, status: 'online' },
      detection({ state: 'blind', reason: 'pose_not_completing', recent_success_rate: 0 }),
    );

    const cell = detectionCell(host);
    expect(cell?.dataset.detectionState).toBe('blind');
    expect(cell?.getAttribute('role')).toBe('alert');
    // 연결 상태는 감지 경보에 가려지지 않는다.
    expect(host.querySelector('[data-testid="connection-history"]')).not.toBeNull();
  });

  it.each([
    ['pose_not_completing'],
    ['decision_not_completing'],
    ['no_completed_cycles'],
  ] as const)('%s 원인은 자기 자신으로 식별되는 안내를 남긴다', (reason) => {
    const { host } = renderWithDetection(camera, detection({ state: 'blind', reason }));

    const cell = detectionCell(host);
    expect(cell?.dataset.detectionState).toBe('blind');
    expect(cell?.dataset.detectionReason).toBe(reason);
    expect(cell?.getAttribute('role')).toBe('alert');
    const guidance = host.querySelector('[data-testid="detection-guidance"]');
    expect((guidance?.textContent ?? '').trim().length).toBeGreaterThan(0);
  });

  it('세 원인의 안내 문구는 서로 다르다', () => {
    const texts = (['pose_not_completing', 'decision_not_completing', 'no_completed_cycles'] as const).map((reason) => {
      const { host } = renderWithDetection(camera, detection({ state: 'blind', reason }));
      const text = (host.querySelector('[data-testid="detection-guidance"]')?.textContent ?? '').trim();
      document.body.innerHTML = '';
      return text;
    });

    expect(new Set(texts).size).toBe(3);
  });

  it.each([
    ['starting'],
    ['unknown'],
    ['disabled'],
  ] as const)('%s 상태는 경보가 아니라 상태 표시로 보인다', (state) => {
    const { host } = renderWithDetection(camera, detection({ state }));

    const cell = detectionCell(host);
    expect(cell?.dataset.detectionState).toBe(state);
    expect(cell?.getAttribute('role')).toBe('status');
    expect(host.querySelector('[role="alert"]')).toBeNull();
    expect((cell?.textContent ?? '').trim().length).toBeGreaterThan(0);
  });

  it('진단 정보 자체가 없으면 확인 불가로 보수적으로 표시한다', () => {
    const { host } = renderWithDetection(camera, undefined);

    const cell = detectionCell(host);
    expect(cell?.dataset.detectionState).toBe('unknown');
    expect(cell?.getAttribute('role')).toBe('status');
  });

  it('매핑 대기와 감지 경보는 동시에 각자 보인다', () => {
    const { host } = renderWithDetection(
      { ...camera, mapping_pending: true, backend_camera_id: null },
      detection({ state: 'blind', reason: 'no_completed_cycles' }),
    );

    const mapping = host.querySelector('[data-testid="cloud-mapping"]');
    const cell = detectionCell(host);
    expect(mapping?.textContent).toContain('연동 대기');
    expect(cell?.dataset.detectionState).toBe('blind');
    expect(mapping?.contains(cell as Node)).toBe(false);
    expect(cell?.contains(mapping as Node)).toBe(false);
  });

  it('정상으로 회복하면 다음 렌더에서 alert 역할이 사라진다', () => {
    const { host, rerender } = renderWithDetection(
      camera,
      detection({ state: 'blind', reason: 'decision_not_completing' }),
    );
    expect(host.querySelector('[role="alert"]')).not.toBeNull();

    rerender(detection({ state: 'healthy' }));

    expect(host.querySelector('[role="alert"]')).toBeNull();
    expect(detectionCell(host)?.dataset.detectionState).toBe('healthy');
    expect(host.querySelector('[data-testid="detection-guidance"]')).toBeNull();
  });

  it('색만으로 뜻을 전하지 않도록 상태마다 읽을 수 있는 글자를 남긴다', () => {
    (['healthy', 'starting', 'unknown', 'disabled', 'blind'] as const).forEach((state) => {
      const { host } = renderWithDetection(camera, detection({ state, reason: state === 'blind' ? 'pose_not_completing' : null }));
      expect((detectionCell(host)?.textContent ?? '').trim().length).toBeGreaterThan(0);
      document.body.innerHTML = '';
    });
  });
});
