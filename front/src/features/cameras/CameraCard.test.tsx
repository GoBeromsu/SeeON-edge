import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { testCamera, updateCamera, updateCameraDecodeBackend } from '@/shared/api/client';
import { CameraCard } from '@/features/cameras/CameraCard';
import type { Camera } from '@/shared/api/client';

vi.mock('@/shared/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return {
    ...actual,
    testCamera: vi.fn(),
    updateCamera: vi.fn(),
    updateCameraDecodeBackend: vi.fn(),
  };
});

const camera: Camera = {
  id: 'cam-1',
  label: '301호 침대 A',
  rtsp_url_masked: 'rtsp://user:****@camera.local/stream',
  space_id: 'space-301',
  space_name: null,
  floor_name: null,
  backend_camera_id: null,
  status: 'online',
  created_at: '2026-07-06T00:00:00Z',
};

function setInput(host: HTMLElement, name: string, value: string): void {
  const input = host.querySelector(`input[name="${name}"]`);
  if (!(input instanceof HTMLInputElement)) {
    throw new Error(`missing input ${name}`);
  }
  act(() => {
    const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set;
    valueSetter?.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

function clickButton(host: HTMLElement, label: string): void {
  const button = Array.from(host.querySelectorAll('button')).find((candidate) => candidate.textContent === label);
  if (!button) {
    throw new Error(`missing button ${label}`);
  }
  act(() => button.click());
}

beforeEach(() => {
  vi.mocked(updateCamera).mockReset();
  vi.mocked(updateCamera).mockResolvedValue({ ...camera, label: '301호 침대 B' });
  vi.mocked(testCamera).mockReset();
  vi.mocked(testCamera).mockResolvedValue({ ok: true, width: 640, height: 480 });
  vi.mocked(updateCameraDecodeBackend).mockReset();
  vi.mocked(updateCameraDecodeBackend).mockResolvedValue({ ...camera, decode_backend: 'cpu' });
});

describe('CameraCard', () => {
  it('renders operator camera details without transport terminology or masked URLs', () => {
    const host = document.createElement('div');
    const root = createRoot(host);

    act(() => {
      root.render(<CameraCard camera={camera} />);
    });

    expect(host.textContent).toContain('301호 침대 A');
    expect(host.textContent).toContain('온라인');
    expect(host.textContent).toContain('서버 자동 관리');
    expect(host.textContent).toContain('카메라');
    expect(host.textContent).not.toContain('Camera');
    expect(host.textContent).not.toContain('RTSP');
    expect(host.textContent).not.toContain(camera.rtsp_url_masked);
    // Technical settings must be visible without a click for an engineer audience — no disclosure widget.
    expect(host.querySelector('details')).toBeNull();
    expect(host.querySelector('summary')).toBeNull();
    expect(host.textContent).toContain('기술 설정');
    expect(host.textContent).toContain('cam-1');
    const classNames = host.innerHTML;
    expect(host.querySelector('article')?.className).toContain('rounded-2xl');
    expect(host.querySelector('article')?.className).toContain('border-border');
    expect(classNames).not.toMatch(/rounded-(3xl|4xl)|shadow-glow|shadow-soft|bg-white|text-white|slate-|rose-/);

    act(() => root.unmount());
  });

  it('renders missing registration time as Korean operator copy', () => {
    const host = document.createElement('div');
    const root = createRoot(host);

    act(() => root.render(<CameraCard camera={{ ...camera, created_at: null }} />));

    expect(host.textContent).toContain('등록 정보 없음');
    expect(host.textContent).not.toContain('unknown');
    act(() => root.unmount());
  });

  it('edits the camera name without exposing or requiring raw space_id', async () => {
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    const onUpdated = vi.fn();

    act(() => {
      root.render(<CameraCard camera={camera} onUpdated={onUpdated} />);
    });

    const editButton = Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '수정');
    act(() => {
      editButton?.click();
    });

    expect(host.textContent).not.toContain('space_id');

    setInput(host, 'label', '301호 침대 B');
    const saveButton = Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '저장');
    expect(saveButton?.classList.contains('brand-action')).toBe(true);
    await act(async () => {
      saveButton?.click();
    });

    expect(updateCamera).toHaveBeenCalledWith('cam-1', { label: '301호 침대 B' });
    expect(onUpdated).toHaveBeenCalledWith(expect.objectContaining({ id: 'cam-1', label: '301호 침대 B' }), 'cam-1');

    act(() => root.unmount());
    host.remove();
  });

  it('synchronizes idle drafts when polling supplies updated camera values', () => {
    const host = document.createElement('div');
    const root = createRoot(host);
    act(() => root.render(<CameraCard camera={camera} />));

    act(() => root.render(<CameraCard camera={{ ...camera, label: '폴링된 이름', decode_backend: 'cpu' }} />));

    expect((host.querySelector('select') as HTMLSelectElement).value).toBe('cpu');
    clickButton(host, '수정');
    expect((host.querySelector('input[name="label"]') as HTMLInputElement).value).toBe('폴링된 이름');
    act(() => root.unmount());
  });

  it('preserves an active edit across external updates', () => {
    const host = document.createElement('div');
    const root = createRoot(host);
    act(() => root.render(<CameraCard camera={camera} />));
    clickButton(host, '수정');
    setInput(host, 'label', '작성 중인 이름');

    act(() => root.render(<CameraCard camera={{ ...camera, label: '서버의 새 이름' }} />));

    expect((host.querySelector('input[name="label"]') as HTMLInputElement).value).toBe('작성 중인 이름');
    act(() => root.unmount());
  });

  it('resets an active draft to the latest camera value when cancelled', () => {
    const host = document.createElement('div');
    const root = createRoot(host);
    act(() => root.render(<CameraCard camera={camera} />));
    clickButton(host, '수정');
    setInput(host, 'label', '버릴 이름');
    act(() => root.render(<CameraCard camera={{ ...camera, label: '서버의 최신 이름' }} />));

    clickButton(host, '취소');
    clickButton(host, '수정');

    expect((host.querySelector('input[name="label"]') as HTMLInputElement).value).toBe('서버의 최신 이름');
    act(() => root.unmount());
  });

  it('submits the latest polled camera name instead of the initial draft', async () => {
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(<CameraCard camera={camera} />));
    act(() => root.render(<CameraCard camera={{ ...camera, label: '서버의 최신 이름' }} />));
    clickButton(host, '수정');

    await act(async () => {
      Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '저장')?.click();
    });

    expect(updateCamera).toHaveBeenCalledWith('cam-1', { label: '서버의 최신 이름' });
    act(() => root.unmount());
    host.remove();
  });

  it('does not clobber a decode selection while its mutation is pending', async () => {
    let resolveUpdate: ((updated: Camera) => void) | undefined;
    vi.mocked(updateCameraDecodeBackend).mockReturnValue(new Promise((resolve) => {
      resolveUpdate = resolve;
    }));
    const host = document.createElement('div');
    const root = createRoot(host);
    act(() => root.render(<CameraCard camera={camera} />));
    const select = host.querySelector('select') as HTMLSelectElement;

    act(() => {
      const setter = Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, 'value')?.set;
      setter?.call(select, 'cpu');
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });
    act(() => root.render(<CameraCard camera={{ ...camera, decode_backend: 'nvdec' }} />));

    expect(select.value).toBe('cpu');
    expect(select.disabled).toBe(true);

    await act(async () => resolveUpdate?.({ ...camera, decode_backend: 'cpu' }));
    expect(select.value).toBe('nvdec');
    act(() => root.unmount());
  });

  it('keeps edit mode and values after validation or server failure', async () => {
    vi.mocked(updateCamera).mockRejectedValue(new Error('Invalid camera response'));
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    const onUpdated = vi.fn();
    act(() => root.render(<CameraCard camera={camera} onUpdated={onUpdated} />));
    act(() => Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '수정')?.click());

    setInput(host, 'label', '');
    act(() => Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '저장')?.click());
    expect(host.textContent).toContain('카메라 이름을 입력하세요.');

    setInput(host, 'label', '보존할 이름');
    await act(async () => Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '저장')?.click());
    expect(host.textContent).toContain('카메라 수정에 실패했습니다.');
    expect((host.querySelector('input[name="label"]') as HTMLInputElement).value).toBe('보존할 이름');
    expect(onUpdated).not.toHaveBeenCalled();
    act(() => root.unmount());
    host.remove();
  });

  it('never shows a manual connection-test button', () => {
    const host = document.createElement('div');
    const root = createRoot(host);
    act(() => root.render(<CameraCard camera={camera} />));

    expect(Array.from(host.querySelectorAll('button')).some((button) => button.textContent === '연결 테스트')).toBe(false);
    act(() => root.unmount());
  });

  it('shows a retry action only for a camera in a failed state', () => {
    const host = document.createElement('div');
    const root = createRoot(host);
    act(() => root.render(<CameraCard camera={camera} />));
    expect(Array.from(host.querySelectorAll('button')).some((button) => button.textContent === '재시도')).toBe(false);

    act(() => root.render(<CameraCard camera={{ ...camera, status: 'offline' }} />));
    expect(Array.from(host.querySelectorAll('button')).some((button) => button.textContent === '재시도')).toBe(true);
    act(() => root.unmount());
  });

  it('retries the connection from an offline camera, reports the result, and asks the parent to refresh', async () => {
    const host = document.createElement('div');
    const root = createRoot(host);
    const onRetried = vi.fn();
    act(() => root.render(<CameraCard camera={{ ...camera, status: 'offline' }} onRetried={onRetried} />));

    await act(async () => Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '재시도')?.click());
    expect(testCamera).toHaveBeenCalledWith('cam-1');
    expect(host.textContent).toContain('재연결 성공 · 640×480');
    expect(onRetried).toHaveBeenCalledTimes(1);
    act(() => root.unmount());
  });

  it.each([
    ['timeout', '카메라 응답 시간이 초과되었습니다. 네트워크와 전원을 확인하세요.'],
    ['auth', '카메라 인증에 실패했습니다. 아이디와 비밀번호를 확인하세요.'],
    ['decode', '영상 형식을 처리할 수 없습니다. 카메라 스트림 설정을 확인하세요.'],
    ['internal_trace_id=secret', '카메라 연결을 확인할 수 없습니다. 네트워크와 설정을 확인하세요.'],
  ])('given %s connection failure, when retrying an offline camera, then shows a sanitized operator message', async (errorClass, expected) => {
    vi.mocked(testCamera).mockResolvedValue({ ok: false, error_class: errorClass });
    const host = document.createElement('div');
    const root = createRoot(host);
    act(() => root.render(<CameraCard camera={{ ...camera, status: 'offline' }} />));

    await act(async () => Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '재시도')?.click());

    expect(host.textContent).toContain(expected);
    expect(host.textContent).not.toContain(errorClass);
    act(() => root.unmount());
  });

  it('shows no connection line for a camera that never connected, but shows one for a camera that dropped', () => {
    const host = document.createElement('div');
    const root = createRoot(host);
    // Asserted on the element, not on textContent: a stale copy string would still satisfy a
    // "does not contain X" check, so only the absent node proves the line is really gone.
    act(() => root.render(<CameraCard camera={{ ...camera, status: 'offline', never_connected: true }} />));
    expect(host.querySelector('[data-testid="camera-connection-detail"]')).toBeNull();

    act(() => root.render(<CameraCard camera={{ ...camera, status: 'offline', never_connected: false, last_ok_at: '2026-07-20T00:00:00Z' }} />));
    expect(host.querySelector('[data-testid="camera-connection-detail"]')?.textContent).toContain('마지막 연결');
    act(() => root.unmount());
  });

  it('shows humanised heartbeat freshness on an offline card', () => {
    const host = document.createElement('div');
    const root = createRoot(host);
    act(() => root.render(<CameraCard camera={{ ...camera, status: 'offline', heartbeat_age_sec: 245 }} />));

    expect(host.textContent).toContain('마지막 신호 4분 전');
    act(() => root.unmount());
  });

  it('shows nothing for heartbeat freshness when both fields are null, with no placeholder text', () => {
    const host = document.createElement('div');
    const root = createRoot(host);
    act(() => root.render(<CameraCard camera={{ ...camera, status: 'offline', last_heartbeat_at: null, heartbeat_age_sec: null }} />));

    expect(host.textContent).not.toContain('마지막 신호');
    expect(host.textContent).not.toContain('N/A');
    act(() => root.unmount());
  });

  it('does not show heartbeat freshness on an online card to avoid noise on the expected-healthy state', () => {
    const host = document.createElement('div');
    const root = createRoot(host);
    act(() => root.render(<CameraCard camera={{ ...camera, status: 'online', heartbeat_age_sec: 3 }} />));

    expect(host.textContent).not.toContain('마지막 신호');
    act(() => root.unmount());
  });

  it('suppresses the heartbeat age on a never_connected card, even if the backend sent one', () => {
    const host = document.createElement('div');
    const root = createRoot(host);
    act(() => root.render(<CameraCard camera={{ ...camera, status: 'offline', never_connected: true, heartbeat_age_sec: 0 }} />));

    expect(host.textContent).not.toContain('마지막 신호');
    act(() => root.unmount());
  });

  it('links to the camera-scoped clips view only when a handler is supplied', () => {
    const host = document.createElement('div');
    const root = createRoot(host);
    act(() => root.render(<CameraCard camera={camera} />));
    expect(Array.from(host.querySelectorAll('button')).some((button) => button.textContent === '클립 보기')).toBe(false);

    const onViewClips = vi.fn();
    act(() => root.render(<CameraCard camera={camera} onViewClips={onViewClips} />));
    act(() => Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '클립 보기')?.click());
    expect(onViewClips).toHaveBeenCalledWith(camera);
    act(() => root.unmount());
  });
});
