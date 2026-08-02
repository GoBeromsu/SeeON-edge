import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createCamera, recognizeBedZone } from '@/shared/api/client';
import { HttpError } from '@/shared/api/http';
import { CameraRegisterModal } from '@/features/settings/CameraRegisterModal';
import { toast } from '@/shared/ui/Toast';
import type { BedZone, Camera } from '@/shared/api/client';

vi.mock('@/shared/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return { ...actual, createCamera: vi.fn(), recognizeBedZone: vi.fn() };
});

const createdCamera: Camera = {
  id: 'cam-9',
  label: '101호',
  rtsp_url_masked: 'rtsp://***@10.0.0.5/stream',
  floor_name: null,
  status: 'starting',
  created_at: null,
  bed_zone: null,
};

const bedZone: BedZone = {
  polygon: [[0, 0], [100, 0], [100, 100], [0, 100]],
  image_width: 1920,
  image_height: 1080,
  recognized_at: '2026-08-01T00:00:00Z',
};

function render(open = true, onClose = vi.fn(), onCreated = vi.fn()) {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<CameraRegisterModal open={open} onClose={onClose} onCreated={onCreated} />));
  return { host, root, onClose, onCreated };
}

function setInput(name: string, value: string): void {
  const input = document.querySelector(`input[name="${name}"]`);
  if (!(input instanceof HTMLInputElement)) throw new Error(`missing input ${name}`);
  act(() => {
    const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set;
    valueSetter?.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

function findButton(label: string): HTMLButtonElement {
  const button = Array.from(document.querySelectorAll('button')).find((candidate) => candidate.textContent === label);
  if (!button) throw new Error(`missing button ${label}`);
  return button;
}

beforeEach(() => {
  vi.mocked(createCamera).mockReset();
  vi.mocked(recognizeBedZone).mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

describe('CameraRegisterModal', () => {
  it('blocks advancing to step 1 submission with a toast when the name is blank', async () => {
    const errorSpy = vi.spyOn(toast, 'error');
    render();
    setInput('rtsp_url', 'rtsp://cam/1');

    await act(async () => findButton('다음').click());

    expect(createCamera).not.toHaveBeenCalled();
    expect(errorSpy).toHaveBeenCalledWith('카메라 이름을 입력하세요.');
  });

  it('creates the camera via step 1 and advances to the bed-zone recognition step', async () => {
    vi.mocked(createCamera).mockResolvedValue(createdCamera);
    render();
    setInput('label', '101호');
    setInput('rtsp_url', 'rtsp://cam/1');

    await act(async () => findButton('다음').click());

    expect(createCamera).toHaveBeenCalledWith({ label: '101호', rtsp_url: 'rtsp://cam/1' }, { forceRegister: false });
    expect(document.body.textContent).toContain('침대 영역 인식이 필요합니다.');
  });

  it('shows an inline probe-failure message mapped from the error_class on a 422 rejection', async () => {
    vi.mocked(createCamera).mockRejectedValue(
      new HttpError(422, { detail: { error: 'probe_failed', error_class: 'timeout' } }),
    );
    render();
    setInput('label', '101호');
    setInput('rtsp_url', 'rtsp://cam/1');

    await act(async () => findButton('다음').click());

    expect(document.querySelector('[role="alert"]')?.textContent).toContain('시간이 초과');
  });

  it('shows the existing camera label and a force-register retry on a 409 duplicate rejection', async () => {
    vi.mocked(createCamera).mockRejectedValueOnce(
      new HttpError(409, {
        detail: { error: 'duplicate_camera', existing_camera_id: 'cam-1', existing_label: '기존 카메라' },
      }),
    );
    vi.mocked(createCamera).mockResolvedValueOnce(createdCamera);
    render();
    setInput('label', '101호');
    setInput('rtsp_url', 'rtsp://cam/1');

    await act(async () => findButton('다음').click());

    expect(document.querySelector('[role="alert"]')?.textContent).toContain('기존 카메라');

    await act(async () => findButton('그래도 등록').click());

    expect(createCamera).toHaveBeenLastCalledWith({ label: '101호', rtsp_url: 'rtsp://cam/1' }, { forceRegister: true });
    expect(document.body.textContent).toContain('침대 영역 인식이 필요합니다.');
  });

  it('blocks completion with a toast until the bed zone is recognized, then completes after recognition', async () => {
    vi.mocked(createCamera).mockResolvedValue(createdCamera);
    vi.mocked(recognizeBedZone).mockResolvedValue(bedZone);
    const errorSpy = vi.spyOn(toast, 'error');
    const successSpy = vi.spyOn(toast, 'success');
    const { onCreated } = render();
    setInput('label', '101호');
    setInput('rtsp_url', 'rtsp://cam/1');
    await act(async () => findButton('다음').click());

    act(() => findButton('저장하고 완료').click());
    expect(errorSpy).toHaveBeenCalledWith('침대 영역 인식을 완료해야 저장할 수 있습니다.');
    expect(onCreated).not.toHaveBeenCalled();

    await act(async () => findButton('인식 시작').click());
    act(() => findButton('저장하고 완료').click());

    expect(successSpy).toHaveBeenCalledWith('카메라를 등록했습니다.');
    expect(onCreated).toHaveBeenCalled();
  });

  it('resets its draft the next time it is reopened', async () => {
    vi.mocked(createCamera).mockResolvedValue(createdCamera);
    const { root, onClose } = render(true);
    setInput('label', '101호');
    setInput('rtsp_url', 'rtsp://cam/1');
    await act(async () => findButton('다음').click());
    expect(document.body.textContent).toContain('침대 영역 인식이 필요합니다.');

    act(() => root.render(<CameraRegisterModal open={false} onClose={onClose} onCreated={vi.fn()} />));
    act(() => root.render(<CameraRegisterModal open onClose={onClose} onCreated={vi.fn()} />));

    expect((document.querySelector('input[name="label"]') as HTMLInputElement).value).toBe('');
    expect(document.body.textContent).not.toContain('침대 영역 인식이 필요합니다.');
  });
});
