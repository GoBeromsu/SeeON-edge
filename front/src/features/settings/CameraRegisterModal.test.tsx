import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createCamera, recognizeBedZone } from '@/shared/api/client';
import { HttpError } from '@/shared/api/http';
import { CameraRegisterModal } from '@/features/settings/CameraRegisterModal';
import { toast } from '@/shared/ui/Toast';
import type { BedZone, Camera } from '@/shared/api/client';

vi.mock('@/shared/api/client', async () => {
  const { withOverrides } = await vi.importActual<typeof import('@/test/moduleMock')>('@/test/moduleMock');
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return withOverrides(actual, { createCamera: vi.fn(), recognizeBedZone: vi.fn() });
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

function render(open = true, onClose = vi.fn(), onCreated = vi.fn(), cameras: Camera[] = []) {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<CameraRegisterModal open={open} cameras={cameras} onClose={onClose} onCreated={onCreated} />));
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

function setSelect(name: string, value: string): void {
  const select = document.querySelector(`select[name="${name}"]`);
  if (!(select instanceof HTMLSelectElement)) throw new Error(`missing select ${name}`);
  act(() => {
    const setter = Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, 'value')?.set;
    setter?.call(select, value);
    select.dispatchEvent(new Event('change', { bubbles: true }));
  });
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

    expect(createCamera).toHaveBeenCalledWith({ label: '101호', rtsp_url: 'rtsp://cam/1', floor: 1 }, { forceRegister: false });
    expect(document.body.textContent).toContain('침대 영역 인식이 필요합니다.');
  });

  it('widens the dialog from step 1 (440px) to step 2 (640px) per the design spec', async () => {
    vi.mocked(createCamera).mockResolvedValue(createdCamera);
    render();
    expect(document.querySelector('[role="dialog"]')?.getAttribute('data-size')).toBe('md');

    setInput('label', '101호');
    setInput('rtsp_url', 'rtsp://cam/1');
    await act(async () => findButton('다음').click());

    expect(document.querySelector('[role="dialog"]')?.getAttribute('data-size')).toBe('lg');
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

    expect(createCamera).toHaveBeenLastCalledWith({ label: '101호', rtsp_url: 'rtsp://cam/1', floor: 1 }, { forceRegister: true });
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

    await act(async () => findButton('▶ 인식 시작').click());
    act(() => findButton('저장하고 완료').click());

    expect(successSpy).toHaveBeenCalledWith('카메라를 등록했습니다.');
    expect(onCreated).toHaveBeenCalled();
  });

  it('층 드롭다운의 기본값은 1층이고, 건드리지 않으면 정수 1을 등록 요청에 포함시킨다 (issue #155)', async () => {
    vi.mocked(createCamera).mockResolvedValue(createdCamera);
    render();
    expect((document.querySelector('select[name="floor"]') as HTMLSelectElement).value).toBe('1');
    setInput('label', '101호');
    setInput('rtsp_url', 'rtsp://cam/1');

    await act(async () => findButton('다음').click());

    expect(createCamera).toHaveBeenCalledWith(
      { label: '101호', rtsp_url: 'rtsp://cam/1', floor: 1 },
      { forceRegister: false },
    );
  });

  it('선택한 층을 정수로 등록 요청에 포함시킨다', async () => {
    vi.mocked(createCamera).mockResolvedValue(createdCamera);
    render();
    setInput('label', '101호');
    setInput('rtsp_url', 'rtsp://cam/1');
    setSelect('floor', '2');

    await act(async () => findButton('다음').click());

    expect(createCamera).toHaveBeenCalledWith(
      { label: '101호', rtsp_url: 'rtsp://cam/1', floor: 2 },
      { forceRegister: false },
    );
  });

  it('지하층을 선택하면 음수로 보낸다', async () => {
    vi.mocked(createCamera).mockResolvedValue(createdCamera);
    render();
    setInput('label', '101호');
    setInput('rtsp_url', 'rtsp://cam/1');
    setSelect('floor', '-1');

    await act(async () => findButton('다음').click());

    expect(createCamera).toHaveBeenCalledWith(
      { label: '101호', rtsp_url: 'rtsp://cam/1', floor: -1 },
      { forceRegister: false },
    );
  });

  it('resets its draft the next time it is reopened', async () => {
    vi.mocked(createCamera).mockResolvedValue(createdCamera);
    const { root, onClose } = render(true);
    setInput('label', '101호');
    setInput('rtsp_url', 'rtsp://cam/1');
    await act(async () => findButton('다음').click());
    expect(document.body.textContent).toContain('침대 영역 인식이 필요합니다.');

    act(() => root.render(<CameraRegisterModal open={false} cameras={[]} onClose={onClose} onCreated={vi.fn()} />));
    act(() => root.render(<CameraRegisterModal open cameras={[]} onClose={onClose} onCreated={vi.fn()} />));

    expect((document.querySelector('input[name="label"]') as HTMLInputElement).value).toBe('');
    expect(document.body.textContent).not.toContain('침대 영역 인식이 필요합니다.');
  });

  it('warns (without blocking) when the RTSP address looks like an IDIS main stream (issue #154)', async () => {
    vi.mocked(createCamera).mockResolvedValue(createdCamera);
    render();
    setInput('label', '101호');
    setInput('rtsp_url', 'rtsp://admin:pw@192.0.2.10:554/trackID=1');

    expect(document.querySelector('[data-testid="rtsp-substream-guidance"]')?.textContent).toContain('trackID=2');

    await act(async () => findButton('다음').click());

    expect(createCamera).toHaveBeenCalledWith(
      { label: '101호', rtsp_url: 'rtsp://admin:pw@192.0.2.10:554/trackID=1', floor: 1 },
      { forceRegister: false },
    );
  });

  it('shows no substream warning once the address already selects the IDIS substream (trackID=2)', () => {
    render();
    setInput('label', '101호');
    setInput('rtsp_url', 'rtsp://admin:pw@192.0.2.10:554/trackID=2');

    expect(document.querySelector('[data-testid="rtsp-substream-guidance"]')).toBeNull();
  });
});

describe('I7 — 클라우드 방(space) 선택', () => {
  /** 엣지가 pull한 roster의 미매칭 항목 = 아직 카메라가 없는 방. */
  const rosterSpace: Camera = {
    id: 'backend-cam-205',
    label: '205호',
    rtsp_url_masked: 'rtsp://***',
    space_id: 'sp_205',
    space_name: '205호',
    floor_name: '2층',
    backend_camera_id: 'backend-cam-205',
    mapping_pending: true,
    status: 'unknown',
    created_at: null,
  };

  it('빈 방이 있으면 선택 UI를 보여준다', () => {
    render(true, vi.fn(), vi.fn(), [rosterSpace]);

    const select = document.querySelector('select[name="space_id"]');
    expect(select).not.toBeNull();
    expect(document.body.textContent).toContain('2층 · 205호');
  });

  it('방을 고르지 않으면 등록을 막는다', async () => {
    const errorSpy = vi.spyOn(toast, 'error');
    render(true, vi.fn(), vi.fn(), [rosterSpace]);
    setInput('label', '205호');
    setInput('rtsp_url', 'rtsp://cam/1');

    await act(async () => findButton('다음').click());

    expect(createCamera).not.toHaveBeenCalled();
    expect(errorSpy).toHaveBeenCalledWith('이 카메라가 설치된 방을 선택하세요.');
  });

  it('선택한 space_id를 등록 요청에 함께 보낸다', async () => {
    vi.mocked(createCamera).mockResolvedValue(createdCamera);
    render(true, vi.fn(), vi.fn(), [rosterSpace]);
    setInput('label', '205호');
    setInput('rtsp_url', 'rtsp://cam/1');
    setSelect('space_id', 'sp_205');

    await act(async () => findButton('다음').click());

    expect(createCamera).toHaveBeenCalledWith(
      { label: '205호', rtsp_url: 'rtsp://cam/1', floor: 1, space_id: 'sp_205' },
      { forceRegister: false },
    );
  });

  it('이미 카메라가 붙은 방은 후보에서 빠진다 (카메라 1대 = 방 1개)', () => {
    const occupied: Camera = {
      ...rosterSpace,
      id: 'local-cam',
      mapping_pending: false,
      backend_camera_id: 'backend-cam-205',
    };
    render(true, vi.fn(), vi.fn(), [occupied]);

    const select = document.querySelector('select[name="space_id"]');
    expect(select).toBeNull();
  });

  it('roster가 비어 있으면 선택을 요구하지 않는다', async () => {
    vi.mocked(createCamera).mockResolvedValue(createdCamera);
    render(true, vi.fn(), vi.fn(), []);
    setInput('label', '101호');
    setInput('rtsp_url', 'rtsp://cam/1');

    await act(async () => findButton('다음').click());

    expect(createCamera).toHaveBeenCalledWith(
      { label: '101호', rtsp_url: 'rtsp://cam/1', floor: 1 },
      { forceRegister: false },
    );
  });

  it('roster가 비어 있으면 그 사실과 결과를 알린다', () => {
    // 등록은 통과시키되 침묵하지 않는다. 기사가 정상 등록으로 알고 떠나면
    // 클라우드 현황판에는 영영 나타나지 않는다.
    render(true, vi.fn(), vi.fn(), []);

    const notice = document.querySelector('[data-testid="no-spaces-notice"]');
    expect(notice).not.toBeNull();
    expect(notice?.textContent).toContain('클라우드 현황판에는 나타나지 않습니다');
  });

  it('roster가 비어 있으면 안내를 한 줄로 압축하고 세부 설명은 접어 둔다', () => {
    render(true, vi.fn(), vi.fn(), []);

    const notice = document.querySelector('[data-testid="no-spaces-notice"]') as HTMLElement;
    expect(notice.textContent).toContain('방 목록을 받지 못해 방 지정 없이 저장됩니다. 등록 후 지정할 수 있습니다.');

    const details = notice.querySelector('details');
    expect(details).not.toBeNull();
    expect(details?.open).toBe(false);
    expect(details?.textContent).toContain('클라우드 현황판에는 나타나지 않습니다');
  });

  it('빈 방이 있으면 안내 대신 선택 UI를 보여준다', () => {
    render(true, vi.fn(), vi.fn(), [rosterSpace]);

    expect(document.querySelector('[data-testid="no-spaces-notice"]')).toBeNull();
    expect(document.querySelector('select[name="space_id"]')).not.toBeNull();
  });
});
