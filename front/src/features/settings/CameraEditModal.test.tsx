import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { recognizeBedZone, testCamera, updateCamera } from '@/shared/api/client';
import { CameraEditModal } from '@/features/settings/CameraEditModal';
import { toast } from '@/shared/ui/Toast';
import type { BedZone, Camera } from '@/shared/api/client';

vi.mock('@/shared/api/client', async () => {
  const { withOverrides } = await vi.importActual<typeof import('@/test/moduleMock')>('@/test/moduleMock');
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return withOverrides(actual, { testCamera: vi.fn(), updateCamera: vi.fn(), recognizeBedZone: vi.fn() });
});

const camera: Camera = {
  id: 'cam-1',
  label: '101호',
  rtsp_url_masked: 'rtsp://***@10.0.0.5/stream',
  floor_name: '1층',
  status: 'online',
  created_at: null,
  bed_zone: null,
};

const bedZone: BedZone = {
  polygon: [[0, 0], [100, 0], [100, 100], [0, 100]],
  image_width: 1920,
  image_height: 1080,
  recognized_at: '2026-08-01T00:00:00Z',
};

function render(
  cam: Camera | null,
  onClose = vi.fn(),
  onUpdated = vi.fn(),
  onRequestDelete = vi.fn(),
  cams: Camera[] = cam ? [cam] : [],
) {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(
    <CameraEditModal camera={cam} cameras={cams} onClose={onClose} onUpdated={onUpdated} onRequestDelete={onRequestDelete} />,
  ));
  const rerender = (nextCam: Camera | null) => {
    act(() => root.render(
      <CameraEditModal camera={nextCam} cameras={cams} onClose={onClose} onUpdated={onUpdated} onRequestDelete={onRequestDelete} />,
    ));
  };
  return { host, root, onClose, onUpdated, onRequestDelete, rerender };
}

function findButton(label: string): HTMLButtonElement {
  const button = Array.from(document.querySelectorAll('button')).find((candidate) => candidate.textContent === label);
  if (!button) throw new Error(`missing button ${label}`);
  return button;
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
  vi.mocked(testCamera).mockReset();
  vi.mocked(updateCamera).mockReset();
  vi.mocked(recognizeBedZone).mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

describe('CameraEditModal', () => {
  it('renders nothing when no camera is targeted', () => {
    render(null);
    expect(document.querySelector('[role="dialog"]')).toBeNull();
  });

  it('pre-fills the name, shows the cloud floor_name as the fallback select option, and the 인식 필요 bed-zone badge', () => {
    render(camera);
    expect((document.querySelector('input[name="label"]') as HTMLInputElement).value).toBe('101호');
    const select = document.querySelector('select[name="floor"]') as HTMLSelectElement;
    // No local override yet -- the select shows the "use the cloud value" option, not a fixed floor.
    expect(select.value).toBe('');
    expect(select.selectedOptions[0]?.textContent).toContain('1층');
    expect(document.body.textContent).toContain('인식 필요');
  });

  it('selects a different floor and sends the integer in the patch (issue #155)', async () => {
    vi.mocked(updateCamera).mockResolvedValue(camera);
    render(camera);

    setSelect('floor', '2');
    await act(async () => findButton('저장').click());

    expect(updateCamera).toHaveBeenCalledWith('cam-1', { floor: 2 });
  });

  it('선택한 층이 지하면 음수로 patch에 포함시킨다', async () => {
    vi.mocked(updateCamera).mockResolvedValue(camera);
    render(camera);

    setSelect('floor', '-1');
    await act(async () => findButton('저장').click());

    expect(updateCamera).toHaveBeenCalledWith('cam-1', { floor: -1 });
  });

  it('returns the select to "클라우드 값 사용" and sends floor: null to clear an existing override', async () => {
    vi.mocked(updateCamera).mockResolvedValue(camera);
    render({ ...camera, floor: 3 });

    setSelect('floor', '');
    await act(async () => findButton('저장').click());

    expect(updateCamera).toHaveBeenCalledWith('cam-1', { floor: null });
  });

  it('omits floor from the patch when the select is left untouched, avoiding an accidental override', async () => {
    vi.mocked(updateCamera).mockResolvedValue({ ...camera, label: '새 이름' });
    render(camera);
    setInput('label', '새 이름');

    await act(async () => findButton('저장').click());

    expect(updateCamera).toHaveBeenCalledWith('cam-1', { label: '새 이름' });
  });

  it('saves only the changed fields and omits rtsp_url when left untouched', async () => {
    vi.mocked(updateCamera).mockResolvedValue({ ...camera, label: '새 이름' });
    const { onUpdated, onClose } = render(camera);
    setInput('label', '새 이름');

    await act(async () => findButton('저장').click());

    expect(updateCamera).toHaveBeenCalledWith('cam-1', { label: '새 이름' });
    expect(onUpdated).toHaveBeenCalled();
    expect(onClose).toHaveBeenCalled();
  });

  it('includes rtsp_url in the patch after the technician verifies the replacement', async () => {
    vi.mocked(updateCamera).mockResolvedValue(camera);
    vi.mocked(testCamera).mockResolvedValue({ ok: true, width: 1920, height: 1080 });
    render(camera);
    setInput('rtsp_url', 'rtsp://new-address/stream');

    // 새 주소는 연결 확인을 통과해야 저장된다. 확인 없이 저장하면 오타가
    // 그대로 들어가 카메라가 조용히 죽는다.
    await act(async () => findButton('연결 확인').click());
    await act(async () => findButton('저장').click());

    expect(updateCamera).toHaveBeenCalledWith('cam-1', { rtsp_url: 'rtsp://new-address/stream' });
  });

  it('shows an inline error and keeps the modal open when save fails', async () => {
    vi.mocked(updateCamera).mockRejectedValue(new Error('boom'));
    const { onClose } = render(camera);
    setInput('label', '새 이름');

    await act(async () => findButton('저장').click());

    expect(document.querySelector('[role="alert"]')?.textContent).toContain('저장하지 못했습니다');
    expect(onClose).not.toHaveBeenCalled();
  });

  it('toasts and refreshes on a successful connection test without closing the modal', async () => {
    vi.mocked(testCamera).mockResolvedValue({ ok: true, width: 1920, height: 1080 });
    const successSpy = vi.spyOn(toast, 'success');
    const { onUpdated, onClose } = render(camera);

    await act(async () => findButton('연결 확인').click());

    expect(testCamera).toHaveBeenCalledWith('cam-1', undefined);
    expect(successSpy).toHaveBeenCalledWith('연결 성공');
    expect(onUpdated).toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });

  it('shows a failure message when the connection test fails', async () => {
    vi.mocked(testCamera).mockResolvedValue({ ok: false, error_class: 'timeout' });
    render(camera);

    await act(async () => findButton('연결 확인').click());

    expect(document.querySelector('[role="alert"]')?.textContent).toContain('연결 테스트에 실패했습니다');
  });

  // 이슈 #151: worker의 /probe에 닿지 못한 경우(probe_unavailable)는 실제로 디코드를
  // 확인한 적이 없으므로, 그 실패 메시지가 "디코드" 관련 문구를 담아서는 안 된다.
  it('does not claim a decode failure when the worker probe is unreachable', async () => {
    vi.mocked(testCamera).mockResolvedValue({ ok: false, probe_unavailable: true });
    render(camera);

    await act(async () => findButton('연결 확인').click());

    const message = document.querySelector('[role="alert"]')?.textContent ?? '';
    expect(message).toContain('연결 테스트에 실패했습니다');
    expect(message).not.toContain('디코드');
  });

  it('delegates deletion to the parent instead of opening a nested dialog', () => {
    const { onRequestDelete } = render(camera);
    act(() => findButton('삭제').click());
    expect(onRequestDelete).toHaveBeenCalledWith(camera);
    expect(document.querySelectorAll('[role="dialog"]').length).toBe(1);
  });

  it('switches into the re-recognition sub-view and back without opening a second dialog', async () => {
    vi.mocked(recognizeBedZone).mockResolvedValue(bedZone);
    const { onUpdated } = render(camera);

    act(() => document.querySelector<HTMLButtonElement>('[aria-label="침대 영역 다시 인식"]')?.click());
    expect(document.querySelectorAll('[role="dialog"]').length).toBe(1);
    expect(document.body.textContent).toContain('침대 영역 인식이 필요합니다.');

    await act(async () => findButton('▶ 인식 시작').click());
    expect(onUpdated).toHaveBeenCalled();

    act(() => findButton('완료').click());
    expect(document.body.textContent).toContain('인식 완료');
  });

  it('widens the dialog to 640px for the re-recognition sub-view, matching step 2 of registration', () => {
    render(camera);
    expect(document.querySelector('[role="dialog"]')?.getAttribute('data-size')).toBe('md');

    act(() => document.querySelector<HTMLButtonElement>('[aria-label="침대 영역 다시 인식"]')?.click());
    expect(document.querySelector('[role="dialog"]')?.getAttribute('data-size')).toBe('lg');

    act(() => findButton('취소').click());
    expect(document.querySelector('[role="dialog"]')?.getAttribute('data-size')).toBe('md');
  });

  it('does not reset the re-recognition flow when the operations page polls in a fresh camera object for the same id', () => {
    const { rerender } = render(camera);

    act(() => document.querySelector<HTMLButtonElement>('[aria-label="침대 영역 다시 인식"]')?.click());
    expect(document.querySelector('[role="dialog"]')?.getAttribute('data-size')).toBe('lg');
    expect(document.body.textContent).toContain('침대 영역 인식이 필요합니다.');

    // Simulate a 5s poll tick handing down a brand-new object for the same camera id.
    rerender({ ...camera });

    expect(document.querySelector('[role="dialog"]')?.getAttribute('data-size')).toBe('lg');
    expect(document.body.textContent).toContain('침대 영역 인식이 필요합니다.');
  });

  it('resets to the view mode when the modal is retargeted at a different camera id', () => {
    const otherCamera: Camera = { ...camera, id: 'cam-2', label: '102호', bed_zone: null };
    const { rerender } = render(camera);

    act(() => document.querySelector<HTMLButtonElement>('[aria-label="침대 영역 다시 인식"]')?.click());
    expect(document.querySelector('[role="dialog"]')?.getAttribute('data-size')).toBe('lg');

    rerender(otherCamera);

    expect(document.querySelector('[role="dialog"]')?.getAttribute('data-size')).toBe('md');
    expect((document.querySelector('input[name="label"]') as HTMLInputElement).value).toBe('102호');
  });

  it('shows the masked address as helper text instead of leaving the operator guessing whether one is registered', () => {
    render(camera);
    const rtspInput = document.querySelector('input[name="rtsp_url"]') as HTMLInputElement;
    expect(rtspInput.value).toBe('');
    expect(document.body.textContent).toContain('현재 등록됨:');
    expect(document.body.textContent).toContain(camera.rtsp_url_masked);
  });

  it('shows no substream warning until the operator actually types a replacement address (issue #154)', () => {
    render(camera);
    expect(document.querySelector('[data-testid="rtsp-substream-guidance"]')).toBeNull();
  });

  it('warns (without blocking the save) when a replacement RTSP address looks like an IDIS main stream (issue #154)', async () => {
    vi.mocked(updateCamera).mockResolvedValue({ ...camera, rtsp_url_masked: 'rtsp://***@10.0.0.5/trackID=1' });
    render(camera);
    setInput('rtsp_url', 'rtsp://admin:pw@192.0.2.10:554/trackID=1');

    expect(document.querySelector('[data-testid="rtsp-substream-guidance"]')?.textContent).toContain('trackID=2');

    await act(async () => findButton('저장').click());

    expect(updateCamera).toHaveBeenCalledWith('cam-1', { rtsp_url: 'rtsp://admin:pw@192.0.2.10:554/trackID=1' });
  });

  it('I8 — 저장된 URL이 아니라 입력창의 새 URL을 검사한다', async () => {
    vi.mocked(testCamera).mockResolvedValue({ ok: true, width: 1920, height: 1080 });
    render(camera);
    setInput('rtsp_url', 'rtsp://typo.local/live');

    await act(async () => findButton('연결 확인').click());

    expect(testCamera).toHaveBeenCalledWith('cam-1', 'rtsp://typo.local/live');
  });

  it('연결 확인 없이도 새 RTSP를 저장한다', async () => {
    // 저장을 연결 성공에 묶으면 주소를 고칠 방법이 사라진다: 주소가 틀려서
    // 연결이 안 되는 카메라야말로 주소를 고쳐야 하는데, 고친 주소를 저장하려면
    // 먼저 연결에 성공해야 했다.
    vi.mocked(updateCamera).mockResolvedValue(camera);
    render(camera);
    setInput('rtsp_url', 'rtsp://typo.local/live');

    await act(async () => findButton('저장').click());

    expect(updateCamera).toHaveBeenCalledWith('cam-1', { rtsp_url: 'rtsp://typo.local/live' });
  });

  it('주소를 고치면 앞선 연결 성공 표시를 지운다', async () => {
    // 검사받지 않은 주소 위에 "연결 성공"이 남아 있으면 그게 방금 그 주소의
    // 결과처럼 읽힌다.
    vi.mocked(testCamera).mockResolvedValue({ ok: true, width: 1920, height: 1080 });
    render(camera);

    setInput('rtsp_url', 'rtsp://good.local/live');
    await act(async () => findButton('연결 확인').click());
    expect(document.body.textContent).toContain('연결 성공');

    setInput('rtsp_url', 'rtsp://typo.local/live');

    expect(document.body.textContent).not.toContain('연결 성공');
  });

  it('I8 — 통과한 주소 그대로면 저장된다', async () => {
    vi.mocked(testCamera).mockResolvedValue({ ok: true, width: 1920, height: 1080 });
    vi.mocked(updateCamera).mockResolvedValue(camera);
    render(camera);

    setInput('rtsp_url', 'rtsp://good.local/live');
    await act(async () => findButton('연결 확인').click());
    await act(async () => findButton('저장').click());

    expect(updateCamera).toHaveBeenCalledWith('cam-1', { rtsp_url: 'rtsp://good.local/live' });
  });
});
