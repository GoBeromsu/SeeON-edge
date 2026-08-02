import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { recognizeBedZone, testCamera, updateCamera } from '@/shared/api/client';
import { CameraEditModal } from '@/features/settings/CameraEditModal';
import { toast } from '@/shared/ui/Toast';
import type { BedZone, Camera } from '@/shared/api/client';

vi.mock('@/shared/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return { ...actual, testCamera: vi.fn(), updateCamera: vi.fn(), recognizeBedZone: vi.fn() };
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
) {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(
    <CameraEditModal camera={cam} onClose={onClose} onUpdated={onUpdated} onRequestDelete={onRequestDelete} />,
  ));
  return { host, root, onClose, onUpdated, onRequestDelete };
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

  it('pre-fills the name and shows the read-only floor and 인식 필요 bed-zone badge', () => {
    render(camera);
    expect((document.querySelector('input[name="label"]') as HTMLInputElement).value).toBe('101호');
    expect((document.querySelector('input[name="floor"]') as HTMLInputElement).value).toBe('1층');
    expect(document.body.textContent).toContain('인식 필요');
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

  it('includes rtsp_url in the patch only when the technician types a replacement', async () => {
    vi.mocked(updateCamera).mockResolvedValue(camera);
    render(camera);
    setInput('rtsp_url', 'rtsp://new-address/stream');

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

    await act(async () => findButton('연결').click());

    expect(testCamera).toHaveBeenCalledWith('cam-1');
    expect(successSpy).toHaveBeenCalledWith('연결 성공');
    expect(onUpdated).toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });

  it('shows a failure message when the connection test fails', async () => {
    vi.mocked(testCamera).mockResolvedValue({ ok: false, error_class: 'timeout' });
    render(camera);

    await act(async () => findButton('연결').click());

    expect(document.querySelector('[role="alert"]')?.textContent).toContain('연결 테스트에 실패했습니다');
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

    await act(async () => findButton('인식 시작').click());
    expect(onUpdated).toHaveBeenCalled();

    act(() => findButton('완료').click());
    expect(document.body.textContent).toContain('인식 완료');
  });
});
