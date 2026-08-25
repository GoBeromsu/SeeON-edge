import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { deleteCamera } from '@/shared/api/client';
import { DeleteCameraDialog } from '@/features/settings/DeleteCameraDialog';
import { toast } from '@/shared/ui/Toast';
import type { Camera } from '@/shared/api/client';

vi.mock('@/shared/api/client', async () => {
  const { withOverrides } = await vi.importActual<typeof import('@/test/moduleMock')>('@/test/moduleMock');
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return withOverrides(actual, { deleteCamera: vi.fn() });
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

// AccessibleDialog portals into document.body, so assertions and lookups target the document, not the render host.
function render(cam: Camera | null, onClose = vi.fn(), onDeleted = vi.fn()) {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<DeleteCameraDialog camera={cam} onClose={onClose} onDeleted={onDeleted} />));
  return { host, root, onClose, onDeleted };
}

function findButton(label: string): HTMLButtonElement {
  const button = Array.from(document.querySelectorAll('button')).find((candidate) => candidate.textContent === label);
  if (!button) throw new Error(`missing button ${label}`);
  return button;
}

beforeEach(() => {
  vi.mocked(deleteCamera).mockReset();
  vi.mocked(deleteCamera).mockResolvedValue(undefined);
});

afterEach(() => {
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

describe('DeleteCameraDialog', () => {
  it('renders nothing when no camera is targeted', () => {
    render(null);
    expect(document.querySelector('[role="dialog"]')).toBeNull();
  });

  it('shows the camera label in the confirmation copy', () => {
    render(camera);
    expect(document.querySelector('[role="dialog"]')?.textContent).toContain('101호');
  });

  it('calls onClose without deleting when 취소 is clicked', () => {
    const { onClose } = render(camera);
    act(() => findButton('취소').click());
    expect(deleteCamera).not.toHaveBeenCalled();
    expect(onClose).toHaveBeenCalled();
  });

  it('deletes the camera, toasts success, and notifies the parent on confirm', async () => {
    const successSpy = vi.spyOn(toast, 'success');
    const { onDeleted } = render(camera);

    await act(async () => findButton('삭제').click());

    expect(deleteCamera).toHaveBeenCalledWith('cam-1');
    expect(successSpy).toHaveBeenCalledWith('카메라를 삭제했습니다.');
    expect(onDeleted).toHaveBeenCalled();
  });

  it('shows an inline error and keeps the dialog open when deletion fails', async () => {
    vi.mocked(deleteCamera).mockRejectedValue(new Error('boom'));
    const { onDeleted } = render(camera);

    await act(async () => findButton('삭제').click());

    expect(document.querySelector('[role="alert"]')?.textContent).toContain('삭제하지 못했습니다');
    expect(onDeleted).not.toHaveBeenCalled();
  });
});
