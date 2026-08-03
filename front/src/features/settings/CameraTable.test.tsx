import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { describe, expect, it, vi } from 'vitest';
import { CameraTable } from '@/features/settings/CameraTable';
import type { Camera } from '@/shared/api/client';

const onlineCamera: Camera = {
  id: 'cam-1',
  label: '101호',
  rtsp_url_masked: 'rtsp://***@10.0.0.5/stream',
  floor_name: '1층',
  status: 'online',
  created_at: null,
  bed_zone: {
    polygon: [[0, 0], [10, 0], [10, 10], [0, 10]],
    image_width: 1920,
    image_height: 1080,
    recognized_at: '2026-08-01T00:00:00Z',
  },
};

const offlineCamera: Camera = {
  id: 'cam-2',
  label: '102호',
  rtsp_url_masked: 'rtsp://***@10.0.0.6/stream',
  floor_name: null,
  status: 'offline',
  created_at: null,
  bed_zone: null,
};

function render(cameras: Camera[], onEdit = vi.fn(), onRequestDelete = vi.fn()) {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<CameraTable cameras={cameras} onEdit={onEdit} onRequestDelete={onRequestDelete} />));
  return { host, root, onEdit, onRequestDelete };
}

describe('CameraTable', () => {
  it('shows an empty-state message when there are no cameras', () => {
    const { host, root } = render([]);
    expect(host.textContent).toContain('등록된 카메라가 없습니다.');
    act(() => root.unmount());
  });

  it('renders each camera with its label, masked RTSP url, floor, bed-zone and status badges', () => {
    const { host, root } = render([onlineCamera, offlineCamera]);

    expect(host.textContent).toContain('101호');
    expect(host.textContent).toContain('rtsp://***@10.0.0.5/stream');
    expect(host.textContent).toContain('1층');
    expect(host.textContent).toContain('인식 완료');
    expect(host.textContent).toContain('온라인');

    expect(host.textContent).toContain('102호');
    expect(host.textContent).toContain('미지정');
    expect(host.textContent).toContain('인식 필요');
    expect(host.textContent).toContain('오프라인');

    act(() => root.unmount());
  });

  it('tints the offline camera row for visibility', () => {
    const { host, root } = render([onlineCamera, offlineCamera]);
    const rows = Array.from(host.querySelectorAll('tbody tr'));
    expect(rows[0]?.className).not.toContain('bg-status-rejectedBg');
    expect(rows[1]?.className).toContain('bg-status-rejectedBg');
    act(() => root.unmount());
  });

  it('invokes onEdit and onRequestDelete with the clicked camera', () => {
    const { host, root, onEdit, onRequestDelete } = render([onlineCamera]);
    const editButton = Array.from(host.querySelectorAll('button')).find((btn) => btn.textContent === '수정');
    const deleteButton = Array.from(host.querySelectorAll('button')).find((btn) => btn.textContent === '삭제');

    act(() => editButton?.click());
    act(() => deleteButton?.click());

    expect(onEdit).toHaveBeenCalledWith(onlineCamera);
    expect(onRequestDelete).toHaveBeenCalledWith(onlineCamera);
    act(() => root.unmount());
  });
});
