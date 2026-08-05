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

  it('renders each camera with its label, masked RTSP url, floor and status badges in the 4-column layout', () => {
    const { host, root } = render([onlineCamera, offlineCamera]);

    const headers = Array.from(host.querySelectorAll('th')).map((th) => th.textContent);
    expect(headers).toEqual(['카메라', '층', '상태', '작업']);
    expect(host.textContent).not.toContain('침대 영역');

    expect(host.textContent).toContain('101호');
    expect(host.textContent).toContain('rtsp://***@10.0.0.5/stream');
    expect(host.textContent).toContain('1층');
    expect(host.textContent).toContain('온라인');

    expect(host.textContent).toContain('102호');
    expect(host.textContent).toContain('미지정');
    expect(host.textContent).toContain('오프라인');

    act(() => root.unmount());
  });

  it('prefers the user-set floor override over the read-only floor_name (issue #85 precedence), rendering the fixed label at display time (issue #155)', () => {
    const overriddenCamera: Camera = { ...onlineCamera, id: 'cam-3', floor_name: '1층', floor: 3 };
    const { host, root } = render([overriddenCamera]);

    const floorCell = host.querySelector('tbody tr td:nth-child(2)');
    expect(floorCell?.textContent).toBe('3층');

    act(() => root.unmount());
  });

  it('renders a basement floor override as B<n>, not a negative number', () => {
    const basementCamera: Camera = { ...onlineCamera, id: 'cam-4', floor_name: null, floor: -1 };
    const { host, root } = render([basementCamera]);

    const floorCell = host.querySelector('tbody tr td:nth-child(2)');
    expect(floorCell?.textContent).toBe('B1');

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
