import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ClipPlayerModal } from '@/features/operations/ClipPlayerModal';
import type { Clip } from '@/shared/api/client';

const baseClip: Clip = {
  id: 'clip-1',
  camera_id: 'cam-1',
  camera_label: '301호',
  event_type: 'fall',
  created_at: '2026-08-02T03:12:00Z',
  video_path: '/api/v1/clips/clip-1/video',
  video_available: true,
  video_error: null,
};

function render(clip: Clip | null, onClose = vi.fn()) {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<ClipPlayerModal clip={clip} cameraLabel="301호" onClose={onClose} />));
  return { host, root, onClose };
}

function dialog(): HTMLElement {
  return document.querySelector('[role="dialog"]') as HTMLElement;
}

afterEach(() => {
  document.body.innerHTML = '';
});

describe('ClipPlayerModal', () => {
  it('renders at the 720px design-spec width', () => {
    render(baseClip);
    expect(dialog().getAttribute('data-size')).toBe('xl');
  });

  it('omits the 크기 row when size_bytes is not present, never fabricating a value', () => {
    render(baseClip);
    expect(dialog().textContent).not.toContain('크기');
  });

  it('shows the 크기 row formatted in human-readable units when size_bytes is present', () => {
    render({ ...baseClip, size_bytes: 8_400_000 });
    expect(dialog().textContent).toContain('크기');
    expect(dialog().textContent).toContain('8.4 MB');
  });

  it('prefers the manifest duration_s for 길이 over the "측정 중" pending state', () => {
    render({ ...baseClip, duration_s: 12 });
    const rows = Array.from(dialog().querySelectorAll('dt'));
    const durationIndex = rows.findIndex((dt) => dt.textContent === '길이');
    expect(rows[durationIndex]?.nextElementSibling?.textContent).toBe('0:12');
  });

  it('shows the pending state for 길이 when duration_s is absent and metadata has not loaded', () => {
    render(baseClip);
    const rows = Array.from(dialog().querySelectorAll('dt'));
    const durationIndex = rows.findIndex((dt) => dt.textContent === '길이');
    expect(rows[durationIndex]?.nextElementSibling?.textContent).toBe('측정 중');
  });
});
