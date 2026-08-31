import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ClipPlaybackModal } from '@/features/events/ClipPlaybackModal';
import type { Clip } from '@/shared/api/types';

vi.mock('@/shared/api/client', async (importOriginal) => ({ ...(await importOriginal<typeof import('@/shared/api/client')>()), fetchClipScene: vi.fn().mockResolvedValue(null) }));
const roots = new Set<ReturnType<typeof createRoot>>();
afterEach(() => { roots.forEach((root) => root.unmount()); roots.clear(); document.body.replaceChildren(); vi.restoreAllMocks(); });

const clip: Clip = { id: 'clip-1', camera_id: 'cam-1', camera_label: '301호', event_type: 'fall', created_at: null, video_path: '/api/v1/clips/clip-1/video', video_available: true, thumbnail_available: true, video_error: null, scene_available: true, scene_frame_count: 1 };
function render(value: Clip): HTMLElement {
  const host = document.createElement('div'); document.body.append(host); const root = createRoot(host); roots.add(root);
  act(() => root.render(<ClipPlaybackModal clip={value} cameraLabel="301호" open lookupStatus="success" onClose={vi.fn()} onRetry={vi.fn()} onDeleted={vi.fn()} />));
  return host;
}

describe('ClipPlaybackModal scene controls', () => {
  it('uses the jsdom timeupdate plus rAF fallback and allows analysis display to be toggled', () => {
    const raf = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(() => 1);
    const host = render(clip);
    const checkbox = document.querySelector('input[type=checkbox]') as HTMLInputElement;
    expect(checkbox.checked).toBe(true);
    act(() => checkbox.click());
    expect(checkbox.checked).toBe(false);
    const video = document.querySelector('video')!;
    act(() => { video.dispatchEvent(new Event('loadedmetadata')); });
    act(() => { video.dispatchEvent(new Event('timeupdate')); });
    expect(raf).toHaveBeenCalled();
  });

  it('does not expose analysis controls without a sidecar', () => {
    const host = render({ ...clip, scene_available: false, scene_frame_count: null });
    expect(document.querySelector('input[type=checkbox]')).toBeNull();
  });
});
