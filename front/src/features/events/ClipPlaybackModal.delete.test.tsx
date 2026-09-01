import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ClipPlaybackModal } from '@/features/events/ClipPlaybackModal';
import { deleteClip, fetchClipArtifacts } from '@/shared/api/client';
import type { Clip, ClipArtifacts } from '@/shared/api/types';

vi.mock('@/shared/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return { ...actual, deleteClip: vi.fn(), fetchClipArtifacts: vi.fn() };
});

const activeRoots = new Set<ReturnType<typeof createRoot>>();

const baseClip: Clip = {
  id: 'clip-a',
  camera_id: 'cam-1',
  camera_label: '301호',
  event_type: 'fall',
  created_at: '2026-08-02T03:12:00Z',
  video_path: '/api/v1/clips/clip-a/video',
  video_available: true,
  thumbnail_available: true,
  video_error: null,
  scene_available: false,
  scene_frame_count: null,
};

const baseArtifacts: ClipArtifacts = {
  clip_id: 'clip-a',
  clean: 'AVAILABLE',
  snapshot: 'AVAILABLE',
};

function render(clip: Clip | null, onDeleted = vi.fn(), onClose = vi.fn()) {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  activeRoots.add(root);
  act(() => root.render(
    <ClipPlaybackModal
      clip={clip}
      cameraLabel="301호"
      open
      onClose={onClose}
      lookupStatus="success"
      onRetry={vi.fn()}
      onDeleted={onDeleted}
    />,
  ));
  return { host, root, onDeleted, onClose };
}

function dialogs(): HTMLElement[] {
  return Array.from(document.querySelectorAll('[role="dialog"]'));
}

function typeConfirm(input: HTMLInputElement, value: string): void {
  act(() => {
    const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set;
    valueSetter?.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

function findButton(root: ParentNode, label: string): HTMLButtonElement {
  const button = Array.from(root.querySelectorAll('button')).find((candidate) => candidate.textContent === label);
  if (!button) throw new Error(`missing button ${label}`);
  return button;
}

beforeEach(() => {
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
  vi.mocked(fetchClipArtifacts).mockResolvedValue(baseArtifacts);
  vi.mocked(deleteClip).mockReset();
});

afterEach(() => {
  act(() => activeRoots.forEach((root) => root.unmount()));
  activeRoots.clear();
  document.body.innerHTML = '';
  vi.restoreAllMocks();
});

describe('ClipPlaybackModal clip deletion', () => {
  it('opens an exact-match confirmation dialog showing the clip id', async () => {
    render(baseClip);
    await act(async () => Promise.resolve());

    act(() => findButton(document.body, '클립 삭제').click());

    const confirmDialog = dialogs().find((dialog) => dialog.textContent?.includes('클립 삭제 확인'));
    expect(confirmDialog).toBeDefined();
    expect(document.querySelector('[data-testid="delete-confirm-clip-id"]')?.textContent).toBe('clip-a');
    expect(deleteClip).not.toHaveBeenCalled();
  });

  it('keeps the confirm button disabled until the exact clip id is typed', async () => {
    render(baseClip);
    await act(async () => Promise.resolve());
    act(() => findButton(document.body, '클립 삭제').click());
    const confirmDialog = dialogs().find((dialog) => dialog.textContent?.includes('클립 삭제 확인')) as HTMLElement;
    const input = confirmDialog.querySelector('#delete-confirm-input') as HTMLInputElement;
    const confirmButton = findButton(confirmDialog, '삭제');

    expect(confirmButton.disabled).toBe(true);
    typeConfirm(input, 'wrong-id');
    expect(confirmButton.disabled).toBe(true);
    typeConfirm(input, 'clip-a');
    expect(confirmButton.disabled).toBe(false);
  });

  it('cancel closes the dialog without calling the API', async () => {
    render(baseClip);
    await act(async () => Promise.resolve());
    act(() => findButton(document.body, '클립 삭제').click());
    const confirmDialog = dialogs().find((dialog) => dialog.textContent?.includes('클립 삭제 확인')) as HTMLElement;

    act(() => findButton(confirmDialog, '취소').click());

    expect(dialogs().some((dialog) => dialog.textContent?.includes('클립 삭제 확인'))).toBe(false);
    expect(deleteClip).not.toHaveBeenCalled();
  });

  it('sends the exact clip id, disables playback, and notifies the parent on accepted (PURGED) deletion', async () => {
    vi.mocked(deleteClip).mockResolvedValue({ clip_id: 'clip-a', status: 'PURGED' });
    const { onDeleted } = render(baseClip);
    await act(async () => Promise.resolve());
    act(() => findButton(document.body, '클립 삭제').click());
    const confirmDialog = dialogs().find((dialog) => dialog.textContent?.includes('클립 삭제 확인')) as HTMLElement;
    const input = confirmDialog.querySelector('#delete-confirm-input') as HTMLInputElement;
    typeConfirm(input, 'clip-a');

    await act(async () => findButton(confirmDialog, '삭제').click());

    expect(deleteClip).toHaveBeenCalledWith('clip-a', 'clip-a');
    expect(dialogs().some((dialog) => dialog.textContent?.includes('클립 삭제 확인'))).toBe(false);
    expect(document.querySelector('video')).toBeNull();
    expect(document.querySelector('[data-testid="clip-modal-unavailable"]')?.textContent).toContain('삭제되어');
    expect(document.querySelector('[data-testid="clip-delete-status"]')?.textContent).toContain('삭제했습니다');
    expect(onDeleted).toHaveBeenCalled();
    expect(document.querySelector('[aria-label="클립 삭제"]')).toBeNull();
    expect(document.querySelector('[aria-label="증거 보기 선택"]')).toBeNull();
    expect(document.querySelector('[aria-label="파생 증거 제어"]')).toBeNull();
    expect(document.querySelector('[aria-label="적용 실행 증명"]')).toBeNull();
  });

  it('requests only the slimmed artifacts contract and renders no retired evidence control', async () => {
    render(baseClip);
    await act(async () => Promise.resolve());

    expect(fetchClipArtifacts).toHaveBeenCalledWith('clip-a');
    // The clip route surface is clip identity + clean media + optional snapshot; nothing else.
    expect(document.querySelector('[aria-label="증거 보기 선택"]')).toBeNull();
    expect(document.querySelector('[aria-label="파생 증거 제어"]')).toBeNull();
    expect(document.querySelector('[aria-label="적용 실행 증명"]')).toBeNull();
    const videos = document.querySelectorAll('video');
    expect(videos).toHaveLength(1);
    expect(videos[0]?.getAttribute('src')).toBe('/api/v1/clips/clip-a/video');
    expect(document.querySelector('[data-testid="clip-artifact-status"]')).not.toBeNull();
  });

  it('keeps the bounded artifact-unknown state when the artifacts read fails', async () => {
    vi.mocked(fetchClipArtifacts).mockRejectedValue(new Error('artifacts unavailable'));
    render(baseClip);
    await act(async () => Promise.resolve());

    expect(document.querySelector('[data-testid="clip-artifact-status"]')).not.toBeNull();
    expect(document.querySelectorAll('video')).toHaveLength(1);
    expect(document.querySelector('[aria-label="파생 증거 제어"]')).toBeNull();
  });

  it.each([
    ['MISSING', '찾을 수 없습니다'],
    ['UNVERIFIABLE', '확인할 수 없어'],
    ['DELETE_FAILED', '삭제에 실패'],
    ['VERIFICATION_FAILED', '삭제 확인에 실패'],
  ] as const)('shows terminal %s feedback outside the confirmation dialog without notifying the parent', async (status, copy) => {
    vi.mocked(deleteClip).mockResolvedValue({ clip_id: 'clip-a', status });
    const { onDeleted } = render(baseClip);
    await act(async () => Promise.resolve());
    act(() => findButton(document.body, '클립 삭제').click());
    const confirmDialog = dialogs().find((dialog) => dialog.textContent?.includes('클립 삭제 확인')) as HTMLElement;
    const input = confirmDialog.querySelector('#delete-confirm-input') as HTMLInputElement;
    typeConfirm(input, 'clip-a');

    await act(async () => findButton(confirmDialog, '삭제').click());

    expect(dialogs().some((dialog) => dialog.textContent?.includes('클립 삭제 확인'))).toBe(false);
    expect(document.querySelector('[data-testid="clip-delete-status"]')?.textContent).toContain(copy);
    expect(onDeleted).not.toHaveBeenCalled();
    expect(document.querySelector('video')).not.toBeNull();
  });

  it('shows a HELD message, keeps playback available, and does not notify the parent', async () => {
    vi.mocked(deleteClip).mockResolvedValue({ clip_id: 'clip-a', status: 'HELD' });
    const { onDeleted } = render(baseClip);
    await act(async () => Promise.resolve());
    act(() => findButton(document.body, '클립 삭제').click());
    const confirmDialog = dialogs().find((dialog) => dialog.textContent?.includes('클립 삭제 확인')) as HTMLElement;
    const input = confirmDialog.querySelector('#delete-confirm-input') as HTMLInputElement;
    typeConfirm(input, 'clip-a');

    await act(async () => findButton(confirmDialog, '삭제').click());

    expect(document.querySelector('video')).not.toBeNull();
    expect(document.querySelector('[data-testid="clip-delete-status"]')?.textContent).toContain('다시 시도');
    expect(onDeleted).not.toHaveBeenCalled();
    expect(findButton(document.body, '클립 삭제')).toBeDefined();
  });

  it('keeps the parent open when Escape is pressed during an in-flight confirmation', async () => {
    let resolveDelete: (value: { clip_id: string; status: 'PURGED' }) => void = () => undefined;
    vi.mocked(deleteClip).mockReturnValue(new Promise((resolve) => { resolveDelete = resolve; }));
    const { onClose } = render(baseClip);
    await act(async () => Promise.resolve());
    act(() => findButton(document.body, '클립 삭제').click());
    const confirmDialog = dialogs().find((dialog) => dialog.textContent?.includes('클립 삭제 확인')) as HTMLElement;
    typeConfirm(confirmDialog.querySelector('#delete-confirm-input') as HTMLInputElement, 'clip-a');
    act(() => findButton(confirmDialog, '삭제').click());
    act(() => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));

    expect(onClose).not.toHaveBeenCalled();
    expect(dialogs()).toHaveLength(2);
    await act(async () => { resolveDelete({ clip_id: 'clip-a', status: 'PURGED' }); });
  });

  it('shows a network-failure message and keeps the confirmation dialog open on transport error', async () => {
    vi.mocked(deleteClip).mockRejectedValue(new Error('network down'));
    render(baseClip);
    await act(async () => Promise.resolve());
    act(() => findButton(document.body, '클립 삭제').click());
    const confirmDialog = dialogs().find((dialog) => dialog.textContent?.includes('클립 삭제 확인')) as HTMLElement;
    const input = confirmDialog.querySelector('#delete-confirm-input') as HTMLInputElement;
    typeConfirm(input, 'clip-a');

    await act(async () => findButton(confirmDialog, '삭제').click());

    expect(dialogs().some((dialog) => dialog.textContent?.includes('클립 삭제 확인'))).toBe(true);
    expect(document.querySelector('[role="alert"]')?.textContent).toContain('보내지 못했습니다');
  });

  it('ignores a stale fetchClipArtifacts response that resolves after an accepted deletion', async () => {
    let resolveArtifacts: (value: ClipArtifacts) => void = () => undefined;
    vi.mocked(fetchClipArtifacts).mockReturnValue(new Promise((resolve) => { resolveArtifacts = resolve; }));
    vi.mocked(deleteClip).mockResolvedValue({ clip_id: 'clip-a', status: 'PURGED' });
    render(baseClip);
    await act(async () => Promise.resolve());
    act(() => findButton(document.body, '클립 삭제').click());
    const confirmDialog = dialogs().find((dialog) => dialog.textContent?.includes('클립 삭제 확인')) as HTMLElement;
    const input = confirmDialog.querySelector('#delete-confirm-input') as HTMLInputElement;
    typeConfirm(input, 'clip-a');
    await act(async () => findButton(confirmDialog, '삭제').click());

    // The artifacts request that was already in flight when the delete was accepted resolves late.
    await act(async () => { resolveArtifacts(baseArtifacts); await Promise.resolve(); });

    expect(document.querySelector('video')).toBeNull();
    expect(document.querySelector('[data-testid="clip-modal-unavailable"]')?.textContent).toContain('삭제되어');
  });
});
