import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { browseClipStorage, saveClipStorageLocation } from '@/shared/api/client';
import { ClipStorageCard } from '@/features/settings/ClipStorageCard';
import { toast } from '@/shared/ui/Toast';
import type { ClipStorageBrowseResult, ClipStorageInfo } from '@/shared/api/client';
import type { PollingResource } from '@/shared/api/usePollingResource';

vi.mock('@/shared/api/client', async () => {
  const { withOverrides } = await vi.importActual<typeof import('@/test/moduleMock')>('@/test/moduleMock');
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return withOverrides(actual, { saveClipStorageLocation: vi.fn(), browseClipStorage: vi.fn() });
});

const storage: ClipStorageInfo = {
  mount_label: 'clip-store',
  selected_path: '',
  total_bytes: 10 * 1024 ** 3,
  used_bytes: 2 * 1024 ** 3,
  used_pct: 20,
};

const rootListing: ClipStorageBrowseResult = {
  path: '',
  parent: null,
  directories: [{ name: 'cam-101a', path: 'cam-101a' }],
};

function makeResource(overrides: Partial<PollingResource<ClipStorageInfo>> = {}): PollingResource<ClipStorageInfo> {
  return {
    status: 'success',
    data: storage,
    error: null,
    lastSuccessAt: Date.now(),
    refreshing: false,
    retry: vi.fn(),
    replace: vi.fn(),
    ...overrides,
  };
}

function render(resource: PollingResource<ClipStorageInfo> = makeResource()) {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<ClipStorageCard resource={resource} />));
  return { host, root, resource };
}

function findButton(container: ParentNode, label: string): HTMLButtonElement {
  const button = Array.from(container.querySelectorAll('button')).find((candidate) => candidate.textContent === label);
  if (!button) throw new Error(`missing button ${label}`);
  return button;
}

beforeEach(() => {
  vi.mocked(saveClipStorageLocation).mockReset();
  vi.mocked(browseClipStorage).mockReset();
  vi.mocked(browseClipStorage).mockResolvedValue(rootListing);
});

afterEach(() => {
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

describe('ClipStorageCard', () => {
  it('shows a loading message before the first successful fetch', () => {
    const { host, root } = render(makeResource({ status: 'loading', data: null }));
    expect(host.textContent).toContain('불러오는 중입니다');
    act(() => root.unmount());
  });

  it('shows a retry affordance when the initial fetch fails', () => {
    const { host, root, resource } = render(makeResource({ status: 'error', data: null }));
    act(() => findButton(host, '다시 시도').click());
    expect(resource.retry).toHaveBeenCalled();
    act(() => root.unmount());
  });

  it('renders the usage gauge and current path', () => {
    const { host, root } = render();
    expect(host.textContent).toContain('2.0 / 10.0 GB');
    expect(host.textContent).toContain('clip-store');
    const bar = host.querySelector('[role="progressbar"]');
    expect(bar?.getAttribute('aria-valuenow')).toBe('20');
    act(() => root.unmount());
  });

  it('shows the joined subdirectory path when a subdirectory is selected', () => {
    const nested: ClipStorageInfo = { ...storage, selected_path: 'cam-101a' };
    const { host, root } = render(makeResource({ data: nested }));
    expect(host.textContent).toContain('clip-store/cam-101a');
    act(() => root.unmount());
  });

  it('opens the folder browser modal from the 변경 button', async () => {
    const { host, root } = render();
    await act(async () => findButton(host, '변경').click());
    await act(async () => Promise.resolve());

    expect(browseClipStorage).toHaveBeenCalledWith('');
    expect(document.querySelector('[role="dialog"]')).not.toBeNull();
    act(() => root.unmount());
  });

  it('saves the selected location, refreshes, and toasts on success', async () => {
    vi.mocked(saveClipStorageLocation).mockResolvedValue({ ...storage, selected_path: 'cam-101a' });
    const successSpy = vi.spyOn(toast, 'success');
    const { host, root, resource } = render();

    await act(async () => findButton(host, '변경').click());
    await act(async () => Promise.resolve());
    await act(async () => findButton(document.body, '이 위치 사용').click());

    expect(saveClipStorageLocation).toHaveBeenCalledWith('');
    expect(resource.retry).toHaveBeenCalled();
    expect(successSpy).toHaveBeenCalledWith('저장 위치를 변경했습니다.');
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    act(() => root.unmount());
  });

  it('shows an error toast and keeps the modal open when saving fails', async () => {
    vi.mocked(saveClipStorageLocation).mockRejectedValue(new Error('boom'));
    const errorSpy = vi.spyOn(toast, 'error');
    const { host, root } = render();

    await act(async () => findButton(host, '변경').click());
    await act(async () => Promise.resolve());
    await act(async () => findButton(document.body, '이 위치 사용').click());

    expect(errorSpy).toHaveBeenCalledWith('저장 위치를 변경하지 못했습니다.');
    expect(document.querySelector('[role="dialog"]')).not.toBeNull();
    act(() => root.unmount());
  });
});
