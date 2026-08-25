import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { browseClipStorage } from '@/shared/api/client';
import { FolderBrowserModal } from '@/features/settings/FolderBrowserModal';
import type { ClipStorageBrowseResult } from '@/shared/api/client';

vi.mock('@/shared/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return withOverrides(actual, { browseClipStorage: vi.fn() });
});

const rootListing: ClipStorageBrowseResult = {
  path: '/mnt/data/clips',
  parent: null,
  directories: [{ name: 'cam-101a', path: '/mnt/data/clips/cam-101a' }],
};

const childListing: ClipStorageBrowseResult = {
  path: '/mnt/data/clips/cam-101a',
  parent: '/mnt/data/clips',
  directories: [],
};

function render(open = true, initialPath = '/mnt/data/clips', onClose = vi.fn(), onSelect = vi.fn()) {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(
    <FolderBrowserModal open={open} initialPath={initialPath} onClose={onClose} onSelect={onSelect} />,
  ));
  return { host, root, onClose, onSelect };
}

function findButton(label: string): HTMLButtonElement {
  const button = Array.from(document.querySelectorAll('button')).find((candidate) => candidate.textContent === label);
  if (!button) throw new Error(`missing button ${label}`);
  return button;
}

beforeEach(() => {
  vi.mocked(browseClipStorage).mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

describe('FolderBrowserModal', () => {
  it('renders at the 420px design-spec width', async () => {
    vi.mocked(browseClipStorage).mockResolvedValue(rootListing);
    render();
    await act(async () => Promise.resolve());

    expect(document.querySelector('[role="dialog"]')?.getAttribute('data-size')).toBe('sm');
  });

  it('loads the initial path listing on open', async () => {
    vi.mocked(browseClipStorage).mockResolvedValue(rootListing);
    render();
    await act(async () => Promise.resolve());

    expect(browseClipStorage).toHaveBeenCalledWith('/mnt/data/clips');
    expect(document.body.textContent).toContain('cam-101a');
  });

  it('drills into a subfolder when a directory row is clicked', async () => {
    vi.mocked(browseClipStorage).mockResolvedValueOnce(rootListing).mockResolvedValueOnce(childListing);
    render();
    await act(async () => Promise.resolve());

    await act(async () => findButton('cam-101a').click());

    expect(browseClipStorage).toHaveBeenLastCalledWith('/mnt/data/clips/cam-101a');
    expect(document.body.textContent).toContain('하위 폴더 없음');
  });

  it('navigates back up via the parent button', async () => {
    vi.mocked(browseClipStorage).mockResolvedValueOnce(rootListing).mockResolvedValueOnce(childListing).mockResolvedValueOnce(rootListing);
    render();
    await act(async () => Promise.resolve());
    await act(async () => findButton('cam-101a').click());

    await act(async () => document.querySelector<HTMLButtonElement>('[aria-label="상위 폴더로 이동"]')?.click());

    expect(browseClipStorage).toHaveBeenLastCalledWith('/mnt/data/clips');
  });

  it('disables the up button when there is no parent', async () => {
    vi.mocked(browseClipStorage).mockResolvedValue(rootListing);
    render();
    await act(async () => Promise.resolve());

    expect(document.querySelector<HTMLButtonElement>('[aria-label="상위 폴더로 이동"]')?.disabled).toBe(true);
  });

  it('shows an error state when the listing request fails', async () => {
    vi.mocked(browseClipStorage).mockRejectedValue(new Error('boom'));
    render();
    await act(async () => Promise.resolve());

    expect(document.querySelector('[role="alert"]')?.textContent).toContain('불러오지 못했습니다');
  });

  it('calls onSelect with the current listing path when 이 위치 사용 is clicked', async () => {
    vi.mocked(browseClipStorage).mockResolvedValue(rootListing);
    const { onSelect } = render();
    await act(async () => Promise.resolve());

    act(() => findButton('이 위치 사용').click());

    expect(onSelect).toHaveBeenCalledWith('/mnt/data/clips');
  });

  it('resets to the initial path the next time it is reopened', async () => {
    vi.mocked(browseClipStorage).mockImplementation(async (requestedPath) =>
      requestedPath === childListing.path ? childListing : rootListing);
    const { root, onClose, onSelect } = render(true, '/mnt/data/clips');
    await act(async () => Promise.resolve());
    await act(async () => findButton('cam-101a').click());

    act(() => root.render(<FolderBrowserModal open={false} initialPath="/mnt/data/clips" onClose={onClose} onSelect={onSelect} />));
    act(() => root.render(<FolderBrowserModal open initialPath="/mnt/data/clips" onClose={onClose} onSelect={onSelect} />));
    await act(async () => Promise.resolve());

    expect(browseClipStorage).toHaveBeenLastCalledWith('/mnt/data/clips');
  });
});
