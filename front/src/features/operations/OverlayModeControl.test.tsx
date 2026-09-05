import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { OverlaySelectionControl } from '@/features/operations/OverlayModeControl';
import { toast } from '@/shared/ui/Toast';
import type { OverlaySelection } from '@/shared/api/client';

type MockResponse = { ok: boolean; status: number; selection?: OverlaySelection };

function installFetchMock(initial: MockResponse): {
  fetchMock: ReturnType<typeof vi.fn>;
  setPostResult: (result: MockResponse) => void;
} {
  let postResult: MockResponse = { ok: true, status: 200, selection: { person: true, bed: true } };
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input.toString();
    if (!url.includes('/streams/') || !url.includes('/pose')) {
      return Promise.reject(new Error(`unexpected fetch: ${url}`));
    }
    const response = init?.method === 'POST' ? postResult : initial;
    return Promise.resolve({
      ok: response.ok,
      status: response.status,
      json: async () => response.selection ?? {},
    });
  });
  vi.stubGlobal('fetch', fetchMock);
  return { fetchMock, setPostResult: (result) => { postResult = result; } };
}

async function flush(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  });
}

function render(onSelectionChange?: (selection: OverlaySelection | null) => void): { host: HTMLDivElement; root: Root } {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<OverlaySelectionControl cameraId="cam-1" onSelectionChange={onSelectionChange} />));
  return { host, root };
}

function toggle(host: HTMLElement, label: '사람' | '침대'): HTMLButtonElement {
  return Array.from(host.querySelectorAll<HTMLButtonElement>('button')).find((button) => button.textContent === label)!;
}

afterEach(() => {
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('OverlaySelectionControl', () => {
  it('shows both server-default subjects as enabled icon toggles on entry', async () => {
    installFetchMock({ ok: true, status: 200, selection: { person: true, bed: true } });
    const { host } = render();
    await flush();

    const person = toggle(host, '사람');
    const bed = toggle(host, '침대');
    expect(person.getAttribute('aria-pressed')).toBe('true');
    expect(bed.getAttribute('aria-pressed')).toBe('true');
    expect(person.querySelector('[data-overlay-target="person"]')).not.toBeNull();
    expect(bed.querySelector('[data-overlay-target="bed"]')).not.toBeNull();
  });

  it('renders both independently pressed selections from the confirmed GET response', async () => {
    installFetchMock({ ok: true, status: 200, selection: { person: true, bed: false } });
    const onSelectionChange = vi.fn();
    const { host } = render(onSelectionChange);
    await flush();

    expect(toggle(host, '사람').getAttribute('aria-pressed')).toBe('true');
    expect(toggle(host, '침대').getAttribute('aria-pressed')).toBe('false');
    expect(onSelectionChange).toHaveBeenLastCalledWith({ person: true, bed: false });
  });

  it('toggles one target without changing the other and sends both required fields', async () => {
    const { fetchMock, setPostResult } = installFetchMock({ ok: true, status: 200, selection: { person: true, bed: true } });
    setPostResult({ ok: true, status: 200, selection: { person: false, bed: true } });
    const { host } = render();
    await flush();

    await act(async () => {
      toggle(host, '사람').click();
      await Promise.resolve();
    });
    await flush();

    expect(fetchMock).toHaveBeenLastCalledWith('/api/v1/streams/cam-1/pose', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ person: false, bed: true }),
    }));
    expect(toggle(host, '사람').getAttribute('aria-pressed')).toBe('false');
    expect(toggle(host, '침대').getAttribute('aria-pressed')).toBe('true');
  });

  it('keeps the confirmed selection and shows an error toast when an update fails', async () => {
    const toastError = vi.spyOn(toast, 'error').mockImplementation(() => undefined);
    const { setPostResult } = installFetchMock({ ok: true, status: 200, selection: { person: false, bed: true } });
    setPostResult({ ok: false, status: 500 });
    const { host } = render();
    await flush();

    await act(async () => {
      toggle(host, '사람').click();
      await Promise.resolve();
    });
    await flush();

    expect(toastError).toHaveBeenCalledTimes(1);
    expect(toggle(host, '사람').getAttribute('aria-pressed')).toBe('false');
    expect(toggle(host, '침대').getAttribute('aria-pressed')).toBe('true');
    expect(toggle(host, '사람').disabled).toBe(false);
  });
});
