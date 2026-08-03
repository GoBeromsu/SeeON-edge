import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { OverlayModeControl } from '@/features/operations/OverlayModeControl';
import { toast } from '@/shared/ui/Toast';

function installFetchMock(overlayResponses: Array<{ ok: boolean; status: number; mode?: string }>): { fetchMock: ReturnType<typeof vi.fn>; setPostResult: (result: { ok: boolean; status: number; mode?: string }) => void } {
  let getIndex = 0;
  let postResult: { ok: boolean; status: number; mode?: string } = { ok: true, status: 200, mode: 'none' };
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input.toString();
    if (url.includes('/streams/') && url.includes('/pose')) {
      if (init?.method === 'POST') {
        return Promise.resolve({
          ok: postResult.ok,
          status: postResult.status,
          json: async () => (postResult.ok ? { mode: postResult.mode } : {}),
        });
      }
      const response = overlayResponses[Math.min(getIndex, overlayResponses.length - 1)];
      getIndex += 1;
      return Promise.resolve({
        ok: response.ok,
        status: response.status,
        json: async () => (response.ok ? { mode: response.mode } : {}),
      });
    }
    return Promise.reject(new Error(`unexpected fetch: ${url}`));
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

function render(cameraId = 'cam-1'): { host: HTMLDivElement; root: Root } {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<OverlayModeControl cameraId={cameraId} />));
  return { host, root };
}

afterEach(() => {
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('OverlayModeControl', () => {
  it('shows the overlay chips once the initial fetch resolves', async () => {
    installFetchMock([{ ok: true, status: 200, mode: 'none' }]);
    const { host } = render();
    await flush();

    expect(host.querySelector('[role="group"]')).not.toBeNull();
    expect(host.textContent).toContain('오버레이 없음');
  });

  it('shows an error state with a retry button when the initial fetch is rejected, and recovers on retry', async () => {
    const { fetchMock } = installFetchMock([
      { ok: false, status: 500 },
      { ok: true, status: 200, mode: 'bedexit' },
    ]);
    const { host } = render();
    await flush();

    expect(host.querySelector('[role="alert"]')?.textContent).toBe('오버레이 모드를 불러오지 못했습니다.');
    const retryButton = Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '다시 시도');
    expect(retryButton).not.toBeUndefined();

    await act(async () => {
      retryButton?.click();
      await Promise.resolve();
    });
    await flush();

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(host.querySelector('[role="alert"]')).toBeNull();
    expect(host.querySelector('[role="group"]')).not.toBeNull();
    expect(host.textContent).toContain('침대 이탈');
  });

  it('shows a toast and resets pending state when setCameraOverlay is rejected', async () => {
    const toastErrorSpy = vi.spyOn(toast, 'error').mockImplementation(() => undefined);
    const { setPostResult } = installFetchMock([{ ok: true, status: 200, mode: 'none' }]);
    setPostResult({ ok: false, status: 500 });
    const { host } = render();
    await flush();

    const fallButton = Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '낙상') as HTMLButtonElement;
    expect(fallButton).not.toBeUndefined();

    await act(async () => {
      fallButton.click();
      await Promise.resolve();
    });
    await flush();

    expect(toastErrorSpy).toHaveBeenCalledWith('오버레이 모드를 변경하지 못했습니다.');
    // Reverted: still shows the previously-confirmed mode ('none'), not the rejected selection, and re-enabled.
    expect(fallButton.getAttribute('aria-pressed')).toBe('false');
    expect(fallButton.disabled).toBe(false);
  });
});
