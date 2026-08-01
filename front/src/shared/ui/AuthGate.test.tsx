import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { fetchCameras } from '@/shared/api/client';
import { AuthGate, useAuthSession } from '@/shared/ui/AuthGate';

afterEach(() => {
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function okResponse(body: unknown = {}): { ok: true; status: number; json: () => Promise<unknown> } {
  return { ok: true, status: 200, json: async () => body };
}

function errorResponse(status: number): { ok: false; status: number; json: () => Promise<unknown> } {
  return { ok: false, status, json: async () => ({}) };
}

function LogoutProbe(): JSX.Element {
  const session = useAuthSession();
  return (
    <div>
      대시보드
      <button type="button" data-testid="logout" onClick={() => void session?.logout()}>로그아웃</button>
    </div>
  );
}

function renderGate(children: JSX.Element = <div>대시보드</div>): { host: HTMLDivElement; root: ReturnType<typeof createRoot> } {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<AuthGate>{children}</AuthGate>));
  return { host, root };
}

async function settle(): Promise<void> {
  await act(async () => {});
}

async function submit(host: HTMLElement, username: string, password: string): Promise<void> {
  const inputs = host.querySelectorAll('input');
  await act(async () => {
    for (const [input, value] of [[inputs[0], username], [inputs[1], password]] as const) {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set?.call(input, value);
      input.dispatchEvent(new Event('input', { bubbles: true }));
    }
    host.querySelector('form')?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
  });
}

describe('AuthGate', () => {
  it('shows checking copy while the initial GET session request is in flight', async () => {
    let resolveSession: ((value: unknown) => void) | undefined;
    const fetchMock = vi.fn().mockImplementation(() => new Promise((resolve) => {
      resolveSession = resolve;
    }));
    vi.stubGlobal('fetch', fetchMock);
    const { host, root } = renderGate();
    expect(host.textContent).toContain('로그인 상태를 확인하고 있습니다.');
    expect(host.querySelector('#login-title')).toBeNull();
    expect(host.textContent).not.toContain('대시보드');
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/auth/session', expect.objectContaining({
      credentials: 'same-origin',
    }));
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit | undefined;
    expect(init?.method ?? 'GET').toBe('GET');
    resolveSession?.(okResponse());
    await settle();
    expect(host.textContent).toContain('대시보드');
    act(() => root.unmount());
  });

  it('restores an existing server session without showing the login form', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okResponse());
    vi.stubGlobal('fetch', fetchMock);
    const { host, root } = renderGate();
    await settle();
    expect(host.textContent).toContain('대시보드');
    expect(host.querySelector('#login-title')).toBeNull();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/auth/session', expect.objectContaining({
      credentials: 'same-origin',
    }));
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit | undefined;
    expect(init?.method ?? 'GET').toBe('GET');
    act(() => root.unmount());
  });

  it('shows the login form when session restoration fails', async () => {
    const fetchMock = vi.fn().mockResolvedValue(errorResponse(401));
    vi.stubGlobal('fetch', fetchMock);
    const { host, root } = renderGate();
    await settle();
    expect(host.querySelector('#login-title')?.textContent).toBe('로그인');
    expect(host.querySelector('form')).not.toBeNull();
    expect(host.textContent).not.toContain('대시보드');
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/auth/session', expect.objectContaining({
      credentials: 'same-origin',
    }));
    act(() => root.unmount());
  });


  it('shows an unavailable state for operational session-probe failures and retries', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(errorResponse(503))
      .mockResolvedValueOnce(errorResponse(401));
    vi.stubGlobal('fetch', fetchMock);
    const { host, root } = renderGate();
    await settle();

    expect(host.textContent).toContain('로그인 서비스 연결 실패');
    expect(host.querySelector('#login-title')).toBeNull();
    await act(async () => {
      Array.from(host.querySelectorAll<HTMLButtonElement>('button'))
        .find((button) => button.textContent === '다시 시도')
        ?.click();
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(host.querySelector('#login-title')?.textContent).toBe('로그인');
    act(() => root.unmount());
  });

  it('shows required empty username/password fields without credential or token disclosure', async () => {
    const fetchMock = vi.fn().mockResolvedValue(errorResponse(401));
    vi.stubGlobal('fetch', fetchMock);
    const { host, root } = renderGate();
    await settle();
    expect(host.querySelectorAll('input')).toHaveLength(2);
    expect(host.querySelector('input[type="password"]')).not.toBeNull();
    expect(host.querySelector('input[name="token"]')).toBeNull();
    expect(Array.from(host.querySelectorAll('input')).every((input) => input.required && input.value === '')).toBe(true);
    expect(host.querySelector('button[type="submit"]')?.classList.contains('brand-action')).toBe(true);
    expect(host.textContent).not.toMatch(/token|토큰|relay/iu);
    act(() => root.unmount());
  });


  it('rejects empty credentials locally without calling the login endpoint', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(errorResponse(401));
    vi.stubGlobal('fetch', fetchMock);
    const { host, root } = renderGate();
    await settle();

    await submit(host, '', '');

    expect(host.textContent).toContain('아이디와 비밀번호를 입력해 주세요.');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    act(() => root.unmount());
  });

  it('keeps the dashboard hidden and reports credential error copy when login is rejected with 401', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(errorResponse(401))
      .mockResolvedValueOnce(errorResponse(401));
    vi.stubGlobal('fetch', fetchMock);
    const { host, root } = renderGate();
    await settle();
    await submit(host, 'operator', 'wrong-password');
    expect(host.textContent).toContain('아이디 또는 비밀번호가 올바르지 않습니다.');
    expect(host.textContent).not.toContain('로그인 서비스에 연결하지 못했습니다');
    expect(host.textContent).not.toContain('대시보드');
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/v1/auth/session', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ username: 'operator', password: 'wrong-password' }),
      credentials: 'same-origin',
    }));
    act(() => root.unmount());
  });

  it('reports operational error copy for 5xx and network login failures', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(errorResponse(401))
      .mockResolvedValueOnce(errorResponse(503))
      .mockRejectedValueOnce(new TypeError('network down'));
    vi.stubGlobal('fetch', fetchMock);
    const { host, root } = renderGate();
    await settle();

    await submit(host, 'operator', 'correct-password');
    expect(host.textContent).toContain('로그인 서비스에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.');
    expect(host.textContent).not.toContain('아이디 또는 비밀번호가 올바르지 않습니다.');
    expect(host.textContent).not.toContain('대시보드');

    await submit(host, 'operator', 'correct-password');
    expect(host.textContent).toContain('로그인 서비스에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.');
    expect(host.textContent).not.toContain('아이디 또는 비밀번호가 올바르지 않습니다.');
    expect(host.textContent).not.toContain('대시보드');
    act(() => root.unmount());
  });

  it('mounts only after the server accepts entered credentials and sends subsequent requests with the session cookie', async () => {
    let resolveLogin: (() => void) | undefined;
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(errorResponse(401))
      .mockImplementationOnce(() => new Promise((resolve) => {
        resolveLogin = () => resolve(okResponse());
      }))
      .mockResolvedValueOnce(okResponse({ registry_version: 0, cameras: [] }));
    vi.stubGlobal('fetch', fetchMock);
    const { host, root } = renderGate();
    await settle();
    const login = submit(host, 'operator', 'correct-password');
    expect(host.textContent).not.toContain('대시보드');
    expect(host.querySelector('#login-title')?.textContent).toBe('로그인');
    resolveLogin?.();
    await login;
    expect(host.textContent).toContain('대시보드');
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/v1/auth/session', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ username: 'operator', password: 'correct-password' }),
      credentials: 'same-origin',
    }));
    await fetchCameras();
    expect(fetchMock).toHaveBeenNthCalledWith(3, '/api/v1/cameras', expect.objectContaining({ credentials: 'same-origin' }));
    act(() => root.unmount());
  });

  it('keeps the dashboard authorized and shows operational copy when logout fails', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(okResponse())
      .mockResolvedValueOnce(errorResponse(500));
    vi.stubGlobal('fetch', fetchMock);
    const { host, root } = renderGate(<LogoutProbe />);
    await settle();
    expect(host.textContent).toContain('대시보드');
    await act(async () => {
      host.querySelector<HTMLButtonElement>('button[data-testid="logout"]')?.click();
    });
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/v1/auth/session', expect.objectContaining({
      method: 'DELETE',
      credentials: 'same-origin',
    }));
    expect(host.textContent).toContain('대시보드');
    expect(host.textContent).toContain('로그아웃하지 못했습니다. 다시 시도해 주세요.');
    expect(host.querySelector('#login-title')).toBeNull();
    act(() => root.unmount());
  });
});
