import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { AuthGate } from '@/shared/ui/AuthGate';
import { DashboardShell } from '@/shared/ui/DashboardShell';

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

async function settle(): Promise<void> {
  await act(async () => {});
}

async function submitLogin(host: HTMLElement): Promise<void> {
  const inputs = host.querySelectorAll('input');
  await act(async () => {
    for (const [input, value] of [[inputs[0], 'operator'], [inputs[1], 'correct-password']] as const) {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set?.call(input, value);
      input.dispatchEvent(new Event('input', { bubbles: true }));
    }
    host.querySelector('form')?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
  });
}

async function clickLogout(host: HTMLElement): Promise<void> {
  await act(async () => {
    host.querySelector<HTMLButtonElement>('header button')?.click();
  });
}

describe('DashboardShell', () => {
  it('renders four semantic destinations, active state, skip link, SA brand, and no implementation copy', () => {
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(
      <DashboardShell screen="cameras" onScreenChange={vi.fn()} backendStatus={{ label: '연결됨', className: '' }} onAddCamera={vi.fn()}>
        <h1>관제 본문</h1>
      </DashboardShell>,
    ));
    expect(host.querySelector('a[href="#main-content"]')?.textContent).toContain('본문으로');
    expect(host.querySelector('#main-content')).not.toBeNull();
    expect(host.querySelectorAll('nav').length).toBe(2);
    for (const nav of host.querySelectorAll('nav')) {
      const buttons = nav.querySelectorAll('button');
      expect(buttons).toHaveLength(4);
      for (const button of buttons) {
        const mark = button.querySelector('.nav-mark');
        expect(mark?.getAttribute('aria-hidden')).toBe('true');
        expect(mark?.querySelector('svg')).not.toBeNull();
        expect(mark?.textContent?.trim()).toBe('');
      }
    }
    expect(host.querySelectorAll('[aria-current="page"]')).toHaveLength(2);
    expect(Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '카메라 추가')?.classList.contains('brand-action')).toBe(true);
    expect(host.textContent).toContain('Senior AI Lab');
    expect(host.textContent).not.toMatch(/API|token|토큰|다크|실시간 알림/iu);
    act(() => root.unmount());
  });

  it('deletes the authenticated session and returns to login on logout', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(errorResponse(401))
      .mockResolvedValueOnce(okResponse())
      .mockResolvedValueOnce({ ok: true, status: 204, json: async () => undefined });
    vi.stubGlobal('fetch', fetchMock);
    window.history.replaceState(null, '', '/?page=system');
    const replace = vi.spyOn(window.history, 'replaceState');
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(
      <AuthGate>
        <DashboardShell screen="system" onScreenChange={vi.fn()} backendStatus={{ label: '연결됨', className: '' }} onAddCamera={vi.fn()}>
          <div>시스템 본문</div>
        </DashboardShell>
      </AuthGate>,
    ));
    await settle();
    await submitLogin(host);
    expect(host.querySelector('header button')?.textContent).toBe('로그아웃');
    await clickLogout(host);
    expect(fetchMock).toHaveBeenNthCalledWith(3, '/api/v1/auth/session', expect.objectContaining({
      method: 'DELETE',
      credentials: 'same-origin',
    }));
    expect(host.querySelector('#login-title')?.textContent).toBe('로그인');
    expect(replace).toHaveBeenLastCalledWith(null, '', '/?page=operations&wallPage=1');
    act(() => root.unmount());
  });

  it('keeps the dashboard authorized when logout fails', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(okResponse())
      .mockResolvedValueOnce(errorResponse(500));
    vi.stubGlobal('fetch', fetchMock);
    window.history.replaceState(null, '', '/?page=system');
    const replace = vi.spyOn(window.history, 'replaceState');
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(
      <AuthGate>
        <DashboardShell screen="system" onScreenChange={vi.fn()} backendStatus={{ label: '연결됨', className: '' }} onAddCamera={vi.fn()}>
          <div>시스템 본문</div>
        </DashboardShell>
      </AuthGate>,
    ));
    await settle();
    expect(host.querySelector('header button')?.textContent).toBe('로그아웃');
    replace.mockClear();
    await clickLogout(host);
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/v1/auth/session', expect.objectContaining({
      method: 'DELETE',
      credentials: 'same-origin',
    }));
    expect(host.textContent).toContain('시스템 본문');
    expect(host.textContent).toContain('로그아웃하지 못했습니다. 다시 시도해 주세요.');
    expect(host.querySelector('#login-title')).toBeNull();
    expect(host.querySelector('header button')?.textContent).toBe('로그아웃');
    expect(replace).not.toHaveBeenCalled();
    act(() => root.unmount());
  });

  it('gives logout a distinguishing class and visual separation from the primary action, without changing its exact label', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(okResponse());
    vi.stubGlobal('fetch', fetchMock);
    window.history.replaceState(null, '', '/?page=cameras');
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(
      <AuthGate>
        <DashboardShell screen="cameras" onScreenChange={vi.fn()} backendStatus={{ label: '연결됨', className: '' }} onAddCamera={vi.fn()}>
          <div>카메라 관리 본문</div>
        </DashboardShell>
      </AuthGate>,
    ));
    await settle();

    const addCameraButton = Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '카메라 추가');
    const logoutButton = host.querySelector<HTMLButtonElement>('header button.logout-action');

    expect(logoutButton?.textContent).toBe('로그아웃');
    expect(logoutButton?.classList.contains('brand-action')).toBe(false);
    expect(addCameraButton?.classList.contains('logout-action')).toBe(false);
    // Not a bare sibling of equal visual weight — grouped separately with a divider.
    expect(addCameraButton?.parentElement).not.toBe(logoutButton?.parentElement);
    expect(host.querySelector('.shell-divider')).not.toBeNull();
    act(() => root.unmount());
  });
});
