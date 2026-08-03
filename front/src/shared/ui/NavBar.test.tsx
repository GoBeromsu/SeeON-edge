import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { NavBar } from '@/shared/ui/NavBar';
import { AuthGate } from '@/shared/ui/AuthGate';

afterEach(() => {
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function okResponse(body: unknown = {}): { ok: true; status: number; json: () => Promise<unknown> } {
  return { ok: true, status: 200, json: async () => body };
}

async function settle(): Promise<void> {
  await act(async () => {});
}

describe('NavBar', () => {
  it('renders the brand text and 3 nav chips, marking the active page current', () => {
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(
      <NavBar page="events" onNavigate={vi.fn()} onOpenAccountSettings={vi.fn()} />,
    ));

    expect(host.querySelector('.app-navbar-brand')?.textContent).toBe('Senior AI Lab Edge');
    const buttons = Array.from(host.querySelectorAll<HTMLButtonElement>('.app-nav button'));
    expect(buttons.map((button) => button.textContent)).toEqual(['관제', '이벤트', '설정']);
    expect(buttons.find((button) => button.textContent === '이벤트')?.getAttribute('aria-current')).toBe('page');
    expect(buttons.find((button) => button.textContent === '관제')?.getAttribute('aria-current')).toBeNull();
    act(() => root.unmount());
  });

  it('calls onNavigate with the clicked page id', () => {
    const onNavigate = vi.fn();
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(
      <NavBar page="operations" onNavigate={onNavigate} onOpenAccountSettings={vi.fn()} />,
    ));

    const settingsButton = Array.from(host.querySelectorAll<HTMLButtonElement>('.app-nav button'))
      .find((button) => button.textContent === '설정');
    act(() => settingsButton?.click());
    expect(onNavigate).toHaveBeenCalledWith('settings');
    act(() => root.unmount());
  });

  it('opens account settings via the account icon button', () => {
    const onOpenAccountSettings = vi.fn();
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(
      <NavBar page="operations" onNavigate={vi.fn()} onOpenAccountSettings={onOpenAccountSettings} />,
    ));

    act(() => host.querySelector<HTMLButtonElement>('.icon-button')?.click());
    expect(onOpenAccountSettings).toHaveBeenCalledTimes(1);
    act(() => root.unmount());
  });

  it('shows no backend-status pill and no add-camera action', () => {
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(
      <NavBar page="operations" onNavigate={vi.fn()} onOpenAccountSettings={vi.fn()} />,
    ));

    expect(host.textContent).not.toMatch(/카메라 추가/);
    expect(host.querySelector('[class*="backend"]')).toBeNull();
    act(() => root.unmount());
  });

  it('renders a working logout button when mounted inside an authorized AuthGate session', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(okResponse())
      .mockResolvedValueOnce(okResponse());
    vi.stubGlobal('fetch', fetchMock);
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(
      <AuthGate>
        <NavBar page="operations" onNavigate={vi.fn()} onOpenAccountSettings={vi.fn()} />
      </AuthGate>,
    ));
    await settle();

    const logoutButton = host.querySelector<HTMLButtonElement>('.logout-button');
    expect(logoutButton?.textContent).toBe('로그아웃');
    await act(async () => {
      logoutButton?.click();
    });
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/v1/auth/session', expect.objectContaining({ method: 'DELETE' }));
    act(() => root.unmount());
  });
});
