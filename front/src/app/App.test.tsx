import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App, { Dashboard } from '@/app/App';
import { AuthGate } from '@/shared/ui/AuthGate';
import { fetchDashboardSession, loginDashboard, logoutDashboard } from '@/shared/api/client';
import { HttpError } from '@/shared/api/http';

vi.mock('@/shared/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return withOverrides(actual, {
    fetchDashboardSession: vi.fn(),
    loginDashboard: vi.fn(),
    logoutDashboard: vi.fn(),
  });
});

function mockAuthorizedSession(): void {
  vi.mocked(fetchDashboardSession).mockResolvedValue(undefined);
  vi.mocked(loginDashboard).mockResolvedValue();
  vi.mocked(logoutDashboard).mockResolvedValue();
}

async function renderDashboard(element: JSX.Element = <AuthGate><Dashboard /></AuthGate>): Promise<{ host: HTMLDivElement; root: ReturnType<typeof createRoot> }> {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(element));
  await act(async () => {});
  return { host, root };
}

afterEach(() => {
  document.body.innerHTML = '';
  window.history.replaceState(null, '', '/');
  vi.restoreAllMocks();
});

describe('Dashboard shell', () => {
  it('renders the top NavBar with brand, three nav destinations, and defaults to the operations page', async () => {
    mockAuthorizedSession();
    const { host, root } = await renderDashboard();

    const navbar = host.querySelector('.app-navbar');
    expect(navbar).not.toBeNull();
    expect(navbar?.textContent).toContain('Senior AI Lab Edge');

    const navButtons = [...host.querySelectorAll('.app-nav button')].map((button) => button.textContent);
    expect(navButtons).toEqual(['관제', '이벤트', '설정']);

    const activeButton = host.querySelector('.app-nav button[aria-current="page"]');
    expect(activeButton?.textContent).toBe('관제');
    expect(host.querySelector('#main-content h1')?.textContent).toBe('관제');
    act(() => root.unmount());
  });

  it('navigates to the events page and updates the URL when the 이벤트 nav chip is clicked', async () => {
    mockAuthorizedSession();
    const { host, root } = await renderDashboard();

    const eventsButton = [...host.querySelectorAll('.app-nav button')].find((button) => button.textContent === '이벤트') as HTMLButtonElement;
    await act(async () => { eventsButton.click(); });

    expect(host.querySelector('#main-content h1')?.textContent).toBe('이벤트');
    expect(host.querySelector('.app-nav button[aria-current="page"]')?.textContent).toBe('이벤트');
    expect(window.location.search).toBe('?page=events');
    act(() => root.unmount());
  });

  it('navigates to the settings page when the 설정 nav chip is clicked', async () => {
    mockAuthorizedSession();
    const { host, root } = await renderDashboard();

    const settingsButton = [...host.querySelectorAll('.app-nav button')].find((button) => button.textContent === '설정') as HTMLButtonElement;
    await act(async () => { settingsButton.click(); });

    expect(host.querySelector('#main-content h1')?.textContent).toBe('설정');
    expect(window.location.search).toBe('?page=settings');
    act(() => root.unmount());
  });

  it('opens AccountSettingsModal from the account icon button and closes it on cancel', async () => {
    mockAuthorizedSession();
    const { host, root } = await renderDashboard();

    expect(document.querySelector('[role="dialog"]')).toBeNull();
    const accountButton = host.querySelector('button[aria-label="계정 설정"]') as HTMLButtonElement;
    await act(async () => { accountButton.click(); });

    const dialog = document.querySelector('[role="dialog"]');
    expect(dialog).not.toBeNull();
    expect(dialog?.textContent).toContain('계정 설정');

    const cancelButton = [...(dialog?.querySelectorAll('button') ?? [])].find((button) => button.textContent === '취소') as HTMLButtonElement;
    await act(async () => { cancelButton.click(); });

    expect(document.querySelector('[role="dialog"]')).toBeNull();
    act(() => root.unmount());
  });

  it('renders a logout button that calls logoutDashboard when clicked', async () => {
    mockAuthorizedSession();
    const { host, root } = await renderDashboard();

    const logoutButton = host.querySelector('.logout-button') as HTMLButtonElement;
    expect(logoutButton?.textContent).toBe('로그아웃');
    await act(async () => { logoutButton.click(); });

    expect(logoutDashboard).toHaveBeenCalledTimes(1);
    act(() => root.unmount());
  });

  it('renders a skip link targeting #main-content', async () => {
    mockAuthorizedSession();
    const { host, root } = await renderDashboard();

    const skipLink = host.querySelector('.skip-link');
    expect(skipLink?.getAttribute('href')).toBe('#main-content');
    expect(host.querySelector('#main-content')).not.toBeNull();
    act(() => root.unmount());
  });
});

describe('App', () => {
  it('shows the login form via AuthGate when no session exists, and the dashboard after signing in', async () => {
    vi.mocked(fetchDashboardSession).mockRejectedValue(new HttpError(401));
    vi.mocked(loginDashboard).mockResolvedValue();
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(<App />));
    await act(async () => {});

    expect(host.querySelector('form')).not.toBeNull();
    expect(host.querySelector('.app-navbar')).toBeNull();

    const inputs = host.querySelectorAll('input');
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set?.call(inputs[0], 'admin');
      inputs[0].dispatchEvent(new Event('input', { bubbles: true }));
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set?.call(inputs[1], 'password');
      inputs[1].dispatchEvent(new Event('input', { bubbles: true }));
      host.querySelector('form')?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    });

    expect(loginDashboard).toHaveBeenCalledWith('admin', 'password');
    expect(host.querySelector('.app-navbar')).not.toBeNull();
    act(() => root.unmount());
  });
});
