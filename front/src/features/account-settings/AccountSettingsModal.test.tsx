import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { AccountSettingsModal } from '@/features/account-settings/AccountSettingsModal';
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

function Harness({ open, onClose }: { open: boolean; onClose: () => void }): JSX.Element {
  const session = useAuthSession();
  return (
    <div>
      <span data-testid="session-username">{session?.username ?? ''}</span>
      <AccountSettingsModal open={open} onClose={onClose} />
    </div>
  );
}

async function renderAuthorized(open = true, onClose: () => void = () => {}): Promise<{ host: HTMLDivElement; root: ReturnType<typeof createRoot> }> {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => {
    root.render(
      <AuthGate>
        <Harness open={open} onClose={onClose} />
      </AuthGate>,
    );
  });
  await act(async () => {});
  return { host, root };
}

function dialog(): HTMLElement {
  return document.querySelector('[role="dialog"]') as HTMLElement;
}

function fields(): { username: HTMLInputElement; newPassword: HTMLInputElement; confirmPassword: HTMLInputElement } {
  const scope = dialog();
  return {
    username: scope.querySelector('input[name="newUsername"]') as HTMLInputElement,
    newPassword: scope.querySelector('input[name="newPassword"]') as HTMLInputElement,
    confirmPassword: scope.querySelector('input[name="confirmPassword"]') as HTMLInputElement,
  };
}

function setValue(input: HTMLInputElement, value: string): void {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set?.call(input, value);
  input.dispatchEvent(new Event('input', { bubbles: true }));
}

describe('AccountSettingsModal', () => {
  it('renders username, new password, and confirm password fields with no current-password field', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okResponse());
    vi.stubGlobal('fetch', fetchMock);
    const { root } = await renderAuthorized();

    const scope = dialog();
    expect(scope.querySelector('input[name="currentPassword"]')).toBeNull();
    expect(scope.querySelector('input[type="password"][autocomplete="current-password"]')).toBeNull();
    const { username, newPassword, confirmPassword } = fields();
    expect(username).not.toBeNull();
    expect(newPassword.type).toBe('password');
    expect(confirmPassword.type).toBe('password');
    expect(scope.textContent).toContain('취소');
    expect(scope.textContent).toContain('변경하기');
    act(() => root.unmount());
  });

  it('focuses the username field on open', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okResponse());
    vi.stubGlobal('fetch', fetchMock);
    const { root } = await renderAuthorized();

    expect(document.activeElement).toBe(fields().username);
    act(() => root.unmount());
  });

  it('rejects a new password shorter than 4 characters without calling the credentials endpoint', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okResponse());
    vi.stubGlobal('fetch', fetchMock);
    const { root } = await renderAuthorized();
    const { newPassword, confirmPassword } = fields();

    await act(async () => {
      setValue(newPassword, 'abc');
      setValue(confirmPassword, 'abc');
      dialog().querySelector('form')?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    });

    expect(dialog().textContent).toContain('새 비밀번호는 4자 이상이어야 합니다.');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    act(() => root.unmount());
  });

  it('rejects mismatched confirmation without calling the credentials endpoint', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okResponse());
    vi.stubGlobal('fetch', fetchMock);
    const { root } = await renderAuthorized();
    const { newPassword, confirmPassword } = fields();

    await act(async () => {
      setValue(newPassword, 'abcdef');
      setValue(confirmPassword, 'abcdefg');
      dialog().querySelector('form')?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    });

    expect(dialog().textContent).toContain('새 비밀번호가 일치하지 않습니다.');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    act(() => root.unmount());
  });

  it('submits a PUT to /auth/credentials with the trimmed username and new password, and shows a success message', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(okResponse())
      .mockResolvedValueOnce(okResponse());
    vi.stubGlobal('fetch', fetchMock);
    const { host, root } = await renderAuthorized();
    const { username, newPassword, confirmPassword } = fields();

    await act(async () => {
      setValue(username, '  newadmin  ');
      setValue(newPassword, 'newpass123');
      setValue(confirmPassword, 'newpass123');
      dialog().querySelector('form')?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    });

    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/v1/auth/credentials', expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify({ new_password: 'newpass123', username: 'newadmin' }),
      credentials: 'same-origin',
    }));
    expect(dialog().textContent).toContain('계정 정보가 변경되었습니다.');
    expect(host.querySelector('[data-testid="session-username"]')?.textContent).toBe('newadmin');
    act(() => root.unmount());
  });

  it('omits username from the request body when left blank, keeping the current account username', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(okResponse()).mockResolvedValueOnce(okResponse());
    vi.stubGlobal('fetch', fetchMock);
    const { root } = await renderAuthorized();
    const { username, newPassword, confirmPassword } = fields();

    await act(async () => {
      setValue(username, '');
      setValue(newPassword, 'newpass123');
      setValue(confirmPassword, 'newpass123');
      dialog().querySelector('form')?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    });

    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/v1/auth/credentials', expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify({ new_password: 'newpass123' }),
    }));
    act(() => root.unmount());
  });

  it('shows a generic error message on failure without a current-password-specific branch', async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(okResponse()).mockResolvedValueOnce(errorResponse(500));
    vi.stubGlobal('fetch', fetchMock);
    const { root } = await renderAuthorized();
    const { newPassword, confirmPassword } = fields();

    await act(async () => {
      setValue(newPassword, 'newpass123');
      setValue(confirmPassword, 'newpass123');
      dialog().querySelector('form')?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    });

    expect(dialog().textContent).toContain('계정 정보를 변경하지 못했습니다. 잠시 후 다시 시도해 주세요.');
    expect(dialog().textContent).not.toContain('현재 비밀번호');
    act(() => root.unmount());
  });

  it('calls onClose and clears fields when cancel is clicked', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okResponse());
    vi.stubGlobal('fetch', fetchMock);
    const onClose = vi.fn();
    const { root } = await renderAuthorized(true, onClose);
    const { newPassword } = fields();

    await act(async () => {
      setValue(newPassword, 'abcdef');
      dialog().querySelectorAll('button').forEach((button) => {
        if (button.textContent === '취소') button.click();
      });
    });

    expect(onClose).toHaveBeenCalledTimes(1);
    act(() => root.unmount());
  });
});
