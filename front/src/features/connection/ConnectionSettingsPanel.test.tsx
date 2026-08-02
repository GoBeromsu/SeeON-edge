import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { saveConnection, testConnection } from '@/shared/api/client';
import { ConnectionSettingsPanel } from '@/features/connection/ConnectionSettingsPanel';
import { toast } from '@/shared/ui/Toast';
import type { ConnectionView } from '@/shared/api/client';
import type { PollingResource } from '@/shared/api/usePollingResource';

vi.mock('@/shared/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return {
    ...actual,
    saveConnection: vi.fn(),
    testConnection: vi.fn(),
  };
});

const baseView: ConnectionView = {
  events_url: 'https://backend.example.com/v1/events',
  config_url: 'https://backend.example.com/v1/config',
  facility_id: 'facility-42',
  facility_token_set: true,
  facility_token_masked: '****ab12',
  configured: true,
  reachable: true,
  last_ok_at: '2026-08-01T00:00:00Z',
  updated_at: '2026-08-01T00:00:00Z',
};

function makeResource(overrides: Partial<PollingResource<ConnectionView>> = {}): PollingResource<ConnectionView> {
  return {
    status: 'success',
    data: baseView,
    error: null,
    lastSuccessAt: Date.now(),
    refreshing: false,
    retry: vi.fn(),
    ...overrides,
  };
}

function renderPanel(resource: PollingResource<ConnectionView> = makeResource()): { host: HTMLDivElement; root: ReturnType<typeof createRoot> } {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<ConnectionSettingsPanel resource={resource} />));
  return { host, root };
}

function setInput(host: HTMLElement, name: string, value: string): void {
  const input = host.querySelector(`input[name="${name}"]`);
  if (!(input instanceof HTMLInputElement)) {
    throw new Error(`missing input ${name}`);
  }
  act(() => {
    const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set;
    valueSetter?.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

function findButton(host: HTMLElement, label: string): HTMLButtonElement {
  const button = Array.from(host.querySelectorAll('button')).find((candidate) => candidate.textContent === label);
  if (!button) {
    throw new Error(`missing button ${label}`);
  }
  return button;
}

function clickButton(host: HTMLElement, label: string): void {
  act(() => findButton(host, label).click());
}

function openEdit(host: HTMLElement): void {
  act(() => host.querySelector<HTMLButtonElement>('[aria-label="서버 연결 편집"]')?.click());
}

beforeEach(() => {
  vi.mocked(saveConnection).mockReset();
  vi.mocked(saveConnection).mockResolvedValue(baseView);
  vi.mocked(testConnection).mockReset();
  vi.mocked(testConnection).mockResolvedValue({ ok: true, error_class: null, detail: '연결 성공', probed_url: null });
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('ConnectionSettingsPanel read mode', () => {
  it('renders facility id, masked token, and last sync as a read-only dl, with no form yet', () => {
    const { host, root } = renderPanel();

    expect(host.textContent).toContain('facility-42');
    expect(host.textContent).toContain('****ab12');
    expect(host.querySelector('form')).toBeNull();
    act(() => root.unmount());
  });

  it('opens the edit form pre-filled with facility_id and a blank token when the pencil button is clicked', () => {
    const { host, root } = renderPanel();

    openEdit(host);

    expect((host.querySelector('input[name="facility_id"]') as HTMLInputElement).value).toBe('facility-42');
    expect((host.querySelector('input[name="facility_token"]') as HTMLInputElement).value).toBe('');
    act(() => root.unmount());
  });

  it('closes the edit form and discards a draft change when the pencil is toggled again', () => {
    const { host, root } = renderPanel();
    openEdit(host);
    setInput(host, 'facility_id', '작성 중인 ID');

    act(() => host.querySelector<HTMLButtonElement>('[aria-label="서버 연결 편집 닫기"]')?.click());

    expect(host.querySelector('form')).toBeNull();
    openEdit(host);
    expect((host.querySelector('input[name="facility_id"]') as HTMLInputElement).value).toBe('facility-42');
    act(() => root.unmount());
  });
});

describe('ConnectionSettingsPanel busy states', () => {
  it('disables both actions while a save is in flight, then closes the form on success', async () => {
    let resolveSave: ((view: ConnectionView) => void) | undefined;
    vi.mocked(saveConnection).mockReturnValue(new Promise((resolve) => {
      resolveSave = resolve;
    }));
    const { host, root } = renderPanel();
    openEdit(host);

    act(() => clickButton(host, '저장'));

    expect(findButton(host, '연결').disabled).toBe(true);
    expect(findButton(host, '저장 중...').disabled).toBe(true);

    await act(async () => resolveSave?.(baseView));

    expect(host.querySelector('form')).toBeNull();
    act(() => root.unmount());
  });

  it('disables both actions while a connection test is in flight, then re-enables them', async () => {
    let resolveTest: ((result: Awaited<ReturnType<typeof testConnection>>) => void) | undefined;
    vi.mocked(testConnection).mockReturnValue(new Promise((resolve) => {
      resolveTest = resolve;
    }));
    const { host, root } = renderPanel();
    openEdit(host);

    act(() => clickButton(host, '연결'));

    expect(findButton(host, '확인 중...').disabled).toBe(true);
    expect(findButton(host, '저장').disabled).toBe(true);

    await act(async () => resolveTest?.({ ok: true, error_class: null, detail: '연결 성공', probed_url: null }));

    expect(findButton(host, '연결').disabled).toBe(false);
    expect(findButton(host, '저장').disabled).toBe(false);
    act(() => root.unmount());
  });
});

describe('ConnectionSettingsPanel save behavior', () => {
  it('shows a success toast and refreshes the resource on save, without leaking the raw token', async () => {
    const successSpy = vi.spyOn(toast, 'success');
    const resource = makeResource();
    const { host, root } = renderPanel(resource);
    openEdit(host);
    setInput(host, 'facility_token', 'super-secret-token');

    await act(async () => clickButton(host, '저장'));

    expect(saveConnection).toHaveBeenCalledWith({ facility_id: 'facility-42', facility_token: 'super-secret-token' });
    expect(resource.retry).toHaveBeenCalled();
    expect(successSpy).toHaveBeenCalledWith('연결 설정을 저장했습니다.');
    expect(host.textContent).not.toContain('super-secret-token');
    act(() => root.unmount());
  });

  it('omits facility_token from the payload when the technician never touches it', async () => {
    const { host, root } = renderPanel();
    openEdit(host);

    await act(async () => clickButton(host, '저장'));

    const payload = vi.mocked(saveConnection).mock.calls[0]?.[0];
    expect(payload && 'facility_token' in payload).toBe(false);
    expect(payload).toEqual({ facility_id: 'facility-42' });
    act(() => root.unmount());
  });

  it('keeps the form open and shows an inline error when save fails', async () => {
    vi.mocked(saveConnection).mockRejectedValue(new Error('save failed'));
    const { host, root } = renderPanel();
    openEdit(host);

    await act(async () => clickButton(host, '저장'));

    expect(host.querySelector('form')).not.toBeNull();
    const alert = host.querySelector('[role="alert"]');
    expect(alert?.textContent).toContain('연결 설정 저장에 실패했습니다');
    act(() => root.unmount());
  });

  it('shows the test failure detail inline without leaking the raw token', async () => {
    vi.mocked(testConnection).mockResolvedValue({ ok: false, error_class: 'auth', detail: '인증에 실패했습니다.', probed_url: null });
    const { host, root } = renderPanel();
    openEdit(host);
    setInput(host, 'facility_token', 'super-secret-token');

    await act(async () => clickButton(host, '연결'));

    const alert = host.querySelector('[role="alert"]');
    expect(alert?.textContent).toContain('인증에 실패했습니다.');
    expect(host.textContent).not.toContain('super-secret-token');
    act(() => root.unmount());
  });
});
