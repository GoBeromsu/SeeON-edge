import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { saveConnection, testConnection } from '@/shared/api/client';
import { ConnectionSettingsPanel } from '@/features/connection/ConnectionSettingsPanel';
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
  events_url: 'https://backend.example.com/events',
  config_url: 'https://backend.example.com/config',
  facility_id: 'facility-42',
  facility_token_set: true,
  facility_token_masked: '****ab12',
  configured: true,
  reachable: true,
  last_ok_at: '2026-08-01T00:00:00Z',
  updated_at: '2026-08-01T00:00:00Z',
  heartbeat_relay: { enabled: true, last_success_at: '2026-08-02T00:00:00Z', last_error_class: null, detail: null },
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

beforeEach(() => {
  vi.mocked(saveConnection).mockReset();
  vi.mocked(saveConnection).mockResolvedValue(baseView);
  vi.mocked(testConnection).mockReset();
  vi.mocked(testConnection).mockResolvedValue({ ok: true, error_class: null, detail: '연결 성공', probed_url: null });
});

describe('ConnectionSettingsPanel draft sync', () => {
  it('does not clobber an in-progress edit when a fresh poll delivers different values', () => {
    const { host, root } = renderPanel();
    setInput(host, 'facility_id', '작성 중인 ID');

    act(() => root.render(<ConnectionSettingsPanel resource={makeResource({ data: { ...baseView, facility_id: 'polled-facility' } })} />));

    expect((host.querySelector('input[name="facility_id"]') as HTMLInputElement).value).toBe('작성 중인 ID');
    act(() => root.unmount());
  });

  it('syncs the draft from a fresh poll result while the form is pristine', () => {
    const { host, root } = renderPanel();

    act(() => root.render(<ConnectionSettingsPanel resource={makeResource({ data: { ...baseView, facility_id: 'polled-facility' } })} />));

    expect((host.querySelector('input[name="facility_id"]') as HTMLInputElement).value).toBe('polled-facility');
    act(() => root.unmount());
  });
});

describe('ConnectionSettingsPanel busy states', () => {
  it('disables both actions while a save is in flight, then re-enables them', async () => {
    let resolveSave: ((view: ConnectionView) => void) | undefined;
    vi.mocked(saveConnection).mockReturnValue(new Promise((resolve) => {
      resolveSave = resolve;
    }));
    const { host, root } = renderPanel();

    act(() => clickButton(host, '저장'));

    expect(findButton(host, '연결 테스트').disabled).toBe(true);
    expect(findButton(host, '저장 중...').disabled).toBe(true);

    await act(async () => resolveSave?.(baseView));

    expect(findButton(host, '연결 테스트').disabled).toBe(false);
    expect(findButton(host, '저장').disabled).toBe(false);
    act(() => root.unmount());
  });

  it('disables both actions while a connection test is in flight, then re-enables them', async () => {
    let resolveTest: ((result: Awaited<ReturnType<typeof testConnection>>) => void) | undefined;
    vi.mocked(testConnection).mockReturnValue(new Promise((resolve) => {
      resolveTest = resolve;
    }));
    const { host, root } = renderPanel();

    act(() => clickButton(host, '연결 테스트'));

    expect(findButton(host, '확인 중...').disabled).toBe(true);
    expect(findButton(host, '저장').disabled).toBe(true);

    await act(async () => resolveTest?.({ ok: true, error_class: null, detail: '연결 성공', probed_url: null }));

    expect(findButton(host, '연결 테스트').disabled).toBe(false);
    expect(findButton(host, '저장').disabled).toBe(false);
    act(() => root.unmount());
  });
});

describe('ConnectionSettingsPanel token tri-state payload', () => {
  it('omits facility_token when the token field is left untouched', async () => {
    const { host, root } = renderPanel();

    await act(async () => clickButton(host, '저장'));

    const payload = vi.mocked(saveConnection).mock.calls[0]?.[0];
    expect(payload && 'facility_token' in payload).toBe(false);
    act(() => root.unmount());
  });

  it('sends the typed token when the technician replaces it', async () => {
    const { host, root } = renderPanel();
    setInput(host, 'facility_token', 'new-secret-value');

    await act(async () => clickButton(host, '저장'));

    expect(saveConnection).toHaveBeenCalledWith(expect.objectContaining({ facility_token: 'new-secret-value' }));
    act(() => root.unmount());
  });

  it('sends an explicit null when the technician clears a previously-set token', async () => {
    const { host, root } = renderPanel();
    clickButton(host, '지우기');

    await act(async () => clickButton(host, '저장'));

    expect(saveConnection).toHaveBeenCalledWith(expect.objectContaining({ facility_token: null }));
    act(() => root.unmount());
  });
});

describe('ConnectionSettingsPanel save and test failures', () => {
  it('renders the save failure message and never leaks the raw token into the DOM', async () => {
    vi.mocked(saveConnection).mockRejectedValue(new Error('save failed'));
    const { host, root } = renderPanel();
    setInput(host, 'facility_token', 'super-secret-token');

    await act(async () => clickButton(host, '저장'));

    const status = host.querySelector('[role="status"]');
    expect(status?.textContent).toContain('연결 설정 저장에 실패했습니다');
    expect(host.textContent).not.toContain('super-secret-token');
    act(() => root.unmount());
  });

  it('renders the test failure message and never leaks the raw token into the DOM', async () => {
    vi.mocked(testConnection).mockResolvedValue({ ok: false, error_class: 'auth', detail: '인증에 실패했습니다.', probed_url: null });
    const { host, root } = renderPanel();
    setInput(host, 'facility_token', 'super-secret-token');

    await act(async () => clickButton(host, '연결 테스트'));

    const alert = host.querySelector('[role="alert"]');
    expect(alert?.textContent).toContain('인증에 실패했습니다.');
    expect(host.textContent).not.toContain('super-secret-token');
    act(() => root.unmount());
  });
});

describe('ConnectionSettingsPanel heartbeat relay status', () => {
  it('shows the normal relay message with the last success time when enabled without an error', () => {
    const { host, root } = renderPanel(makeResource({
      data: { ...baseView, heartbeat_relay: { enabled: true, last_success_at: '2026-08-02T00:00:00Z', last_error_class: null, detail: null } },
    }));

    expect(host.textContent).toContain('하트비트 릴레이 정상');
    act(() => root.unmount());
  });

  it('shows the server-provided detail when the relay is enabled but erroring', () => {
    const { host, root } = renderPanel(makeResource({
      data: { ...baseView, heartbeat_relay: { enabled: true, last_success_at: null, last_error_class: 'timeout', detail: '릴레이 응답 시간이 초과되었습니다.' } },
    }));

    expect(host.textContent).toContain('릴레이 응답 시간이 초과되었습니다.');
    act(() => root.unmount());
  });

  it('shows the inactive relay message when disabled', () => {
    const { host, root } = renderPanel(makeResource({
      data: { ...baseView, heartbeat_relay: { enabled: false, last_success_at: null, last_error_class: null, detail: null } },
    }));

    expect(host.textContent).toContain('하트비트 릴레이 비활성');
    act(() => root.unmount());
  });

  it('defaults to the inactive relay message when the view omits heartbeat_relay entirely (older backend)', () => {
    const { heartbeat_relay: _omitted, ...legacyView } = baseView;
    const { host, root } = renderPanel(makeResource({ data: legacyView as ConnectionView }));

    expect(host.textContent).toContain('하트비트 릴레이 비활성');
    act(() => root.unmount());
  });
});
