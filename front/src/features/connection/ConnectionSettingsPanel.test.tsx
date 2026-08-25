import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { saveConnection, testConnection, type ConnectionView } from '@/shared/api/client';
import { HttpError } from '@/shared/api/http';
import { ConnectionSettingsPanel } from '@/features/connection/ConnectionSettingsPanel';
import type { PollingResource } from '@/shared/api/usePollingResource';

vi.mock('@/shared/api/client', async () => {
  const { withOverrides } = await vi.importActual<typeof import('@/test/moduleMock')>('@/test/moduleMock');
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return withOverrides(actual, { saveConnection: vi.fn(), testConnection: vi.fn() });
});

const enrolledView: ConnectionView = {
  events_url: 'https://api.eldercare.example/api/v1/events',
  config_url: 'https://api.eldercare.example/api/v1/ml-config',
  facility_code: 'NH-7H2K9M4QXP',
  client_installation_ref: 'aa83ea3f-6e5f-4f45-a401-fb36c38835b6',
  facility_id: 'facility-canonical-42',
  edge_installation_id: 'c72bd9a7-3e04-47ba-a8cd-a56e54f98152',
  enrollment_generation: 3,
  facility_token_set: true,
  facility_token_masked: '****ab12',
  enrolled: true,
  configured: true,
  reachable: true,
  last_ok_at: '2026-08-01T00:00:00Z',
  updated_at: '2026-08-01T00:00:00Z',
};

function resource(data: ConnectionView = enrolledView): PollingResource<ConnectionView> {
  return { status: 'success', data, error: null, lastSuccessAt: Date.now(), refreshing: false, retry: vi.fn(), replace: vi.fn() };
}

function renderPanel(value = resource()): { readonly host: HTMLDivElement; readonly root: Root } {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<ConnectionSettingsPanel resource={value} />));
  return { host, root };
}

function setInput(host: HTMLElement, name: string, value: string): void {
  const input = host.querySelector(`input[name="${name}"]`);
  if (!(input instanceof HTMLInputElement)) throw new Error(`missing input ${name}`);
  act(() => {
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set?.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

function button(host: HTMLElement, label: string): HTMLButtonElement {
  const match = Array.from(host.querySelectorAll('button')).find((candidate) => candidate.textContent === label);
  if (!match) throw new Error(`missing button ${label}`);
  return match;
}

function openEnrollment(host: HTMLElement): void {
  act(() => host.querySelector<HTMLButtonElement>('[aria-label="등록 정보 변경"]')?.click());
}

beforeEach(() => {
  // Mirrors a plain-HTTP LAN dashboard: getRandomValues exists, randomUUID does
  // not (it is secure-context only). The panel must still generate a ref.
  vi.stubGlobal('crypto', {
    getRandomValues: (bytes: Uint8Array) => {
      for (let index = 0; index < bytes.length; index += 1) bytes[index] = (index * 37 + 11) & 0xff;
      return bytes;
    },
  });
  vi.mocked(saveConnection).mockReset();
  vi.mocked(saveConnection).mockResolvedValue(enrolledView);
  vi.mocked(testConnection).mockReset();
  vi.mocked(testConnection).mockResolvedValue({
    ok: true,
    error_class: null,
    detail: '등록 정보를 확인했습니다.',
    facility_id: 'facility-canonical-42',
    edge_installation_id: 'c72bd9a7-3e04-47ba-a8cd-a56e54f98152',
    enrollment_generation: 3,
  });
});

afterEach(() => {
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('ConnectionSettingsPanel', () => {
  it('shows the deployment backend, canonical facility, masked token, and generation without exposing installation internals', () => {
    const { host } = renderPanel();

    expect(host.textContent).toContain('api.eldercare.example');
    expect(host.textContent).toContain('facility-canonical-42');
    expect(host.textContent).toContain('3회');
    expect(host.textContent).toContain('****ab12');
    expect(host.textContent).not.toContain('aa83ea3f');
    expect(host.textContent).not.toContain('c72bd9a7');
  });

  it('accepts exactly facility code and token and submits one stable generated installation reference', async () => {
    const { host } = renderPanel(resource({ ...enrolledView, enrolled: false, facility_code: null, client_installation_ref: null }));
    openEnrollment(host);
    setInput(host, 'facility_code', 'NH-7H2K9M4QXP');
    setInput(host, 'facility_token', 'one-time-secret');

    await act(async () => button(host, '등록 저장').click());

    expect(saveConnection).toHaveBeenCalledWith({
      facility_code: 'NH-7H2K9M4QXP',
      facility_token: 'one-time-secret',
      // Exact value is generated; assert the provisioning contract instead of a constant.
      client_installation_ref: expect.stringMatching(
        /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
      ),
    });
    expect(host.querySelector('input[name="facility_id"]')).toBeNull();
    expect(host.textContent).not.toContain('one-time-secret');
    openEnrollment(host);
    expect((host.querySelector('input[name="facility_token"]') as HTMLInputElement).value).toBe('');
  });

  it('preserves unsaved code and token when fresh masked state arrives during editing', () => {
    const first = resource();
    const { host, root } = renderPanel(first);
    openEnrollment(host);
    setInput(host, 'facility_code', 'NH-9H2K9M4QXP');
    setInput(host, 'facility_token', 'draft-secret');

    act(() => root.render(<ConnectionSettingsPanel resource={resource({ ...enrolledView, enrollment_generation: 4 })} />));

    expect((host.querySelector('input[name="facility_code"]') as HTMLInputElement).value).toBe('NH-9H2K9M4QXP');
    expect((host.querySelector('input[name="facility_token"]') as HTMLInputElement).value).toBe('draft-secret');
  });

  it('classifies a relink conflict without rendering backend bodies or the token', async () => {
    vi.mocked(saveConnection).mockRejectedValue(new HttpError(409, { detail: 'internal conflict body' }));
    const { host } = renderPanel();
    openEnrollment(host);
    setInput(host, 'facility_token', 'conflict-secret');

    await act(async () => button(host, '등록 저장').click());

    expect(host.querySelector('[role="alert"]')?.textContent).toContain('다른 장치');
    expect(host.textContent).not.toContain('internal conflict body');
    expect(host.textContent).not.toContain('conflict-secret');
  });

  it('distinguishes a wrong/unreachable Hub address from a credential failure and shows the address on save', async () => {
    vi.mocked(saveConnection).mockRejectedValue(new HttpError(503, { detail: '외부 백엔드에 연결할 수 없습니다.' }));
    const { host } = renderPanel();
    openEnrollment(host);
    setInput(host, 'facility_token', 'unreachable-secret');

    await act(async () => button(host, '등록 저장').click());

    const alert = host.querySelector('[role="alert"]')?.textContent ?? '';
    expect(alert).toContain('연결할 수 없습니다');
    expect(alert).toContain('api.eldercare.example');
    expect(alert).not.toContain('시설 코드');
    expect(alert).not.toContain('외부 백엔드');
  });

  it('distinguishes a Hub timeout the same way as unreachable, showing the address rather than a generic save failure', async () => {
    vi.mocked(saveConnection).mockRejectedValue(new HttpError(504, { detail: '외부 백엔드 응답 시간이 초과되었습니다.' }));
    const { host } = renderPanel();
    openEnrollment(host);
    setInput(host, 'facility_token', 'timeout-secret');

    await act(async () => button(host, '등록 저장').click());

    const alert = host.querySelector('[role="alert"]')?.textContent ?? '';
    expect(alert).toContain('연결할 수 없습니다');
    expect(alert).toContain('api.eldercare.example');
  });

  it('shows the wrong/unreachable Hub address distinctly on the inline connection test, not the raw backend detail', async () => {
    vi.mocked(testConnection).mockResolvedValue({
      ok: false,
      error_class: 'unreachable',
      detail: '외부 백엔드에 연결할 수 없습니다.',
      facility_id: null,
      edge_installation_id: null,
      enrollment_generation: null,
    });
    const { host } = renderPanel();
    openEnrollment(host);
    setInput(host, 'facility_token', 'test-secret');

    await act(async () => button(host, '등록 확인').click());

    const alert = host.querySelector('[role="alert"]')?.textContent ?? '';
    expect(alert).toContain('연결할 수 없습니다');
    expect(alert).toContain('api.eldercare.example');
    expect(alert).not.toContain('외부 백엔드');
  });

  it('still shows an auth failure from the inline connection test as a credential problem, not a server-address one', async () => {
    vi.mocked(testConnection).mockResolvedValue({
      ok: false,
      error_class: 'auth',
      detail: '등록 코드 또는 토큰을 확인해 주세요.',
      facility_id: null,
      edge_installation_id: null,
      enrollment_generation: null,
    });
    const { host } = renderPanel();
    openEnrollment(host);
    setInput(host, 'facility_token', 'bad-token');

    await act(async () => button(host, '등록 확인').click());

    const alert = host.querySelector('[role="alert"]')?.textContent ?? '';
    expect(alert).toContain('등록 코드 또는 토큰');
    expect(alert).not.toContain('연결할 수 없습니다');
  });

  it('prevents duplicate saves while enrollment is in flight', async () => {
    let finish: ((value: ConnectionView) => void) | undefined;
    vi.mocked(saveConnection).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
    const { host } = renderPanel();
    openEnrollment(host);
    setInput(host, 'facility_token', 'busy-secret');

    const submit = button(host, '등록 저장');
    act(() => { submit.click(); submit.click(); });

    expect(saveConnection).toHaveBeenCalledTimes(1);
    await act(async () => finish?.(enrolledView));
  });
});
