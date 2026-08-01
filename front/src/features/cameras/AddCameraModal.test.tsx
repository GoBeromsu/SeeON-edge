import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createCamera, type Camera } from '@/shared/api/client';
import { HttpError } from '@/shared/api/http';
import { AddCameraModal } from '@/features/cameras/AddCameraModal';

vi.mock('@/shared/api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/shared/api/client')>();
  return { ...actual, createCamera: vi.fn() };
});

function cameraFixture(overrides: Partial<Camera> = {}): Camera {
  return {
    id: 'server-issued-camera-id',
    label: '301호 침대 A',
    rtsp_url_masked: 'rtsp://operator:***@192.0.2.10:8554/live',
    space_id: null,
    space_name: null,
    floor_name: null,
    backend_camera_id: null,
    status: 'offline',
    created_at: '2026-07-07T00:00:00.000Z',
    ...overrides,
  };
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

function findButton(text: string): HTMLButtonElement | undefined {
  return Array.from(document.querySelectorAll('button')).find((button) => button.textContent === text);
}

function renderModal(
  onCreated = vi.fn(),
  onClose = vi.fn(),
  cameras: Camera[] = [],
): { readonly host: HTMLDivElement; readonly root: ReturnType<typeof createRoot> } {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => {
    root.render(<AddCameraModal open onClose={onClose} onCreated={onCreated} cameras={cameras} />);
  });
  return { host, root };
}

function fillValidForm(): void {
  setInput(document.body, 'label', '301호 침대 A');
  setInput(document.body, 'rtspHost', '192.0.2.10');
}

beforeEach(() => {
  vi.mocked(createCamera).mockReset();
  vi.mocked(createCamera).mockResolvedValue(cameraFixture());
});

describe('AddCameraModal', () => {
  it('blocks submission and explains required camera label validation', () => {
    const { host, root } = renderModal();

    const submitButton = findButton('카메라 등록');
    expect(submitButton?.classList.contains('brand-action')).toBe(true);

    act(() => {
      submitButton?.click();
    });

    expect(document.body.textContent).toContain('카메라 이름을 입력하세요.');

    act(() => root.unmount());
    host.remove();
  });

  it('creates a camera from name and structured RTSP fields without asking for a camera id or space_id', async () => {
    const onClose = vi.fn();
    const onCreated = vi.fn();
    const { host, root } = renderModal(onCreated, onClose);

    expect(document.body.textContent).not.toContain('space_id');
    expect(document.body.textContent).not.toContain('카메라 ID');
    expect((document.querySelector('input[name="rtspPath"]') as HTMLInputElement | null)?.value).toBe('/trackID=1');

    setInput(document.body, 'label', '301호 침대 A');
    setInput(document.body, 'rtspHost', '192.0.2.10');
    setInput(document.body, 'rtspPort', '8554');
    setInput(document.body, 'rtspUsername', 'operator');
    setInput(document.body, 'rtspPassword', 'secret');
    setInput(document.body, 'rtspQuery', 'profile=main');

    expect(document.body.textContent).toContain('rtsp://***@192.0.2.10:8554/trackID=1?profile=***');
    expect(document.body.textContent).not.toContain('operator');
    expect(document.body.textContent).not.toContain('secret');

    await act(async () => {
      findButton('카메라 등록')?.click();
    });

    expect(createCamera).toHaveBeenCalledWith(
      { label: '301호 침대 A', rtsp_url: 'rtsp://operator:secret@192.0.2.10:8554/trackID=1?profile=main' },
      { forceRegister: false },
    );
    expect(onCreated).toHaveBeenCalledWith(expect.objectContaining({ id: 'server-issued-camera-id' }));
    expect(onClose).toHaveBeenCalledTimes(1);

    act(() => root.unmount());
    host.remove();
  });

  it.each([
    ['username only', 'operator', '', 'camera.local', 'rtsp://***@camera.local:554/trackID=1'],
    ['username and password', 'operator', 'secret', 'camera.local', 'rtsp://***@camera.local:554/trackID=1'],
    ['encoded credentials', 'operator name', 'p@ss/word', 'camera.local', 'rtsp://***@camera.local:554/trackID=1'],
    ['no credentials', '', '', 'camera.local', 'rtsp://camera.local:554/trackID=1'],
    ['malformed URL', 'operator', 'secret', '[invalid-host', 'rtsp://***@[invalid-host:554/trackID=1'],
  ])('masks the entire RTSP userinfo for %s', (_case, username, password, hostName, expectedPreview) => {
    const { host, root } = renderModal();
    setInput(document.body, 'rtspHost', hostName);
    setInput(document.body, 'rtspUsername', username);
    setInput(document.body, 'rtspPassword', password);

    const preview = document.querySelector('[data-testid="rtsp-preview"]');
    expect(preview?.textContent).toBe(expectedPreview);
    expect(preview?.closest('form')).not.toBeNull();
    expect(preview?.textContent).not.toContain('operator');
    expect(preview?.textContent).not.toContain('secret');
    expect(preview?.textContent).not.toContain('operator%20name');
    expect(preview?.textContent).not.toContain('p%40ss%2Fword');

    act(() => root.unmount());
    host.remove();
  });

  it.each([
    [
      'every ordinary and credential-shaped value',
      'profile=main&pwd=p%40ss&credential=scope&X-Goog-Signature=deadbeef&AWSAccessKeyId=AKIAEXAMPLE&camera=west',
      'profile=***&pwd=***&credential=***&X-Goog-Signature=***&AWSAccessKeyId=***&camera=***',
    ],
    [
      'repeated, encoded, empty, and key-only parameters',
      'profile=first&profile=second&%70rofile=encoded%2Fvalue&empty=&flag&bad%ZZ=value',
      'profile=***&profile=***&%70rofile=***&empty=***&***&bad%ZZ=***',
    ],
    [
      'values containing encoded and literal equals signs',
      'opaque=one%26two%3Dthree&signed=a=b=c',
      'opaque=***&signed=***',
    ],
  ])('redacts every query value for %s while preserving query key structure', (_case, query, expectedQuery) => {
    const { host, root } = renderModal();
    setInput(document.body, 'rtspHost', 'camera.local');
    setInput(document.body, 'rtspQuery', query);

    const preview = document.querySelector('[data-testid="rtsp-preview"]')?.textContent ?? '';
    expect(preview).toBe(`rtsp://camera.local:554/trackID=1?${expectedQuery}`);
    expect(['main', 'p%40ss', 'scope', 'deadbeef', 'AKIAEXAMPLE', 'encoded%2Fvalue', 'one%26two%3Dthree', 'a=b=c'].some((secret) => preview.includes(secret))).toBe(false);

    act(() => root.unmount());
    host.remove();
  });

  it.each([
    [
      'mixed values, multiple key-only segments, and a fragment',
      'camera.local',
      'operator',
      'camera-password',
      'profile=main&opaque-secret-token&another%2Dsecret#token=fragment-secret',
      'rtsp://***@camera.local:554/trackID=1?profile=***&***&***#***',
      'rtsp://operator:camera-password@camera.local:554/trackID=1?profile=main&opaque-secret-token&another%2Dsecret#token=fragment-secret',
    ],
    [
      'encoded fragment on a malformed URL',
      '[invalid-host',
      '',
      '',
      'first-secret&second%20secret#encoded%2Ffragment%3Dvalue',
      'rtsp://[invalid-host:554/trackID=1?***&***#***',
      'rtsp://[invalid-host:554/trackID=1?first-secret&second%20secret#encoded%2Ffragment%3Dvalue',
    ],
    [
      'a single key-only query segment',
      'camera.local',
      '',
      '',
      'opaque-secret-token',
      'rtsp://camera.local:554/trackID=1?***',
      'rtsp://camera.local:554/trackID=1?opaque-secret-token',
    ],
  ])('redacts %s without changing the submitted RTSP URL', async (_case, hostName, username, password, query, expectedPreview, expectedSubmittedUrl) => {
    const { host, root } = renderModal();
    setInput(document.body, 'label', 'Privacy camera');
    setInput(document.body, 'rtspHost', hostName);
    setInput(document.body, 'rtspUsername', username);
    setInput(document.body, 'rtspPassword', password);
    setInput(document.body, 'rtspQuery', query);

    const preview = document.querySelector('[data-testid="rtsp-preview"]')?.textContent ?? '';
    expect(preview).toBe(expectedPreview);
    expect(['opaque-secret-token', 'another%2Dsecret', 'fragment-secret', 'first-secret', 'second%20secret', 'encoded%2Ffragment%3Dvalue'].some((secret) => preview.includes(secret))).toBe(false);

    await act(async () => findButton('카메라 등록')?.click());
    expect(vi.mocked(createCamera).mock.calls[0]?.[0].rtsp_url).toBe(expectedSubmittedUrl);

    act(() => root.unmount());
    host.remove();
  });

  it('keeps userinfo and query secrets masked for a malformed URL without changing the submitted RTSP value', async () => {
    const { host, root } = renderModal();
    setInput(document.body, 'label', 'Malformed camera');
    setInput(document.body, 'rtspHost', '[invalid-host');
    setInput(document.body, 'rtspUsername', 'operator');
    setInput(document.body, 'rtspPassword', 'camera-password');
    setInput(document.body, 'rtspQuery', 'profile=main&token=query-secret&token=second-secret');

    const preview = document.querySelector('[data-testid="rtsp-preview"]')?.textContent ?? '';
    expect(preview).toBe('rtsp://***@[invalid-host:554/trackID=1?profile=***&token=***&token=***');
    expect(['operator', 'camera-password', 'main', 'query-secret', 'second-secret'].some((secret) => preview.includes(secret))).toBe(false);

    await act(async () => findButton('카메라 등록')?.click());
    const submittedUrl = vi.mocked(createCamera).mock.calls[0]?.[0].rtsp_url;
    expect(submittedUrl === 'rtsp://operator:camera-password@[invalid-host:554/trackID=1?profile=main&token=query-secret&token=second-secret').toBe(true);

    act(() => root.unmount());
    host.remove();
  });

  it('completes registration immediately on a successful probe-before-persist create', async () => {
    const onClose = vi.fn();
    const onCreated = vi.fn();
    const { host, root } = renderModal(onCreated, onClose);
    fillValidForm();

    await act(async () => findButton('카메라 등록')?.click());

    expect(onCreated).toHaveBeenCalledWith(expect.objectContaining({ id: 'server-issued-camera-id' }));
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(document.body.textContent).not.toContain('등록되지 않았습니다');
    act(() => root.unmount());
    host.remove();
  });

  it('keeps values on validation and create failure, validates the port, and suppresses repeated submit', async () => {
    let rejectCreate: ((reason?: unknown) => void) | undefined;
    vi.mocked(createCamera).mockImplementation(() => new Promise((_, reject) => { rejectCreate = reject; }));
    const onCreated = vi.fn();
    const { host, root } = renderModal(onCreated);
    fillValidForm();
    setInput(document.body, 'rtspPort', '70000');

    const submit = findButton('카메라 등록');
    act(() => submit?.click());
    expect(document.body.textContent).toContain('1부터 65535');
    expect((document.querySelector('input[name="label"]') as HTMLInputElement).value).toBe('301호 침대 A');

    setInput(document.body, 'rtspPort', '554');
    act(() => { submit?.click(); submit?.click(); });
    expect(createCamera).toHaveBeenCalledTimes(1);
    await act(async () => rejectCreate?.(new Error('network unreachable')));
    expect(document.body.textContent).toContain('카메라 등록에 실패했습니다.');
    expect(document.querySelector('[role="dialog"]')).not.toBeNull();
    expect((document.querySelector('input[name="label"]') as HTMLInputElement).value).toBe('301호 침대 A');
    expect(onCreated).not.toHaveBeenCalled();

    act(() => root.unmount());
    host.remove();
  });

  it.each(['1e3', '0x50', ' 554', '554 ', '+554', '-554', '554.0', '５５４'])('rejects non-ASCII-decimal RTSP port input %j', (port) => {
    const { host, root } = renderModal();
    fillValidForm();
    setInput(document.body, 'rtspPort', port);

    act(() => findButton('카메라 등록')?.click());

    expect(document.body.textContent).toContain('1부터 65535');
    expect(createCamera).not.toHaveBeenCalled();

    act(() => root.unmount());
    host.remove();
  });

  it('keeps a pending create dialog open across every close attempt and prevents a duplicate submit', async () => {
    let resolveCreate: ((camera: Camera) => void) | undefined;
    vi.mocked(createCamera).mockImplementation(() => new Promise((resolve) => { resolveCreate = resolve; }));
    const onClose = vi.fn();
    const onCreated = vi.fn();
    const { host, root } = renderModal(onCreated, onClose);
    fillValidForm();

    act(() => findButton('카메라 등록')?.click());
    const close = Array.from(document.querySelectorAll<HTMLButtonElement>('[role="dialog"] button')).find((button) => button.textContent === '닫기');
    expect(close?.disabled).toBe(true);
    act(() => close?.click());
    act(() => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
    act(() => document.querySelector('[data-dialog-backdrop]')?.dispatchEvent(new MouseEvent('click', { bubbles: true })));
    act(() => findButton('등록 중...')?.click());

    expect(onClose).not.toHaveBeenCalled();
    expect(document.querySelector('[role="dialog"]')).not.toBeNull();
    expect(createCamera).toHaveBeenCalledTimes(1);

    await act(async () => resolveCreate?.(cameraFixture()));
    expect(onCreated).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
    act(() => root.unmount());
    host.remove();
  });

  it.each([
    ['timeout', '카메라 응답 시간이 초과되었습니다. 네트워크와 전원을 확인하세요.'],
    ['auth', '카메라 인증에 실패했습니다. 아이디와 비밀번호를 확인하세요.'],
    ['decode', '영상 형식을 처리할 수 없습니다. 카메라 스트림 설정을 확인하세요.'],
    [undefined, '카메라 연결을 확인할 수 없습니다. 네트워크와 설정을 확인하세요.'],
  ] as const)('given a 422 probe_failed rejection with error_class %s, blocks the save and shows the reason', async (errorClass, expected) => {
    vi.mocked(createCamera).mockRejectedValue(new HttpError(422, { detail: { error: 'probe_failed', error_class: errorClass } }));
    const onCreated = vi.fn();
    const { host, root } = renderModal(onCreated);
    fillValidForm();

    await act(async () => findButton('카메라 등록')?.click());

    expect(document.body.textContent).toContain('등록되지 않았습니다');
    expect(document.body.textContent).toContain(expected);
    if (errorClass) expect(document.body.textContent).not.toContain(errorClass);
    expect(onCreated).not.toHaveBeenCalled();
    expect(document.querySelector('[role="dialog"]')).not.toBeNull();

    act(() => root.unmount());
    host.remove();
  });

  it('retries the same probe-before-persist request via the primary submit button after a 422', async () => {
    vi.mocked(createCamera)
      .mockRejectedValueOnce(new HttpError(422, { detail: { error: 'probe_failed', error_class: 'timeout' } }))
      .mockResolvedValueOnce(cameraFixture());
    const onCreated = vi.fn();
    const onClose = vi.fn();
    const { host, root } = renderModal(onCreated, onClose);
    fillValidForm();

    await act(async () => findButton('카메라 등록')?.click());
    expect(findButton('다시 시도')).toBeDefined();

    await act(async () => findButton('다시 시도')?.click());

    expect(createCamera).toHaveBeenCalledTimes(2);
    expect(vi.mocked(createCamera).mock.calls[1]).toEqual([
      { label: '301호 침대 A', rtsp_url: 'rtsp://192.0.2.10:554/trackID=1' },
      { forceRegister: false },
    ]);
    expect(onCreated).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);

    act(() => root.unmount());
    host.remove();
  });

  it('offers a subordinate "연결 없이 등록" escape hatch that resubmits with force_register after a 422', async () => {
    vi.mocked(createCamera)
      .mockRejectedValueOnce(new HttpError(422, { detail: { error: 'probe_failed', error_class: 'timeout' } }))
      .mockResolvedValueOnce(cameraFixture({ status: 'offline' }));
    const onCreated = vi.fn();
    const onClose = vi.fn();
    const { host, root } = renderModal(onCreated, onClose);
    fillValidForm();

    await act(async () => findButton('카메라 등록')?.click());
    const forceButton = findButton('연결 없이 등록');
    expect(forceButton).toBeDefined();
    // Visually subordinate to the primary retry action: no brand-action styling.
    expect(forceButton?.className).not.toContain('brand-action');
    expect(findButton('다시 시도')?.className).toContain('brand-action');

    await act(async () => forceButton?.click());

    expect(createCamera).toHaveBeenCalledTimes(2);
    expect(vi.mocked(createCamera).mock.calls[1]).toEqual([
      { label: '301호 침대 A', rtsp_url: 'rtsp://192.0.2.10:554/trackID=1' },
      { forceRegister: true },
    ]);
    expect(onCreated).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);

    act(() => root.unmount());
    host.remove();
  });

  it('given a 409 duplicate rejection, shows the duplicate message with the existing camera label prominently displayed and no dead navigation link', async () => {
    vi.mocked(createCamera).mockRejectedValue(
      new HttpError(409, { detail: { error: 'duplicate_camera', existing_camera_id: 'cam-existing-1', existing_label: '302호 침대 B' } }),
    );
    const onCreated = vi.fn();
    const { host, root } = renderModal(onCreated);
    fillValidForm();

    await act(async () => findButton('카메라 등록')?.click());

    expect(document.body.textContent).toContain('이미 등록된 카메라입니다');
    expect(document.body.textContent).toContain('302호 침대 B');
    expect(onCreated).not.toHaveBeenCalled();

    // The 409 alert must not offer a full-page-reload link to a destination that drops the
    // camera id during canonicalization (see dashboardLocation.ts) — it lost its only reason to
    // exist, so the label itself is the source of truth for "which camera" instead.
    expect(document.querySelector('a')).toBeNull();
    expect(document.body.textContent).not.toContain('기존 카메라로 이동');

    act(() => root.unmount());
    host.remove();
  });

  it('warns, without blocking submission, when the entered label case-insensitively matches an already-registered camera (R2.2)', async () => {
    const roster = [cameraFixture({ id: 'cam-existing', label: '301호 침대 A' })];
    const onCreated = vi.fn();
    const { host, root } = renderModal(onCreated, vi.fn(), roster);

    setInput(document.body, 'label', '301호 침대 a');
    setInput(document.body, 'rtspHost', '192.0.2.11');

    const warning = document.querySelector('[role="status"]');
    expect(warning?.textContent).toContain('301호 침대 A');
    expect(warning?.className).not.toContain('status-danger');

    const submitButton = findButton('카메라 등록');
    expect(submitButton?.disabled).toBe(false);

    await act(async () => submitButton?.click());
    expect(createCamera).toHaveBeenCalledWith(
      { label: '301호 침대 a', rtsp_url: 'rtsp://192.0.2.11:554/trackID=1' },
      { forceRegister: false },
    );
    expect(onCreated).toHaveBeenCalled();

    act(() => root.unmount());
    host.remove();
  });

  it('shows no label-duplicate warning when the label does not match any registered camera', () => {
    const roster = [cameraFixture({ id: 'cam-existing', label: '301호 침대 A' })];
    const { host, root } = renderModal(vi.fn(), vi.fn(), roster);

    setInput(document.body, 'label', '302호 침대 B');

    expect(document.body.textContent).not.toContain('이름이 같습니다');

    act(() => root.unmount());
    host.remove();
  });

  it('defaults to an empty roster and shows no label warning when the camera prop is omitted', () => {
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => {
      root.render(<AddCameraModal open onClose={vi.fn()} onCreated={vi.fn()} />);
    });

    setInput(document.body, 'label', '301호 침대 A');
    expect(document.body.textContent).not.toContain('이름이 같습니다');

    act(() => root.unmount());
    host.remove();
  });

  it('parses a pasted full RTSP URL into the structured fields and collapses them', async () => {
    const { host, root } = renderModal();
    setInput(document.body, 'label', '301호 침대 A');
    setInput(document.body, 'rtspPaste', 'rtsp://operator:secret@192.0.2.10:8554/Streaming/Channels/101?subtype=0');

    expect(document.querySelector('input[name="rtspHost"]')).toBeNull();
    const preview = document.querySelector('[data-testid="rtsp-preview"]')?.textContent ?? '';
    expect(preview).toBe('rtsp://***@192.0.2.10:8554/Streaming/Channels/101?subtype=***');
    expect(preview).not.toContain('operator');
    expect(preview).not.toContain('secret');

    await act(async () => findButton('카메라 등록')?.click());
    expect(vi.mocked(createCamera).mock.calls[0]?.[0].rtsp_url).toBe('rtsp://operator:secret@192.0.2.10:8554/Streaming/Channels/101?subtype=0');

    act(() => root.unmount());
    host.remove();
  });

  it('leaves fields expanded and editable, and never discards the pasted text, when the paste cannot be parsed', () => {
    const { host, root } = renderModal();
    setInput(document.body, 'rtspPaste', 'not-a-real-rtsp://url with spaces');

    expect(document.body.textContent).toContain('RTSP 주소를 인식하지 못했습니다.');
    expect((document.querySelector('input[name="rtspPaste"]') as HTMLInputElement).value).toBe('not-a-real-rtsp://url with spaces');
    expect(document.querySelector('input[name="rtspHost"]')).not.toBeNull();

    setInput(document.body, 'rtspHost', 'camera.local');
    expect((document.querySelector('input[name="rtspHost"]') as HTMLInputElement).value).toBe('camera.local');

    act(() => root.unmount());
    host.remove();
  });

  it('uses the bounded shared dialog surface at a 375px viewport', () => {
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 375 });
    const { host, root } = renderModal();
    expect(document.querySelector('.accessible-dialog')).not.toBeNull();
    expect(document.querySelector('[data-dialog-backdrop]')).not.toBeNull();
    expect(document.body.scrollWidth).toBeLessThanOrEqual(375);
    act(() => root.unmount());
    host.remove();
  });

  it('initially focuses the camera name input instead of the close button', () => {
    const { host, root } = renderModal();
    expect(document.activeElement).toBe(document.querySelector('input[name="label"]'));
    act(() => root.unmount());
    host.remove();
  });
});
