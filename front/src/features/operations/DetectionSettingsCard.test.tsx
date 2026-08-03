import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { DetectionSettingsCard } from '@/features/operations/DetectionSettingsCard';
import type { Camera, DetectionSettings } from '@/shared/api/client';

const onlineCamera: Camera = {
  id: 'cam-1',
  label: '101호',
  rtsp_url_masked: 'rtsp://redacted-camera/a',
  floor_name: '1층',
  status: 'online',
  created_at: null,
};

const offlineCamera: Camera = { ...onlineCamera, id: 'cam-2', status: 'offline' };

const detectionSettings: DetectionSettings = {
  domains: {
    fall: { on: true, mode: 'always', start: null, end: null },
    bed_exit: { on: false, mode: 'always', start: null, end: null },
  },
};

let overlayMode = 'none';

function installFetchMock(): void {
  vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input.toString();
    if (url.includes('/detection-settings')) {
      return Promise.resolve({ ok: true, status: 200, json: async () => detectionSettings });
    }
    if (url.includes('/streams/') && url.includes('/pose')) {
      if (init?.method === 'POST') {
        const body = init.body ? JSON.parse(init.body as string) : {};
        overlayMode = body.mode ?? overlayMode;
      }
      return Promise.resolve({ ok: true, status: 200, json: async () => ({ mode: overlayMode }) });
    }
    return Promise.reject(new Error(`unexpected fetch: ${url}`));
  }));
}

async function flush(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  });
}

async function render(camera: Camera): Promise<{ host: HTMLDivElement; root: Root }> {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<DetectionSettingsCard camera={camera} />));
  await flush();
  return { host, root };
}

afterEach(() => {
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
  overlayMode = 'none';
});

describe('DetectionSettingsCard (operations)', () => {
  it('shows "탐지 중" for an on domain on an online camera, and "꺼짐" for an off domain', async () => {
    installFetchMock();
    const { host } = await render(onlineCamera);

    expect(host.textContent).toContain('탐지 이벤트');
    expect(host.textContent).toContain('탐지 중');
    expect(host.textContent).toContain('꺼짐');
    expect(host.textContent).not.toContain('중단됨');
  });

  it('shows "중단됨" for every domain when the camera is offline, regardless of the domain\'s own on/off flag', async () => {
    installFetchMock();
    const { host } = await render(offlineCamera);

    expect(host.textContent).toContain('중단됨');
    expect(host.textContent).not.toContain('탐지 중');
  });

  it('navigates to the settings page when the header gear icon is clicked', async () => {
    installFetchMock();
    window.history.replaceState(null, '', '/?page=operations&camera=cam-1');
    const { host } = await render(onlineCamera);

    const gearButton = host.querySelector('button[aria-label="탐지 설정으로 이동"]') as HTMLButtonElement;
    expect(gearButton).not.toBeNull();
    act(() => gearButton.click());

    expect(window.location.search).toContain('page=settings');
  });
});
