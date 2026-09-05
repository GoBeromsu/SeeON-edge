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
  bed_zone: {
    polygon: [[0, 0], [10, 0], [10, 10], [0, 10]],
    image_width: 1920,
    image_height: 1080,
    recognized_at: '2026-09-05T00:00:00Z',
  },
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
  detectionSettings.domains.bed_exit.on = false;
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

  it('never marks enabled bed-exit as detecting without persisted geometry, while fall remains detecting', async () => {
    installFetchMock();
    detectionSettings.domains.bed_exit.on = true;
    const cameraWithoutBedZone: Camera = { ...onlineCamera, bed_zone: null };
    const { host } = await render(cameraWithoutBedZone);

    expect(host.textContent).toContain('침대 영역 미설정');
    expect(host.textContent).toContain('침대 영역 인식');
    expect(host.textContent?.match(/탐지 중/g)).toHaveLength(1);
  });

  it('marks enabled bed-exit as detecting when persisted geometry exists', async () => {
    installFetchMock();
    detectionSettings.domains.bed_exit.on = true;
    const { host } = await render(onlineCamera);

    expect(host.textContent).not.toContain('침대 영역 미설정');
    expect(host.textContent?.match(/탐지 중/g)).toHaveLength(2);
  });

  it('keeps the disabled state ahead of missing bed geometry', async () => {
    installFetchMock();
    detectionSettings.domains.bed_exit.on = false;
    const { host } = await render({ ...onlineCamera, bed_zone: null });

    expect(host.textContent).toContain('꺼짐');
    expect(host.textContent).not.toContain('침대 영역 미설정');
  });

  it('makes recognition actionable without treating the click itself as successful geometry', async () => {
    installFetchMock();
    detectionSettings.domains.bed_exit.on = true;
    const onRecognizeBedZone = vi.fn();
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(
      <DetectionSettingsCard
        camera={{ ...onlineCamera, bed_zone: null }}
        onRecognizeBedZone={onRecognizeBedZone}
      />,
    ));
    await flush();

    const action = Array.from(host.querySelectorAll('button')).find((button) => button.textContent === '침대 영역 인식');
    expect(action).toBeTruthy();
    act(() => action?.click());
    expect(onRecognizeBedZone).toHaveBeenCalledOnce();
    expect(host.textContent).toContain('침대 영역 미설정');
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
