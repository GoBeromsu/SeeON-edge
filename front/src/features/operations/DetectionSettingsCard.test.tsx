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
    regions: [
      { id: 'bed-1', polygon: [[0, 0], [10, 0], [10, 10]], origin: 'model' },
      { id: 'bed-2', polygon: [[20, 0], [30, 0], [30, 10]], origin: 'manual' },
    ],
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

let overlaySelection = { person: true, bed: true };
const mountedRoots = new Set<Root>();

function installFetchMock(detectionResponse: 'success' | 'error' | 'pending' = 'success'): void {
  vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input.toString();
    if (url.includes('/detection-settings')) {
      if (detectionResponse === 'error') {
        return Promise.resolve({ ok: false, status: 500, json: async () => ({}) });
      }
      if (detectionResponse === 'pending') {
        return new Promise(() => undefined);
      }
      return Promise.resolve({ ok: true, status: 200, json: async () => detectionSettings });
    }
    if (url.includes('/streams/') && url.includes('/pose')) {
      if (init?.method === 'POST') {
        const body = init.body ? JSON.parse(init.body as string) : {};
        overlaySelection = body;
      }
      return Promise.resolve({ ok: true, status: 200, json: async () => overlaySelection });
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
  mountedRoots.add(root);
  await act(async () => {
    root.render(<DetectionSettingsCard camera={camera} onEditBedZones={vi.fn()} />);
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  });
  return { host, root };
}

afterEach(() => {
  act(() => {
    for (const root of mountedRoots) root.unmount();
  });
  mountedRoots.clear();
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
  overlaySelection = { person: true, bed: true };
  detectionSettings.domains.bed_exit.on = false;
});

describe('DetectionSettingsCard (operations)', () => {
  it('shows accessible active and off status icons for online camera domains', async () => {
    installFetchMock();
    const { host } = await render(onlineCamera);

    expect(host.textContent).toContain('탐지 이벤트');
    expect(host.querySelector('svg[aria-label="탐지 중"]')).not.toBeNull();
    expect(host.querySelector('svg[aria-label="꺼짐"]')).not.toBeNull();
    expect(host.querySelector('svg[aria-label="중단됨"]')).toBeNull();
  });

  it('never marks enabled bed-exit as detecting without persisted geometry, while fall remains detecting', async () => {
    installFetchMock();
    detectionSettings.domains.bed_exit.on = true;
    const cameraWithoutBedZone: Camera = { ...onlineCamera, bed_zone: null };
    const { host } = await render(cameraWithoutBedZone);

    expect(host.querySelector('svg[aria-label="침대 영역 미설정"]')).not.toBeNull();
    expect(host.querySelector('button[aria-label="침대 영역 편집"]')).not.toBeNull();
    expect(host.querySelectorAll('svg[aria-label="탐지 중"]')).toHaveLength(1);
  });

  it('marks enabled bed-exit as detecting when persisted geometry exists', async () => {
    installFetchMock();
    detectionSettings.domains.bed_exit.on = true;
    const { host } = await render(onlineCamera);

    expect(host.querySelector('svg[aria-label="침대 영역 미설정"]')).toBeNull();
    expect(host.querySelectorAll('svg[aria-label="탐지 중"]')).toHaveLength(2);
    expect(host.querySelector('button[aria-label="침대 영역 편집"]')).not.toBeNull();
  });

  it('keeps the disabled state ahead of missing bed geometry', async () => {
    installFetchMock();
    detectionSettings.domains.bed_exit.on = false;
    const { host } = await render({ ...onlineCamera, bed_zone: null });

    expect(host.querySelector('svg[aria-label="꺼짐"]')).not.toBeNull();
    expect(host.querySelector('svg[aria-label="침대 영역 미설정"]')).toBeNull();
    expect(host.querySelector('button[aria-label="침대 영역 편집"]')).not.toBeNull();
  });

  it('keeps bed-zone editing actionable with saved geometry without changing readiness on click', async () => {
    installFetchMock();
    detectionSettings.domains.bed_exit.on = true;
    const onEditBedZones = vi.fn();
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    mountedRoots.add(root);
    act(() => root.render(
      <DetectionSettingsCard
        camera={onlineCamera}
        onEditBedZones={onEditBedZones}
      />,
    ));
    await flush();

    const action = host.querySelector<HTMLButtonElement>('button[aria-label="침대 영역 편집"]');
    expect(action).toBeTruthy();
    act(() => action?.click());
    expect(onEditBedZones).toHaveBeenCalledOnce();
    expect(host.querySelector('svg[aria-label="침대 영역 미설정"]')).toBeNull();
  });

  it('shows an accessible paused icon for every domain when the camera is offline', async () => {
    installFetchMock();
    const { host } = await render(offlineCamera);

    expect(host.querySelectorAll('svg[aria-label="중단됨"]')).toHaveLength(2);
    expect(host.querySelector('svg[aria-label="탐지 중"]')).toBeNull();
    expect(host.querySelector('button[aria-label="침대 영역 편집"]')).not.toBeNull();
  });

  it.each(['pending', 'error'] as const)('keeps bed-zone editing available while settings are %s', async (response) => {
    installFetchMock(response);
    const { host } = await render(onlineCamera);

    const action = host.querySelector<HTMLButtonElement>('button[aria-label="침대 영역 편집"]');
    expect(action).not.toBeNull();
    expect(action?.title).toBe('침대 영역 편집');
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
