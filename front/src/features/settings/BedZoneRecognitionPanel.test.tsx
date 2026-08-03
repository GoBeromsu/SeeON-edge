import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { recognizeBedZone } from '@/shared/api/client';
import { HttpError } from '@/shared/api/http';
import { BedZoneRecognitionPanel } from '@/features/settings/BedZoneRecognitionPanel';
import type { BedZone } from '@/shared/api/client';

vi.mock('@/shared/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return { ...actual, recognizeBedZone: vi.fn() };
});

const bedZone: BedZone = {
  polygon: [[0, 0], [100, 0], [100, 100], [0, 100]],
  image_width: 1920,
  image_height: 1080,
  recognized_at: '2026-08-01T00:00:00Z',
};

function render(zone: BedZone | null, onRecognized = vi.fn()) {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<BedZoneRecognitionPanel cameraId="cam-1" bedZone={zone} onRecognized={onRecognized} />));
  return { host, root, onRecognized };
}

function findButton(host: HTMLElement, label: string): HTMLButtonElement {
  const button = Array.from(host.querySelectorAll('button')).find((candidate) => candidate.textContent === label);
  if (!button) throw new Error(`missing button ${label}`);
  return button;
}

beforeEach(() => {
  vi.mocked(recognizeBedZone).mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('BedZoneRecognitionPanel', () => {
  it('shows "인식 필요" copy and an "인식 시작" trigger when no bed zone is recognized yet', () => {
    const { host, root } = render(null);
    expect(host.textContent).toContain('침대 영역 인식이 필요합니다.');
    expect(findButton(host, '인식 시작')).toBeTruthy();
    act(() => root.unmount());
  });

  it('overlays the polygon and offers "다시 인식" once a bed zone is recognized', () => {
    const { host, root } = render(bedZone);
    expect(host.textContent).toContain('침대 · 자동 인식됨');
    expect(host.querySelector('polygon')).not.toBeNull();
    expect(findButton(host, '다시 인식')).toBeTruthy();
    act(() => root.unmount());
  });

  it('calls recognizeBedZone and reports the result on success', async () => {
    vi.mocked(recognizeBedZone).mockResolvedValue(bedZone);
    const { host, root, onRecognized } = render(null);

    await act(async () => findButton(host, '인식 시작').click());

    expect(recognizeBedZone).toHaveBeenCalledWith('cam-1');
    expect(onRecognized).toHaveBeenCalledWith(bedZone);
    act(() => root.unmount());
  });

  it('shows the "인식 필요" failure message on a 422 bed_not_found rejection', async () => {
    vi.mocked(recognizeBedZone).mockRejectedValue(
      new HttpError(422, { detail: { error_class: 'bed_not_found' } }),
    );
    const { host, root } = render(null);

    await act(async () => findButton(host, '인식 시작').click());

    expect(host.querySelector('[role="alert"]')?.textContent).toContain('침대를 찾지 못했습니다');
    act(() => root.unmount());
  });

  it('shows a generic failure message for other errors', async () => {
    vi.mocked(recognizeBedZone).mockRejectedValue(new Error('network down'));
    const { host, root } = render(null);

    await act(async () => findButton(host, '인식 시작').click());

    expect(host.querySelector('[role="alert"]')?.textContent).toContain('인식에 실패했습니다');
    act(() => root.unmount());
  });
});
