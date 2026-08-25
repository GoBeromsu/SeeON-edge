import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { saveDetectionSettings } from '@/shared/api/client';
import { DetectionSettingsCard } from '@/features/settings/DetectionSettingsCard';
import { toast } from '@/shared/ui/Toast';
import type { DetectionSettings } from '@/shared/api/client';
import type { PollingResource } from '@/shared/api/usePollingResource';

vi.mock('@/shared/api/client', async () => {
  const { withOverrides } = await vi.importActual<typeof import('@/test/moduleMock')>('@/test/moduleMock');
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return withOverrides(actual, { saveDetectionSettings: vi.fn() });
});

const baseSettings: DetectionSettings = {
  domains: {
    fall: { on: true, mode: 'always', start: null, end: null },
    bed_exit: { on: true, mode: 'window', start: '21:00', end: '06:00' },
  },
};

function makeResource(overrides: Partial<PollingResource<DetectionSettings>> = {}): PollingResource<DetectionSettings> {
  return {
    status: 'success',
    data: baseSettings,
    error: null,
    lastSuccessAt: Date.now(),
    refreshing: false,
    retry: vi.fn(),
    replace: vi.fn(),
    ...overrides,
  };
}

function render(resource: PollingResource<DetectionSettings> = makeResource()) {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<DetectionSettingsCard resource={resource} />));
  return { host, root, resource };
}

function findButton(host: HTMLElement, label: string): HTMLButtonElement {
  const button = Array.from(host.querySelectorAll('button')).find((candidate) => candidate.textContent === label);
  if (!button) throw new Error(`missing button ${label}`);
  return button;
}

beforeEach(() => {
  vi.mocked(saveDetectionSettings).mockReset();
  vi.mocked(saveDetectionSettings).mockResolvedValue(baseSettings);
});

afterEach(() => {
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

describe('DetectionSettingsCard', () => {
  it('shows a loading message before the first successful fetch', () => {
    const { host, root } = render(makeResource({ status: 'loading', data: null }));
    expect(host.textContent).toContain('불러오는 중입니다');
    act(() => root.unmount());
  });

  it('shows a retry affordance when the initial fetch fails', () => {
    const { host, root, resource } = render(makeResource({ status: 'error', data: null }));
    act(() => findButton(host, '다시 시도').click());
    expect(resource.retry).toHaveBeenCalled();
    act(() => root.unmount());
  });

  it('renders the read-mode schedule in 침대 이탈 → 낙상 order with 탐지 중 badges', () => {
    const { host, root } = render();
    const text = host.textContent ?? '';
    expect(text.indexOf('침대 이탈')).toBeLessThan(text.indexOf('낙상'));
    expect(text).toContain('21:00–06:00');
    expect(text).toContain('항상');
    expect(host.querySelectorAll('form select, input[type="checkbox"]').length).toBe(0);
    act(() => root.unmount());
  });

  it('shows 미탐지 for a disabled domain', () => {
    const disabled: DetectionSettings = {
      domains: { ...baseSettings.domains, fall: { ...baseSettings.domains.fall, on: false } },
    };
    const { host, root } = render(makeResource({ data: disabled }));
    expect(host.textContent).toContain('미탐지');
    act(() => root.unmount());
  });

  it('opens edit mode from the pencil button and saves immediately on a toggle change', async () => {
    const { host, root, resource } = render();

    act(() => host.querySelector<HTMLButtonElement>('[aria-label="탐지 설정 편집"]')?.click());
    const checkboxes = Array.from(host.querySelectorAll<HTMLInputElement>('input[type="checkbox"]'));
    expect(checkboxes.length).toBe(2);

    await act(async () => checkboxes[0].click());

    expect(saveDetectionSettings).toHaveBeenCalledWith({
      domains: { ...baseSettings.domains, bed_exit: { ...baseSettings.domains.bed_exit, on: false } },
    });
    expect(resource.retry).toHaveBeenCalled();
    act(() => root.unmount());
  });

  it('reveals HH:MM time inputs only for a domain in window mode', () => {
    const { host, root } = render();
    act(() => host.querySelector<HTMLButtonElement>('[aria-label="탐지 설정 편집"]')?.click());

    const timeInputs = host.querySelectorAll('input[type="time"]');
    expect(timeInputs.length).toBe(2);
    act(() => root.unmount());
  });

  it('reverts the draft and shows an error toast when saving fails', async () => {
    vi.mocked(saveDetectionSettings).mockRejectedValue(new Error('boom'));
    const errorSpy = vi.spyOn(toast, 'error');
    const { host, root } = render();

    act(() => host.querySelector<HTMLButtonElement>('[aria-label="탐지 설정 편집"]')?.click());
    const checkboxes = Array.from(host.querySelectorAll<HTMLInputElement>('input[type="checkbox"]'));

    await act(async () => checkboxes[0].click());

    expect(errorSpy).toHaveBeenCalledWith('탐지 설정을 저장하지 못했습니다.');
    expect((host.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')[0]).checked).toBe(true);
    act(() => root.unmount());
  });

  it('closes edit mode via the 완료 button', () => {
    const { host, root } = render();
    act(() => host.querySelector<HTMLButtonElement>('[aria-label="탐지 설정 편집"]')?.click());
    act(() => findButton(host, '완료').click());
    expect(host.querySelectorAll('input[type="checkbox"]').length).toBe(0);
    act(() => root.unmount());
  });
});
