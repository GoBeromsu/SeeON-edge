import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ClipExportSettingsCard } from '@/features/settings/ClipExportSettingsCard';
import { saveRuntimeSettings } from '@/shared/api/client';
import { HttpError } from '@/shared/api/http';
import type { RuntimeSettings } from '@/shared/api/client';
import type { PollingResource } from '@/shared/api/usePollingResource';

vi.mock('@/shared/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return { ...actual, saveRuntimeSettings: vi.fn() };
});

const settings: RuntimeSettings = { clip_export_enabled: false, version: 0 };

function resource(overrides: Partial<PollingResource<RuntimeSettings>> = {}): PollingResource<RuntimeSettings> {
  return {
    status: 'success',
    data: settings,
    error: null,
    lastSuccessAt: Date.now(),
    refreshing: false,
    retry: vi.fn(),
    replace: vi.fn(),
    ...overrides,
  };
}

function render(value = resource()) {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<ClipExportSettingsCard resource={value} />));
  return { host, root, value };
}

function button(host: ParentNode, label: string): HTMLButtonElement {
  const found = Array.from(host.querySelectorAll('button')).find((item) => item.textContent === label);
  if (!found) throw new Error(`missing button ${label}`);
  return found;
}

beforeEach(() => {
  vi.mocked(saveRuntimeSettings).mockReset();
  vi.mocked(saveRuntimeSettings).mockResolvedValue({ clip_export_enabled: true, version: 1 });
});

afterEach(() => {
  document.body.innerHTML = '';
  vi.restoreAllMocks();
});

describe('ClipExportSettingsCard', () => {
  it('defaults OFF and saves only after explicit confirmation', async () => {
    const { host, root, value } = render();
    const toggle = host.querySelector<HTMLInputElement>('input[type="checkbox"]');

    expect(toggle?.checked).toBe(false);
    expect(toggle?.getAttribute('aria-label')).toContain('클립 내보내기');
    act(() => toggle?.click());
    expect(saveRuntimeSettings).not.toHaveBeenCalled();

    await act(async () => button(host, '저장').click());

    expect(saveRuntimeSettings).toHaveBeenCalledWith({
      clip_export_enabled: true,
      expected_version: 0,
    });
    expect(value.replace).toHaveBeenCalledWith({ clip_export_enabled: true, version: 1 });
    expect(value.retry).toHaveBeenCalled();
    act(() => root.unmount());
  });

  it('shows loading, save progress, and an inline save error', async () => {
    const loading = render(resource({ status: 'loading', data: null }));
    expect(loading.host.textContent).toContain('불러오는 중');
    act(() => loading.root.unmount());

    vi.mocked(saveRuntimeSettings).mockRejectedValue(new Error('offline'));
    const rendered = render();
    act(() => rendered.host.querySelector<HTMLInputElement>('input[type="checkbox"]')?.click());
    await act(async () => button(rendered.host, '저장').click());

    expect(rendered.host.querySelector('[role="alert"]')?.textContent).toContain('저장하지 못했습니다');
    expect(rendered.host.querySelector<HTMLInputElement>('input[type="checkbox"]')?.checked).toBe(true);
    act(() => rendered.root.unmount());
  });

  it('reports a stale-write conflict and adopts the server current value', async () => {
    vi.mocked(saveRuntimeSettings).mockRejectedValue(new HttpError(409, {
      detail: {
        code: 'runtime_settings_version_conflict',
        current: { clip_export_enabled: true, version: 3 },
      },
    }));
    const { host, root, value } = render();
    act(() => host.querySelector<HTMLInputElement>('input[type="checkbox"]')?.click());

    await act(async () => button(host, '저장').click());

    expect(host.querySelector('[role="alert"]')?.textContent).toContain('다른 운영자가 먼저 변경했습니다');
    expect(value.replace).toHaveBeenCalledWith({ clip_export_enabled: true, version: 3 });
    expect(value.retry).toHaveBeenCalled();
    act(() => root.unmount());
  });

  it('offers an accessible retry when the initial request fails', () => {
    const { host, root, value } = render(resource({ status: 'error', data: null }));
    act(() => button(host, '다시 시도').click());
    expect(value.retry).toHaveBeenCalled();
    act(() => root.unmount());
  });
});
