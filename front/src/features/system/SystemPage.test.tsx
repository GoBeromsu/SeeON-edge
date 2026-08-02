import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { SystemPage } from '@/features/system/SystemPage';

const { useSystemResource, useStatusResource, useConnectionResource } = vi.hoisted(() => ({
  useSystemResource: vi.fn(() => ({ status: 'idle' as const, data: null, error: null, lastSuccessAt: null, refreshing: false, retry: vi.fn() })),
  useStatusResource: vi.fn(() => ({ status: 'idle' as const, data: null, error: null, lastSuccessAt: null, refreshing: false, retry: vi.fn() })),
  useConnectionResource: vi.fn(() => ({ status: 'idle' as const, data: null, error: null, lastSuccessAt: null, refreshing: false, retry: vi.fn() })),
}));

vi.mock('@/shared/api/usePollingResource', () => ({ useSystemResource, useStatusResource, useConnectionResource }));

afterEach(() => vi.clearAllMocks());

describe('SystemPage', () => {
  it('polls system, typed status, and connection settings only while the page is active', () => {
    const host = document.createElement('div'); document.body.append(host); const root = createRoot(host);
    act(() => root.render(<SystemPage active />));
    expect(useSystemResource).toHaveBeenLastCalledWith(true);
    expect(useStatusResource).toHaveBeenLastCalledWith(true);
    expect(useConnectionResource).toHaveBeenLastCalledWith(true);
    act(() => root.render(<SystemPage active={false} />));
    expect(useSystemResource).toHaveBeenLastCalledWith(false);
    expect(useStatusResource).toHaveBeenLastCalledWith(false);
    expect(useConnectionResource).toHaveBeenLastCalledWith(false);
    act(() => root.unmount()); host.remove();
  });
});
