import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ToastViewport, toast } from '@/shared/ui/Toast';

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  // Flush any pending auto-dismiss timers so module-level toast state doesn't leak between tests.
  act(() => {
    vi.advanceTimersByTime(2200);
  });
  document.body.innerHTML = '';
  vi.useRealTimers();
});

describe('ToastViewport', () => {
  it('renders a success toast and auto-dismisses it after 2.2s', () => {
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(<ToastViewport />));

    act(() => toast.success('저장되었습니다'));
    const item = host.querySelector('.toast');
    expect(item?.textContent).toBe('저장되었습니다');
    expect(item?.className).toContain('toast-success');

    act(() => {
      vi.advanceTimersByTime(2200);
    });
    expect(host.querySelector('.toast')).toBeNull();
    act(() => root.unmount());
  });

  it('renders an error toast with the error variant class', () => {
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(<ToastViewport />));

    act(() => toast.error('오류가 발생했습니다'));
    const item = host.querySelector('.toast');
    expect(item?.textContent).toBe('오류가 발생했습니다');
    expect(item?.className).toContain('toast-error');
    act(() => root.unmount());
  });

  it('stacks multiple toasts', () => {
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(<ToastViewport />));

    act(() => {
      toast.success('첫 번째');
      toast.success('두 번째');
    });
    const items = Array.from(host.querySelectorAll('.toast')).map((el) => el.textContent);
    expect(items).toEqual(['첫 번째', '두 번째']);
    act(() => root.unmount());
  });
});
