import { act, createRef } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { AccessibleDialog } from '@/shared/ui/AccessibleDialog';

afterEach(() => {
  document.body.innerHTML = '';
  document.body.style.overflow = '';
  document.documentElement.removeAttribute('tabindex');
});

describe('AccessibleDialog', () => {
  it('labels, focuses, traps, closes safely, unlocks scroll, and restores focus', () => {
    const onClose = vi.fn();
    const invoker = document.createElement('button');
    invoker.textContent = '열기';
    document.body.append(invoker);
    invoker.focus();
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(<AccessibleDialog open title="삭제 확인" onClose={onClose}><button>취소</button><button>삭제</button></AccessibleDialog>));
    const dialog = document.querySelector('[role="dialog"]');
    const buttons = dialog?.querySelectorAll('button') ?? [];
    expect(dialog?.getAttribute('aria-modal')).toBe('true');
    expect(dialog?.getAttribute('aria-labelledby')).toBeTruthy();
    expect(document.activeElement).toBe(buttons[0]);
    expect(document.body.style.overflow).toBe('hidden');

    act(() => buttons[buttons.length - 1]?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true })));
    expect(document.activeElement).toBe(buttons[0]);
    act(() => buttons[0]?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', shiftKey: true, bubbles: true })));
    expect(document.activeElement).toBe(buttons[buttons.length - 1]);
    act(() => dialog?.dispatchEvent(new MouseEvent('click', { bubbles: true })));
    expect(onClose).not.toHaveBeenCalled();
    act(() => document.querySelector('[data-dialog-backdrop]')?.dispatchEvent(new MouseEvent('click', { bubbles: true })));
    expect(onClose).toHaveBeenCalledTimes(1);
    act(() => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
    expect(onClose).toHaveBeenCalledTimes(2);

    act(() => root.render(<AccessibleDialog open={false} title="삭제 확인" onClose={onClose}><button>취소</button></AccessibleDialog>));
    expect(document.body.style.overflow).toBe('');
    expect(document.activeElement).toBe(invoker);
    act(() => root.unmount());
  });

  it('defaults to the unsized width class and applies a size variant class when requested', () => {
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(<AccessibleDialog open title="기본 크기" onClose={vi.fn()}><button>닫기</button></AccessibleDialog>));
    let dialog = document.querySelector('[role="dialog"]');
    expect(dialog?.getAttribute('data-size')).toBe('default');
    expect(dialog?.className.trim()).toBe('accessible-dialog');

    act(() => root.render(<AccessibleDialog open title="큰 크기" onClose={vi.fn()} size="xl"><button>닫기</button></AccessibleDialog>));
    dialog = document.querySelector('[role="dialog"]');
    expect(dialog?.getAttribute('data-size')).toBe('xl');
    expect(dialog?.classList.contains('accessible-dialog--xl')).toBe(true);
    act(() => root.unmount());
  });

  it('supports bottom-sheet presentation and traps tabbing from the focused heading', () => {
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(<AccessibleDialog open title="카메라 정보" variant="sheet" onClose={vi.fn()} initialFocus="heading"><button>취소</button><button>저장</button></AccessibleDialog>));
    const dialog = document.querySelector('[role="dialog"]');
    const heading = dialog?.querySelector('h2');
    const buttons = dialog?.querySelectorAll('button') ?? [];
    expect(dialog?.getAttribute('data-variant')).toBe('sheet');
    expect(document.activeElement).toBe(heading);

    act(() => heading?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true })));
    expect(document.activeElement).toBe(buttons[0]);
    heading?.focus();
    act(() => heading?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', shiftKey: true, bubbles: true })));
    expect(document.activeElement).toBe(buttons[buttons.length - 1]);
    act(() => root.unmount());
  });

  it('keeps focus on the heading when the dialog has no controls beyond the close button', () => {
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(<AccessibleDialog open title="알림" onClose={vi.fn()} initialFocus="heading"><p>완료되었습니다.</p></AccessibleDialog>));
    const dialog = document.querySelector('[role="dialog"]');
    const heading = dialog?.querySelector('h2');
    const closeButton = dialog?.querySelector('.accessible-dialog-close');
    expect(document.activeElement).toBe(heading);

    // 공용 셸의 닫기(X) 버튼이 유일한 컨트롤이므로, 탭은 그쪽으로 이동한다.
    const tab = new KeyboardEvent('keydown', { key: 'Tab', bubbles: true, cancelable: true });
    act(() => heading?.dispatchEvent(tab));
    expect(tab.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(closeButton);
    act(() => root.unmount());
  });

  it('shows a labeled close (X) button that calls onClose, in every dialog', () => {
    const onClose = vi.fn();
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(<AccessibleDialog open title="클립 재생" onClose={onClose}><p>내용</p></AccessibleDialog>));
    const closeButton = document.querySelector<HTMLButtonElement>('[role="dialog"] .accessible-dialog-close');
    expect(closeButton?.getAttribute('aria-label')).toBe('닫기');

    act(() => closeButton?.click());
    expect(onClose).toHaveBeenCalledTimes(1);
    act(() => root.unmount());
  });

  it('focuses an explicitly referenced control', () => {
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    const initialFocusRef = createRef<HTMLInputElement>();
    act(() => root.render(
      <AccessibleDialog open title="카메라 추가" onClose={vi.fn()} initialFocusRef={initialFocusRef}>
        <button>닫기</button>
        <input ref={initialFocusRef} aria-label="카메라 이름" />
      </AccessibleDialog>,
    ));
    expect(document.activeElement).toBe(initialFocusRef.current);
    act(() => root.unmount());
  });

  it('focuses the guaranteed page fallback when the invoker is removed', () => {
    const main = document.createElement('main');
    main.id = 'main-content';
    const heading = document.createElement('h1');
    heading.tabIndex = -1;
    heading.textContent = '카메라 관리';
    const invoker = document.createElement('button');
    invoker.textContent = '삭제';
    main.append(heading, invoker);
    document.body.append(main);
    invoker.focus();
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);

    act(() => root.render(<AccessibleDialog open title="삭제 확인" onClose={vi.fn()}><button>취소</button></AccessibleDialog>));
    invoker.remove();
    act(() => root.render(<AccessibleDialog open={false} title="삭제 확인" onClose={vi.fn()}><button>취소</button></AccessibleDialog>));

    expect(document.activeElement).toBe(heading);
    act(() => root.unmount());
  });

  it.each([
    ['body', () => document.body],
    ['document element', () => {
      document.documentElement.tabIndex = -1;
      document.documentElement.focus();
      return document.documentElement;
    }],
  ])('focuses the guaranteed page fallback when opened on mount from the %s', (_, prepareInvoker) => {
    const main = document.createElement('main');
    main.id = 'main-content';
    const heading = document.createElement('h1');
    heading.tabIndex = -1;
    heading.dataset.dialogFocusFallback = '';
    heading.textContent = '카메라 관리';
    main.append(heading);
    document.body.append(main);
    const host = document.createElement('div');
    document.body.append(host);
    const invoker = prepareInvoker();
    expect(document.activeElement).toBe(invoker);
    const root = createRoot(host);

    act(() => root.render(<AccessibleDialog open title="직접 링크 대화상자" onClose={vi.fn()}><button>닫기</button></AccessibleDialog>));
    act(() => root.render(<AccessibleDialog open={false} title="직접 링크 대화상자" onClose={vi.fn()}><button>닫기</button></AccessibleDialog>));

    expect(document.activeElement).toBe(heading);
    expect(document.activeElement).not.toBe(document.body);
    act(() => root.unmount());
  });
});
