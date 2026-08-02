import { ReactNode, RefObject, useEffect, useId, useRef } from 'react';
import { createPortal } from 'react-dom';

const FOCUSABLE = 'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/**
 * Width variants matching front/design-handoff/README.md §5 modal specs: xs=380 (계정 설정),
 * sm=420 (폴더 탐색기), md=440 (연결 관리, 카메라 등록 1단계), default=520 (unchanged — other
 * callers, e.g. delete confirmation), lg=640 (카메라 등록 2단계, 연결 관리 다시 인식), xl=720 (클립 재생).
 */
export type DialogSize = 'xs' | 'sm' | 'md' | 'default' | 'lg' | 'xl';

const DIALOG_SIZE_CLASS: Partial<Record<DialogSize, string>> = {
  xs: 'accessible-dialog--xs',
  sm: 'accessible-dialog--sm',
  md: 'accessible-dialog--md',
  lg: 'accessible-dialog--lg',
  xl: 'accessible-dialog--xl',
};

export function AccessibleDialog({
  open,
  title,
  children,
  onClose,
  variant = 'dialog',
  size = 'default',
  initialFocus = 'first-control',
  initialFocusRef,
}: {
  open: boolean;
  title: string;
  children: ReactNode;
  onClose: () => void;
  variant?: 'dialog' | 'sheet';
  size?: DialogSize;
  initialFocus?: 'first-control' | 'heading';
  initialFocusRef?: RefObject<HTMLElement>;
}): JSX.Element | null {
  const titleId = useId();
  const dialogRef = useRef<HTMLDivElement>(null);
  const invokerRef = useRef<HTMLElement | null>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    if (!open) return;
    invokerRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const dialog = dialogRef.current;
    const target = initialFocusRef?.current
      ?? (initialFocus === 'heading' ? dialog?.querySelector<HTMLElement>('h2') : dialog?.querySelector<HTMLElement>(FOCUSABLE));
    target?.focus();

    const background = Array.from(document.body.children).filter((element) => !element.contains(dialog));
    const previousAria = background.map((element) => element.getAttribute('aria-hidden'));
    background.forEach((element) => { element.setAttribute('aria-hidden', 'true'); (element as HTMLElement).inert = true; });

    function handleKeyDown(event: KeyboardEvent): void {
      if (event.key === 'Escape') {
        event.preventDefault();
        onCloseRef.current();
        return;
      }
      if (event.key !== 'Tab' || !dialog) return;
      const focusable = Array.from(dialog.querySelectorAll<HTMLElement>(FOCUSABLE));
      if (!focusable.length) {
        event.preventDefault();
        dialog.querySelector<HTMLElement>('h2')?.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable.at(-1) as HTMLElement;
      const activeElement = document.activeElement;
      if (!focusable.includes(activeElement as HTMLElement)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      } else if (event.shiftKey && activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = previousOverflow;
      background.forEach((element, index) => {
        (element as HTMLElement).inert = false;
        const aria = previousAria[index];
        if (aria === null) element.removeAttribute('aria-hidden');
        else element.setAttribute('aria-hidden', aria);
      });
      const invoker = invokerRef.current;
      if (invoker?.isConnected && invoker !== document.body && invoker !== document.documentElement) invoker.focus();
      else document.querySelector<HTMLElement>('#main-content [data-dialog-focus-fallback], #main-content h1[tabindex="-1"]')?.focus();
    };
  }, [initialFocus, initialFocusRef, open]);

  if (!open) return null;
  const className = ['accessible-dialog', variant === 'sheet' ? 'accessible-sheet' : '', DIALOG_SIZE_CLASS[size] ?? '']
    .filter(Boolean)
    .join(' ');
  return createPortal(
    <div className="dialog-backdrop" data-dialog-backdrop onClick={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <div ref={dialogRef} className={className} data-variant={variant} data-size={size} role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <h2 id={titleId} tabIndex={-1}>{title}</h2>
        {children}
      </div>
    </div>,
    document.body,
  );
}
