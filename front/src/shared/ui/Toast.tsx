import { useEffect, useState } from 'react';

export type ToastVariant = 'success' | 'error';

export type ToastItem = {
  id: string;
  message: string;
  variant: ToastVariant;
};

const AUTO_DISMISS_MS = 2200;

let toasts: ToastItem[] = [];
let listeners: Array<(items: ToastItem[]) => void> = [];
let nextId = 0;

function emit(): void {
  for (const listener of listeners) listener(toasts);
}

/** Imperative API: `toast.success('저장되었습니다')` / `toast.error('...')`, usable outside React (e.g. API error handlers). */
export function showToast(message: string, variant: ToastVariant = 'success'): void {
  const id = `toast-${(nextId += 1)}`;
  toasts = [...toasts, { id, message, variant }];
  emit();
  setTimeout(() => {
    toasts = toasts.filter((item) => item.id !== id);
    emit();
  }, AUTO_DISMISS_MS);
}

export const toast = {
  success: (message: string) => showToast(message, 'success'),
  error: (message: string) => showToast(message, 'error'),
};

export function useToasts(): ToastItem[] {
  const [items, setItems] = useState(toasts);

  useEffect(() => {
    listeners.push(setItems);
    setItems(toasts);
    return () => {
      listeners = listeners.filter((listener) => listener !== setItems);
    };
  }, []);

  return items;
}

/** Mount once near the app root. Renders the bottom-right auto-dismissing toast stack. */
export function ToastViewport(): JSX.Element {
  const items = useToasts();

  return (
    <div className="toast-stack" role="status" aria-live="polite">
      {items.map((item) => (
        <div key={item.id} className={`toast toast-${item.variant}`}>
          {item.message}
        </div>
      ))}
    </div>
  );
}
