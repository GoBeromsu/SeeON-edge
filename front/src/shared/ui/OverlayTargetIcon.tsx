import type { OverlaySelection } from '@/shared/api/types';

export function OverlayTargetIcon({ target }: { target: keyof OverlaySelection }): JSX.Element {
  return <svg aria-hidden="true" data-overlay-target={target} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4">
    {target === 'person' ? <><circle cx="12" cy="5" r="2.5" /><path d="M8 21v-5l1-6h6l1 6v5M9 12l-3 4M15 12l3 4" /></> : <><path d="M3 6v12M3 14h18v4M7 10h4a3 3 0 0 1 3 3v1M7 10V7h4a3 3 0 0 1 3 3" /></>}
  </svg>;
}
