import { EVENTS_PAGE_SIZE, getPageCount, getPagerItems } from '@/features/events/paging';

type EventsPagerProps = {
  readonly pageIndex: number;
  readonly total: number;
  readonly visibleCount: number;
  readonly pendingPageIndex: number | null;
  readonly onNavigate: (pageIndex: number) => void;
  readonly onRefresh: () => void;
};

const CONTROL_CLASS = 'inline-flex min-h-11 min-w-11 items-center justify-center rounded-control border border-border bg-card px-3 text-sm font-semibold text-foreground hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60';

export function EventsPager({
  pageIndex,
  total,
  visibleCount,
  pendingPageIndex,
  onNavigate,
  onRefresh,
}: EventsPagerProps): JSX.Element | null {
  const pageCount = getPageCount(total);
  if (pageCount === 1) return null;
  const firstResult = pageIndex * EVENTS_PAGE_SIZE + 1;
  const lastResult = Math.min(pageIndex * EVENTS_PAGE_SIZE + visibleCount, total);
  const pending = pendingPageIndex !== null;

  return (
    <nav className="mt-6 flex flex-wrap items-center justify-between gap-3" aria-label="이벤트 페이지" aria-busy={pending}>
      <p className="tabular-nums text-sm text-muted-foreground">
        {firstResult}–{lastResult} / {total}건 · {pageIndex + 1} / {pageCount} 페이지
      </p>
      <div className="flex flex-wrap items-center justify-end gap-2">
        {pageIndex > 0 ? (
          <button type="button" className={CONTROL_CLASS} aria-label="현재 페이지 새로 고침" disabled={pending} onClick={onRefresh}>
            새로 고침
          </button>
        ) : null}
        <button
          type="button"
          className={CONTROL_CLASS}
          aria-label="이전 페이지"
          disabled={pageIndex === 0 || pending}
          onClick={() => onNavigate(pageIndex - 1)}
        >
          이전
        </button>
        {getPagerItems(pageIndex, pageCount).map((item) => item.kind === 'ellipsis' ? (
          <span key={item.key} className="px-1 text-sm text-muted-foreground" aria-hidden="true">…</span>
        ) : (
          <button
            type="button"
            key={item.pageIndex}
            className={`${CONTROL_CLASS} ${item.pageIndex === pageIndex ? 'border-primary/30 bg-primary/10 text-primary' : ''}`}
            aria-label={`${item.pageIndex + 1}페이지`}
            aria-current={item.pageIndex === pageIndex ? 'page' : undefined}
            disabled={item.pageIndex === pageIndex || pending}
            onClick={() => onNavigate(item.pageIndex)}
          >
            {item.pageIndex + 1}
          </button>
        ))}
        <button
          type="button"
          className={CONTROL_CLASS}
          aria-label="다음 페이지"
          disabled={pageIndex + 1 >= pageCount || pending}
          onClick={() => onNavigate(pageIndex + 1)}
        >
          다음
        </button>
      </div>
    </nav>
  );
}
