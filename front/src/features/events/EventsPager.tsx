type EventsPagerProps = {
  readonly pageIndex: number;
  readonly total: number;
  readonly visibleCount: number;
  readonly hasNextPage: boolean;
  readonly pendingPageIndex: number | null;
  readonly onNext: () => void;
  readonly onPrevious: () => void;
  readonly onFirst: () => void;
  readonly onRefresh: () => void;
};

const CONTROL_CLASS = 'inline-flex min-h-11 min-w-11 items-center justify-center rounded-control border border-border bg-card px-3 text-sm font-semibold text-foreground hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60';

/**
 * Keyset pager. The backend orders clips by `(started_at DESC, clip_id DESC)` and hands back an
 * opaque cursor, so there is no addressable page number to jump to; navigation is strictly one
 * boundary forward or back along the trail the operator already walked.
 */
export function EventsPager({
  pageIndex,
  total,
  visibleCount,
  hasNextPage,
  pendingPageIndex,
  onNext,
  onPrevious,
  onFirst,
  onRefresh,
}: EventsPagerProps): JSX.Element | null {
  if (pageIndex === 0 && !hasNextPage) return null;
  const pending = pendingPageIndex !== null;

  return (
    <nav className="mt-6 flex flex-wrap items-center justify-between gap-3" aria-label="이벤트 페이지" aria-busy={pending}>
      {/* A keyset page has no addressable global offset, so only the honest counts are shown: how
          many rows this page holds, the backend's total, and how far the operator has walked. */}
      <p className="tabular-nums text-sm text-muted-foreground" data-testid="events-page-range">
        {visibleCount}건 표시 / 전체 {total}건 · {pageIndex + 1} 페이지
      </p>
      <div className="flex flex-wrap items-center justify-end gap-2">
        {pageIndex > 0 ? (
          <button type="button" className={CONTROL_CLASS} aria-label="현재 페이지 새로 고침" disabled={pending} onClick={onRefresh}>
            새로 고침
          </button>
        ) : null}
        {pageIndex > 0 ? (
          <button type="button" className={CONTROL_CLASS} aria-label="최신 페이지" disabled={pending} onClick={onFirst}>
            최신
          </button>
        ) : null}
        <button
          type="button"
          className={CONTROL_CLASS}
          aria-label="이전 페이지"
          disabled={pageIndex === 0 || pending}
          onClick={onPrevious}
        >
          이전
        </button>
        <button
          type="button"
          className={CONTROL_CLASS}
          aria-label="다음 페이지"
          disabled={!hasNextPage || pending}
          onClick={onNext}
        >
          다음
        </button>
      </div>
    </nav>
  );
}
