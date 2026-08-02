import { getPageLabel } from '@/shared/ui/NavBar';

/** Placeholder — replaced by the event history/clip review view in the page-implementation wave. */
export function EventsPage(): JSX.Element {
  return (
    <section className="page-placeholder">
      <h1 className="shell-page-title" tabIndex={-1} data-dialog-focus-fallback>
        {getPageLabel('events')}
      </h1>
      <p>이벤트 화면은 다음 작업에서 구현됩니다.</p>
    </section>
  );
}
