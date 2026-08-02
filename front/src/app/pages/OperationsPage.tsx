import { getPageLabel } from '@/shared/ui/NavBar';

/** Placeholder — replaced by the camera-wall/floor view in the page-implementation wave. */
export function OperationsPage(): JSX.Element {
  return (
    <section className="page-placeholder">
      <h1 className="shell-page-title" tabIndex={-1} data-dialog-focus-fallback>
        {getPageLabel('operations')}
      </h1>
      <p>관제 화면은 다음 작업에서 구현됩니다.</p>
    </section>
  );
}
