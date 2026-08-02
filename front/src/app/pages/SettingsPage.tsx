import { getPageLabel } from '@/shared/ui/NavBar';

/** Placeholder — replaced by camera/connection/system settings views in the page-implementation wave. */
export function SettingsPage(): JSX.Element {
  return (
    <section className="page-placeholder">
      <h1 className="shell-page-title" tabIndex={-1} data-dialog-focus-fallback>
        {getPageLabel('settings')}
      </h1>
      <p>설정 화면은 다음 작업에서 구현됩니다.</p>
    </section>
  );
}
