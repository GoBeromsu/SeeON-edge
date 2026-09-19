import { DASHBOARD_PAGES, type DashboardPage } from '@/app/dashboardLocation';

/**
 * Cross-page navigation from inside the operations feature, which has no access to App.tsx's
 * NavBar-owned DashboardLocationController instance (App.tsx is out of ownership scope here).
 * Pushes a clean `?page=<page>` URL, then dispatches a synthetic `popstate` so App's controller —
 * which already listens for real back/forward navigation — picks it up and renders the target page.
 */
export function navigateToPage(page: DashboardPage): void {
  if (!DASHBOARD_PAGES.includes(page)) return;
  const params = new URLSearchParams();
  params.set('page', page);
  window.history.pushState(null, '', `${window.location.pathname}?${params.toString()}${window.location.hash}`);
  window.dispatchEvent(new PopStateEvent('popstate'));
}

/** Fixed activation path for an empty camera wall. No camera or installation data enters the URL. */
export function navigateToEdgeSetup(): void {
  window.history.pushState(null, '', `${window.location.pathname}?page=settings#edge-setup`);
  window.dispatchEvent(new PopStateEvent('popstate'));
}
