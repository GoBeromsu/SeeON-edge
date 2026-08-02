import { useCallback, useEffect, useState } from 'react';
import { DashboardLocationController, canonicalizeDashboardLocation, type DashboardPage } from '@/app/dashboardLocation';
import { EventsPage } from '@/app/pages/EventsPage';
import { OperationsPage } from '@/app/pages/OperationsPage';
import { SettingsPage } from '@/app/pages/SettingsPage';
import { AccountSettingsModal } from '@/features/account-settings/AccountSettingsModal';
import { AuthGate } from '@/shared/ui/AuthGate';
import { NavBar } from '@/shared/ui/NavBar';
import { ToastViewport } from '@/shared/ui/Toast';

export function Dashboard(): JSX.Element {
  const [location, setLocation] = useState(() => canonicalizeDashboardLocation(window.location.search).location);
  const [controller] = useState(() => new DashboardLocationController(window, setLocation));
  const [accountSettingsOpen, setAccountSettingsOpen] = useState(false);

  useEffect(() => {
    controller.start();
    return () => controller.dispose();
  }, [controller]);

  const navigate = useCallback((page: DashboardPage): void => {
    controller.navigate({ page });
  }, [controller]);

  let page: JSX.Element;
  switch (location.page) {
    case 'events':
      page = <EventsPage />;
      break;
    case 'settings':
      page = <SettingsPage />;
      break;
    default:
      page = <OperationsPage />;
  }

  return (
    <div className="app-shell">
      <a href="#main-content" className="skip-link">본문으로 건너뛰기</a>
      <NavBar
        page={location.page}
        onNavigate={navigate}
        onOpenAccountSettings={() => setAccountSettingsOpen(true)}
      />
      <main id="main-content" className="app-main">
        {page}
      </main>
      <AccountSettingsModal open={accountSettingsOpen} onClose={() => setAccountSettingsOpen(false)} />
      <ToastViewport />
    </div>
  );
}

export default function App(): JSX.Element {
  return (
    <AuthGate>
      <Dashboard />
    </AuthGate>
  );
}
