import { ComponentType, SVGProps } from 'react';
import type { DashboardPage } from '@/app/dashboardLocation';
import { useAuthSession } from '@/shared/ui/AuthGate';

const BRAND_TEXT = 'Senior AI Lab Edge';

type NavIcon = ComponentType<SVGProps<SVGSVGElement>>;

function iconProps(props: SVGProps<SVGSVGElement>): SVGProps<SVGSVGElement> {
  return {
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.75,
    strokeLinecap: 'round',
    strokeLinejoin: 'round',
    ...props,
  };
}

/** 2x2 wall — 관제 (camera wall). */
function OperationsIcon(props: SVGProps<SVGSVGElement>): JSX.Element {
  return (
    <svg {...iconProps(props)}>
      <rect x="3.5" y="3.5" width="7" height="7" rx="1.25" />
      <rect x="13.5" y="3.5" width="7" height="7" rx="1.25" />
      <rect x="3.5" y="13.5" width="7" height="7" rx="1.25" />
      <rect x="13.5" y="13.5" width="7" height="7" rx="1.25" />
    </svg>
  );
}

/** Clock — 이벤트 (historical evidence). */
function EventsIcon(props: SVGProps<SVGSVGElement>): JSX.Element {
  return (
    <svg {...iconProps(props)}>
      <circle cx="12" cy="12" r="8.25" />
      <path d="M12 7.5V12l3 2" />
    </svg>
  );
}

/** Gear — 설정 (cameras + system + connection settings). */
function SettingsIcon(props: SVGProps<SVGSVGElement>): JSX.Element {
  return (
    <svg {...iconProps(props)}>
      <circle cx="12" cy="12" r="2.75" />
      <path d="M12 3.75v2.1M12 18.15v2.1M20.25 12h-2.1M5.85 12h-2.1M17.66 6.34l-1.49 1.49M7.83 16.17l-1.49 1.49M17.66 17.66l-1.49-1.49M7.83 7.83 6.34 6.34" />
    </svg>
  );
}

/** Person silhouette — account settings entry point. */
function AccountIcon(props: SVGProps<SVGSVGElement>): JSX.Element {
  return (
    <svg {...iconProps(props)}>
      <circle cx="12" cy="8.25" r="3.25" />
      <path d="M4.75 19.25c1.2-3.35 4.1-5 7.25-5s6.05 1.65 7.25 5" />
    </svg>
  );
}

const destinations: Array<{ id: DashboardPage; label: string; Icon: NavIcon }> = [
  { id: 'operations', label: '관제', Icon: OperationsIcon },
  { id: 'events', label: '이벤트', Icon: EventsIcon },
  { id: 'settings', label: '설정', Icon: SettingsIcon },
];

export function getPageLabel(page: DashboardPage): string {
  return destinations.find((entry) => entry.id === page)?.label ?? page;
}

export function NavBar({
  page,
  onNavigate,
  onOpenAccountSettings,
}: {
  page: DashboardPage;
  onNavigate: (page: DashboardPage) => void;
  onOpenAccountSettings: () => void;
}): JSX.Element {
  const session = useAuthSession();

  return (
    <header className="app-navbar">
      <span className="app-navbar-brand">{BRAND_TEXT}</span>
      <nav className="app-nav" aria-label="주요 내비게이션">
        {destinations.map((entry) => (
          <button
            key={entry.id}
            type="button"
            aria-current={page === entry.id ? 'page' : undefined}
            onClick={() => onNavigate(entry.id)}
          >
            <span aria-hidden="true" className="nav-mark"><entry.Icon /></span>
            <span>{entry.label}</span>
          </button>
        ))}
      </nav>
      <div className="app-navbar-actions">
        <button type="button" className="icon-button" aria-label="계정 설정" onClick={onOpenAccountSettings}>
          <AccountIcon />
        </button>
        {session ? (
          <button type="button" className="logout-button" onClick={() => void session.logout()}>로그아웃</button>
        ) : null}
      </div>
    </header>
  );
}
