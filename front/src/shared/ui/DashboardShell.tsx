import { ComponentType, ReactNode, SVGProps } from 'react';
import type { DashboardPage } from '@/app/dashboardLocation';
import { SeniorAiLabBrand } from '@/shared/ui/SeniorAiLabBrand';
import { useAuthSession } from '@/shared/ui/AuthGate';

type BackendStatusView = { label: string; className: string };
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

/** 2×2 wall — operations / camera wall. */
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

/** Clock — historical event evidence. */
function EventsIcon(props: SVGProps<SVGSVGElement>): JSX.Element {
  return (
    <svg {...iconProps(props)}>
      <circle cx="12" cy="12" r="8.25" />
      <path d="M12 7.5V12l3 2" />
    </svg>
  );
}

/** Simple camera body — registry management. */
function CamerasIcon(props: SVGProps<SVGSVGElement>): JSX.Element {
  return (
    <svg {...iconProps(props)}>
      <path d="M3.75 8.25h11.5a1.5 1.5 0 0 1 1.5 1.5v7.5a1.5 1.5 0 0 1-1.5 1.5H3.75a1.5 1.5 0 0 1-1.5-1.5v-7.5a1.5 1.5 0 0 1 1.5-1.5Z" />
      <path d="M16.75 11.25 20.75 9v9l-4-2.25" />
      <circle cx="9.5" cy="13.5" r="2.25" />
    </svg>
  );
}

/** Activity pulse — system diagnostics. */
function SystemIcon(props: SVGProps<SVGSVGElement>): JSX.Element {
  return (
    <svg {...iconProps(props)}>
      <path d="M4 13h3.25l2-5.5 3.5 9 2.25-5H20" />
    </svg>
  );
}

const destinations: Array<{ id: DashboardPage; label: string; Icon: NavIcon }> = [
  { id: 'operations', label: '관제', Icon: OperationsIcon },
  { id: 'events', label: '이벤트 기록', Icon: EventsIcon },
  { id: 'cameras', label: '카메라 관리', Icon: CamerasIcon },
  { id: 'system', label: '시스템', Icon: SystemIcon },
];

export function getScreenLabel(screen: DashboardPage): string {
  return destinations.find((entry) => entry.id === screen)?.label ?? screen;
}

function Navigation({ active, onChange, label }: { active: DashboardPage; onChange: (screen: DashboardPage) => void; label: string }): JSX.Element {
  return (
    <nav aria-label={label}>
      {destinations.map((entry) => (
        <button key={entry.id} type="button" aria-current={active === entry.id ? 'page' : undefined} onClick={() => onChange(entry.id)}>
          <span aria-hidden="true" className="nav-mark"><entry.Icon /></span>
          <span>{entry.label}</span>
        </button>
      ))}
    </nav>
  );
}

export function DashboardShell({
  backendStatus,
  screen,
  onScreenChange,
  onAddCamera,
  children,
}: {
  apiBase?: string;
  backendStatus?: BackendStatusView;
  screen: DashboardPage;
  onScreenChange: (screen: DashboardPage) => void;
  onAddCamera?: () => void;
  children: ReactNode;
}): JSX.Element {
  const session = useAuthSession();
  return (
    <div className="dashboard-shell">
      <a className="skip-link" href="#main-content">본문으로 건너뛰기</a>
      <aside className="desktop-rail">
        <SeniorAiLabBrand />
        <Navigation active={screen} onChange={onScreenChange} label="주요 내비게이션" />
      </aside>
      <div className="dashboard-workspace">
        <header className="dashboard-topbar">
          <div className="mobile-brand"><SeniorAiLabBrand compact /></div>
          <div>
            <p className="eyebrow">운영 콘솔</p>
            <p className="shell-page-title">{getScreenLabel(screen)}</p>
          </div>
          <div className="shell-actions">
            {backendStatus ? <span className={backendStatus.className}>{backendStatus.label}</span> : null}
            {screen === 'cameras' && onAddCamera ? <button type="button" className="brand-action" onClick={onAddCamera}>카메라 추가</button> : null}
            {session ? <button type="button" onClick={session.logout}>로그아웃</button> : null}
          </div>
        </header>
        <main id="main-content" tabIndex={-1}>{children}</main>
      </div>
      <div className="mobile-tabs"><Navigation active={screen} onChange={onScreenChange} label="하단 내비게이션" /></div>
    </div>
  );
}
