import { useCallback, useEffect, useState } from 'react';
import {
  DashboardLocationController,
  canonicalizeDashboardLocation,
  type DashboardLocation,
  type DashboardLocationData,
  type DashboardPage,
} from '@/app/dashboardLocation';
import { CameraManagementPage } from '@/features/camera-management/CameraManagementPage';
import { EventHistoryPage } from '@/features/events/EventHistoryPage';
import type { EventHistoryLocation } from '@/features/events/cameraEventLogic';
import { OperationsView } from '@/features/operations/OperationsView';
import { SystemPage } from '@/features/system/SystemPage';
import { useCamerasResource, useClipsResource, useStatusResource } from '@/shared/api/usePollingResource';
import { AuthGate } from '@/shared/ui/AuthGate';
import { DashboardShell } from '@/shared/ui/DashboardShell';

type LocationUpdate = Partial<Record<keyof DashboardLocation, string | null>>;

function OperationsRoute({ location, onNavigate, onValidate }: {
  location: DashboardLocation;
  onNavigate: (update: LocationUpdate) => void;
  onValidate: (data: DashboardLocationData) => void;
}): JSX.Element {
  const cameras = useCamerasResource(true);
  const status = useStatusResource(true);

  const validateLocation = useCallback((): void => {
    if (cameras.status === 'success' && cameras.data !== null) {
      onValidate({ cameras: { status: 'success', data: cameras.data.cameras } });
    }
  }, [cameras.data, cameras.status, onValidate]);

  useEffect(validateLocation, [location.camera, location.floor, location.room, location.wallPage, validateLocation]);

  return (
    <OperationsView
      active
      cameras={cameras.data?.cameras ?? []}
      camerasState={cameras.status}
      camerasLastSuccessAt={cameras.lastSuccessAt}
      camerasRefreshing={cameras.refreshing}
      status={status.data}
      statusState={status.status}
      location={location}
      onNavigate={onNavigate}
      onReplace={validateLocation}
      onRetryCameras={cameras.retry}
      onRetryStatus={status.retry}
    />
  );
}

function EventsRoute({ location, onNavigate, onValidate }: {
  location: EventHistoryLocation;
  onNavigate: (update: LocationUpdate) => void;
  onValidate: (data: DashboardLocationData) => void;
}): JSX.Element {
  const cameras = useCamerasResource(true);
  const clips = useClipsResource(true);

  useEffect(() => {
    if (cameras.status === 'success' && cameras.data !== null && clips.status === 'success' && clips.data !== null) {
      onValidate({
        cameras: { status: 'success', data: cameras.data.cameras },
        clips: { status: 'success', data: clips.data },
      });
    }
  }, [cameras.data, cameras.status, clips.data, clips.status, location.camera, location.clip, location.event, location.floor, location.room, onValidate]);

  let status: 'loading' | 'success' | 'error' = 'loading';
  if (cameras.status === 'error' || clips.status === 'error') status = 'error';
  else if (cameras.status === 'success' && clips.status === 'success') status = 'success';
  const lastSuccessAt = cameras.lastSuccessAt === null || clips.lastSuccessAt === null
    ? null
    : Math.min(cameras.lastSuccessAt, clips.lastSuccessAt);

  return (
    <EventHistoryPage
      cameras={cameras.data?.cameras ?? []}
      clips={clips.data ?? []}
      status={status}
      lastSuccessAt={lastSuccessAt}
      refreshing={cameras.refreshing || clips.refreshing}
      location={location}
      onNavigate={onNavigate}
      onRetry={() => { cameras.retry(); clips.retry(); }}
      onClipChanged={() => undefined}
    />
  );
}

export function Dashboard(): JSX.Element {
  const [location, setLocation] = useState(() => canonicalizeDashboardLocation(window.location.search).location);
  const [controller] = useState(() => new DashboardLocationController(window, setLocation));

  useEffect(() => {
    controller.start();
    return () => controller.dispose();
  }, [controller]);

  const navigate = useCallback((update: LocationUpdate): void => {
    controller.navigate(update);
  }, [controller]);

  const validate = useCallback((data: DashboardLocationData): void => {
    controller.canonicalize(data);
  }, [controller]);

  let page: JSX.Element;
  switch (location.page) {
    case 'events':
      page = <EventsRoute location={{ ...location, page: 'events' }} onNavigate={navigate} onValidate={validate} />;
      break;
    case 'cameras':
      page = <><h1 className="sr-only" data-dialog-focus-fallback tabIndex={-1}>카메라 관리</h1><CameraManagementPage /></>;
      break;
    case 'system':
      page = <SystemPage active />;
      break;
    default:
      page = <OperationsRoute location={location} onNavigate={navigate} onValidate={validate} />;
  }

  return (
    <DashboardShell
      screen={location.page}
      onScreenChange={(page: DashboardPage) => navigate({ page })}
    >
      {page}
    </DashboardShell>
  );
}

export default function App(): JSX.Element {
  return <AuthGate><Dashboard /></AuthGate>;
}
