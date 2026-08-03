import { useEffect, useRef, useState } from 'react';
import {
  canonicalizeDashboardLocation,
  DashboardLocationController,
  type DashboardLocation,
  type DashboardLocationData,
} from '@/app/dashboardLocation';
import type { PollingResourceStatus } from '@/shared/api/usePollingResource';
import type { Camera } from '@/shared/api/client';

export type OperationsLocation = {
  floor: string | undefined;
  cameraId: string | undefined;
  setFloor: (floor: string | undefined) => void;
  selectCamera: (cameraId: string | undefined) => void;
  backToWall: () => void;
};

function dataFor(status: PollingResourceStatus, cameras: readonly Camera[] | undefined): DashboardLocationData {
  return { cameras: { status, data: cameras } };
}

/**
 * Owns a scoped `DashboardLocationController` instance for this page so `?floor=&camera=` can be
 * read/written without App.tsx threading its shared controller down as a prop (App.tsx is out of
 * ownership scope here). Mirrors the controller's designed `data` revalidation hook by re-running
 * canonicalization whenever the camera list resolves, so a stale `camera`/`floor` param gets
 * dropped from the URL exactly as the controller already does for App-level navigation.
 */
export function useOperationsLocation(camerasStatus: PollingResourceStatus, cameras: readonly Camera[] | undefined): OperationsLocation {
  const [location, setLocation] = useState<DashboardLocation>(
    () => canonicalizeDashboardLocation(window.location.search).location,
  );
  const controllerRef = useRef<DashboardLocationController | null>(null);

  useEffect(() => {
    const controller = new DashboardLocationController(window, setLocation);
    controllerRef.current = controller;
    controller.start(dataFor(camerasStatus, cameras));
    return () => {
      controller.dispose();
      controllerRef.current = null;
    };
    // Intentional mount-only setup: the effect below re-validates against fresh camera data.
  }, []);

  useEffect(() => {
    controllerRef.current?.canonicalize(dataFor(camerasStatus, cameras));
  }, [camerasStatus, cameras]);

  return {
    floor: location.floor,
    cameraId: location.camera,
    setFloor: (floor) => controllerRef.current?.navigate({ floor: floor ?? null, camera: null }),
    selectCamera: (cameraId) => controllerRef.current?.navigate({ camera: cameraId ?? null }),
    backToWall: () => controllerRef.current?.navigate({ camera: null }),
  };
}
