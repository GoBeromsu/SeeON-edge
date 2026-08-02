import { getPageLabel } from '@/shared/ui/NavBar';
import { useCamerasResource } from '@/shared/api/usePollingResource';
import { useOperationsLocation } from '@/features/operations/useOperationsLocation';
import { CameraWall } from '@/features/operations/CameraWall';
import { RoomDetail } from '@/features/operations/RoomDetail';

export function OperationsPage(): JSX.Element {
  const { status, data, retry } = useCamerasResource(true);
  const cameras = data?.cameras;
  const location = useOperationsLocation(status, cameras);

  const selectedCamera = location.cameraId
    ? cameras?.find((camera) => camera.id === location.cameraId)
    : undefined;

  return (
    <section className="page-placeholder">
      <h1 className="shell-page-title" tabIndex={-1} data-dialog-focus-fallback>
        {getPageLabel('operations')}
      </h1>

      {selectedCamera ? (
        <RoomDetail camera={selectedCamera} onBack={location.backToWall} />
      ) : (
        <CameraWall
          status={status}
          cameras={cameras}
          floor={location.floor}
          onFloorChange={location.setFloor}
          onSelectCamera={location.selectCamera}
          onRetry={retry}
        />
      )}
    </section>
  );
}
