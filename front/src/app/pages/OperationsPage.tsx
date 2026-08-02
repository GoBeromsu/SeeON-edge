import { useCamerasResource } from '@/shared/api/usePollingResource';
import { useOperationsLocation } from '@/features/operations/useOperationsLocation';
import { CameraWall } from '@/features/operations/CameraWall';
import { RoomDetail } from '@/features/operations/RoomDetail';

/**
 * No page-level h1 here: the wall and room-detail views each own their own primary heading
 * (front/design-handoff/README.md §2 헤더 줄 / §3 브레드크럼) instead of duplicating "관제" above
 * a second, view-specific title. Each view still exposes exactly one `[data-dialog-focus-fallback]`
 * h1 so AccessibleDialog's close-focus fallback keeps working.
 */
export function OperationsPage(): JSX.Element {
  const { status, data, retry } = useCamerasResource(true);
  const cameras = data?.cameras;
  const location = useOperationsLocation(status, cameras);

  const selectedCamera = location.cameraId
    ? cameras?.find((camera) => camera.id === location.cameraId)
    : undefined;

  return (
    <section className="page-placeholder">
      {selectedCamera ? (
        <RoomDetail camera={selectedCamera} onBack={location.backToWall} onRetryConnection={retry} />
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
