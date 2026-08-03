import { useState } from 'react';
import { LiveStreamPanel } from '@/features/operations/LiveStreamPanel';
import { CameraInfoCard } from '@/features/operations/CameraInfoCard';
import { DetectionSettingsCard } from '@/features/operations/DetectionSettingsCard';
import { EventHistoryList } from '@/features/operations/EventHistoryList';
import { ClipPlayerModal } from '@/features/operations/ClipPlayerModal';
import { CameraEditModal } from '@/features/settings/CameraEditModal';
import { DeleteCameraDialog } from '@/features/settings/DeleteCameraDialog';
import { useStatusResource } from '@/shared/api/usePollingResource';
import { getCameraStatusMeta } from '@/shared/ui/StatusBadge';
import type { Camera, Clip, OverlayMode } from '@/shared/api/client';

type RoomDetailProps = {
  camera: Camera;
  onBack: () => void;
  onRetryConnection: () => void;
};

/**
 * 연결 관리 is the settings feature's edit-camera modal (front/design-handoff/README.md 모달 2) —
 * imported rather than duplicated. Delete is delegated the same way settings/CameraSection does:
 * CameraEditModal never confirms its own delete (AccessibleDialog-on-AccessibleDialog would inert
 * the first dialog), so the request bubbles up to a sibling DeleteCameraDialog instead. A delete
 * here removes the camera being viewed, so it also navigates back to the wall.
 */
export function RoomDetail({ camera, onBack, onRetryConnection }: RoomDetailProps): JSX.Element {
  const { data: status } = useStatusResource(true);
  const diagnostics = status?.runtime.cameras[camera.id];
  const [selectedClip, setSelectedClip] = useState<Clip | null>(null);
  const [connectionModalOpen, setConnectionModalOpen] = useState(false);
  const [deletingCamera, setDeletingCamera] = useState<Camera | null>(null);
  // Sourced from OverlayModeControl's own fetch/selection state (via its onModeChange callback)
  // rather than re-fetched here, so the live badge below never disagrees with what the operator
  // picked in the detection settings card (issue #102).
  const [overlayMode, setOverlayMode] = useState<OverlayMode | null>(null);
  const statusMeta = getCameraStatusMeta(camera.status);

  return (
    <section aria-label="호실 상세">
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-1.5 text-sm">
          <button
            type="button"
            onClick={onBack}
            className="font-semibold text-muted-foreground hover:text-foreground hover:underline"
          >
            관제
          </button>
          <span aria-hidden="true" className="text-muted-foreground">/</span>
          <h1 tabIndex={-1} data-dialog-focus-fallback className="text-base font-semibold text-foreground">
            {camera.label}
          </h1>
        </div>
        <span className={statusMeta.className}>
          <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-current" />
          {statusMeta.label}
        </span>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <LiveStreamPanel
            camera={camera}
            diagnostics={diagnostics}
            overlayMode={overlayMode}
            onRetryConnection={onRetryConnection}
            onManageConnection={() => setConnectionModalOpen(true)}
          />
        </div>
        <div className="flex flex-col gap-4 lg:col-span-1">
          <CameraInfoCard camera={camera} onManageConnection={() => setConnectionModalOpen(true)} />
          <DetectionSettingsCard camera={camera} onOverlayModeChange={setOverlayMode} />
        </div>
      </div>

      <div className="mt-8">
        <EventHistoryList cameraId={camera.id} onSelectClip={setSelectedClip} />
      </div>

      <ClipPlayerModal clip={selectedClip} cameraLabel={camera.label} onClose={() => setSelectedClip(null)} />

      <CameraEditModal
        camera={connectionModalOpen ? camera : null}
        onClose={() => setConnectionModalOpen(false)}
        onUpdated={onRetryConnection}
        onRequestDelete={(target) => {
          setConnectionModalOpen(false);
          setDeletingCamera(target);
        }}
      />

      <DeleteCameraDialog
        camera={deletingCamera}
        onClose={() => setDeletingCamera(null)}
        onDeleted={() => {
          setDeletingCamera(null);
          onRetryConnection();
          onBack();
        }}
      />
    </section>
  );
}
