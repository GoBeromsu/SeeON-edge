import { useEffect, useRef, useState } from 'react';
import { deleteCamera, type Camera, type CameraRegistry } from '@/shared/api/client';
import { useCamerasResource } from '@/shared/api/usePollingResource';
import { AddCameraModal } from '@/features/cameras/AddCameraModal';
import { DeleteCameraDialog } from '@/features/cameras/DeleteCameraDialog';
import { CameraManagementPanel } from '@/features/camera-management/CameraManagementPanel';

const EMPTY_REGISTRY: CameraRegistry = { registry_version: 0, cameras: [] };

type DeleteBarrier = {
  confirmedAtVersion: number;
  identities: Set<string>;
};

function cameraIdentities(camera: Pick<Camera, 'id' | 'backend_camera_id'>): Set<string> {
  return new Set([camera.id, camera.backend_camera_id].filter((identity): identity is string => Boolean(identity)));
}

function identitiesOverlap(left: Set<string>, right: Set<string>): boolean {
  return Array.from(left).some((identity) => right.has(identity));
}

function upsertCamera(registry: CameraRegistry, camera: Camera, previousCameraId?: string): CameraRegistry {
  return {
    registry_version: registry.registry_version,
    cameras: [camera, ...registry.cameras.filter((item) => item.id !== camera.id && item.id !== previousCameraId)],
  };
}

export function CameraManagementPage({ onViewClips }: { onViewClips?: (camera: Camera) => void } = {}): JSX.Element {
  const resource = useCamerasResource(true);
  const [registry, setRegistry] = useState<CameraRegistry>(EMPTY_REGISTRY);
  const [addOpen, setAddOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<Camera | null>(null);
  const [deleteMessage, setDeleteMessage] = useState<string | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const deletingRef = useRef(false);
  const registryVersionRef = useRef(EMPTY_REGISTRY.registry_version);
  const deleteBarriersRef = useRef<DeleteBarrier[]>([]);
  const mutationGenerationsRef = useRef(new Map<string, number>());

  useEffect(() => {
    registryVersionRef.current = registry.registry_version;
  }, [registry.registry_version]);

  useEffect(() => {
    const snapshot = resource.data;
    if (snapshot) {
      setRegistry((current) => {
        if (snapshot.registry_version < current.registry_version) return current;
        const activeBarriers = deleteBarriersRef.current.filter(
          (barrier) => snapshot.registry_version <= barrier.confirmedAtVersion,
        );
        deleteBarriersRef.current = activeBarriers;
        const cameras = snapshot.cameras.filter((camera) => {
          const identities = cameraIdentities(camera);
          return !activeBarriers.some((barrier) => identitiesOverlap(identities, barrier.identities));
        });
        return { ...snapshot, cameras };
      });
    }
  }, [resource.data]);

  function mutationGeneration(camera: Camera, previousCameraId?: string): number {
    const identities = cameraIdentities(camera);
    if (previousCameraId) identities.add(previousCameraId);
    return Math.max(0, ...Array.from(identities, (identity) => mutationGenerationsRef.current.get(identity) ?? 0));
  }

  function handleUpdateStarted(camera: Camera): number {
    const nextGeneration = mutationGeneration(camera) + 1;
    cameraIdentities(camera).forEach((identity) => mutationGenerationsRef.current.set(identity, nextGeneration));
    return nextGeneration;
  }

  function handleUpdated(camera: Camera, previousCameraId?: string, startedAtGeneration?: number): void {
    resource.retry();
    if (startedAtGeneration !== undefined && startedAtGeneration !== mutationGeneration(camera, previousCameraId)) return;
    setRegistry((current) => upsertCamera(current, camera, previousCameraId));
  }

  function handleCreated(camera: Camera): void {
    const identities = cameraIdentities(camera);
    deleteBarriersRef.current = deleteBarriersRef.current.filter(
      (barrier) => !identitiesOverlap(identities, barrier.identities),
    );
    handleUpdated(camera);
  }

  async function confirmDelete(): Promise<void> {
    if (!deleteTarget || deletingRef.current) return;
    deletingRef.current = true;
    setDeleteBusy(true);
    setDeleteMessage(null);
    const cameraId = deleteTarget.id;
    const deletedIdentities = cameraIdentities(deleteTarget);
    try {
      await deleteCamera(cameraId);
      const nextGeneration = mutationGeneration(deleteTarget) + 1;
      deletedIdentities.forEach((identity) => mutationGenerationsRef.current.set(identity, nextGeneration));
      deleteBarriersRef.current.push({
        confirmedAtVersion: registryVersionRef.current,
        identities: deletedIdentities,
      });
      setRegistry((current) => ({
        registry_version: current.registry_version,
        cameras: current.cameras.filter((camera) => !identitiesOverlap(cameraIdentities(camera), deletedIdentities)),
      }));
      resource.retry();
      setDeleteTarget(null);
    } catch {
      setDeleteMessage('카메라 삭제에 실패했습니다. 참조 중인 이벤트나 권한을 확인하세요.');
    } finally {
      deletingRef.current = false;
      setDeleteBusy(false);
    }
  }

  const firstLoad = resource.data === null && resource.status === 'loading';
  const hasLoadedSuccessfully = resource.data !== null;
  const loadError = resource.status === 'error'
    ? resource.data === null
      ? '카메라 목록을 불러오지 못했습니다.'
      : '카메라 목록을 새로 고치지 못했습니다. 마지막 목록을 표시합니다.'
    : null;

  return (
    <>
      <CameraManagementPanel
        registry={registry}
        cameraError={loadError}
        lastSuccessAt={resource.lastSuccessAt}
        refreshing={resource.refreshing}
        loading={firstLoad}
        hasLoadedSuccessfully={hasLoadedSuccessfully}
        onRetry={resource.retry}
        onAddCamera={() => setAddOpen(true)}
        onUpdateStarted={handleUpdateStarted}
        onUpdated={handleUpdated}
        onDelete={(camera) => { setDeleteMessage(null); setDeleteTarget(camera); }}
        onViewClips={onViewClips}
      />
      <AddCameraModal open={addOpen} onClose={() => setAddOpen(false)} onCreated={handleCreated} cameras={registry.cameras} />
      {deleteTarget ? (
        <DeleteCameraDialog
          camera={deleteTarget}
          message={deleteMessage}
          busy={deleteBusy}
          onCancel={() => { if (!deleteBusy) setDeleteTarget(null); }}
          onConfirm={() => void confirmDelete()}
        />
      ) : null}
    </>
  );
}
