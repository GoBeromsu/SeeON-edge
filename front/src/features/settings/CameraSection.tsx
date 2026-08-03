import { useState } from 'react';
import { CameraTable } from '@/features/settings/CameraTable';
import { CameraRegisterModal } from '@/features/settings/CameraRegisterModal';
import { CameraEditModal } from '@/features/settings/CameraEditModal';
import { DeleteCameraDialog } from '@/features/settings/DeleteCameraDialog';
import type { Camera, CameraRegistry, SystemSnapshot } from '@/shared/api/client';
import type { PollingResource } from '@/shared/api/usePollingResource';

type CameraSectionProps = {
  camerasResource: PollingResource<CameraRegistry>;
  systemResource: PollingResource<SystemSnapshot>;
};

function CameraPlusIcon(): JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4">
      <path d="M4 8a2 2 0 0 1 2-2h1.5l1-1.5h7l1 1.5H18a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2Z" />
      <circle cx="12" cy="12.5" r="3.2" />
      <path d="M18.5 5.5v4M16.5 7.5h4" />
    </svg>
  );
}

function versionLine(system: SystemSnapshot | null): string {
  if (!system) return '버전 정보 확인 중';
  const parts = [system.version ? `v${system.version.replace(/^v/, '')}` : null];
  if (system.image_digests?.ml_api) parts.push(`ml-api@${system.image_digests.ml_api}`);
  if (system.image_digests?.ml_worker) parts.push(`ml-worker@${system.image_digests.ml_worker}`);
  const filtered = parts.filter((part): part is string => Boolean(part));
  return filtered.length > 0 ? filtered.join(' · ') : '버전 정보 확인 중';
}

/** 설정 페이지 좌측 "카메라" 카드: 헤더 + 등록 버튼 + 테이블 + 버전 라인(front/design-handoff/README.md §5). */
export function CameraSection({ camerasResource, systemResource }: CameraSectionProps): JSX.Element {
  const [registerOpen, setRegisterOpen] = useState(false);
  const [editingCamera, setEditingCamera] = useState<Camera | null>(null);
  const [deletingCamera, setDeletingCamera] = useState<Camera | null>(null);

  const cameras = camerasResource.data?.cameras ?? [];

  function refresh(): void {
    camerasResource.retry();
  }

  return (
    <article className="rounded-card border border-border bg-card p-5">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold text-foreground">카메라</h2>
        <button
          type="button"
          className="inline-flex h-8 items-center gap-1.5 rounded-control bg-primary px-3 text-xs font-semibold text-primary-foreground"
          onClick={() => setRegisterOpen(true)}
        >
          <CameraPlusIcon />
          카메라 등록
        </button>
      </div>

      <div className="mt-4">
        {camerasResource.status === 'loading' && !camerasResource.data ? (
          <p className="py-6 text-center text-sm text-muted-foreground">카메라 목록을 불러오는 중입니다...</p>
        ) : camerasResource.status === 'error' && !camerasResource.data ? (
          <div className="py-6 text-center">
            <p className="text-sm text-destructive">카메라 목록을 불러오지 못했습니다.</p>
            <button type="button" className="dialog-secondary-action mt-2" onClick={refresh}>
              다시 시도
            </button>
          </div>
        ) : (
          <CameraTable
            cameras={cameras}
            onEdit={(camera) => setEditingCamera(camera)}
            onRequestDelete={(camera) => setDeletingCamera(camera)}
          />
        )}
      </div>

      <p className="mt-4 font-mono text-xs text-muted-foreground">{versionLine(systemResource.data)}</p>

      <CameraRegisterModal
        open={registerOpen}
        cameras={cameras}
        onClose={() => setRegisterOpen(false)}
        onCreated={() => {
          setRegisterOpen(false);
          refresh();
        }}
      />

      <CameraEditModal
        camera={editingCamera}
        cameras={cameras}
        onClose={() => setEditingCamera(null)}
        onUpdated={refresh}
        onRequestDelete={(camera) => {
          setEditingCamera(null);
          setDeletingCamera(camera);
        }}
      />

      <DeleteCameraDialog
        camera={deletingCamera}
        onClose={() => setDeletingCamera(null)}
        onDeleted={() => {
          setDeletingCamera(null);
          refresh();
        }}
      />
    </article>
  );
}
