import { useRef, useState } from 'react';
import { deleteCamera } from '@/shared/api/client';
import { AccessibleDialog } from '@/shared/ui/AccessibleDialog';
import { toast } from '@/shared/ui/Toast';
import type { Camera } from '@/shared/api/client';

type DeleteCameraDialogProps = {
  camera: Camera | null;
  onClose: () => void;
  onDeleted: () => void;
};

/** Destructive-action confirm dialog for the 카메라 테이블's 삭제 action. */
export function DeleteCameraDialog({ camera, onClose, onDeleted }: DeleteCameraDialogProps): JSX.Element | null {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const busyRef = useRef(false);

  function requestClose(): void {
    if (busyRef.current) return;
    setError(null);
    onClose();
  }

  async function handleConfirm(): Promise<void> {
    if (!camera || busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    setError(null);
    try {
      await deleteCamera(camera.id);
      toast.success('카메라를 삭제했습니다.');
      onDeleted();
    } catch {
      setError('카메라를 삭제하지 못했습니다. 잠시 후 다시 시도하세요.');
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }

  return (
    <AccessibleDialog open={camera !== null} title="카메라 삭제" onClose={requestClose}>
      <p className="text-sm text-foreground">
        {camera ? <>&ldquo;{camera.label}&rdquo; 카메라를 삭제하시겠습니까?</> : null} 이 작업은 되돌릴 수 없습니다.
      </p>
      {error ? <p role="alert" className="auth-error">{error}</p> : null}
      <div className="dialog-actions mt-4">
        <button type="button" className="dialog-secondary-action" disabled={busy} onClick={requestClose}>
          취소
        </button>
        <button
          type="button"
          className="inline-flex h-9 items-center justify-center rounded-control bg-destructive px-4 text-sm font-semibold text-destructive-foreground disabled:opacity-60"
          disabled={busy}
          onClick={() => void handleConfirm()}
        >
          {busy ? '삭제 중...' : '삭제'}
        </button>
      </div>
    </AccessibleDialog>
  );
}
