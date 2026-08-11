import { useState } from 'react';
import type { CameraRegistry } from '@/shared/api/client';
import type { CameraTopology, TopologyPreview } from '@/shared/api/topologyClient';
import { TopologyConfirmationDialog } from '@/features/cameras/TopologyConfirmationDialog';
import { cameraTotal, hasCameraCountMismatch, mappedCameraTotal } from '@/features/connection/wizardSteps';

type Props = {
  readonly cameras: CameraRegistry | null;
  readonly topology: CameraTopology;
  readonly preview: TopologyPreview | null;
  readonly onPreviewChanged: (next: TopologyPreview | null) => void;
  readonly onChanged: () => Promise<void>;
};

/**
 * 3단계 본문: 서버(Hub)에 반영될 스냅샷을 확인하고, 기사님의 명시적 확인 후에만 확정합니다.
 * `preview.cameras/rooms/floors`는 TopologyConfirmationDialog와 동일하게 "비활성화될 항목" 수이며
 * (새로 생성되는 항목이 아님 -- 기존 확인 다이얼로그 문구 "삭제되지 않고 비활성화됩니다"를 그대로 유지),
 * 이 화면에서 그 의미를 창작 diff로 재해석하지 않는다.
 */
export function ServerSyncStep({ cameras, topology, preview, onPreviewChanged, onChanged }: Props): JSX.Element {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const total = cameraTotal(cameras);
  const mapped = mappedCameraTotal(topology);
  // 브리프의 "마지막 방어선": 확정 이후 서버에 반영된 카메라 수가 이 장치의 등록 카메라 수와
  // 어긋나면(예: 확정으로 방/카메라가 비활성화된 뒤) 조용히 넘어가지 않고 경고로 드러낸다.
  const mismatch = hasCameraCountMismatch(cameras, topology);

  async function afterConfirm(revision: number): Promise<void> {
    onPreviewChanged(preview ? { ...preview, confirmed: true } : null);
    setMessage(`변경 사항을 서버에 반영했습니다. (리비전 ${revision})`);
    await onChanged();
  }

  return (
    <div>
      <div className="rounded-control bg-muted p-3 text-sm text-foreground">
        이 장치에 등록된 카메라 <span className="font-semibold tabular-nums">{total}</span>대가 서버의 같은 시설
        아래 층/방 구조로 반영됩니다. (배치 완료 {mapped}대)
      </div>

      {mismatch ? (
        <p role="alert" className="auth-error mt-3">
          서버에 반영되는 카메라 수({mapped}대)가 이 장치의 등록 카메라 수({total}대)와 다릅니다. 2단계로 돌아가
          카메라 배치를 다시 확인하세요.
        </p>
      ) : null}

      {preview ? (
        preview.confirmed ? (
          <p role="status" className="dialog-success mt-4">이전 변경 사항을 이미 확정했습니다.</p>
        ) : (
          <section className="mt-4 rounded-control bg-status-pendingBg p-4 text-status-pendingFg" aria-labelledby="wizard-preview-title">
            <h3 id="wizard-preview-title" className="text-sm font-semibold">반영 전 확인이 필요합니다</h3>
            <p className="mt-1 text-sm">
              아래 항목은 삭제되지 않고 비활성화됩니다: 카메라 {preview.cameras}대 · 방 {preview.rooms}개 · 층 {preview.floors}개.
            </p>
            <div className="mt-3 flex justify-end">
              <button
                type="button"
                className="inline-flex h-11 items-center justify-center rounded-control bg-destructive px-4 text-sm font-semibold text-destructive-foreground sm:h-9"
                onClick={() => setDialogOpen(true)}
              >
                변경 사항 확인 후 반영
              </button>
            </div>
          </section>
        )
      ) : (
        <p role="status" className="dialog-success mt-4">서버에 반영할 새 변경 사항이 없습니다. 이미 최신 상태입니다.</p>
      )}

      {message ? <p role="status" className="dialog-success mt-4">{message}</p> : null}

      <TopologyConfirmationDialog
        open={dialogOpen}
        preview={preview}
        onClose={() => setDialogOpen(false)}
        onChanged={onPreviewChanged}
        onConfirmed={(revision) => { void afterConfirm(revision); }}
      />
    </div>
  );
}
