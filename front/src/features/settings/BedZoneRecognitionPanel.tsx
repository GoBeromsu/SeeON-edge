import { useState } from 'react';
import { bedZoneRecognitionFailureDetail, getCameraSnapshotUrl, recognizeBedZone } from '@/shared/api/client';
import type { BedZone } from '@/shared/api/client';

type BedZoneRecognitionPanelProps = {
  cameraId: string;
  bedZone: BedZone | null;
  onRecognized: (bedZone: BedZone) => void;
};

function polygonPoints(bedZone: BedZone): string {
  return bedZone.polygon
    .map(([x, y]) => `${(x / bedZone.image_width) * 100},${(y / bedZone.image_height) * 100}`)
    .join(' ');
}

/**
 * Shared 침대 영역 인식 UI, reused by the 카메라 등록 모달's step 2 and the 연결 관리 모달's 다시 인식
 * sub-flow. Recognition is server-side YOLO segmentation (POST /cameras/{id}/bed-zone/recognize) — the
 * technician never draws the polygon themselves, only triggers/reviews it.
 */
export function BedZoneRecognitionPanel({ cameraId, bedZone, onRecognized }: BedZoneRecognitionPanelProps): JSX.Element {
  const [recognizing, setRecognizing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  async function handleRecognize(): Promise<void> {
    if (recognizing) return;
    setRecognizing(true);
    setError(null);
    try {
      const result = await recognizeBedZone(cameraId);
      onRecognized(result);
      setRefreshKey((key) => key + 1);
    } catch (caught) {
      const detail = bedZoneRecognitionFailureDetail(caught);
      setError(
        detail?.error_class === 'bed_not_found'
          ? '침대를 찾지 못했습니다. 카메라 각도를 확인한 뒤 다시 시도하세요.'
          : '침대 영역 인식에 실패했습니다. 잠시 후 다시 시도하세요.',
      );
    } finally {
      setRecognizing(false);
    }
  }

  return (
    <div>
      <div className="event-media-frame relative">
        <img
          src={getCameraSnapshotUrl(cameraId, refreshKey)}
          alt=""
          className="h-full w-full object-cover"
        />
        {bedZone ? (
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
            <polygon points={polygonPoints(bedZone)} fill="rgba(43,182,163,0.25)" stroke="var(--overlay-teal)" strokeWidth={1} />
          </svg>
        ) : null}
        {recognizing ? (
          <div className="media-status-overlay absolute inset-0 flex items-center justify-center text-sm font-semibold">
            인식 중...
          </div>
        ) : null}
        {bedZone && !recognizing ? (
          <span className="media-status-overlay absolute bottom-2 left-2 rounded-full px-2.5 py-1 text-xs font-semibold">
            침대 · 자동 인식됨
          </span>
        ) : null}
      </div>

      <p aria-live="polite" className="mt-2 text-sm text-muted-foreground">
        {recognizing ? '침대 영역을 인식하는 중입니다...' : bedZone ? '침대 영역이 인식되었습니다.' : '침대 영역 인식이 필요합니다.'}
      </p>

      {error ? <p role="alert" className="auth-error">{error}</p> : null}

      <button
        type="button"
        className="dialog-secondary-action mt-2"
        disabled={recognizing}
        onClick={() => void handleRecognize()}
      >
        {recognizing ? '인식 중...' : bedZone ? '다시 인식' : '인식 시작'}
      </button>
    </div>
  );
}
