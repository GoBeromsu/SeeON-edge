import { useEffect, useRef, useState } from 'react';
import { bedZoneRecognitionFailureDetail, getCameraStreamUrl, recognizeBedZone } from '@/shared/api/client';
import type { BedZone } from '@/shared/api/client';
import { useMjpegStream } from '@/shared/api/useMjpegStream';

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
 *
 * #157: 화면은 정지 스냅샷이 아니라 운영 화면과 같은 MJPEG 라이브 스트림이다(`useMjpegStream` 재사용
 * -- 직접 `<img>` 를 새로 만들지 않는다). 카메라 각도를 조정하는 작업이라 화면이 실시간으로
 * 따라와야 하기 때문. 인식은 버튼을 누를 때마다 한 번만 실행한다. 자동 반복은 매번 Flow를
 * 재시작하므로 사용하지 않으며, 결과를 확인한 기술자가 필요할 때 명시적으로 다시 실행한다.
 *
 * 폴리곤 오버레이는 캔버스가 아니라 그 위에 겹친 별도의 `<svg>` 로 그린다: `useMjpegStream` 은
 * 프레임마다 캔버스에 `drawImage` 하므로, 오버레이를 같은 캔버스에 그리면 다음 프레임에 지워진다.
 */
export function BedZoneRecognitionPanel({ cameraId, bedZone, onRecognized }: BedZoneRecognitionPanelProps): JSX.Element {
  const [recognizing, setRecognizing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const generationRef = useRef(0);
  const requestRef = useRef<object | null>(null);
  const onRecognizedRef = useRef(onRecognized);
  onRecognizedRef.current = onRecognized;

  const stream = useMjpegStream(getCameraStreamUrl(cameraId));

  useEffect(() => {
    generationRef.current += 1;
    requestRef.current = null;
    setRecognizing(false);
    setError(null);
    return () => {
      generationRef.current += 1;
      requestRef.current = null;
    };
  }, [cameraId]);

  async function runRecognition(): Promise<void> {
    if (requestRef.current) return;
    const generation = generationRef.current;
    const request = {};
    requestRef.current = request;
    setRecognizing(true);
    setError(null);
    try {
      const result = await recognizeBedZone(cameraId);
      if (generation === generationRef.current && requestRef.current === request) {
        onRecognizedRef.current(result);
      }
    } catch (caught) {
      if (generation !== generationRef.current || requestRef.current !== request) return;
      const detail = bedZoneRecognitionFailureDetail(caught);
      setError(
        detail?.error_class === 'bed_not_found'
          ? '침대를 찾지 못했습니다. 카메라 각도를 확인한 뒤 다시 시도하세요.'
          : '침대 영역 인식에 실패했습니다. 잠시 후 다시 시도하세요.',
      );
    } finally {
      if (generation === generationRef.current && requestRef.current === request) {
        requestRef.current = null;
        setRecognizing(false);
      }
    }
  }

  return (
    <div>
      <div className="event-media-frame relative">
        <canvas ref={stream.canvasRef} role="img" aria-label="카메라 영상" className="h-full w-full object-cover" />
        {bedZone ? (
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
            <polygon points={polygonPoints(bedZone)} fill="rgba(43,182,163,0.25)" stroke="var(--overlay-teal)" strokeWidth={1} />
          </svg>
        ) : null}
        {recognizing && !bedZone ? (
          <div className="media-status-overlay absolute inset-0 flex items-center justify-center text-sm font-semibold">
            인식 중...
          </div>
        ) : null}
        {recognizing && bedZone ? (
          <span
            role="status"
            className="media-status-overlay absolute right-2 top-2 rounded-full px-2.5 py-1 text-xs font-semibold"
          >
            재인식 중…
          </span>
        ) : null}
        {bedZone ? (
          <span className="media-status-overlay absolute bottom-2 left-2 rounded-full px-2.5 py-1 text-xs font-semibold">
            침대 · 자동 인식됨
          </span>
        ) : null}
      </div>

      <p aria-live="polite" className="mt-2 text-sm text-muted-foreground">
        {recognizing
          ? '침대 영역을 인식하는 중입니다...'
          : bedZone
            ? '침대 영역이 인식되었습니다.'
            : '침대 영역 인식이 필요합니다.'}
      </p>

      {error ? <p role="alert" className="auth-error">{error}</p> : null}

      <button
        type="button"
        className="dialog-secondary-action mt-2"
        onClick={() => void runRecognition()}
        disabled={recognizing}
      >
        {recognizing ? '인식 중...' : bedZone ? '다시 인식' : '▶ 인식 시작'}
      </button>
    </div>
  );
}
