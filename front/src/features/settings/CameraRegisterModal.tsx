import { useEffect, useRef, useState } from 'react';
import {
  cameraDuplicateDetail,
  cameraProbeFailureDetail,
  createCamera,
  type Camera,
  type BedZone,
} from '@/shared/api/client';
import { AccessibleDialog } from '@/shared/ui/AccessibleDialog';
import { BedZoneRecognitionPanel } from '@/features/settings/BedZoneRecognitionPanel';
import { toast } from '@/shared/ui/Toast';

type CameraRegisterModalProps = {
  open: boolean;
  onClose: () => void;
  onCreated: () => void;
};

const PROBE_FAILURE_MESSAGE: Record<string, string> = {
  timeout: 'RTSP 연결 시간이 초과되었습니다. 주소와 네트워크를 확인하세요.',
  auth: 'RTSP 인증에 실패했습니다. 계정 정보를 확인하세요.',
  decode: '영상 스트림을 디코드하지 못했습니다. 카메라 설정을 확인하세요.',
};

/**
 * 카메라 등록 모달 (front/design-handoff/README.md §5 모달 1): 1단계에서 연결 정보를 입력하면 그 자리에서
 * POST /cameras가 프로브 후 저장까지 수행하므로(별도 프로브 엔드포인트 없음), 2단계는 이미 생성된
 * 카메라에 대해 침대 영역을 인식하는 단계다. 층은 외부 space-sync 필드라 이 모달에서 입력받지 않는다
 * (스펙 이탈: floor chip 선택 UI 없음, floor_name은 서버가 읽기 전용으로 채운다).
 */
export function CameraRegisterModal({ open, onClose, onCreated }: CameraRegisterModalProps): JSX.Element {
  const [step, setStep] = useState<1 | 2>(1);
  const [label, setLabel] = useState('');
  const [rtspUrl, setRtspUrl] = useState('');
  const [busy, setBusy] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [duplicateLabel, setDuplicateLabel] = useState<string | null>(null);
  const [camera, setCamera] = useState<Camera | null>(null);
  const [bedZone, setBedZone] = useState<BedZone | null>(null);
  const busyRef = useRef(false);
  const wasOpen = useRef(open);

  useEffect(() => {
    if (open && !wasOpen.current) {
      setStep(1);
      setLabel('');
      setRtspUrl('');
      setErrorMessage(null);
      setDuplicateLabel(null);
      setCamera(null);
      setBedZone(null);
      setBusy(false);
      busyRef.current = false;
    }
    wasOpen.current = open;
  }, [open]);

  function requestClose(): void {
    if (busyRef.current) return;
    onClose();
  }

  async function submitStep1(force = false): Promise<void> {
    if (busyRef.current) return;
    if (!label.trim()) {
      toast.error('카메라 이름을 입력하세요.');
      return;
    }
    if (!rtspUrl.trim()) {
      toast.error('RTSP 주소를 입력하세요.');
      return;
    }

    busyRef.current = true;
    setBusy(true);
    setErrorMessage(null);
    setDuplicateLabel(null);
    try {
      const created = await createCamera({ label, rtsp_url: rtspUrl }, { forceRegister: force });
      setCamera(created);
      setBedZone(created.bed_zone ?? null);
      setStep(2);
    } catch (error) {
      const probeFailure = cameraProbeFailureDetail(error);
      if (probeFailure) {
        setErrorMessage(
          (probeFailure.error_class && PROBE_FAILURE_MESSAGE[probeFailure.error_class])
            ?? 'RTSP 연결을 확인하지 못했습니다. 주소를 다시 확인하세요.',
        );
        return;
      }
      const duplicate = cameraDuplicateDetail(error);
      if (duplicate) {
        setErrorMessage(`이미 등록된 스트림입니다 (기존 카메라: ${duplicate.existing_label}).`);
        setDuplicateLabel(duplicate.existing_label);
        return;
      }
      setErrorMessage('카메라를 등록하지 못했습니다. 잠시 후 다시 시도하세요.');
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }

  function handleComplete(): void {
    if (!bedZone) {
      toast.error('침대 영역 인식을 완료해야 저장할 수 있습니다.');
      return;
    }
    toast.success('카메라를 등록했습니다.');
    onCreated();
  }

  return (
    <AccessibleDialog open={open} title="카메라 등록" onClose={requestClose}>
      <p className="mb-3 text-xs font-semibold text-muted-foreground">
        <span className={step === 1 ? 'text-foreground' : ''}>1 연결 정보</span>
        {' → '}
        <span className={step === 2 ? 'text-foreground' : ''}>2 침대 영역</span>
      </p>

      {step === 1 ? (
        <form
          onSubmit={(event) => {
            event.preventDefault();
            if (camera) {
              setStep(2);
            } else {
              void submitStep1();
            }
          }}
          noValidate
        >
          <label>
            이름
            <input
              name="label"
              value={label}
              disabled={busy || camera !== null}
              onChange={(event) => setLabel(event.target.value)}
              placeholder="예: 101호"
            />
          </label>
          <label>
            RTSP 주소
            <input
              name="rtsp_url"
              value={rtspUrl}
              disabled={busy || camera !== null}
              onChange={(event) => setRtspUrl(event.target.value)}
              placeholder="rtsp://user:pass@host:554/stream"
            />
          </label>

          {errorMessage ? (
            <div>
              <p role="alert" className="auth-error">{errorMessage}</p>
              {duplicateLabel ? (
                <button
                  type="button"
                  className="dialog-secondary-action"
                  disabled={busy}
                  onClick={() => void submitStep1(true)}
                >
                  그래도 등록
                </button>
              ) : null}
            </div>
          ) : null}

          <div className="dialog-actions">
            <button type="button" className="dialog-secondary-action" disabled={busy} onClick={requestClose}>
              취소
            </button>
            <button
              type="submit"
              className="brand-action inline-flex h-9 items-center justify-center rounded-control px-4 text-sm font-semibold"
              disabled={busy}
            >
              {busy ? '등록 중...' : '다음'}
            </button>
          </div>
        </form>
      ) : null}

      {step === 2 && camera ? (
        <div>
          <BedZoneRecognitionPanel cameraId={camera.id} bedZone={bedZone} onRecognized={setBedZone} />
          <div className="dialog-actions mt-4">
            <button type="button" className="dialog-secondary-action" onClick={() => setStep(1)}>
              이전
            </button>
            <button
              type="button"
              className="brand-action inline-flex h-9 items-center justify-center rounded-control px-4 text-sm font-semibold"
              onClick={handleComplete}
            >
              저장하고 완료
            </button>
          </div>
        </div>
      ) : null}
    </AccessibleDialog>
  );
}
