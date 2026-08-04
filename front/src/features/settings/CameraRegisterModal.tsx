import { useEffect, useMemo, useRef, useState } from 'react';
import {
  cameraDuplicateDetail,
  cameraProbeFailureDetail,
  createCamera,
  type Camera,
  type BedZone,
} from '@/shared/api/client';
import { AccessibleDialog } from '@/shared/ui/AccessibleDialog';
import { BedZoneRecognitionPanel } from '@/features/settings/BedZoneRecognitionPanel';
import { FloorChips } from '@/features/settings/FloorChips';
import { deriveFloorOptions } from '@/features/settings/floorOptions';
import { toast } from '@/shared/ui/Toast';

type CameraRegisterModalProps = {
  open: boolean;
  cameras: Camera[];
  onClose: () => void;
  onCreated: () => void;
};

const PROBE_FAILURE_MESSAGE: Record<string, string> = {
  timeout: 'RTSP 연결 시간이 초과되었습니다. 주소와 네트워크를 확인하세요.',
  auth: 'RTSP 인증에 실패했습니다. 계정 정보를 확인하세요.',
  decode: '영상 스트림을 디코드하지 못했습니다. 카메라 설정을 확인하세요.',
};

/**
 * 카메라 등록 모달 (front/design-handoff/Eldercare Prototype.dc.html:305-336 모달 1): 1단계에서 연결
 * 정보를 입력하면 그 자리에서 POST /cameras가 프로브 후 저장까지 수행하므로(별도 프로브 엔드포인트
 * 없음), 2단계는 이미 생성된 카메라에 대해 침대 영역을 인식하는 단계다. 층은 사용자가 이 모달에서
 * 선택하는 로컬 override(`floor`)이며, 외부 space-sync가 채우는 읽기 전용 `floor_name`과는 별개다
 * (issue #85; 정밀 계약은 CameraResponse.floor 문서 주석 참고).
 */
export function CameraRegisterModal({ open, cameras, onClose, onCreated }: CameraRegisterModalProps): JSX.Element {
  const [step, setStep] = useState<1 | 2>(1);
  const [label, setLabel] = useState('');
  const [rtspUrl, setRtspUrl] = useState('');
  const [floor, setFloor] = useState<string | null>(null);
  const [extraFloors, setExtraFloors] = useState<string[]>([]);
  // 클라우드 방 선택. 미지정으로 저장하면 roster_sync가 이 카메라를 push
  // 대상에서 제외해, 엣지에는 보이는데 클라우드에는 영영 안 나타난다.
  const [spaceId, setSpaceId] = useState<string>('');

  /**
   * 선택 가능한 클라우드 방.
   *
   * 엣지가 pull한 roster 중 아직 로컬 카메라와 매칭되지 않은 항목이
   * "빈 방"이다. 카메라 1대 = 방 1개이므로 이미 점유된 방은 제외한다.
   */
  const availableSpaces = useMemo(() => {
    const taken = new Set(
      cameras
        .filter((cam) => cam.mapping_pending === false && cam.space_id)
        .map((cam) => cam.space_id as string),
    );
    const seen = new Set<string>();
    return cameras
      .filter((cam) => cam.space_id && !taken.has(cam.space_id))
      .filter((cam) => {
        if (seen.has(cam.space_id as string)) return false;
        seen.add(cam.space_id as string);
        return true;
      })
      .map((cam) => ({
        id: cam.space_id as string,
        name: cam.space_name ?? cam.label,
        floorName: cam.floor_name ?? null,
      }));
  }, [cameras]);
  const [busy, setBusy] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [duplicateLabel, setDuplicateLabel] = useState<string | null>(null);
  const [camera, setCamera] = useState<Camera | null>(null);
  const [bedZone, setBedZone] = useState<BedZone | null>(null);
  const busyRef = useRef(false);
  const wasOpen = useRef(open);

  const floorOptions = deriveFloorOptions(cameras, extraFloors);

  useEffect(() => {
    if (open && !wasOpen.current) {
      setStep(1);
      setLabel('');
      setRtspUrl('');
      setFloor(null);
      setExtraFloors([]);
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
    // 방을 고르지 않으면 roster_sync가 이 카메라를 클라우드 push에서
    // 제외한다. 엣지에는 보이는데 클라우드 현황판에는 안 나타나므로,
    // 기사님이 현장을 떠난 뒤에야 발견된다. 등록 시점에 막는다.
    if (availableSpaces.length > 0 && !spaceId) {
      toast.error('이 카메라가 설치된 방을 선택하세요.');
      return;
    }

    busyRef.current = true;
    setBusy(true);
    setErrorMessage(null);
    setDuplicateLabel(null);
    try {
      const created = await createCamera(
        {
          label,
          rtsp_url: rtspUrl,
          floor: floor ?? undefined,
          space_id: spaceId || undefined,
        },
        { forceRegister: force },
      );
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
    <AccessibleDialog open={open} title="카메라 등록" onClose={requestClose} size={step === 1 ? 'md' : 'lg'}>
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
          <div>
            <span className="text-sm font-medium">층</span>
            <FloorChips
              options={floorOptions}
              selected={floor}
              onSelect={setFloor}
              onAddFloor={(newFloor) => {
                setExtraFloors((prev) => (prev.includes(newFloor) ? prev : [...prev, newFloor]));
                setFloor(newFloor);
              }}
            />
          </div>
          <label>
            RTSP 주소
            <input
              name="rtsp_url"
              value={rtspUrl}
              disabled={busy || camera !== null}
              onChange={(event) => setRtspUrl(event.target.value)}
              placeholder="rtsp://192.0.2.10:554/stream"
            />
          </label>

          {availableSpaces.length > 0 ? (
            <label>
              설치된 방
              <select
                name="space_id"
                value={spaceId}
                disabled={busy || camera !== null}
                onChange={(event) => setSpaceId(event.target.value)}
              >
                <option value="">방을 선택하세요</option>
                {availableSpaces.map((space) => (
                  <option key={space.id} value={space.id}>
                    {space.floorName ? `${space.floorName} · ${space.name}` : space.name}
                  </option>
                ))}
              </select>
              <span className="text-sm text-muted-foreground">
                방을 지정해야 클라우드 현황판에 이 카메라가 나타납니다.
              </span>
            </label>
          ) : (
            /* 클라우드 roster가 아직 안 왔을 때. 등록 자체를 막으면 오프라인
               설치가 불가능해지므로 통과시키되, 침묵하지는 않는다 — 기사가
               정상 등록으로 알고 현장을 떠나면 클라우드에는 영영 안 나타난다. */
            <p data-testid="no-spaces-notice" className="text-sm text-amber-600">
              클라우드에서 방 목록을 아직 받지 못했습니다. 지금 등록하면 방 지정
              없이 저장되며, 클라우드 현황판에는 나타나지 않습니다. 연결 설정을
              먼저 확인하거나, 등록 후 수정 화면에서 방을 지정하세요.
            </p>
          )}

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
