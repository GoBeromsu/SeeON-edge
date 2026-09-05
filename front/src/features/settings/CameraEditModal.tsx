import { useEffect, useMemo, useRef, useState } from 'react';
import { testCamera, updateCamera, type BedZone, type Camera, type CameraTestResult } from '@/shared/api/client';
import { AccessibleDialog } from '@/shared/ui/AccessibleDialog';
import { BedZoneRecognitionPanel } from '@/shared/ui/BedZoneRecognitionPanel';
import { FloorSelect } from '@/features/settings/FloorSelect';
import { assessRtspSubstreamGuidance } from '@/features/settings/rtspSubstreamGuidance';
import { getCameraStatusMeta, statusBadgeClassName } from '@/shared/ui/StatusBadge';
import { toast } from '@/shared/ui/Toast';

type CameraEditModalProps = {
  camera: Camera | null;
  cameras: Camera[];
  onClose: () => void;
  onUpdated: () => void;
  onRequestDelete: (camera: Camera) => void;
};

/**
 * `error_class`별 안내 (CameraRegisterModal의 PROBE_FAILURE_MESSAGE와 같은 값, 이 모달의 실패
 * 메시지에는 지금까지 반영된 적이 없었다 -- 연결 확인이 실패해도 항상 뭉뚱그린 문구 하나만 보여줬다).
 * `probe_unavailable`은 별도로 처리한다 (이슈 #151): worker의 /probe에 닿지 못해 검사 자체를
 * 못 한 것이므로, 실제로 디코드/인증/타임아웃 중 어느 것도 확정된 게 아니다 -- 그걸 "디코드
 * 실패"라고 단정하면 안 된다.
 */
const TEST_FAILURE_DETAIL: Record<string, string> = {
  timeout: 'RTSP 연결 시간이 초과되었습니다. 주소와 네트워크를 확인하세요.',
  auth: 'RTSP 인증에 실패했습니다. 계정 정보를 확인하세요.',
  decode: '영상 스트림을 디코드하지 못했습니다. 카메라 설정을 확인하세요.',
  unavailable: '현재 Flow 미디어 경로에서는 RTSP 연결 검사를 제공하지 않습니다.',
};

function testFailureMessage(result: CameraTestResult): string {
  if (result.probe_unavailable) {
    return '연결 테스트에 실패했습니다. 카메라 검사 기능을 사용할 수 없습니다 (검사 서비스에 연결하지 못했습니다). 잠시 후 다시 시도하세요.';
  }
  const detail = result.error_class ? TEST_FAILURE_DETAIL[result.error_class] : undefined;
  return detail ? `연결 테스트에 실패했습니다. ${detail}` : '연결 테스트에 실패했습니다.';
}

function RefreshIcon(): JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4">
      <path d="M4 4v5h5" />
      <path d="M20 20v-5h-5" />
      <path d="M4.5 15a8 8 0 0 0 13.9 3.2L20 18" />
      <path d="M19.5 9A8 8 0 0 0 5.6 5.8L4 6" />
    </svg>
  );
}

/**
 * 연결 관리 모달 (front/design-handoff/README.md §5 모달 2). 스펙은 침대 영역 재인식을 별도 모달로
 * 띄우지만(2단계와 동일 UI), 이 구현에서는 같은 모달 안에서 뷰를 전환한다 — AccessibleDialog는 열려있는
 * 다이얼로그 밖 body 자식들을 inert 처리하므로, 모달 위에 모달을 띄우면 먼저 연 다이얼로그가 그 안에
 * 포함되지 않는 두 번째 다이얼로그 때문에 자기 자신이 inert 처리되는 포커스 트랩 버그가 생긴다.
 * 삭제도 같은 이유로 이 모달 안에서 직접 확인창을 띄우지 않고, 부모(CameraSection)의 공용
 * DeleteCameraDialog로 위임한다(onRequestDelete) — 그래야 한 번에 다이얼로그가 하나만 열린다.
 * 층은 이 모달에서 편집 가능한 로컬 override다(issue #85 design handoff; 고정 목록 B1~10층의
 * 정수, issue #155). 드롭다운을 건드리지 않고 저장하면 floor 패치 자체를 보내지 않아, 조용히
 * space-sync floor_name을 고정 override로 바꿔버리는 부작용을 피한다 — 자세한 내용은 handleSave의
 * floor 비교 로직 참고. 드롭다운의 "클라우드 값 사용" 옵션을 고르면 null을 보내 override를 지우고
 * floor_name로 되돌린다.
 */
export function CameraEditModal({ camera, onClose, onUpdated, onRequestDelete }: CameraEditModalProps): JSX.Element | null {
  const [label, setLabel] = useState('');
  const [rtspUrl, setRtspUrl] = useState('');
  const [floor, setFloor] = useState<number | null>(null);
  const [initialFloor, setInitialFloor] = useState<number | null>(null);
  const [bedZone, setBedZone] = useState<BedZone | null>(null);
  const [mode, setMode] = useState<'view' | 'reseg'>('view');
  const [busy, setBusy] = useState<'save' | 'test' | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  /**
   * 마지막 연결 확인 결과. 저장을 막지 않고 화면에만 보여준다. 주소를
   * 고치면 지운다 — 검사받지 않은 주소 위에 "연결 성공"이 남아 있으면
   * 그게 방금 그 주소의 결과처럼 읽힌다.
   */
  const [testResult, setTestResult] = useState<CameraTestResult | null>(null);
  const busyRef = useRef(false);
  const openCameraIdRef = useRef<string | null>(null);
  // 등록 모달과 같은 안내다 (issue #154): 새 주소를 입력할 때만 평가하므로,
  // 손대지 않은 기존 카메라(빈 draft)에는 아무것도 뜨지 않는다.
  const substreamGuidance = useMemo(() => assessRtspSubstreamGuidance(rtspUrl), [rtspUrl]);

  // Reset the form only when the targeted camera's identity changes (opening the modal for a
  // camera, or switching to a different one) — not on every re-render caused by the operations
  // page's polling passing a fresh `camera` object for the same id. Otherwise an in-progress
  // re-recognition flow (mode === 'reseg') gets silently snapped back to 'view' mid-flow.
  useEffect(() => {
    if (!camera) {
      openCameraIdRef.current = null;
      return;
    }
    if (openCameraIdRef.current === camera.id) return;
    openCameraIdRef.current = camera.id;
    setLabel(camera.label);
    setRtspUrl('');
    // floor_name(공간 동기화가 채우는 표시용 문자열)과 섞지 않는다 -- 로컬
    // override가 없으면 null로 두고, 드롭다운은 unsetLabel로 floor_name을
    // 대체 표시만 한다(FloorSelect 참고).
    const effectiveFloor = camera.floor ?? null;
    setFloor(effectiveFloor);
    setInitialFloor(effectiveFloor);
    setBedZone(camera.bed_zone ?? null);
    setMode('view');
    setSaveError(null);
    setTestResult(null);
    setBusy(null);
    busyRef.current = false;
  }, [camera]);

  function requestClose(): void {
    if (busyRef.current) return;
    onClose();
  }

  async function handleTest(): Promise<void> {
    if (!camera || busyRef.current) return;
    busyRef.current = true;
    setBusy('test');
    setTestResult(null);
    try {
      // 저장된 값이 아니라 지금 입력창에 있는 값을 검사한다. 예전에는
      // 저장본만 검사해 오타를 넣어도 "연결 성공"이 뜬 뒤 그대로 저장됐다.
      const draft = rtspUrl.trim();
      const result = await testCamera(camera.id, draft || undefined);
      setTestResult(result);
      if (result.ok) {
        toast.success('연결 성공');
        onUpdated();
      }
    } catch {
      setTestResult({ ok: false, error_class: undefined });
    } finally {
      busyRef.current = false;
      setBusy(null);
    }
  }

  async function handleSave(): Promise<void> {
    if (!camera || busyRef.current) return;
    if (!label.trim()) {
      setSaveError('카메라 이름을 입력하세요.');
      return;
    }
    busyRef.current = true;
    setBusy('save');
    setSaveError(null);
    try {
      const patch: { label?: string; rtsp_url?: string; floor?: number | null } = {};
      if (label.trim() !== camera.label) patch.label = label.trim();
      const draft = rtspUrl.trim();
      if (draft) {
        // 저장은 저장이다. 연결 확인은 "연결 확인" 버튼으로 언제든 할 수
        // 있고, 그 결과는 정보로만 보여준다.
        //
        // 예전에는 연결 테스트를 통과해야만 저장할 수 있었는데, 그러면
        // 주소를 고칠 방법 자체가 사라진다: 주소가 틀려서 연결이 안 되는
        // 카메라야말로 주소를 고쳐야 하는데, 고친 주소를 저장하려면 먼저
        // 연결에 성공해야 했다. 틀린 주소는 목록에 오프라인으로 그대로
        // 보이므로 조용히 죽지 않는다.
        patch.rtsp_url = draft;
      }
      // Only send floor when the user actually changed the selection (issue #85): saving without
      // touching the dropdown must not silently freeze the current space-sync floor_name as a
      // local override -- see FloorSelect's doc comment and the floor precedence contract on
      // Camera.floor.
      if (floor !== initialFloor) patch.floor = floor;
      await updateCamera(camera.id, patch);
      onUpdated();
      toast.success('카메라 정보를 저장했습니다.');
      onClose();
    } catch {
      setSaveError('카메라 정보를 저장하지 못했습니다. 잠시 후 다시 시도하세요.');
    } finally {
      busyRef.current = false;
      setBusy(null);
    }
  }

  if (!camera) return null;
  const statusMeta = getCameraStatusMeta(camera.status);
  const bedZoneBadge = bedZone
    ? { label: '인식 완료', className: statusBadgeClassName('approved') }
    : { label: '인식 필요', className: statusBadgeClassName('rejected') };
  const busyFlag = busy !== null;

  return (
    <AccessibleDialog open={camera !== null} title="연결 관리" onClose={requestClose} size={mode === 'reseg' ? 'lg' : 'md'}>
      <div className="mb-3 flex items-center gap-2">
        <span className={statusMeta.className}>
          <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-current" />
          {statusMeta.label}
        </span>
      </div>

      {mode === 'reseg' ? (
        <div>
          <BedZoneRecognitionPanel
            cameraId={camera.id}
            bedZone={bedZone}
            onRecognized={(zone) => {
              setBedZone(zone);
              onUpdated();
            }}
          />
          <div className="dialog-actions mt-4">
            <button type="button" className="dialog-secondary-action" onClick={() => setMode('view')}>
              취소
            </button>
            <button
              type="button"
              className="brand-action inline-flex h-9 items-center justify-center rounded-control px-4 text-sm font-semibold"
              onClick={() => setMode('view')}
            >
              완료
            </button>
          </div>
        </div>
      ) : (
        <form onSubmit={(event) => { event.preventDefault(); void handleSave(); }} noValidate>
          <label>
            이름
            <input
              name="label"
              value={label}
              disabled={busyFlag}
              onChange={(event) => setLabel(event.target.value)}
            />
          </label>
          <label>
            층
            <FloorSelect
              value={floor}
              onChange={setFloor}
              unsetLabel={camera.floor_name ?? null}
              disabled={busyFlag}
            />
          </label>
          <label>
            RTSP 주소
            <input
              name="rtsp_url"
              value={rtspUrl}
              disabled={busyFlag}
              onChange={(event) => {
                setRtspUrl(event.target.value);
                setTestResult(null);
              }}
              placeholder="새 주소 입력 시에만 작성"
              autoComplete="off"
            />
          </label>
          <p className="mt-1 text-xs text-muted-foreground">
            현재 등록됨: <span className="font-mono">{camera.rtsp_url_masked}</span> — 변경할 때만 새 주소를 입력하세요.
          </p>
          {substreamGuidance ? (
            <p data-testid="rtsp-substream-guidance" className="mt-1 text-sm text-amber-600">
              {substreamGuidance.message}
            </p>
          ) : null}

          <div className="flex items-center justify-between gap-2 text-sm">
            <span className="font-semibold text-muted-foreground">침대 영역</span>
            <div className="flex items-center gap-2">
              <span className={bedZoneBadge.className}>{bedZoneBadge.label}</span>
              <button
                type="button"
                className="icon-button"
                aria-label="침대 영역 다시 인식"
                onClick={() => setMode('reseg')}
              >
                <RefreshIcon />
              </button>
            </div>
          </div>

          {testResult ? (
            <p role={testResult.ok ? 'status' : 'alert'} className={testResult.ok ? 'dialog-success' : 'auth-error'}>
              {testResult.ok ? '연결 성공' : testFailureMessage(testResult)}
            </p>
          ) : null}

          {saveError ? <p role="alert" className="auth-error">{saveError}</p> : null}

          <div className="dialog-actions">
            <button
              type="button"
              className="mr-auto inline-flex h-9 items-center justify-center rounded-control px-4 text-sm font-semibold text-destructive hover:underline"
              disabled={busyFlag}
              onClick={() => onRequestDelete(camera)}
            >
              삭제
            </button>
            <button type="button" className="dialog-secondary-action" disabled={busyFlag} onClick={() => void handleTest()}>
              {busy === 'test' ? '확인 중...' : '연결 확인'}
            </button>
            <button type="submit" className="brand-action inline-flex h-9 items-center justify-center rounded-control px-4 text-sm font-semibold" disabled={busyFlag}>
              {busy === 'save' ? '저장 중...' : '저장'}
            </button>
          </div>
        </form>
      )}
    </AccessibleDialog>
  );
}
