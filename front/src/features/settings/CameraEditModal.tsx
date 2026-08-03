import { useEffect, useRef, useState } from 'react';
import { testCamera, updateCamera, type BedZone, type Camera, type CameraTestResult } from '@/shared/api/client';
import { AccessibleDialog } from '@/shared/ui/AccessibleDialog';
import { BedZoneRecognitionPanel } from '@/features/settings/BedZoneRecognitionPanel';
import { FloorChips } from '@/features/settings/FloorChips';
import { deriveFloorOptions } from '@/features/settings/floorOptions';
import { getCameraStatusMeta, statusBadgeClassName } from '@/shared/ui/StatusBadge';
import { toast } from '@/shared/ui/Toast';

type CameraEditModalProps = {
  camera: Camera | null;
  cameras: Camera[];
  onClose: () => void;
  onUpdated: () => void;
  onRequestDelete: (camera: Camera) => void;
};

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
 * 층은 이 모달에서 편집 가능한 로컬 override다(issue #85 design handoff). 칩을 건드리지 않고
 * 저장하면 floor 패치 자체를 보내지 않아, 조용히 space-sync floor_name을 고정 override로
 * 바꿔버리는 부작용을 피한다 — 자세한 내용은 handleSave의 floor 비교 로직 참고.
 */
export function CameraEditModal({ camera, cameras, onClose, onUpdated, onRequestDelete }: CameraEditModalProps): JSX.Element | null {
  const [label, setLabel] = useState('');
  const [rtspUrl, setRtspUrl] = useState('');
  const [floor, setFloor] = useState<string | null>(null);
  const [initialFloor, setInitialFloor] = useState<string | null>(null);
  const [extraFloors, setExtraFloors] = useState<string[]>([]);
  const [bedZone, setBedZone] = useState<BedZone | null>(null);
  const [mode, setMode] = useState<'view' | 'reseg'>('view');
  const [busy, setBusy] = useState<'save' | 'test' | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<CameraTestResult | null>(null);
  const busyRef = useRef(false);
  const openCameraIdRef = useRef<string | null>(null);

  const floorOptions = deriveFloorOptions(cameras, extraFloors);

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
    const effectiveFloor = camera.floor ?? camera.floor_name ?? null;
    setFloor(effectiveFloor);
    setInitialFloor(effectiveFloor);
    setExtraFloors([]);
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
      const patch: { label?: string; rtsp_url?: string; floor?: string | null } = {};
      if (label.trim() !== camera.label) patch.label = label.trim();
      const draft = rtspUrl.trim();
      if (draft) {
        // RTSP를 바꿨다면 그 값이 실제로 붙는지 확인된 뒤에만 저장한다.
        // 확인 없이 저장하면 오타가 그대로 들어가 카메라가 조용히 죽는다.
        if (!testResult?.ok) {
          setSaveError('먼저 연결을 확인해 주세요. 연결에 성공해야 저장할 수 있습니다.');
          return;
        }
        patch.rtsp_url = draft;
      }
      // Only send floor when the user actually changed the selection (issue #85): saving without
      // touching the chips must not silently freeze the current space-sync floor_name as a local
      // override -- see FloorChips' doc comment and the floor precedence contract on Camera.floor.
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
              disabled={busyFlag}
              onChange={(event) => setRtspUrl(event.target.value)}
              placeholder="새 주소 입력 시에만 작성"
              autoComplete="off"
            />
          </label>
          <p className="mt-1 text-xs text-muted-foreground">
            현재 등록됨: <span className="font-mono">{camera.rtsp_url_masked}</span> — 변경할 때만 새 주소를 입력하세요.
          </p>

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
              {testResult.ok ? '연결 성공' : '연결 테스트에 실패했습니다.'}
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
              {busy === 'test' ? '확인 중...' : '연결'}
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
