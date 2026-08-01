import { ChangeEvent, FormEvent, useMemo, useRef, useState } from 'react';
import { cameraDuplicateDetail, cameraProbeFailureDetail, createCamera, type Camera } from '@/shared/api/client';
import { connectionFailureMessage } from '@/features/cameras/connectionTestMessage';
import {
  buildCameraRegistrationRtspUrl,
  findDuplicateLabelCamera,
  initialCameraRegistrationRtspForm,
  maskCameraRegistrationRtspPreview,
  parseCameraRegistrationRtspUrl,
  validateCameraRegistrationForm,
  type CameraRegistrationRtspForm,
} from '@/features/cameras/cameraRegistrationForm';
import { AccessibleDialog } from '@/shared/ui/AccessibleDialog';

type AddCameraModalProps = {
  open: boolean;
  onClose: () => void;
  onCreated: (camera: Camera) => void;
  /** Registered cameras to compare the entered label against for a non-blocking duplicate-label warning (R2.2). */
  cameras?: readonly Camera[];
};

// Set once createCamera rejects with a structured 409/422 body. Cleared on the next submit attempt.
type RegistrationIssue =
  | { kind: 'duplicate'; existingCameraId: string; existingLabel: string }
  | { kind: 'probe_failed'; errorClass?: string };

export function AddCameraModal({ open, onClose, onCreated, cameras = [] }: AddCameraModalProps): JSX.Element | null {
  const [label, setLabel] = useState('');
  const [rtspForm, setRtspForm] = useState<CameraRegistrationRtspForm>(initialCameraRegistrationRtspForm);
  const [message, setMessage] = useState<string | null>(null);
  const [issue, setIssue] = useState<RegistrationIssue | null>(null);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const nameInputRef = useRef<HTMLInputElement>(null);
  const [pasteInput, setPasteInput] = useState('');
  const [pasteError, setPasteError] = useState<string | null>(null);
  const [fieldsCollapsed, setFieldsCollapsed] = useState(false);

  const rtspUrl = useMemo(() => buildCameraRegistrationRtspUrl(rtspForm), [rtspForm]);
  const rtspPreview = useMemo(() => maskCameraRegistrationRtspPreview(rtspUrl), [rtspUrl]);
  const validationError = useMemo(() => validateCameraRegistrationForm(label, rtspForm), [label, rtspForm]);
  // Non-blocking per R2.2: a label match never disables submit, unlike validationError above.
  const duplicateLabelCamera = useMemo(() => findDuplicateLabelCamera(label, cameras), [label, cameras]);

  function resetAndClose(): void {
    setLabel('');
    setRtspForm(initialCameraRegistrationRtspForm);
    setMessage(null);
    setIssue(null);
    setBusy(false);
    busyRef.current = false;
    setPasteInput('');
    setPasteError(null);
    setFieldsCollapsed(false);
    onClose();
  }

  function requestClose(): void {
    if (busyRef.current) return;
    resetAndClose();
  }

  function updateRtspField(field: keyof CameraRegistrationRtspForm, value: string): void {
    setRtspForm((current) => ({ ...current, [field]: value }));
  }

  function handlePasteChange(event: ChangeEvent<HTMLInputElement>): void {
    const value = event.target.value;
    setPasteInput(value);
    // Only attempt to parse once the text looks like it could be a URL — avoids flashing an
    // error on every keystroke while the user is still typing or mid-paste.
    if (!value.includes('://')) {
      setPasteError(null);
      return;
    }
    const parsed = parseCameraRegistrationRtspUrl(value);
    if (!parsed) {
      setPasteError('RTSP 주소를 인식하지 못했습니다. 아래 필드를 직접 입력하세요.');
      setFieldsCollapsed(false);
      return;
    }
    setRtspForm(parsed);
    setPasteError(null);
    setFieldsCollapsed(true);
  }

  async function submit(forceRegister: boolean): Promise<void> {
    if (busyRef.current) return;
    setMessage(null);
    setIssue(null);
    if (validationError) {
      setMessage(validationError);
      return;
    }

    busyRef.current = true;
    setBusy(true);
    try {
      const camera = await createCamera({ label: label.trim(), rtsp_url: rtspUrl }, { forceRegister });
      onCreated(camera);
      resetAndClose();
      return;
    } catch (error) {
      const duplicate = cameraDuplicateDetail(error);
      if (duplicate) {
        setIssue({ kind: 'duplicate', existingCameraId: duplicate.existing_camera_id, existingLabel: duplicate.existing_label });
      } else {
        const probeFailure = cameraProbeFailureDetail(error);
        if (probeFailure) {
          setIssue({ kind: 'probe_failed', errorClass: probeFailure.error_class });
        } else {
          setMessage('카메라 등록에 실패했습니다. 잠시 후 다시 시도하고 서비스 연결 상태를 확인하세요.');
        }
      }
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }

  async function handleCreate(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    await submit(false);
  }

  async function forceRegisterCamera(): Promise<void> {
    await submit(true);
  }

  const submitLabel = busy ? '등록 중...' : issue?.kind === 'probe_failed' ? '다시 시도' : '카메라 등록';

  return (
    <AccessibleDialog open={open} title="카메라 추가" onClose={requestClose} initialFocusRef={nameInputRef}>
      <div className="flex items-start justify-between gap-4">
        <p className="text-sm font-bold text-brand">새 스트림</p>
        <button type="button" disabled={busy} onClick={requestClose} className="rounded-full bg-surface2 px-4 py-2 text-sm font-bold text-ink-soft hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-60">
          닫기
        </button>
      </div>

      <form className="mt-6 space-y-4" onSubmit={(event) => void handleCreate(event)} noValidate>
          <label className="block text-sm font-bold text-ink-soft">
            카메라 이름
            <input
              ref={nameInputRef}
              name="label"
              value={label}
              onChange={(event) => setLabel(event.target.value)}
              className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4"
              placeholder="예: 301호 침대 A"
            />
          </label>

          {duplicateLabelCamera ? (
            <p className="rounded-2xl bg-status-cautionBg px-4 py-3 text-xs font-bold text-status-caution" role="status">
              &ldquo;{duplicateLabelCamera.label}&rdquo; 카메라와 이름이 같습니다. 등록은 계속 진행할 수 있습니다.
            </p>
          ) : null}

          <label className="block text-sm font-bold text-ink-soft">
            RTSP 주소 붙여넣기
            <input
              name="rtspPaste"
              value={pasteInput}
              onChange={handlePasteChange}
              className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 font-mono text-xs text-ink outline-none ring-brand focus:ring-4"
              placeholder="rtsp://host:port/path?query"
            />
            {pasteError ? <span className="mt-2 block text-xs font-bold text-status-danger">{pasteError}</span> : null}
          </label>

          {fieldsCollapsed ? (
            <div className="flex items-center justify-between rounded-2xl bg-surface2 px-4 py-3">
              <p className="text-xs font-bold text-ink-soft">붙여넣은 주소에서 필드를 인식했습니다.</p>
              <button type="button" onClick={() => setFieldsCollapsed(false)} className="text-xs font-bold text-brand underline">
                직접 입력
              </button>
            </div>
          ) : (
            <>
              <div className="grid gap-3 sm:grid-cols-[120px_minmax(0,1fr)_120px]">
                <label className="block text-sm font-bold text-ink-soft">
                  방식
                  <select
                    name="rtspScheme"
                    value={rtspForm.scheme}
                    onChange={(event) => updateRtspField('scheme', event.target.value === 'rtsps' ? 'rtsps' : 'rtsp')}
                    className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4"
                  >
                    <option value="rtsp">rtsp</option>
                    <option value="rtsps">rtsps</option>
                  </select>
                </label>
                <label className="block text-sm font-bold text-ink-soft">
                  카메라 IP/호스트
                  <input
                    name="rtspHost"
                    value={rtspForm.host}
                    onChange={(event) => updateRtspField('host', event.target.value)}
                    className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4"
                    placeholder="192.0.2.10"
                  />
                </label>
                <label className="block text-sm font-bold text-ink-soft">
                  포트
                  <input
                    name="rtspPort"
                    inputMode="numeric"
                    value={rtspForm.port}
                    onChange={(event) => updateRtspField('port', event.target.value)}
                    className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4"
                  />
                </label>
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <label className="block text-sm font-bold text-ink-soft">
                  RTSP 아이디
                  <input name="rtspUsername" value={rtspForm.username} onChange={(event) => updateRtspField('username', event.target.value)} className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4" />
                </label>
                <label className="block text-sm font-bold text-ink-soft">
                  RTSP 비밀번호
                  <input name="rtspPassword" type="password" value={rtspForm.password} onChange={(event) => updateRtspField('password', event.target.value)} className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4" />
                </label>
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <label className="block text-sm font-bold text-ink-soft">
                  경로
                  <input name="rtspPath" value={rtspForm.path} onChange={(event) => updateRtspField('path', event.target.value)} className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4" placeholder="/trackID=1" />
                </label>
                <label className="block text-sm font-bold text-ink-soft">
                  추가 인자
                  <input name="rtspQuery" value={rtspForm.query} onChange={(event) => updateRtspField('query', event.target.value)} className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4" placeholder="profile=main" />
                </label>
              </div>
            </>
          )}

          <div className="rounded-2xl bg-surface2 px-4 py-3">
            <p className="text-xs font-black uppercase tracking-[0.18em] text-ink-faint">RTSP 미리보기</p>
            <p data-testid="rtsp-preview" className="mt-2 break-all font-mono text-xs text-ink-soft">{rtspPreview}</p>
          </div>

          <button type="submit" disabled={busy} className="brand-action rounded-lg px-6 py-3 text-sm font-black shadow-sm disabled:cursor-not-allowed disabled:opacity-60">
            {submitLabel}
          </button>
      </form>

      {message ? (
        <p className="mt-5 rounded-2xl bg-brand-soft px-4 py-3 text-sm font-bold text-brand" role="status">
          {message}
        </p>
      ) : null}

      {issue?.kind === 'duplicate' ? (
        <div className="mt-5 rounded-2xl bg-status-dangerBg px-4 py-3" role="alert">
          <p className="text-sm font-bold text-status-danger">이미 등록된 카메라입니다</p>
          <p className="mt-1 text-sm text-ink-soft">
            같은 스트림이 이미 <span className="font-bold text-status-danger">&ldquo;{issue.existingLabel}&rdquo;</span> 카메라로 등록되어 있습니다.
          </p>
        </div>
      ) : null}

      {issue?.kind === 'probe_failed' ? (
        <div className="mt-5 rounded-2xl bg-status-dangerBg px-4 py-3" role="alert">
          <p className="text-sm font-bold text-status-danger">등록되지 않았습니다</p>
          <p className="mt-1 text-sm text-ink-soft">{connectionFailureMessage(issue.errorClass)}</p>
          <button
            type="button"
            disabled={busy}
            onClick={() => void forceRegisterCamera()}
            className="mt-3 rounded-lg bg-surface2 px-5 py-3 text-xs font-bold text-ink-faint disabled:cursor-not-allowed disabled:opacity-60"
          >
            연결 없이 등록
          </button>
        </div>
      ) : null}
    </AccessibleDialog>
  );
}
