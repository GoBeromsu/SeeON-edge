import { FormEvent, useEffect, useRef, useState } from 'react';
import { saveConnection, testConnection, type ConnectionTestResult, type ConnectionView } from '@/shared/api/client';
import {
  buildConnectionPayload,
  connectionFormFromView,
  updateConnectionToken,
  updateFacilityId,
  validateConnectionForm,
  type ConnectionFormState,
} from '@/features/connection/connectionSettingsForm';
import type { PollingResource } from '@/shared/api/usePollingResource';
import { toast } from '@/shared/ui/Toast';
import { getConnectionStatus } from '@/shared/ui/StatusBadge';

type ConnectionSettingsPanelProps = {
  resource: PollingResource<ConnectionView>;
};

function formatLastSyncAt(value: string | null | undefined): string {
  if (!value) return '동기화 이력 없음';
  const date = new Date(value);
  return Number.isFinite(date.getTime()) ? date.toLocaleString('ko-KR') : '동기화 이력 없음';
}

const GENERIC_TEST_FAILURE: ConnectionTestResult = {
  ok: false,
  error_class: null,
  detail: '연결 테스트 요청에 실패했습니다. 잠시 후 다시 시도하세요.',
  probed_url: null,
};

function PencilIcon(): JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round">
      <path d="M15.5 4.5 19.5 8.5 8 20H4v-4Z" />
      <path d="M13.5 6.5 17.5 10.5" />
    </svg>
  );
}

/** 설정 페이지의 "서버 연결" 카드: 기본은 읽기 전용 dl, 연필 버튼으로 편집 폼을 연다(front/design-handoff/README.md §5). */
export function ConnectionSettingsPanel({ resource }: ConnectionSettingsPanelProps): JSX.Element {
  const view = resource.data;
  const [editing, setEditing] = useState(false);
  const [form, setForm] = useState<ConnectionFormState>(() => connectionFormFromView(view));
  const [pendingAction, setPendingAction] = useState<'test' | 'save' | null>(null);
  const busyRef = useRef(false);
  const [validationMessage, setValidationMessage] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<ConnectionTestResult | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  // Only resyncs the draft from fresh poll data while the technician isn't mid-edit or mid-submit,
  // mirroring CameraCard's idle-draft-sync pattern.
  useEffect(() => {
    if (!editing && pendingAction === null) {
      setForm(connectionFormFromView(view));
    }
  }, [view, editing, pendingAction]);

  function openEdit(): void {
    setForm(connectionFormFromView(view));
    setValidationMessage(null);
    setTestResult(null);
    setSaveError(null);
    setEditing(true);
  }

  function closeEdit(): void {
    if (busyRef.current) return;
    setEditing(false);
    setForm(connectionFormFromView(view));
    setValidationMessage(null);
    setTestResult(null);
    setSaveError(null);
  }

  function handleFacilityIdChange(value: string): void {
    setForm((current) => updateFacilityId(current, value));
    setValidationMessage(null);
  }

  function handleTokenChange(value: string): void {
    setForm((current) => updateConnectionToken(current, value));
    setValidationMessage(null);
  }

  async function handleTest(): Promise<void> {
    if (busyRef.current) return;
    const error = validateConnectionForm(form);
    setValidationMessage(error);
    if (error) return;

    busyRef.current = true;
    setPendingAction('test');
    setSaveError(null);
    try {
      const result = await testConnection(buildConnectionPayload(form));
      setTestResult(result);
    } catch {
      setTestResult(GENERIC_TEST_FAILURE);
    } finally {
      busyRef.current = false;
      setPendingAction(null);
    }
  }

  async function handleSave(): Promise<void> {
    if (busyRef.current) return;
    const error = validateConnectionForm(form);
    setValidationMessage(error);
    if (error) return;

    busyRef.current = true;
    setPendingAction('save');
    setTestResult(null);
    setSaveError(null);
    try {
      await saveConnection(buildConnectionPayload(form));
      resource.retry();
      setEditing(false);
      toast.success('연결 설정을 저장했습니다.');
    } catch {
      setSaveError('연결 설정 저장에 실패했습니다. 입력값을 확인하고 다시 시도하세요.');
    } finally {
      busyRef.current = false;
      setPendingAction(null);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    void handleSave();
  }

  const busy = pendingAction !== null;
  const statusMeta = getConnectionStatus(view);
  const tokenPlaceholder = view?.facility_token_masked ?? '토큰 없음';

  return (
    <article className="rounded-card border border-border bg-card p-5">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold text-foreground">서버 연결</h2>
        <div className="flex items-center gap-2">
          <span aria-live="polite" className={statusMeta.className}>
            <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-current" />
            {statusMeta.label}
          </span>
          <button
            type="button"
            className="icon-button"
            aria-label={editing ? '서버 연결 편집 닫기' : '서버 연결 편집'}
            aria-pressed={editing}
            onClick={() => (editing ? closeEdit() : openEdit())}
          >
            <PencilIcon />
          </button>
        </div>
      </div>

      <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
        <dt className="text-muted-foreground">시설 ID</dt>
        <dd className="font-mono text-foreground">{view?.facility_id ?? '정보 없음'}</dd>
        <dt className="text-muted-foreground">시설 토큰</dt>
        <dd className="font-mono text-foreground">{view?.facility_token_masked ?? '토큰 없음'}</dd>
        <dt className="text-muted-foreground">마지막 동기화</dt>
        <dd className="tabular-nums text-foreground">{formatLastSyncAt(view?.last_ok_at)}</dd>
      </dl>

      {editing ? (
        <form className="mt-4 space-y-3 border-t border-border pt-4" onSubmit={handleSubmit} noValidate>
          <label>
            시설 ID
            <input
              name="facility_id"
              value={form.facility_id}
              disabled={busy}
              onChange={(event) => handleFacilityIdChange(event.target.value)}
              placeholder="예: facility-301"
            />
          </label>

          <label>
            시설 토큰
            <input
              name="facility_token"
              type="password"
              value={form.facility_token}
              disabled={busy}
              onChange={(event) => handleTokenChange(event.target.value)}
              placeholder={tokenPlaceholder}
              autoComplete="off"
            />
          </label>

          {validationMessage ? (
            <p role="alert" className="auth-error">{validationMessage}</p>
          ) : null}

          {testResult ? (
            <p
              role={testResult.ok ? 'status' : 'alert'}
              className={testResult.ok ? 'dialog-success' : 'auth-error'}
            >
              {testResult.ok ? `연결 성공 · ${testResult.detail}` : testResult.detail}
            </p>
          ) : null}

          {saveError ? (
            <p role="alert" className="auth-error">{saveError}</p>
          ) : null}

          <div className="dialog-actions">
            <button
              type="button"
              className="dialog-secondary-action"
              disabled={busy}
              onClick={() => void handleTest()}
            >
              {pendingAction === 'test' ? '확인 중...' : '연결'}
            </button>
            <button type="submit" className="brand-action inline-flex h-9 items-center justify-center rounded-control px-4 text-sm font-semibold" disabled={busy}>
              {pendingAction === 'save' ? '저장 중...' : '저장'}
            </button>
          </div>
        </form>
      ) : null}
    </article>
  );
}
