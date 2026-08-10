import { FormEvent, useEffect, useRef, useState } from 'react';
import { saveConnection, testConnection, type ConnectionTestResult, type ConnectionView } from '@/shared/api/client';
import { HttpError } from '@/shared/api/http';
import {
  buildConnectionPayload,
  connectionFormFromView,
  updateConnectionToken,
  updateFacilityCode,
  validateConnectionForm,
  type ConnectionFormState,
} from '@/features/connection/connectionSettingsForm';
import type { PollingResource } from '@/shared/api/usePollingResource';
import { toast } from '@/shared/ui/Toast';
import { getConnectionStatus } from '@/shared/ui/StatusBadge';

type Props = { readonly resource: PollingResource<ConnectionView> };
type PendingAction = 'test' | 'save' | null;

function backendAddress(value: string | null | undefined): string {
  if (!value) return '배포 설정 없음';
  try {
    return new URL(value).origin;
  } catch {
    return '배포 설정 확인 필요';
  }
}

function formatTime(value: string | null | undefined): string {
  if (!value) return '동기화 이력 없음';
  const date = new Date(value);
  return Number.isFinite(date.getTime()) ? date.toLocaleString('ko-KR') : '동기화 이력 없음';
}

function saveFailure(error: unknown): string {
  if (!(error instanceof HttpError)) return '등록 서버에 연결하지 못했습니다. 잠시 후 다시 시도하세요.';
  if (error.status === 401) return '시설 코드 또는 토큰을 확인하세요.';
  if (error.status === 403) return '이 시설에 등록할 권한이 없습니다.';
  if (error.status === 409) return '이 설치는 다른 장치에 연결되어 있습니다. 관리자에게 재연결을 요청하세요.';
  return '등록 정보를 저장하지 못했습니다. 잠시 후 다시 시도하세요.';
}

function relayLabel(view: ConnectionView | null | undefined): string {
  const relay = view?.heartbeat_relay;
  if (!relay) return '정보 없음';
  if (!relay.enabled) return '꺼짐';
  if (relay.last_error_class === 'auth') return '인증 확인 필요';
  if (relay.last_error_class === 'timeout' || relay.last_error_class === 'unreachable') return '네트워크 확인 필요';
  return relay.last_success_at ? `정상 · ${formatTime(relay.last_success_at)}` : '전송 기록 없음';
}

function EditIcon(): JSX.Element {
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round"><path d="M15.5 4.5 19.5 8.5 8 20H4v-4Z" /><path d="M13.5 6.5 17.5 10.5" /></svg>;
}

export function ConnectionSettingsPanel({ resource }: Props): JSX.Element {
  const view = resource.data;
  const generatedRef = useRef(crypto.randomUUID());
  const busyRef = useRef(false);
  const [editing, setEditing] = useState(false);
  const [form, setForm] = useState<ConnectionFormState>(() => connectionFormFromView(view ?? null, generatedRef.current));
  const [pending, setPending] = useState<PendingAction>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<ConnectionTestResult | null>(null);

  useEffect(() => {
    if (!editing && pending === null) setForm(connectionFormFromView(view ?? null, generatedRef.current));
  }, [editing, pending, view]);

  function resetFromView(next: ConnectionView | null | undefined = view): void {
    setForm(connectionFormFromView(next ?? null, generatedRef.current));
    setMessage(null);
    setTestResult(null);
  }

  function toggleEdit(): void {
    if (busyRef.current) return;
    if (editing) {
      setEditing(false);
      resetFromView();
      return;
    }
    resetFromView();
    setEditing(true);
  }

  async function run(action: Exclude<PendingAction, null>): Promise<void> {
    if (busyRef.current) return;
    const validation = validateConnectionForm(form);
    setMessage(validation);
    if (validation) return;
    busyRef.current = true;
    setPending(action);
    setTestResult(null);
    try {
      if (action === 'test') {
        setTestResult(await testConnection(buildConnectionPayload(form)));
      } else {
        const saved = await saveConnection(buildConnectionPayload(form));
        setForm(connectionFormFromView(saved, generatedRef.current));
        setEditing(false);
        resource.retry();
        toast.success(saved.enrollment_generation && saved.enrollment_generation > 1 ? '장치 재연결을 저장했습니다.' : '장치 등록을 저장했습니다.');
      }
    } catch (error) {
      setMessage(saveFailure(error));
    } finally {
      busyRef.current = false;
      setPending(null);
    }
  }

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    void run('save');
  }

  const status = getConnectionStatus(view);
  const busy = pending !== null;

  return (
    <article className="rounded-card border border-border bg-card p-5">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-foreground">장치 등록</h2>
          <p className="mt-1 text-sm text-muted-foreground">현장에서는 시설 코드와 토큰만 입력합니다.</p>
        </div>
        <div className="flex items-center gap-2">
          <span aria-live="polite" className={status.className}><span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-current" />{status.label}</span>
          <button type="button" className="icon-button" aria-label={editing ? '등록 정보 변경 닫기' : '등록 정보 변경'} aria-pressed={editing} onClick={toggleEdit}><EditIcon /></button>
        </div>
      </div>

      <dl className="mt-4 grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-2 text-sm">
        <dt className="text-muted-foreground">백엔드</dt><dd className="truncate font-mono">{backendAddress(view?.events_url)}</dd>
        <dt className="text-muted-foreground">시설 코드</dt><dd className="font-mono">{view?.facility_code ?? '등록 전'}</dd>
        <dt className="text-muted-foreground">Canonical 시설</dt><dd className="break-words font-mono">{view?.facility_id ?? '정보 없음'}</dd>
        <dt className="text-muted-foreground">설치 세대</dt><dd className="tabular-nums">{view?.enrollment_generation ? `${view.enrollment_generation}세대` : '등록 전'}</dd>
        <dt className="text-muted-foreground">시설 토큰</dt><dd className="font-mono">{view?.facility_token_masked ?? '토큰 없음'}</dd>
        <dt className="text-muted-foreground">마지막 동기화</dt><dd className="tabular-nums">{formatTime(view?.last_ok_at)}</dd>
        <dt className="text-muted-foreground">클라우드 전송</dt><dd>{relayLabel(view)}</dd>
      </dl>

      {editing ? (
        <form className="mt-4 space-y-3 border-t border-border pt-4" onSubmit={submit} noValidate>
          <label>시설 코드<input name="facility_code" value={form.facility_code} disabled={busy} autoComplete="off" spellCheck={false} placeholder="NH-7H2K9M4QXP" onChange={(event) => { setForm((current) => updateFacilityCode(current, event.target.value)); setMessage(null); }} /></label>
          <label>시설 토큰<input name="facility_token" type="password" value={form.facility_token} disabled={busy} autoComplete="new-password" placeholder={view?.facility_token_masked ?? '발급된 토큰'} onChange={(event) => { setForm((current) => updateConnectionToken(current, event.target.value)); setMessage(null); }} /></label>
          {message ? <p role="alert" className="auth-error">{message}</p> : null}
          {testResult ? <p role={testResult.ok ? 'status' : 'alert'} className={testResult.ok ? 'dialog-success' : 'auth-error'}>{testResult.detail}{testResult.ok && testResult.enrollment_generation ? ` · ${testResult.enrollment_generation}세대` : ''}</p> : null}
          <div className="dialog-actions">
            <button type="button" className="dialog-secondary-action" disabled={busy} onClick={() => void run('test')}>{pending === 'test' ? '확인 중...' : '등록 확인'}</button>
            <button type="submit" className="brand-action inline-flex h-9 items-center justify-center rounded-control px-4 text-sm font-semibold" disabled={busy}>{pending === 'save' ? '저장 중...' : '등록 저장'}</button>
          </div>
        </form>
      ) : null}
    </article>
  );
}
