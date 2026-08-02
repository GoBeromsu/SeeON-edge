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
import type { HeartbeatRelayStatus } from '@/shared/api/types';
import { getConnectionStatus } from '@/shared/ui/StatusBadge';

type ConnectionSettingsPanelProps = {
  resource: PollingResource<ConnectionView>;
};

function formatLastOkAt(value: string | null | undefined): string {
  if (!value) return '연결 이력 없음';
  const date = new Date(value);
  return Number.isFinite(date.getTime()) ? `마지막 연결 확인 ${date.toLocaleString('ko-KR')}` : '연결 이력 없음';
}

const DEFAULT_HEARTBEAT_RELAY: HeartbeatRelayStatus = {
  enabled: false,
  last_success_at: null,
  last_error_class: null,
  detail: null,
};

function formatRelayLastSuccessAt(value: string | null): string {
  if (!value) return '성공 이력 없음';
  const date = new Date(value);
  return Number.isFinite(date.getTime()) ? date.toLocaleString('ko-KR') : '성공 이력 없음';
}

/** Mirrors getConnectionStatus's {enabled, error} -> {label, className} shape, sourced from the relay sub-status instead. */
function heartbeatRelayStatusLine(relay: HeartbeatRelayStatus): { text: string; className: string } {
  if (!relay.enabled) {
    return { text: '하트비트 릴레이 비활성', className: 'text-ink-faint' };
  }
  if (relay.last_error_class) {
    return { text: relay.detail ?? '하트비트 릴레이 오류가 발생했습니다.', className: 'text-status-danger' };
  }
  return { text: `하트비트 릴레이 정상 · ${formatRelayLastSuccessAt(relay.last_success_at)}`, className: 'text-status-stable' };
}

// events_url is a packaging-time constant shaped like "<base>/v1/events"; strip that suffix so the
// technician sees the backend host, not an internal route. An unrecognized shape is shown verbatim
// rather than mangled -- better an odd-looking address than a silently wrong one.
const EVENTS_URL_SUFFIX = '/v1/events';

function connectionAddressMeta(eventsUrl: string | null | undefined): { text: string; className: string } {
  if (!eventsUrl) {
    return { text: '미설정', className: 'text-ink-faint' };
  }
  const derived = eventsUrl.endsWith(EVENTS_URL_SUFFIX) ? eventsUrl.slice(0, -EVENTS_URL_SUFFIX.length) : eventsUrl;
  return { text: derived, className: 'font-mono text-ink-soft' };
}

const GENERIC_TEST_FAILURE: ConnectionTestResult = {
  ok: false,
  error_class: null,
  detail: '연결 테스트 요청에 실패했습니다. 잠시 후 다시 시도하세요.',
  probed_url: null,
};

export function ConnectionSettingsPanel({ resource }: ConnectionSettingsPanelProps): JSX.Element {
  const view = resource.data;
  const [form, setForm] = useState<ConnectionFormState>(() => connectionFormFromView(view));
  const [dirty, setDirty] = useState(false);
  const [pendingAction, setPendingAction] = useState<'test' | 'save' | null>(null);
  const busyRef = useRef(false);
  const [validationMessage, setValidationMessage] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<ConnectionTestResult | null>(null);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  // Never clobbers an in-progress edit: only resyncs the draft from fresh poll data while the
  // technician isn't mid-edit or mid-submit, mirroring CameraCard's idle-draft-sync pattern.
  useEffect(() => {
    if (!dirty && pendingAction === null) {
      setForm(connectionFormFromView(view));
    }
  }, [view, dirty, pendingAction]);

  function handleFacilityIdChange(value: string): void {
    setForm((current) => updateFacilityId(current, value));
    setDirty(true);
    setValidationMessage(null);
  }

  function handleTokenChange(value: string): void {
    setForm((current) => updateConnectionToken(current, value));
    setDirty(true);
    setValidationMessage(null);
  }

  async function handleTest(): Promise<void> {
    if (busyRef.current) return;
    const error = validateConnectionForm(form);
    setValidationMessage(error);
    if (error) return;

    busyRef.current = true;
    setPendingAction('test');
    setSaveMessage(null);
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
    try {
      await saveConnection(buildConnectionPayload(form));
      setDirty(false);
      setSaveMessage('연결 설정을 저장했습니다.');
      resource.retry();
    } catch {
      setSaveMessage('연결 설정 저장에 실패했습니다. 입력값을 확인하고 다시 시도하세요.');
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
  const relayStatus = heartbeatRelayStatusLine(view?.heartbeat_relay ?? DEFAULT_HEARTBEAT_RELAY);
  const addressMeta = connectionAddressMeta(view?.events_url);
  const tokenPlaceholder = view?.facility_token_masked ?? '토큰 없음';

  return (
    <article className="rounded-2xl border border-border bg-surface p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs font-bold tracking-[0.24em] text-brand">외부 백엔드 연결</p>
          <h2 className="mt-2 text-xl font-black text-ink">연결 설정</h2>
        </div>
        <span
          aria-live="polite"
          className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-bold ring-1 ${statusMeta.className}`}
        >
          <span className="mr-1.5 h-2 w-2 rounded-full bg-current" />
          {statusMeta.label}
        </span>
      </div>

      <p className={`mt-2 text-xs font-semibold ${addressMeta.className}`}>연결 대상: {addressMeta.text}</p>
      <p className="mt-1 text-xs font-semibold text-ink-faint">{formatLastOkAt(view?.last_ok_at)}</p>
      <p aria-live="polite" className={`mt-1 text-xs font-semibold ${relayStatus.className}`}>{relayStatus.text}</p>

      <form className="mt-5 space-y-4" onSubmit={handleSubmit} noValidate>
        <label className="block text-sm font-bold text-ink-soft">
          시설 ID
          <input
            name="facility_id"
            value={form.facility_id}
            onChange={(event) => handleFacilityIdChange(event.target.value)}
            className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4"
            placeholder="예: facility-301"
          />
        </label>

        <label className="block text-sm font-bold text-ink-soft">
          시설 토큰
          <input
            name="facility_token"
            type="password"
            value={form.facility_token}
            onChange={(event) => handleTokenChange(event.target.value)}
            className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4"
            placeholder={tokenPlaceholder}
            autoComplete="off"
          />
        </label>

        {validationMessage ? (
          <p role="alert" className="text-xs font-bold text-status-danger">{validationMessage}</p>
        ) : null}

        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            disabled={busy}
            onClick={() => void handleTest()}
            className="rounded-lg border border-border bg-surface px-5 py-3 text-sm font-black text-brand disabled:cursor-not-allowed disabled:opacity-60"
          >
            {pendingAction === 'test' ? '확인 중...' : '연결 테스트'}
          </button>
          <button
            type="submit"
            disabled={busy}
            className="brand-action rounded-lg px-5 py-3 text-sm font-black disabled:cursor-not-allowed disabled:opacity-60"
          >
            {pendingAction === 'save' ? '저장 중...' : '저장'}
          </button>
        </div>
      </form>

      {testResult ? (
        <p
          role={testResult.ok ? 'status' : 'alert'}
          className={`mt-4 rounded-xl px-4 py-3 text-sm font-bold ${
            testResult.ok ? 'bg-status-stableBg text-status-stable' : 'bg-status-dangerBg text-status-danger'
          }`}
        >
          {testResult.detail}
        </p>
      ) : null}

      {saveMessage ? (
        <p role="status" className="mt-4 rounded-xl bg-brand-soft px-4 py-3 text-sm font-bold text-brand">
          {saveMessage}
        </p>
      ) : null}
    </article>
  );
}
