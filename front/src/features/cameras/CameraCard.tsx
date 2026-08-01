import { FormEvent, useEffect, useState } from 'react';
import { testCamera, updateCameraDecodeBackend, updateCamera, type Camera, type DecodeBackend } from '@/shared/api/client';
import { connectionFailureMessage } from '@/features/cameras/connectionTestMessage';
import { formatHeartbeatAge } from '@/features/cameras/heartbeatFreshness';
import { StatusBadge } from '@/shared/ui/StatusBadge';

const DECODE_BACKEND_OPTIONS: Array<{ value: DecodeBackend; label: string }> = [
  { value: 'auto', label: '자동 (GPU→CPU)' },
  { value: 'nvdec', label: 'GPU (NVDEC)' },
  { value: 'cpu', label: 'CPU' },
];

function toDecodeBackendValue(value: Camera['decode_backend']): DecodeBackend {
  if (value === 'nvdec' || value === 'cpu' || value === 'opencv') {
    return value === 'opencv' ? 'cpu' : value;
  }
  return 'auto';
}

// Native select arrows render inconsistently across browsers; replace with a fixed chevron so the
// control reads consistently while keeping the change scoped to this component (no shared CSS edits).
const SELECT_CHEVRON_STYLE = {
  backgroundImage:
    "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='none' stroke='%23595959' stroke-width='1.6'%3E%3Cpath d='M5.5 8L10 12.5L14.5 8' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E\")",
  backgroundRepeat: 'no-repeat',
  backgroundPosition: 'right 0.75rem center',
  backgroundSize: '14px',
} as const;

type CameraCardProps = {
  camera: Camera;
  onUpdateStarted?: (camera: Camera) => number;
  onUpdated?: (camera: Camera, previousCameraId: string, startedAtGeneration?: number) => void;
  onDelete?: (camera: Camera) => void;
  onViewClips?: (camera: Camera) => void;
  onRetried?: () => void;
};

export function CameraCard({ camera, onUpdateStarted, onUpdated, onDelete, onViewClips, onRetried }: CameraCardProps): JSX.Element {
  const mapped = Boolean(camera.space_id || camera.backend_camera_id);
  const neverConnected = camera.never_connected === true;
  const isFailed = camera.status === 'offline' || (neverConnected && camera.status !== 'online');
  const [editing, setEditing] = useState(false);
  const [label, setLabel] = useState(camera.label);
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [decodeBackend, setDecodeBackend] = useState<DecodeBackend>(toDecodeBackendValue(camera.decode_backend));
  const [decodeBusy, setDecodeBusy] = useState(false);
  const [retryBusy, setRetryBusy] = useState(false);

  function notifyUpdated(updated: Camera, startedAtGeneration: number | undefined): void {
    if (startedAtGeneration === undefined) {
      onUpdated?.(updated, camera.id);
    } else {
      onUpdated?.(updated, camera.id, startedAtGeneration);
    }
  }

  useEffect(() => {
    if (!editing && !busy) {
      setLabel(camera.label);
    }
  }, [busy, camera.label, editing]);

  useEffect(() => {
    if (!decodeBusy) {
      setDecodeBackend(toDecodeBackendValue(camera.decode_backend));
    }
  }, [camera.decode_backend, decodeBusy]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setMessage(null);
    if (!label.trim()) {
      setMessage('카메라 이름을 입력하세요.');
      return;
    }
    setBusy(true);
    const startedAtGeneration = onUpdateStarted?.(camera);
    try {
      const updated = await updateCamera(camera.id, {
        label: label.trim(),
      });
      notifyUpdated(updated, startedAtGeneration);
      setEditing(false);
      setMessage('카메라 정보를 수정했습니다.');
    } catch {
      setMessage('카메라 수정에 실패했습니다.');
    } finally {
      setBusy(false);
    }
  }

  async function handleDecodeBackendChange(value: DecodeBackend): Promise<void> {
    setDecodeBackend(value);
    setMessage(null);
    setDecodeBusy(true);
    const startedAtGeneration = onUpdateStarted?.(camera);
    try {
      const updated = await updateCameraDecodeBackend(camera.id, value);
      notifyUpdated(updated, startedAtGeneration);
      setMessage('디코딩 백엔드를 변경했습니다.');
    } catch {
      setDecodeBackend(toDecodeBackendValue(camera.decode_backend));
      setMessage('디코딩 백엔드 변경에 실패했습니다.');
    } finally {
      setDecodeBusy(false);
    }
  }

  function handleCancel(): void {
    if (busy) return;
    setLabel(camera.label);
    setEditing(false);
  }

  async function handleRetry(): Promise<void> {
    if (retryBusy) return;
    setMessage(null);
    setRetryBusy(true);
    try {
      const result = await testCamera(camera.id);
      const size = result.width && result.height ? ` · ${result.width}×${result.height}` : '';
      setMessage(result.ok ? `재연결 성공${size}` : `재연결 실패 · ${connectionFailureMessage(result.error_class)}`);
      onRetried?.();
    } catch {
      setMessage('재연결 실패');
    } finally {
      setRetryBusy(false);
    }
  }

  const connectionDetail = camera.status === 'online'
    ? null
    : neverConnected
      ? '한 번도 연결된 적 없음'
      : camera.last_ok_at
        ? `마지막 연결 ${new Date(camera.last_ok_at).toLocaleString('ko-KR')}`
        : '연결 이력 없음';

  // Edge heartbeat freshness is a separate, more real-time signal than last_ok_at (which only
  // moves on a successful probe). Surface it only where it aids triage: an offline/degraded card
  // benefits from "how long has this actually been dead" at a glance. An online card already
  // reads as alive from the badge, so repeating a heartbeat age there is noise, not signal —
  // matches the console's "normal is silent" principle used elsewhere (System page). Never shown
  // for never_connected, which has its own distinct, non-numeric copy.
  const heartbeatDetail = camera.status !== 'online' && !neverConnected
    ? formatHeartbeatAge(camera.heartbeat_age_sec)
    : null;

  return (
    <article className="rounded-2xl border border-border bg-surface p-5 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-bold tracking-[0.24em] text-brand">카메라</p>
          <h3 className="mt-2 text-xl font-black text-ink">{camera.label}</h3>
        </div>
        <div className="text-right">
          <StatusBadge status={camera.status} />
          {connectionDetail ? <p className="mt-1.5 text-xs font-semibold text-ink-faint">{connectionDetail}</p> : null}
          {heartbeatDetail ? <p className="mt-1 text-xs font-semibold text-ink-faint">{heartbeatDetail}</p> : null}
        </div>
      </div>

      <dl className="mt-4 space-y-2 text-sm">
        <div className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 rounded-2xl bg-surface2 px-3 py-2">
          <dt className="font-semibold text-ink-faint">위치</dt>
          <dd className="text-right font-bold text-ink-soft">{[camera.floor_name, camera.space_name ?? camera.space_id].filter(Boolean).join(' · ') || '미지정'}</dd>
        </div>
      </dl>
      <div className="mt-3 rounded-2xl bg-surface2 px-3 py-3 text-sm">
        <p className="text-xs font-bold text-ink-faint">기술 설정</p>
        <dl className="mt-2 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-2">
          <dt className="font-semibold text-ink-faint">등록 ID</dt>
          <dd className="break-all text-right font-mono text-xs text-ink-soft">{camera.id}</dd>
          <dt className="font-semibold text-ink-faint">관리 방식</dt>
          <dd className={mapped ? 'text-right font-bold text-brand' : 'text-right font-bold text-ink-faint'}>{mapped ? '서버 자동 관리' : '로컬 등록'}</dd>
        </dl>
        <div className="mt-3">
          <label className="sr-only" htmlFor={`decode-backend-${camera.id}`}>디코딩 백엔드</label>
          <select
            id={`decode-backend-${camera.id}`}
            value={decodeBackend}
            disabled={decodeBusy}
            onChange={(event) => void handleDecodeBackendChange(event.target.value as DecodeBackend)}
            style={SELECT_CHEVRON_STYLE}
            className="min-h-11 w-full appearance-none rounded-lg border border-border bg-surface px-3 py-2 pr-9 text-sm font-bold text-ink outline-none ring-brand focus:ring-4 disabled:opacity-60"
          >
            {DECODE_BACKEND_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </div>
      </div>

      {editing ? (
        <form className="mt-5 space-y-3 rounded-xl bg-surface2 p-4" onSubmit={(event) => void handleSubmit(event)} noValidate>
          <label className="block text-sm font-bold text-ink-soft">
            이름
            <input name="label" value={label} onChange={(event) => setLabel(event.target.value)} className="mt-2 w-full rounded-2xl border border-border bg-surface px-4 py-3 text-ink outline-none ring-brand focus:ring-4" />
          </label>
          <div className="flex flex-wrap gap-2">
            <button type="submit" disabled={busy} className="brand-action rounded-lg px-4 py-2 text-xs font-black disabled:opacity-60">{busy ? '저장 중...' : '저장'}</button>
            <button type="button" disabled={busy} onClick={handleCancel} className="rounded-full bg-surface px-4 py-2 text-xs font-black text-ink-soft shadow-sm disabled:opacity-60">취소</button>
          </div>
        </form>
      ) : null}

      {message ? <p className="mt-4 text-xs font-bold text-brand" role="status">{message}</p> : null}

      <div className="mt-5 flex flex-wrap justify-between gap-2">
        <p className="text-xs text-ink-faint">등록 {camera.created_at ? new Date(camera.created_at).toLocaleString('ko-KR') : '정보 없음'}</p>
        <div className="flex flex-wrap gap-2">
          {onViewClips ? (
            <button type="button" onClick={() => onViewClips(camera)} className="rounded-full bg-surface2 px-3 py-2 text-xs font-black text-ink-soft hover:bg-surface2">
              클립 보기
            </button>
          ) : null}
          {isFailed ? (
            <button type="button" disabled={retryBusy} onClick={() => void handleRetry()} className="rounded-full bg-status-cautionBg px-3 py-2 text-xs font-black text-status-caution disabled:opacity-60">
              {retryBusy ? '재시도 중...' : '재시도'}
            </button>
          ) : null}
          <button type="button" disabled={busy} onClick={() => setEditing((value) => !value)} className="rounded-full bg-brand-soft px-3 py-2 text-xs font-black text-brand hover:bg-brand-soft disabled:opacity-60">
            수정
          </button>
          <button type="button" disabled={busy || decodeBusy || retryBusy} onClick={() => onDelete?.(camera)} className="rounded-full bg-status-dangerBg px-3 py-2 text-xs font-black text-status-danger hover:bg-status-dangerBg disabled:opacity-60">
            삭제
          </button>
        </div>
      </div>
    </article>
  );
}
