import { useEffect, useState } from 'react';
import { saveRuntimeSettings } from '@/shared/api/client';
import type { RuntimeSettings } from '@/shared/api/client';
import { HttpError } from '@/shared/api/http';
import type { PollingResource } from '@/shared/api/usePollingResource';

type Props = { resource: PollingResource<RuntimeSettings> };

function conflictCurrent(error: unknown): RuntimeSettings | null {
  if (!(error instanceof HttpError) || error.status !== 409 || typeof error.body !== 'object' || error.body === null) return null;
  const detail = 'detail' in error.body ? error.body.detail : null;
  if (typeof detail !== 'object' || detail === null || !('current' in detail)) return null;
  const current = detail.current;
  if (typeof current !== 'object' || current === null) return null;
  const enabled = 'clip_export_enabled' in current ? current.clip_export_enabled : null;
  const version = 'version' in current ? current.version : null;
  return typeof enabled === 'boolean' && typeof version === 'number' && Number.isInteger(version) && version >= 0
    ? { clip_export_enabled: enabled, version }
    : null;
}

export function ClipExportSettingsCard({ resource }: Props): JSX.Element {
  const [draftEnabled, setDraftEnabled] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    if (resource.data && !saving) setDraftEnabled(resource.data.clip_export_enabled);
  }, [resource.data, saving]);

  if (resource.status === 'loading' && !resource.data) {
    return (
      <article className="rounded-card border border-border bg-card p-5">
        <p className="text-sm text-muted-foreground">클립 내보내기 설정을 불러오는 중입니다...</p>
      </article>
    );
  }

  if (resource.status === 'error' && !resource.data) {
    return (
      <article className="rounded-card border border-border bg-card p-5">
        <h2 className="text-base font-semibold text-foreground">클립 내보내기</h2>
        <p className="mt-2 text-sm text-destructive">클립 내보내기 설정을 불러오지 못했습니다.</p>
        <button type="button" className="dialog-secondary-action mt-2" onClick={() => resource.retry()}>
          다시 시도
        </button>
      </article>
    );
  }

  const current = resource.data;
  const changed = current !== null && draftEnabled !== current.clip_export_enabled;

  async function save(): Promise<void> {
    if (!current || saving || !changed) return;
    setSaving(true);
    setSaveError(null);
    try {
      const saved = await saveRuntimeSettings({
        clip_export_enabled: draftEnabled,
        expected_version: current.version,
      });
      setDraftEnabled(saved.clip_export_enabled);
      resource.replace(saved);
      resource.retry();
    } catch (error: unknown) {
      const latest = conflictCurrent(error);
      if (latest) {
        setDraftEnabled(latest.clip_export_enabled);
        resource.replace(latest);
        resource.retry();
        setSaveError('다른 운영자가 먼저 변경했습니다. 최신 저장값을 반영했습니다.');
      } else {
        setSaveError('클립 내보내기 설정을 저장하지 못했습니다. 잠시 후 다시 시도하세요.');
      }
    } finally {
      setSaving(false);
    }
  }

  return (
    <article className="rounded-card border border-border bg-card p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-foreground">클립 내보내기</h2>
          <p className="mt-1 text-xs leading-5 text-muted-foreground">
            이벤트 알림은 항상 전송됩니다. 이 설정은 새 클립 전송만 제어합니다.
          </p>
        </div>
        <label className="flex shrink-0 items-center gap-2 text-sm font-medium text-foreground">
          <input
            type="checkbox"
            aria-label="클립 내보내기 사용"
            checked={draftEnabled}
            disabled={!current || saving}
            onChange={(event) => {
              setDraftEnabled(event.target.checked);
              setSaveError(null);
            }}
          />
          {draftEnabled ? 'ON' : 'OFF'}
        </label>
      </div>
      <p className="mt-3 text-xs text-muted-foreground">
        저장된 설정: {current?.clip_export_enabled ? 'ON' : 'OFF'} · 버전 {current?.version ?? 0}
      </p>
      {saveError ? <p role="alert" className="mt-3 text-sm text-destructive">{saveError}</p> : null}
      {resource.status === 'error' && current ? (
        <p role="status" className="mt-3 text-sm text-amber-700">
          저장 응답은 반영했지만 최신 상태를 다시 확인하지 못했습니다.
        </p>
      ) : null}
      <div className="mt-4 flex items-center justify-between gap-3">
        <span className="text-xs text-muted-foreground">설정 버전 {current?.version ?? 0}</span>
        <button
          type="button"
          className="brand-action inline-flex h-9 items-center justify-center rounded-control px-4 text-sm font-semibold"
          disabled={!changed || saving}
          onClick={() => void save()}
        >
          {saving ? '저장 중...' : '저장'}
        </button>
      </div>
    </article>
  );
}
