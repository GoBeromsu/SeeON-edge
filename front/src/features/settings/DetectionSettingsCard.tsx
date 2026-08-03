import { useEffect, useState } from 'react';
import { saveDetectionSettings } from '@/shared/api/client';
import {
  DOMAIN_LABELS,
  DOMAIN_ORDER,
  formatDomainSchedule,
  updateDomainMode,
  updateDomainOn,
  updateDomainTime,
} from '@/features/settings/detectionSettingsForm';
import { statusBadgeClassName } from '@/shared/ui/StatusBadge';
import { toast } from '@/shared/ui/Toast';
import type { DetectionDomainKey, DetectionMode, DetectionSettings } from '@/shared/api/client';
import type { PollingResource } from '@/shared/api/usePollingResource';

type DetectionSettingsCardProps = {
  resource: PollingResource<DetectionSettings>;
};

function PencilIcon(): JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round">
      <path d="M15.5 4.5 19.5 8.5 8 20H4v-4Z" />
      <path d="M13.5 6.5 17.5 10.5" />
    </svg>
  );
}

/**
 * 설정 페이지 "탐지 설정" 카드 — 낙상/침대 이탈 스케줄은 전역(모든 카메라 공통)이며, 편집 모드에서는
 * 저장 버튼 없이 변경 즉시 저장된다(front/design-handoff/README.md §5, Interactions & Behavior).
 */
export function DetectionSettingsCard({ resource }: DetectionSettingsCardProps): JSX.Element {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState<DetectionSettings | null>(resource.data);
  const [savingDomain, setSavingDomain] = useState<DetectionDomainKey | null>(null);

  useEffect(() => {
    if (!editing) setDraft(resource.data);
  }, [resource.data, editing]);

  async function persist(next: DetectionSettings, domain: DetectionDomainKey): Promise<void> {
    setDraft(next);
    setSavingDomain(domain);
    try {
      await saveDetectionSettings(next);
      resource.retry();
    } catch {
      toast.error('탐지 설정을 저장하지 못했습니다.');
      setDraft(resource.data);
    } finally {
      setSavingDomain(null);
    }
  }

  if (resource.status === 'loading' && !resource.data) {
    return (
      <article className="rounded-card border border-border bg-card p-5">
        <p className="text-sm text-muted-foreground">탐지 설정을 불러오는 중입니다...</p>
      </article>
    );
  }

  if (resource.status === 'error' && !resource.data) {
    return (
      <article className="rounded-card border border-border bg-card p-5">
        <h2 className="text-base font-semibold text-foreground">탐지 설정</h2>
        <p className="mt-2 text-sm text-destructive">탐지 설정을 불러오지 못했습니다.</p>
        <button type="button" className="dialog-secondary-action mt-2" onClick={() => resource.retry()}>
          다시 시도
        </button>
      </article>
    );
  }

  const settings = editing ? draft ?? resource.data : resource.data;
  if (!settings) return <article className="rounded-card border border-border bg-card p-5" />;

  return (
    <article className="rounded-card border border-border bg-card p-5">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-baseline gap-2">
          <h2 className="text-base font-semibold text-foreground">탐지 설정</h2>
          <span className="text-xs text-muted-foreground">모든 카메라에 적용</span>
        </div>
        {!editing ? (
          <button type="button" className="icon-button" aria-label="탐지 설정 편집" onClick={() => setEditing(true)}>
            <PencilIcon />
          </button>
        ) : null}
      </div>

      {!editing ? (
        <div className="mt-4 grid grid-cols-[1fr_auto_auto] items-center gap-x-3 gap-y-2 text-sm">
          {DOMAIN_ORDER.map((domain) => {
            const setting = settings.domains[domain];
            const badge = setting.on
              ? { label: '탐지 중', className: statusBadgeClassName('approved') }
              : { label: '미탐지', className: statusBadgeClassName('closed') };
            return (
              <div className="contents" key={domain}>
                <span className="text-foreground">{DOMAIN_LABELS[domain]}</span>
                <span className="text-right tabular-nums text-muted-foreground">{formatDomainSchedule(setting)}</span>
                <span className={badge.className}>{badge.label}</span>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="mt-4 space-y-4">
          {DOMAIN_ORDER.map((domain) => {
            const setting = settings.domains[domain];
            const busy = savingDomain === domain;
            return (
              <div key={domain} className="space-y-2 border-b border-border pb-3 last:border-0">
                <label className="flex items-center gap-2 text-sm font-semibold text-foreground">
                  <input
                    type="checkbox"
                    checked={setting.on}
                    disabled={busy}
                    onChange={(event) => void persist(updateDomainOn(settings, domain, event.target.checked), domain)}
                  />
                  {DOMAIN_LABELS[domain]}
                </label>
                <select
                  className="h-9 w-full rounded-control border border-input bg-background px-2 text-sm text-foreground"
                  value={setting.mode}
                  disabled={busy}
                  onChange={(event) => void persist(updateDomainMode(settings, domain, event.target.value as DetectionMode), domain)}
                >
                  <option value="always">항상</option>
                  <option value="window">시간 지정</option>
                </select>
                {setting.mode === 'window' ? (
                  <div className="flex items-center gap-2">
                    <input
                      type="time"
                      className="h-9 flex-1 rounded-control border border-input bg-background px-2 text-sm text-foreground"
                      value={setting.start ?? ''}
                      disabled={busy}
                      onChange={(event) => void persist(updateDomainTime(settings, domain, 'start', event.target.value), domain)}
                    />
                    <span className="text-muted-foreground">–</span>
                    <input
                      type="time"
                      className="h-9 flex-1 rounded-control border border-input bg-background px-2 text-sm text-foreground"
                      value={setting.end ?? ''}
                      disabled={busy}
                      onChange={(event) => void persist(updateDomainTime(settings, domain, 'end', event.target.value), domain)}
                    />
                  </div>
                ) : null}
              </div>
            );
          })}
          <p className="text-xs text-muted-foreground">변경 즉시 모든 카메라에 적용됩니다.</p>
          <div className="dialog-actions">
            <button type="button" className="dialog-secondary-action" onClick={() => setEditing(false)}>
              완료
            </button>
          </div>
        </div>
      )}
    </article>
  );
}
