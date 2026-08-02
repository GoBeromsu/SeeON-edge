import { useState } from 'react';
import { saveClipStorageLocation } from '@/shared/api/client';
import { FolderBrowserModal } from '@/features/settings/FolderBrowserModal';
import { toast } from '@/shared/ui/Toast';
import type { ClipStorageInfo } from '@/shared/api/client';
import type { PollingResource } from '@/shared/api/usePollingResource';

type ClipStorageCardProps = {
  resource: PollingResource<ClipStorageInfo>;
};

const BYTES_PER_GB = 1024 ** 3;

function gbLabel(bytes: number | null): string {
  return bytes === null ? '—' : (bytes / BYTES_PER_GB).toFixed(1);
}

function usagePct(info: ClipStorageInfo): number {
  if (info.used_pct !== null) return Math.min(100, Math.max(0, info.used_pct));
  if (info.total_bytes && info.used_bytes !== null) {
    return Math.min(100, Math.max(0, (info.used_bytes / info.total_bytes) * 100));
  }
  return 0;
}

/** Joins the CLIP_STORE_DIR mount root with the selected relative subdirectory ("" selects the mount root itself). */
function displayPath(info: ClipStorageInfo): string {
  return info.selected_path ? `${info.root}/${info.selected_path}` : info.root;
}

/** 설정 페이지 "클립 저장 공간" 카드 (front/design-handoff/README.md §5). */
export function ClipStorageCard({ resource }: ClipStorageCardProps): JSX.Element {
  const [browserOpen, setBrowserOpen] = useState(false);
  const [saving, setSaving] = useState(false);

  if (resource.status === 'loading' && !resource.data) {
    return (
      <article className="rounded-card border border-border bg-card p-5">
        <p className="text-sm text-muted-foreground">클립 저장 공간 정보를 불러오는 중입니다...</p>
      </article>
    );
  }

  if (resource.status === 'error' && !resource.data) {
    return (
      <article className="rounded-card border border-border bg-card p-5">
        <h2 className="text-base font-semibold text-foreground">클립 저장 공간</h2>
        <p className="mt-2 text-sm text-destructive">클립 저장 공간 정보를 불러오지 못했습니다.</p>
        <button type="button" className="dialog-secondary-action mt-2" onClick={() => resource.retry()}>
          다시 시도
        </button>
      </article>
    );
  }

  const info = resource.data;
  const pct = info ? usagePct(info) : 0;

  const handleSelect = async (path: string): Promise<void> => {
    setSaving(true);
    try {
      await saveClipStorageLocation(path);
      resource.retry();
      toast.success('저장 위치를 변경했습니다.');
      setBrowserOpen(false);
    } catch {
      toast.error('저장 위치를 변경하지 못했습니다.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <article className="rounded-card border border-border bg-card p-5">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold text-foreground">클립 저장 공간</h2>
        <span className="tabular-nums text-sm text-foreground">
          {info ? `${gbLabel(info.used_bytes)} / ${gbLabel(info.total_bytes)} GB` : '—'}
        </span>
      </div>

      <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-muted" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}>
        <div className="h-full rounded-full bg-primary" style={{ width: `${pct}%` }} />
      </div>

      <div className="mt-4 flex items-center justify-between gap-3">
        <p className="min-w-0 truncate font-mono text-xs text-muted-foreground">
          저장 위치 {info ? displayPath(info) : '확인 중'}
        </p>
        <button
          type="button"
          className="shrink-0 rounded-control px-2 py-1 text-xs font-medium text-muted-foreground hover:bg-muted hover:text-foreground"
          disabled={!info || saving}
          onClick={() => setBrowserOpen(true)}
        >
          변경
        </button>
      </div>

      <FolderBrowserModal
        open={browserOpen}
        initialPath={info?.selected_path ?? ''}
        onClose={() => setBrowserOpen(false)}
        onSelect={(path) => void handleSelect(path)}
      />
    </article>
  );
}
