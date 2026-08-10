import { useEffect, useState } from 'react';
import type { CameraRegistry } from '@/shared/api/client';
import {
  fetchTopology,
  fetchTopologyPreview,
  syncTopology,
  type CameraTopology,
  type TopologyPreview,
  type TopologySyncResult,
} from '@/shared/api/topologyClient';
import type { PollingResource } from '@/shared/api/usePollingResource';
import { CameraPairingList } from '@/features/cameras/CameraPairingList';
import { TopologyConfirmationDialog } from '@/features/cameras/TopologyConfirmationDialog';
import { TopologyStructureEditor } from '@/features/cameras/TopologyStructureEditor';
import { statusBadgeClassName } from '@/shared/ui/StatusBadge';

type Props = { readonly camerasResource: PollingResource<CameraRegistry> };

function readinessLabel(topology: CameraTopology): string {
  if (topology.readiness_error === 'LEGACY_MAPPING_REQUIRED') return '매핑 필요';
  if (topology.readiness_error) return '구성 확인 필요';
  if (topology.dirty_registry_version !== null) return '동기화 대기';
  return '준비 완료';
}

function syncMessage(result: TopologySyncResult): string {
  if (result.status === 'synced') return '최신 토폴로지가 서버와 동기화되었습니다.';
  if (result.error_class === 'auth') return '등록 인증을 확인한 뒤 다시 시도하세요.';
  if (result.error_class === 'conflict') return '서버 리비전 충돌이 발생했습니다. 상태를 새로 확인하세요.';
  if (result.error_class === 'timeout' || result.error_class === 'unreachable') return '서버 연결을 확인한 뒤 다시 시도하세요.';
  if (result.error_class === 'unconfigured') return '장치 등록을 먼저 완료하세요.';
  return result.detail ?? '동기화 대기 중입니다.';
}

function StatusDot({ topology }: { readonly topology: CameraTopology }): JSX.Element {
  const ready = topology.readiness_error === null && topology.dirty_registry_version === null;
  return <span className={statusBadgeClassName(ready ? 'approved' : 'pending')}><span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-current" />{readinessLabel(topology)}</span>;
}

export function TopologyEditor({ camerasResource }: Props): JSX.Element {
  const [topology, setTopology] = useState<CameraTopology | null>(null);
  const [preview, setPreview] = useState<TopologyPreview | null>(null);
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [messageKind, setMessageKind] = useState<'status' | 'alert'>('status');

  async function reload(): Promise<void> {
    const [nextTopology, nextPreview] = await Promise.all([fetchTopology(), fetchTopologyPreview()]);
    setTopology(nextTopology);
    setPreview(nextPreview.preview);
  }

  useEffect(() => {
    let active = true;
    void Promise.all([fetchTopology(), fetchTopologyPreview()])
      .then(([nextTopology, nextPreview]) => {
        if (!active) return;
        setTopology(nextTopology);
        setPreview(nextPreview.preview);
      })
      .catch(() => {
        if (active) {
          setMessageKind('alert');
          setMessage('토폴로지 상태를 불러오지 못했습니다.');
        }
      })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, []);

  async function manualSync(): Promise<void> {
    if (syncing) return;
    setSyncing(true);
    setMessage(null);
    try {
      const result = await syncTopology();
      setMessageKind(result.status === 'failed' ? 'alert' : 'status');
      setMessage(syncMessage(result));
      await reload();
    } catch {
      setMessageKind('alert');
      setMessage('동기화 요청에 실패했습니다. 잠시 후 다시 시도하세요.');
    } finally {
      setSyncing(false);
    }
  }

  if (loading) return <article className="rounded-card border border-border bg-card p-5" aria-busy="true"><h2 className="text-base font-semibold">시설 토폴로지</h2><p className="mt-3 text-sm text-muted-foreground">토폴로지 상태를 확인하고 있습니다.</p></article>;
  if (!topology) return <article className="rounded-card border border-border bg-card p-5"><h2 className="text-base font-semibold">시설 토폴로지</h2><p role="alert" className="auth-error mt-3">{message}</p><button type="button" className="dialog-secondary-action mt-3" onClick={() => { setLoading(true); void reload().finally(() => setLoading(false)); }}>다시 시도</button></article>;

  return (
    <article className="rounded-card border border-border bg-card p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div><h2 className="text-base font-semibold">시설 토폴로지</h2><p className="mt-1 text-sm text-muted-foreground">층과 방을 만든 뒤 기존 카메라를 명시적으로 배정합니다.</p></div>
        <StatusDot topology={topology} />
      </div>

      <div className="mt-4 flex flex-wrap gap-2 text-xs tabular-nums">
        <span className="rounded-control bg-muted px-2 py-1">현재 {topology.registry_version}</span>
        <span className="rounded-control bg-muted px-2 py-1">{topology.dirty_registry_version === null ? '동기화 완료' : `동기화 대기 ${topology.dirty_registry_version}`}</span>
        {preview ? <><span className="rounded-control bg-muted px-2 py-1">클라이언트 {preview.client_revision}</span><span className="rounded-control bg-muted px-2 py-1">서버 {preview.server_revision}</span></> : null}
      </div>

      <section className="mt-5" aria-labelledby="structure-title"><h3 id="structure-title" className="mb-3 text-sm font-semibold">층과 방</h3><TopologyStructureEditor topology={topology} onChanged={reload} onError={(value) => { setMessageKind('alert'); setMessage(value || null); }} /></section>
      <section className="mt-5 border-t border-border pt-5" aria-labelledby="pairing-title"><div className="mb-3 flex items-center justify-between gap-3"><h3 id="pairing-title" className="text-sm font-semibold">기존 카메라 매핑</h3><span className="text-xs text-muted-foreground">미매핑 {topology.unmapped_camera_ids.length}대</span></div><CameraPairingList topology={topology} camerasResource={camerasResource} onChanged={reload} onError={(value) => { setMessageKind('alert'); setMessage(value); }} /></section>

      {preview ? (
        <section className="mt-5 rounded-control bg-status-pendingBg p-4 text-status-pendingFg" aria-labelledby="preview-title">
          <div className="flex flex-wrap items-center justify-between gap-3"><div><h3 id="preview-title" className="text-sm font-semibold">변경 제외 미리보기</h3><p className="mt-1 text-xs">카메라 {preview.cameras}대 · 방 {preview.rooms}개 · 층 {preview.floors}개</p></div>{preview.confirmed ? <span className="text-sm font-semibold">확정 완료</span> : <button type="button" className="h-11 rounded-control bg-destructive px-3 text-sm font-semibold text-destructive-foreground sm:h-9" onClick={() => setDialogOpen(true)}>변경 제외 확인</button>}</div>
        </section>
      ) : null}

      {message ? <p role={messageKind} className={messageKind === 'alert' ? 'auth-error mt-4' : 'dialog-success mt-4'}>{message}</p> : null}
      <div className="mt-5 flex justify-end"><button type="button" className="brand-action h-11 rounded-control px-4 text-sm font-semibold disabled:opacity-50 sm:h-9" disabled={syncing || topology.unmapped_camera_ids.length > 0} onClick={() => void manualSync()}>{syncing ? '동기화 중...' : '지금 동기화'}</button></div>

      <TopologyConfirmationDialog open={dialogOpen} preview={preview} onClose={() => setDialogOpen(false)} onChanged={(next) => setPreview(next)} onConfirmed={(revision) => { setPreview((current) => current ? { ...current, confirmed: true } : null); setMessageKind('status'); setMessage(`변경 제외를 확정했습니다. 서버 리비전 ${revision}`); }} />
    </article>
  );
}
