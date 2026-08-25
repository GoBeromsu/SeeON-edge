import { useEffect, useState } from 'react';
import { fetchIncidents, reviewIncident } from '@/shared/api/client';
import { AccessibleDialog } from '@/shared/ui/AccessibleDialog';
import { toast } from '@/shared/ui/Toast';
import type { Incident } from '@/shared/api/types';

const STATE_LABEL: Record<string, string> = { COMPLETE: '완료', STAGING: '증거 수집 중', FAILED: '실패', RECOVERING: '복구 중' };

/** Central incident projection: never renders local file paths, credentials, or opaque relay inputs. */
export function IncidentList(): JSX.Element {
  const [incidents, setIncidents] = useState<Incident[] | null>(null);
  const [error, setError] = useState(false);
  const [selected, setSelected] = useState<Incident | null>(null);
  const [busy, setBusy] = useState(false);
  const load = async (): Promise<void> => { try { setError(false); setIncidents(await fetchIncidents()); } catch { setError(true); } };
  useEffect(() => { void load(); }, []);
  const review = async (disposition: 'TRUE_POSITIVE' | 'FALSE_POSITIVE'): Promise<void> => { if (!selected) return; setBusy(true); try { const next = await reviewIncident(selected.incident_id, { expected_version: selected.review?.version ?? 0, disposition, notes: null }); setSelected(next); setIncidents((current) => current?.map((item) => item.incident_id === next.incident_id ? next : item) ?? null); toast.success('검토 상태를 저장했습니다.'); } catch { toast.error('다른 운영자의 검토와 충돌했습니다. 최신 상태를 확인하세요.'); await load(); } finally { setBusy(false); } };
  return <section className="mt-8 border-t border-border pt-5" aria-labelledby="incident-title">
    <div className="flex items-center justify-between gap-3"><div><h2 id="incident-title" className="text-lg font-semibold">중앙 인시던트</h2><p className="mt-1 text-sm text-muted-foreground">증거 수명주기, 전송 및 복구 상태</p></div><button type="button" className="dialog-secondary-action" onClick={() => void load()}>새로 고침</button></div>
    {incidents === null && !error ? <p className="mt-4 text-sm text-muted-foreground">인시던트를 불러오는 중입니다...</p> : null}
    {error ? <p className="mt-4 text-sm text-destructive" role="alert">오프라인이거나 중앙 인시던트를 불러오지 못했습니다.</p> : null}
    {incidents?.length === 0 ? <p className="mt-4 text-sm text-muted-foreground">표시할 중앙 인시던트가 없습니다.</p> : null}
    <div className="mt-4 grid gap-3 md:grid-cols-2">{incidents?.map((incident) => <button type="button" key={incident.incident_id} className="rounded-card border border-border bg-card p-4 text-left hover:bg-muted" onClick={() => setSelected(incident)}><div className="flex justify-between gap-3"><strong>{incident.event_type === 'fall' ? '낙상' : incident.event_type}</strong><span className="text-xs text-muted-foreground">{STATE_LABEL[incident.lifecycle_state] ?? incident.lifecycle_state}</span></div><p className="mt-2 text-sm text-muted-foreground">{incident.camera_id} · {new Date(incident.detected_at).toLocaleString('ko-KR')}</p><p className="mt-2 text-xs text-muted-foreground">원본 {incident.primary_artifact_state ?? '미상'} · 스냅샷 {incident.snapshot_artifact_state ?? '미상'} · 전송 {incident.event_delivery_state}</p></button>)}</div>
    <AccessibleDialog open={selected !== null} title="인시던트 증거" onClose={() => setSelected(null)} size="lg" initialFocus="heading">{selected ? <div className="space-y-4"><dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm"><dt className="text-muted-foreground">수명주기</dt><dd>{STATE_LABEL[selected.lifecycle_state] ?? selected.lifecycle_state}</dd><dt className="text-muted-foreground">원본 / 스냅샷</dt><dd data-testid="incident-artifact-states">{selected.primary_artifact_state ?? '미상'} / {selected.snapshot_artifact_state ?? '미상'}</dd><dt className="text-muted-foreground">릴레이 / 게시 / 보존</dt><dd>{selected.event_delivery_state} / {selected.clip_publish_state ?? '미상'} / {selected.retention_state ?? '미상'}</dd><dt className="text-muted-foreground">적용 증명</dt><dd className="break-all">{selected.module_qualified_id ?? '기록되지 않음'} · {selected.policy_qualified_id ?? '기록되지 않음'}</dd><dt className="text-muted-foreground">런타임 / 결정 추적</dt><dd className="break-all">{selected.runtime_manifest_sha256 ?? '기록되지 않음'} · {selected.decision_trace_id ?? '기록되지 않음'}</dd></dl><div className="border-t border-border pt-4"><p className="text-sm font-semibold">검토 상태: {selected.review?.disposition === 'TRUE_POSITIVE' ? '실제 알림' : selected.review?.disposition === 'FALSE_POSITIVE' ? '오탐' : '미검토'}</p><div className="mt-3 flex flex-wrap gap-2"><button type="button" className="dialog-primary-action" disabled={busy} onClick={() => void review('TRUE_POSITIVE')}>실제 알림으로 검토</button><button type="button" className="dialog-secondary-action" disabled={busy} onClick={() => void review('FALSE_POSITIVE')}>오탐으로 검토</button></div></div>{selected.failure_reason ? <p className="text-sm text-destructive">실패 사유: {selected.failure_reason}</p> : null}</div> : null}</AccessibleDialog>
  </section>;
}
