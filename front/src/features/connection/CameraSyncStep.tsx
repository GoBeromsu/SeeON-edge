import { useState } from 'react';
import type { CameraRegistry } from '@/shared/api/client';
import { syncTopology, type CameraTopology, type TopologySyncResult } from '@/shared/api/topologyClient';
import type { PollingResource } from '@/shared/api/usePollingResource';
import { CameraPairingList } from '@/features/cameras/CameraPairingList';
import { TopologyStructureEditor } from '@/features/cameras/TopologyStructureEditor';
import { cameraTotal } from '@/features/connection/wizardSteps';

type Props = {
  readonly camerasResource: PollingResource<CameraRegistry>;
  readonly topology: CameraTopology;
  readonly onChanged: () => Promise<void>;
};

/** null return means the sync itself succeeded and needs no failure copy. */
function syncFailureMessage(result: TopologySyncResult): string | null {
  if (result.status === 'synced') return null;
  if (result.error_class === 'auth') return '등록 인증을 확인한 뒤 다시 시도하세요.';
  if (result.error_class === 'conflict') return '서버 상태와 어긋났습니다. 화면을 새로고침한 뒤 다시 시도하세요.';
  if (result.error_class === 'timeout' || result.error_class === 'unreachable') return '서버에 연결할 수 없습니다. 네트워크를 확인한 뒤 다시 시도하세요.';
  if (result.error_class === 'unconfigured') return '1단계 장치 연결을 먼저 완료하세요.';
  return result.detail ?? '동기화가 아직 완료되지 않았습니다. 잠시 후 다시 시도하세요.';
}

/** 2단계 본문: 이 엣지가 알고 있는 카메라 총계 + 층/방 배치 + 서버로의 카메라 동기화. */
export function CameraSyncStep({ camerasResource, topology, onChanged }: Props): JSX.Element {
  const [syncing, setSyncing] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [messageKind, setMessageKind] = useState<'status' | 'alert'>('status');

  const total = cameraTotal(camerasResource.data ?? null);
  const unmapped = topology.unmapped_camera_ids.length;

  async function manualSync(): Promise<void> {
    if (syncing) return;
    setSyncing(true);
    setMessage(null);
    try {
      const result = await syncTopology();
      const failure = syncFailureMessage(result);
      setMessageKind(failure ? 'alert' : 'status');
      setMessage(failure ?? '카메라 목록을 서버와 동기화했습니다.');
      await onChanged();
    } catch {
      setMessageKind('alert');
      setMessage('동기화 요청에 실패했습니다. 잠시 후 다시 시도하세요.');
    } finally {
      setSyncing(false);
    }
  }

  const syncDisabled = syncing || total === 0 || unmapped > 0;

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-control bg-muted p-3">
        <p className="text-sm text-foreground">이 장치에 등록된 카메라 <span className="font-semibold tabular-nums">{total}</span>대</p>
        {unmapped > 0 ? <span className="text-xs text-status-pendingFg">배치 필요 카메라 {unmapped}대</span> : null}
      </div>

      {total === 0 ? (
        <p role="alert" className="auth-error mt-3">
          등록된 카메라가 없습니다. 왼쪽 "카메라" 카드에서 CCTV를 먼저 등록한 뒤 이 단계로 돌아오세요.
        </p>
      ) : null}

      <section className="mt-5" aria-labelledby="wizard-structure-title">
        <h3 id="wizard-structure-title" className="mb-3 text-sm font-semibold">층과 방</h3>
        <TopologyStructureEditor topology={topology} onChanged={onChanged} onError={(value) => { setMessageKind('alert'); setMessage(value || null); }} />
      </section>

      <section className="mt-5 border-t border-border pt-5" aria-labelledby="wizard-pairing-title">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h3 id="wizard-pairing-title" className="text-sm font-semibold">카메라 배치</h3>
          <span className="text-xs text-muted-foreground">미배치 {unmapped}대</span>
        </div>
        <CameraPairingList topology={topology} camerasResource={camerasResource} onChanged={onChanged} onError={(value) => { setMessageKind('alert'); setMessage(value); }} />
      </section>

      <details className="mt-4 text-sm">
        <summary className="cursor-pointer text-muted-foreground">자세히</summary>
        <dl className="mt-2 grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-2">
          <dt className="text-muted-foreground">등록 리비전</dt><dd className="tabular-nums">{topology.registry_version}</dd>
          <dt className="text-muted-foreground">동기화 대기 리비전</dt><dd className="tabular-nums">{topology.dirty_registry_version ?? '없음'}</dd>
        </dl>
      </details>

      {message ? <p role={messageKind === 'alert' ? 'alert' : 'status'} className={messageKind === 'alert' ? 'auth-error mt-4' : 'dialog-success mt-4'}>{message}</p> : null}

      <div className="mt-5 flex justify-end">
        <button type="button" className="brand-action h-11 rounded-control px-4 text-sm font-semibold disabled:opacity-50 sm:h-9" disabled={syncDisabled} onClick={() => void manualSync()}>
          {syncing ? '동기화 중...' : '카메라 동기화'}
        </button>
      </div>
    </div>
  );
}
