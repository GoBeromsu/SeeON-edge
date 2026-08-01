import type { StatusSnapshot, SystemSnapshot } from '@/shared/api/client';
import type { PollingResource } from '@/shared/api/usePollingResource';

type SystemPanelProps = {
  systemResource: PollingResource<SystemSnapshot>;
  statusResource: PollingResource<StatusSnapshot>;
};

function formatDate(value: string | number | null): string {
  if (value === null || value === '') return '정보 없음';
  const date = new Date(typeof value === 'number' ? value * 1_000 : value);
  return Number.isFinite(date.getTime()) ? date.toLocaleString('ko-KR') : '정보 없음';
}

function formatPollingDate(value: number): string {
  return new Date(value).toLocaleString('ko-KR');
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  return `${(value / 1024 ** 3).toFixed(1)} GB`;
}

function isNonNegative(value: number | null | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0;
}

function formatCount(value: number | null, suffix: string): string {
  return isNonNegative(value) ? `${value}${suffix}` : '정보 없음';
}

function formatVersion(value: string | null | undefined): string {
  return !value || value.trim().toLowerCase() === 'unknown' ? '정보 없음' : value;
}

function formatElapsedSince(value: string | null): string | null {
  if (value === null || value === '') return null;
  const then = new Date(value).getTime();
  if (!Number.isFinite(then)) return null;
  const diffSec = Math.max(0, Math.round((Date.now() - then) / 1_000));
  if (diffSec < 60) return `${diffSec}초`;
  const diffMin = Math.round(diffSec / 60);
  if (diffMin < 60) return `${diffMin}분`;
  const diffHour = Math.round(diffMin / 60);
  if (diffHour < 24) return `${diffHour}시간`;
  return `${Math.round(diffHour / 24)}일`;
}

// 30초 서버측 갱신 루프(backend/app/lifespan.py)의 산출물이므로 "연결됨" 같은 실시간 불리언이 아니라
// 마지막으로 확인된 시점을 경과 시간으로 정직하게 표시한다.
function formatBackendConnection(backend: SystemSnapshot['backend']): string {
  if (!backend.configured) return '설정되지 않음';
  const elapsed = formatElapsedSince(backend.last_ok_at);
  if (elapsed === null) return backend.reachable === false ? '동기화 이력 없음 · 연결 실패' : '동기화 이력 없음';
  if (backend.reachable === false) return `${elapsed}째 실패`;
  if (backend.reachable === true) return `${elapsed} 전 동기화`;
  return `${elapsed} 전 · 상태 불확실`;
}

function Availability({ label, value }: { label: string; value: string }): JSX.Element {
  return (
    <div className="rounded-xl border border-border bg-surface2 p-4">
      <dt className="text-sm font-bold text-ink-soft">{label}</dt>
      <dd className="mt-1 text-base font-black text-ink">{value}</dd>
    </div>
  );
}
function HealthBadge({ value, availableLabel, unavailableLabel }: { value: boolean | null | undefined; availableLabel: string; unavailableLabel: string }): JSX.Element {
  const label = value === true ? availableLabel : value === false ? unavailableLabel : '정보 없음';
  const className = value === true
    ? 'bg-status-stableBg text-status-stable ring-status-stable'
    : value === false
      ? 'bg-status-dangerBg text-status-danger ring-status-danger'
      : 'bg-surface2 text-ink-soft ring-border';
  return (
    <span className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-bold ring-1 ${className}`}>
      <span className="mr-1.5 h-2 w-2 rounded-full bg-current" />
      {label}
    </span>
  );
}

function formatFps(value: number | null | undefined): string {
  return isNonNegative(value) ? `${value.toFixed(1)} FPS` : '정보 없음';
}

function formatSeconds(value: number | null | undefined): string {
  return isNonNegative(value) ? `${value.toFixed(2)}초` : '정보 없음';
}


function ResourceState<T>({ label, resource }: { label: string; resource: PollingResource<T> }): JSX.Element {
  if (resource.data === null) {
    return <p>{label}{resource.status === 'error' ? '를 불러오지 못했습니다' : '를 불러오는 중'}</p>;
  }
  const checkedAt = resource.lastSuccessAt === null ? '' : ` · 마지막 확인 ${formatPollingDate(resource.lastSuccessAt)}`;
  if (resource.status === 'error') return <p>{label} 갱신 지연{checkedAt}</p>;
  if (resource.refreshing) return <p>{label} 새로 확인 중{checkedAt}</p>;
  return <p>{label}{checkedAt}</p>;
}

export function StorageGauge({ system }: { system: SystemSnapshot | null }): JSX.Element {
  const store = system?.storage?.clip_store;
  const usedBytes = isNonNegative(store?.used_bytes) ? store.used_bytes : null;
  const totalBytes = isNonNegative(store?.total_bytes) ? store.total_bytes : null;
  const usedPercentage = isNonNegative(store?.used_pct) && store.used_pct <= 100 ? store.used_pct : null;
  const hasAmount = usedBytes !== null && totalBytes !== null;
  const hasPercentage = usedPercentage !== null;

  return (
    <article className="rounded-2xl border border-border bg-surface p-5">
      <h3 className="text-lg font-black text-ink">클립 저장 공간</h3>
      {!hasAmount && !hasPercentage ? (
        <p className="mt-3 text-sm font-bold text-ink-soft">저장 공간 정보 없음</p>
      ) : (
        <>
          {hasAmount ? <p className="mt-3 text-sm font-bold tabular-nums text-ink">{formatBytes(usedBytes)} / {formatBytes(totalBytes)}</p> : null}
          {hasPercentage ? (
            <>
              <div className="mt-3 h-3 overflow-hidden rounded-full bg-surface2" role="meter" aria-label="클립 저장 공간 사용률" aria-valuemin={0} aria-valuemax={100} aria-valuenow={usedPercentage}>
                <div className="h-full rounded-full bg-brand" style={{ width: `${usedPercentage}%` }} />
              </div>
              <p className="mt-2 text-sm font-bold tabular-nums text-ink-soft">{usedPercentage.toFixed(1)}%</p>
            </>
          ) : <p className="mt-2 text-sm text-ink-soft">사용률 정보 없음</p>}
        </>
      )}
    </article>
  );
}

function BackendPanel({ system }: { system: SystemSnapshot }): JSX.Element {
  return (
    <article className="rounded-2xl border border-border bg-surface p-5">
      <h3 className="text-lg font-black text-ink">서비스 연결</h3>
      <dl className="mt-4 grid gap-3 sm:grid-cols-2">
        <Availability label="설정" value={system.backend.configured ? '설정됨' : '설정되지 않음'} />
        <Availability label="연결" value={formatBackendConnection(system.backend)} />
      </dl>
    </article>
  );
}

function HeartbeatPanel({ status }: { status: StatusSnapshot }): JSX.Element {
  const heartbeats = Object.values(status.cameras);
  const online = heartbeats.filter((item) => item.status === 'online').length;
  const stale = heartbeats.filter((item) => item.status === 'stale').length;
  const neverSeen = heartbeats.filter((item) => item.status === 'never_seen').length;
  // 신선도 필드는 개별 카메라마다 부재할 수 있으므로 항상 isNonNegative로 방어한다.
  const atRisk = heartbeats.filter((item) => item.status !== 'online');
  return (
    <article className="rounded-2xl border border-border bg-surface p-5">
      <h3 className="text-lg font-black text-ink">카메라 연결 상태</h3>
      {heartbeats.length === 0 ? <p className="mt-3 text-sm font-bold text-ink-soft">카메라 연결 이력이 없습니다</p> : (
        <>
          <p className="mt-3 text-sm font-bold tabular-nums text-ink-soft">정상 {online}대 · 연결 지연 {stale}대 · 연결 이력 없음 {neverSeen}대</p>
          {atRisk.length > 0 ? (
            <ul className="mt-2 space-y-1 text-sm text-ink-soft">
              {atRisk.map((item) => (
                <li key={item.camera_id}>
                  <span className="font-bold text-ink">{item.camera_id}</span>
                  <span className="ml-2 tabular-nums">
                    {item.status === 'never_seen' ? '연결 이력 없음' : `마지막 신호 ${formatCount(item.age_sec, '초 전')}`}
                  </span>
                </li>
              ))}
            </ul>
          ) : null}
        </>
      )}
    </article>
  );
}

function RecoveryPanel({ facility }: { facility: StatusSnapshot['runtime']['facilities'][string] }): JSX.Element {
  const gpu = facility.gpu;
  const worker = facility.worker;
  const latency = facility.latency;
  return (
    <section aria-label="복구 상태" className="mt-4 rounded-xl border border-border bg-surface2 p-4">
      <h5 className="font-black text-ink">GPU 및 워커 복구 상태</h5>
      <dl className="mt-3 grid gap-3 sm:grid-cols-2">
        <div>
          <dt className="text-sm font-bold text-ink-soft">GPU NVML</dt>
          <dd className="mt-1"><HealthBadge value={gpu?.nvml_available} availableLabel="NVML 사용 가능" unavailableLabel="NVML 사용 불가" /></dd>
        </div>
        <div>
          <dt className="text-sm font-bold text-ink-soft">CUDA 컨텍스트</dt>
          <dd className="mt-1"><HealthBadge value={gpu?.cuda_context_ok} availableLabel="생성 가능" unavailableLabel="생성 실패" /></dd>
        </div>
        <div>
          <dt className="text-sm font-bold text-ink-soft">GPU 드라이버</dt>
          <dd className="mt-1 font-bold text-ink">{gpu ? formatVersion(gpu.driver_version) : '정보 없음'}</dd>
        </div>
        <div>
          <dt className="text-sm font-bold text-ink-soft">GPU 디바이스</dt>
          <dd className="mt-1 font-bold text-ink">{gpu ? formatVersion(gpu.device_name) : '정보 없음'}</dd>
        </div>
        <div>
          <dt className="text-sm font-bold text-ink-soft">워커</dt>
          <dd className="mt-1"><HealthBadge value={worker?.alive} availableLabel="워커 실행 중" unavailableLabel="워커 중지" /></dd>
        </div>
        <div>
          <dt className="text-sm font-bold text-ink-soft">워커 PID</dt>
          <dd className="mt-1 font-bold tabular-nums text-ink">{isNonNegative(worker?.pid) ? worker.pid : '정보 없음'}</dd>
        </div>
      </dl>
      <div className="mt-4">
        <h6 className="text-sm font-bold text-ink-soft">카메라 디코드 및 실측 FPS</h6>
        {facility.cameras.length === 0 ? <p className="mt-1 text-sm text-ink-soft">카메라 진단 정보 없음</p> : (
          <ul className="mt-2 space-y-2">
            {facility.cameras.map((camera) => (
              <li key={camera.camera_id} className="rounded-lg border border-border bg-surface p-3 text-sm text-ink-soft">
                <span className="font-bold text-ink">{camera.camera_id}</span>
                <span className="ml-2">실제 디코드 {camera.decode.selected ?? '정보 없음'} · 폴백 {formatCount(camera.decode.fallback_count, '회')} · 실측 {formatFps(camera.measured_fps)}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-2 text-sm text-ink-soft">
        <span>최초 전송 지연 표본 {formatCount(latency?.first_attempt_samples ?? null, '건')} · 최대 지연 {formatSeconds(latency?.max_sec)}</span>
        <HealthBadge
          value={latency?.first_attempt_samples && latency.first_attempt_samples > 0 && isNonNegative(latency.max_sec) ? latency.max_sec <= 5 : null}
          availableLabel="5초 이내"
          unavailableLabel="5초 초과"
        />
      </div>
    </section>
  );
}

function RuntimePanel({ status }: { status: StatusSnapshot }): JSX.Element {
  const facilities = Object.values(status.runtime.facilities);
  return (
    <section aria-labelledby="runtime-heading" className="rounded-2xl border border-border bg-surface p-5">
      <h3 id="runtime-heading" className="text-lg font-black text-ink">시설 처리 상태</h3>
      {facilities.length === 0 ? <p className="mt-3 text-sm font-bold text-ink-soft">시설 처리 정보가 없습니다</p> : (
        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          {facilities.map((facility, index) => {
            const freshness = facility.stale === false ? '최신' : facility.stale === true ? '연결 지연' : '신선도 정보 없음';
            const recorder = facility.clip_recorder;
            return (
              <article key={facility.facility_id} className="rounded-xl border border-border bg-surface2 p-4">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <h4 className="font-black text-ink">시설 {index + 1}</h4>
                  <span className="rounded-full border border-border px-3 py-1 text-xs font-bold text-ink-soft">{freshness}</span>
                </div>
                <p className="mt-3 text-sm font-bold tabular-nums text-ink-soft">카메라 {facility.cameras.length}대</p>
                {recorder ? (
                  <p className="mt-1 text-sm font-bold tabular-nums text-ink-soft">
                    활성 클립 {formatCount(recorder.active_clips, '개')} · 완료 클립 {formatCount(recorder.finalized_clips, '개')} · 기록 실패 {formatCount(recorder.failed_writes, '건')}
                  </p>
                ) : <p className="mt-1 text-sm text-ink-soft">클립 기록 정보 없음</p>}
                <p className="mt-2 text-sm text-ink-soft">마지막 확인 {formatDate(facility.received_at)}</p>
                <RecoveryPanel facility={facility} />
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}

function TechnicalInformation({ system, status }: { system: SystemSnapshot | null; status: StatusSnapshot | null }): JSX.Element {
  const facilities = status ? Object.values(status.runtime.facilities) : [];
  const configVersions = status
    ? [...new Set(Object.values(status.cameras).map((camera) => camera.config_version).filter(isNonNegative))]
    : [];
  return (
    <details className="rounded-2xl border border-border bg-surface p-5">
      <summary className="min-h-11 cursor-pointer py-2 font-black text-brand">기술 정보</summary>
      <dl className="mt-3 grid gap-3 text-sm text-ink-soft">
        <div><dt className="font-bold">버전</dt><dd className="break-all">{formatVersion(system?.version)}</dd></div>
        <div><dt className="font-bold">API 이미지 식별값</dt><dd className="break-all">{system?.image_digests?.ml_api ?? '정보 없음'}</dd></div>
        <div><dt className="font-bold">워커 이미지 식별값</dt><dd className="break-all">{system?.image_digests?.ml_worker ?? '정보 없음'}</dd></div>
        <div><dt className="font-bold">카메라 설정</dt><dd>{configVersions.length ? configVersions.map((version) => `설정 버전 ${version}`).join(' · ') : '정보 없음'}</dd></div>
        {facilities.map((facility) => (
          <div key={facility.facility_id}>
            <dt className="font-bold">런타임 시설 식별자</dt>
            <dd className="break-all">{facility.facility_id} · 세대 {facility.generation ?? '정보 없음'} · 순번 {facility.seq ?? '정보 없음'}</dd>
          </div>
        ))}
      </dl>
    </details>
  );
}

export function SystemPanel(props: SystemPanelProps): JSX.Element {
  const { systemResource, statusResource: runtimeResource } = props;
  const system = systemResource.data;
  const status = runtimeResource.data;
  const errors = [systemResource, runtimeResource].filter((resource) => resource.status === 'error');
  const available = Number(system !== null) + Number(status !== null);
  const pending = [systemResource, runtimeResource].some((resource) => resource.data === null && resource.status !== 'error');
  const updatedAt = system ? formatDate(system.updated_at ?? null) : null;

  function retry(): void {
    if (systemResource.status === 'error') systemResource.retry();
    if (runtimeResource.status === 'error') runtimeResource.retry();
  }

  if (available === 0 && errors.length === 0) {
    return <section aria-labelledby="system-heading"><h1 id="system-heading" tabIndex={-1} className="text-xl font-black text-ink">시스템</h1><p className="mt-3 text-sm font-bold text-ink-soft">시스템 상태를 불러오는 중</p></section>;
  }

  if (available === 0 && errors.length === 2) {
    return (
      <section aria-labelledby="system-heading" className="rounded-2xl border border-border bg-surface p-5">
        <h1 id="system-heading" tabIndex={-1} className="text-xl font-black text-ink">시스템</h1>
        <p role="alert" className="mt-3 text-sm font-bold text-status-danger">시스템 상태를 불러오지 못했습니다</p>
        <button type="button" onClick={retry} className="brand-action mt-4 min-h-11 rounded-lg px-4 font-bold">다시 시도</button>
      </section>
    );
  }

  // 정상 상태(오류 없음, 대기 없음)는 침묵한다 — "확인 완료" 같은 문구는 아무것도 보증하지 않는다.
  const headline = errors.length
    ? (available === 2 ? '마지막 성공 상태를 표시합니다' : '일부 상태만 확인됨')
    : pending ? '일부 상태를 불러오는 중'
      : null;
  return (
    <section aria-labelledby="system-heading" className="space-y-4">
      <header>
        <h1 id="system-heading" tabIndex={-1} className="text-xl font-black text-ink">시스템</h1>
        {headline !== null ? <p aria-live="polite" className="mt-2 text-sm font-bold text-ink-soft">{headline}</p> : null}
        <div className="mt-2 text-sm text-ink-soft">
          <ResourceState label="시스템 정보" resource={systemResource} />
          <ResourceState label="런타임 상태" resource={runtimeResource} />
          {updatedAt !== null && updatedAt !== '정보 없음' ? <p>시스템 기준 시각 {updatedAt}</p> : null}
        </div>
        {errors.length ? <button type="button" onClick={retry} className="mt-3 min-h-11 rounded-lg border border-border bg-surface px-4 font-bold text-brand">다시 시도</button> : null}
      </header>
      {system ? <BackendPanel system={system} /> : <p className="rounded-xl border border-border bg-surface p-4 text-sm font-bold text-ink-soft">{systemResource.status === 'error' ? '백엔드와 저장 공간 상태를 불러오지 못했습니다' : '시스템 정보를 불러오는 중입니다'}</p>}
      {system ? <StorageGauge system={system} /> : null}
      {status ? <HeartbeatPanel status={status} /> : null}
      {status ? <RuntimePanel status={status} /> : <p className="rounded-xl border border-border bg-surface p-4 text-sm font-bold text-ink-soft">런타임 상태를 {runtimeResource.status === 'error' ? '불러오지 못했습니다' : '불러오는 중입니다'}</p>}
      <TechnicalInformation system={system} status={status} />
    </section>
  );
}
