import { useEffect, useMemo, useRef, useState } from 'react';
import {
  applyDetectionPolicy,
  diffDetectionPolicy,
  fetchDetectionPolicies,
  rollbackDetectionPolicy,
} from '@/shared/api/client';
import { toast } from '@/shared/ui/Toast';
import type {
  CameraRegistry,
  DetectionPolicyCatalog,
  DetectionPolicyComparedPayload,
  DetectionPolicyDiff,
} from '@/shared/api/types';

type Props = { cameras: CameraRegistry | null };

function draftValues(draft: Record<string, string>): Record<string, number> {
  return Object.fromEntries(Object.entries(draft).map(([key, value]) => [key, Number(value)]));
}

function scopeConcurrencyToken(
  policy: { camera_revision_id: number | null; facility_revision_id: number | null },
  cameraId: string | null,
): number {
  const revision = cameraId ? policy.camera_revision_id : policy.facility_revision_id;
  return revision ?? 0;
}

/** Policy values remain API-owned numeric documents; this card only renders their qualified contract. */
export function PolicyEvidenceCard({ cameras }: Props): JSX.Element {
  const [catalog, setCatalog] = useState<DetectionPolicyCatalog | null>(null);
  const [status, setStatus] = useState<'loading' | 'error' | 'success'>('loading');
  const [moduleId, setModuleId] = useState('fall');
  const [cameraId, setCameraId] = useState('');
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [comparison, setComparison] = useState<DetectionPolicyDiff | null>(null);
  const [busy, setBusy] = useState(false);
  const compareGenerationRef = useRef(0);
  const compareAbortRef = useRef<AbortController | null>(null);

  const invalidateComparison = (): void => {
    compareGenerationRef.current += 1;
    compareAbortRef.current?.abort();
    compareAbortRef.current = null;
    setComparison(null);
  };

  const load = async (): Promise<void> => {
    setStatus('loading');
    invalidateComparison();
    try {
      setCatalog(await fetchDetectionPolicies());
      setStatus('success');
    } catch {
      setStatus('error');
    }
  };

  useEffect(() => {
    void load();
    return () => {
      compareAbortRef.current?.abort();
    };
  }, []);

  const policy = useMemo(
    () =>
      catalog &&
      (cameraId
        ? catalog.effective.cameras[cameraId]?.[moduleId]
        : catalog.effective.defaults[moduleId]),
    [cameraId, catalog, moduleId],
  );
  const module = catalog?.modules.find((item) => item.qualified_id.startsWith(`${moduleId}.`));

  useEffect(() => {
    invalidateComparison();
    if (policy) {
      setDraft(
        Object.fromEntries(Object.entries(policy.values).map(([key, value]) => [key, String(value)])),
      );
    }
  }, [policy, moduleId, cameraId]);

  if (status === 'loading') {
    return (
      <article className="rounded-card border border-border bg-card p-5">
        <p className="text-sm text-muted-foreground">정책 증명을 불러오는 중입니다...</p>
      </article>
    );
  }
  if (status === 'error' || !catalog || !policy || !module) {
    return (
      <article className="rounded-card border border-border bg-card p-5">
        <h2 className="text-base font-semibold">정책 증명</h2>
        <p className="mt-2 text-sm text-destructive">정책 증명을 불러오지 못했습니다.</p>
        <button type="button" className="dialog-secondary-action mt-3" onClick={() => void load()}>
          다시 시도
        </button>
      </article>
    );
  }

  const cameraOptions = cameras?.cameras ?? [];
  const selectedCamera = cameraOptions.find(
    (camera) => (camera.backend_camera_id ?? camera.id) === cameraId,
  );
  const effectiveCameraId = cameraId || null;
  const draftPayload: DetectionPolicyComparedPayload = {
    module_id: policy.module_id,
    module_version: policy.module_version,
    schema_id: policy.schema_id,
    schema_version: policy.schema_version,
    camera_id: effectiveCameraId,
    values: draftValues(draft),
  };

  const updateDraftField = (field: string, value: string): void => {
    invalidateComparison();
    setDraft((current) => ({ ...current, [field]: value }));
  };

  const changeModule = (nextModuleId: string): void => {
    invalidateComparison();
    setModuleId(nextModuleId);
  };

  const changeCamera = (nextCameraId: string): void => {
    invalidateComparison();
    setCameraId(nextCameraId);
  };

  const compare = async (): Promise<void> => {
    const generation = compareGenerationRef.current + 1;
    compareGenerationRef.current = generation;
    compareAbortRef.current?.abort();
    const controller = new AbortController();
    compareAbortRef.current = controller;
    setBusy(true);
    setComparison(null);
    try {
      const result = await diffDetectionPolicy(draftPayload, controller.signal);
      if (generation !== compareGenerationRef.current) return;
      setComparison(result);
    } catch {
      if (controller.signal.aborted || generation !== compareGenerationRef.current) return;
      toast.error('정책 차이를 확인하지 못했습니다.');
    } finally {
      if (generation === compareGenerationRef.current) setBusy(false);
    }
  };

  const apply = async (): Promise<void> => {
    if (!comparison?.changed) return;
    setBusy(true);
    try {
      await applyDetectionPolicy({
        ...comparison.compared_payload,
        expected_revision_id: comparison.concurrency_token,
      });
      toast.success('정책 변경을 대기열에 반영했습니다. 워커 재시작/적용 확인이 필요합니다.');
      invalidateComparison();
      await load();
    } catch {
      toast.error('정책 변경이 충돌했거나 저장되지 않았습니다. 최신 값을 확인하세요.');
    } finally {
      setBusy(false);
    }
  };

  const rollback = async (): Promise<void> => {
    setBusy(true);
    try {
      await rollbackDetectionPolicy({
        module_id: policy.module_id,
        module_version: policy.module_version,
        camera_id: effectiveCameraId,
        expected_revision_id: scopeConcurrencyToken(policy, effectiveCameraId),
      });
      toast.success('이전 정책 개정으로 되돌리기를 요청했습니다. 적용 확인이 필요합니다.');
      invalidateComparison();
      await load();
    } catch {
      toast.error('되돌릴 이전 정책 개정이 없거나 충돌했습니다.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <article className="rounded-card border border-border bg-card p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold">정책 증명</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            활성 세대 {catalog.activation_generation} · {module.qualified_id} · {module.policy_qualified_id}
          </p>
        </div>
        <span className="text-xs text-muted-foreground">구성요소는 적용 런타임 증거에서만 표시됩니다.</span>
      </div>
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <label className="text-sm">
          탐지 모듈
          <select
            className="mt-1 h-9 w-full rounded-control border border-input bg-background px-2"
            value={moduleId}
            onChange={(event) => changeModule(event.target.value)}
          >
            {catalog.modules.map((item) => (
              <option key={item.qualified_id} value={item.qualified_id.split('.v')[0]}>
                {item.qualified_id}
              </option>
            ))}
          </select>
        </label>
        <label className="text-sm">
          적용 범위
          <select
            className="mt-1 h-9 w-full rounded-control border border-input bg-background px-2"
            value={cameraId}
            onChange={(event) => changeCamera(event.target.value)}
          >
            <option value="">시설 기본값</option>
            {cameraOptions.map((camera) => (
              <option key={camera.id} value={camera.backend_camera_id ?? camera.id}>
                {camera.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p className="mt-3 text-xs text-muted-foreground">
        현재 출처: {policy.source} · 개정{' '}
        {effectiveCameraId ? policy.camera_revision_id ?? '-' : policy.facility_revision_id ?? '-'}
        {selectedCamera ? ` · ${selectedCamera.label}` : ''}
      </p>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        {Object.entries(draft).map(([field, value]) => (
          <label className="text-sm" key={field}>
            {field} <span className="text-xs text-muted-foreground">{module.units[field]}</span>
            <input
              className="mt-1 h-9 w-full rounded-control border border-input bg-background px-2 tabular-nums"
              type="number"
              step="any"
              value={value}
              onChange={(event) => updateDraftField(field, event.target.value)}
            />
          </label>
        ))}
      </div>
      {comparison ? (
        <div className="mt-4 rounded-control bg-muted p-3 text-sm" role="status">
          {comparison.changed
            ? `변경 확인: ${comparison.current.source} → ${comparison.proposed.source}`
            : '현재 유효 정책과 차이가 없습니다.'}
        </div>
      ) : null}
      <div className="mt-4 flex flex-wrap gap-2">
        <button type="button" className="dialog-secondary-action" disabled={busy} onClick={() => void compare()}>
          차이 비교
        </button>
        <button
          type="button"
          className="dialog-primary-action"
          disabled={busy || !comparison?.changed}
          onClick={() => void apply()}
        >
          검토한 변경 적용
        </button>
        <button type="button" className="dialog-secondary-action" disabled={busy} onClick={() => void rollback()}>
          이전 개정으로 되돌리기
        </button>
      </div>
    </article>
  );
}
