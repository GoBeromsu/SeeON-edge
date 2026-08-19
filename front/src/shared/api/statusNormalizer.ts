import {
  isRecord,
  pickBoolean,
  pickNonNegativeNumber,
  pickNullableString,
  pickString,
} from '@/shared/api/normalizerFields';
import type {
  CameraHeartbeat,
  DetectionReason,
  DetectionState,
  RuntimeCameraDiagnostics,
  RuntimeClipExportApplied,
  RuntimeClipRecorder,
  RuntimeDecodeDiagnostics,
  RuntimeDetectionDiagnostics,
  RuntimeDetectionRawCounters,
  RuntimeDeviceDiagnostics,
  RuntimeLatencyDiagnostics,
  RuntimeWorkerDiagnostics,
  StatusSnapshot,
} from '@/shared/api/types';

function normalizeHeartbeat(value: unknown): CameraHeartbeat | null {
  if (!isRecord(value)) return null;
  const cameraId = pickString(value, ['camera_id', 'cameraId']);
  const status = pickString(value, ['status']);
  if (!cameraId || (status !== 'online' && status !== 'stale' && status !== 'never_seen')) return null;
  return {
    camera_id: cameraId,
    facility_id: pickNullableString(value, ['facility_id', 'facilityId']),
    status,
    last_heartbeat_at: pickNonNegativeNumber(value, ['last_heartbeat_at', 'lastHeartbeatAt']),
    age_sec: pickNonNegativeNumber(value, ['age_sec', 'ageSec']),
    config_version: pickNonNegativeNumber(value, ['config_version', 'configVersion']),
  };
}

function normalizeRuntimeDecode(value: unknown): RuntimeDecodeDiagnostics {
  const decode = isRecord(value) ? value : {};
  return {
    requested: pickNullableString(decode, ['requested']),
    selected: pickNullableString(decode, ['selected']),
    fallback_count: pickNonNegativeNumber(decode, ['fallback_count', 'fallbackCount']),
    last_reason: pickNullableString(decode, ['last_reason', 'lastReason']),
    updated_at_sec: pickNonNegativeNumber(decode, ['updated_at_sec', 'updatedAtSec']),
  };
}

function normalizeRuntimeLatency(value: unknown): RuntimeLatencyDiagnostics | null {
  if (!isRecord(value)) return null;
  return {
    first_attempt_samples: pickNonNegativeNumber(value, ['first_attempt_samples', 'firstAttemptSamples']),
    max_sec: pickNonNegativeNumber(value, ['max_sec', 'maxSec']),
    since_sec: pickNonNegativeNumber(value, ['since_sec', 'sinceSec']),
  };
}

const UNKNOWN_DETECTION: RuntimeDetectionDiagnostics = {
  state: 'unknown',
  reason: null,
  recent_success_rate: null,
  last_completed_at_sec: null,
  evaluation_window_sec: 0,
  timeout_sec: 0,
};

function isDetectionState(value: string): value is DetectionState {
  return value === 'starting'
    || value === 'healthy'
    || value === 'blind'
    || value === 'unknown'
    || value === 'disabled';
}

function isDetectionReason(value: string): value is DetectionReason {
  return value === 'pose_not_completing'
    || value === 'decision_not_completing'
    || value === 'no_completed_cycles'
    || value === 'telemetry_stale'
    || value === 'telemetry_missing'
    || value === 'counter_reset';
}

function pickRequiredNonNegative(
  record: Record<string, unknown>,
  keys: string[],
): number | null {
  for (const key of keys) {
    if (!(key in record)) continue;
    const value = record[key];
    if (typeof value === 'number' && Number.isFinite(value) && value >= 0) {
      return value;
    }
    return null;
  }
  return null;
}

function pickNullableNonNegativeMetric(
  record: Record<string, unknown>,
  keys: string[],
): { ok: true; value: number | null } | { ok: false } {
  for (const key of keys) {
    if (!(key in record)) continue;
    const value = record[key];
    if (value === null) return { ok: true, value: null };
    if (typeof value === 'number' && Number.isFinite(value) && value >= 0) {
      return { ok: true, value };
    }
    return { ok: false };
  }
  return { ok: true, value: null };
}

function pickRawDetectionCounters(
  record: Record<string, unknown>,
): RuntimeDetectionRawCounters | null | undefined {
  const rawKeys = [
    'expected',
    'inference_admitted',
    'inference_succeeded',
    'inference_overwritten',
    'decision_completed',
  ];
  if (!rawKeys.some((key) => key in record)) return undefined;
  const expected = pickBoolean(record, ['expected']);
  const admitted = pickNonNegativeNumber(record, ['inference_admitted']);
  const succeeded = pickNonNegativeNumber(record, ['inference_succeeded']);
  const overwritten = pickNonNegativeNumber(record, ['inference_overwritten']);
  const completed = pickNonNegativeNumber(record, ['decision_completed']);
  if (
    expected === null
    || admitted === null
    || succeeded === null
    || overwritten === null
    || completed === null
    || !Number.isInteger(admitted)
    || !Number.isInteger(succeeded)
    || !Number.isInteger(overwritten)
    || !Number.isInteger(completed)
  ) {
    return null;
  }
  return {
    expected,
    inference_admitted: admitted,
    inference_succeeded: succeeded,
    inference_overwritten: overwritten,
    decision_completed: completed,
  };
}

function normalizeRuntimeDetection(value: unknown): RuntimeDetectionDiagnostics {
  if (!isRecord(value)) return { ...UNKNOWN_DETECTION };
  const state = pickString(value, ['state']);
  if (!isDetectionState(state)) return { ...UNKNOWN_DETECTION };
  if ('reason' in value && value.reason !== null && typeof value.reason !== 'string') {
    return { ...UNKNOWN_DETECTION };
  }
  const reasonText = pickString(value, ['reason']);
  const reason = reasonText === '' ? null : reasonText;
  if (reason !== null && !isDetectionReason(reason)) return { ...UNKNOWN_DETECTION };
  const successRate = pickNullableNonNegativeMetric(value, ['recent_success_rate', 'recentSuccessRate']);
  const lastCompleted = pickNullableNonNegativeMetric(value, ['last_completed_at_sec', 'lastCompletedAtSec']);
  const evaluationWindow = pickRequiredNonNegative(value, ['evaluation_window_sec', 'evaluationWindowSec']);
  const timeout = pickRequiredNonNegative(value, ['timeout_sec', 'timeoutSec']);
  if (!successRate.ok || !lastCompleted.ok || evaluationWindow === null || timeout === null) {
    return { ...UNKNOWN_DETECTION };
  }
  const rawCounters = pickRawDetectionCounters(value);
  if (rawCounters === null) return { ...UNKNOWN_DETECTION };
  return {
    state,
    reason,
    recent_success_rate: successRate.value,
    last_completed_at_sec: lastCompleted.value,
    evaluation_window_sec: evaluationWindow,
    timeout_sec: timeout,
    ...rawCounters,
  };
}

function normalizeRuntimeCameraDiagnostics(value: unknown): RuntimeCameraDiagnostics | null {
  if (!isRecord(value)) return null;
  const cameraId = pickString(value, ['camera_id', 'cameraId']);
  if (!cameraId) return null;
  return {
    camera_id: cameraId,
    decode: normalizeRuntimeDecode(value.decode),
    measured_fps: pickNonNegativeNumber(value, ['measured_fps', 'measuredFps']),
    latency: normalizeRuntimeLatency(value.latency),
    stale: pickBoolean(value, ['stale']),
    detection: normalizeRuntimeDetection(value.detection),
  };
}

/** Device-adaptive: `backend` names whatever acceleration path is active (nvdec, opencv, cpu, ...), not CUDA-only. */
function normalizeRuntimeDevice(value: unknown): RuntimeDeviceDiagnostics | null {
  if (!isRecord(value)) return null;
  return {
    backend: pickNullableString(value, ['backend']),
    available: pickBoolean(value, ['available']),
    device_name: pickNullableString(value, ['device_name', 'deviceName']),
    captured_at_sec: pickNonNegativeNumber(value, ['captured_at_sec', 'capturedAtSec']),
  };
}

function normalizeRuntimeWorker(value: unknown): RuntimeWorkerDiagnostics | null {
  if (!isRecord(value)) return null;
  const pid = pickNonNegativeNumber(value, ['pid']);
  return {
    alive: pickBoolean(value, ['alive']),
    pid: pid !== null && Number.isInteger(pid) ? pid : null,
    started_at_sec: pickNonNegativeNumber(value, ['started_at_sec', 'startedAtSec']),
  };
}

function normalizeRuntimeClipExportApplied(value: unknown): RuntimeClipExportApplied {
  if (!isRecord(value)) return { enabled: null, version: null, freshness: 'unknown' };
  const freshness = pickString(value, ['freshness']);
  const version = pickNonNegativeNumber(value, ['version']);
  return {
    enabled: pickBoolean(value, ['enabled']),
    version: version !== null && Number.isInteger(version) ? version : null,
    freshness: freshness === 'fresh' || freshness === 'stale' || freshness === 'offline'
      ? freshness
      : 'unknown',
  };
}

function normalizeRuntimeClipRecorder(value: unknown): RuntimeClipRecorder | null {
  if (!isRecord(value)) return null;
  return {
    available: pickBoolean(value, ['available']),
    dropped_frames: pickNonNegativeNumber(value, ['dropped_frames', 'droppedFrames']),
    dropped_events: pickNonNegativeNumber(value, ['dropped_events', 'droppedEvents']),
    failed_writes: pickNonNegativeNumber(value, ['failed_writes', 'failedWrites']),
    finalized_clips: pickNonNegativeNumber(value, ['finalized_clips', 'finalizedClips']),
    video_unavailable_clips: pickNonNegativeNumber(value, ['video_unavailable_clips', 'videoUnavailableClips']),
    active_clips: pickNonNegativeNumber(value, ['active_clips', 'activeClips']),
    encoder: pickNullableString(value, ['encoder']),
  };
}

export function normalizeStatusSnapshot(value: unknown): StatusSnapshot {
  const record = isRecord(value) ? value : null;
  const rawCameras = record?.cameras;
  const rawRuntime = record?.runtime;
  const rawRuntimeCameras = isRecord(rawRuntime) ? rawRuntime.cameras : null;
  if (!record || !isRecord(rawCameras) || !isRecord(rawRuntime) || !isRecord(rawRuntimeCameras)) {
    throw new Error('Invalid status response');
  }
  const cameras: Record<string, CameraHeartbeat> = {};
  Object.entries(rawCameras).forEach(([key, entry]) => {
    const heartbeat = normalizeHeartbeat(entry);
    if (heartbeat) cameras[key] = heartbeat;
  });

  const runtimeCameras: Record<string, RuntimeCameraDiagnostics> = {};
  Object.entries(rawRuntimeCameras).forEach(([key, entry]) => {
    const diagnostics = normalizeRuntimeCameraDiagnostics(entry);
    if (diagnostics) runtimeCameras[key] = diagnostics;
  });

  return {
    cameras,
    stale_after_sec: pickNonNegativeNumber(record, ['stale_after_sec', 'staleAfterSec']),
    runtime: {
      cameras: runtimeCameras,
      worker: normalizeRuntimeWorker(rawRuntime.worker),
      device: normalizeRuntimeDevice(rawRuntime.device),
      clip_export_applied: normalizeRuntimeClipExportApplied(rawRuntime.clip_export_applied),
      clip_recorder: normalizeRuntimeClipRecorder(rawRuntime.clip_recorder),
      stale_after_sec: pickNonNegativeNumber(rawRuntime, ['stale_after_sec', 'staleAfterSec']),
    },
  };
}
