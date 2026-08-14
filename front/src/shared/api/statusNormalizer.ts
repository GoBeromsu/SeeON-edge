import {
  isRecord,
  pickBoolean,
  pickNonNegativeNumber,
  pickNullableString,
  pickString,
} from '@/shared/api/normalizerFields';
import type {
  CameraHeartbeat,
  RuntimeCameraDiagnostics,
  RuntimeClipExportApplied,
  RuntimeClipRecorder,
  RuntimeDecodeDiagnostics,
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
