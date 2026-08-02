export type CameraStatus = 'online' | 'offline' | 'starting' | 'unknown';

export type DecodeBackend = 'auto' | 'nvdec' | 'opencv' | 'cpu';

export type Camera = {
  id: string;
  label: string;
  rtsp_url_masked: string;
  floor_name: string | null;
  status: CameraStatus;
  created_at: string | null;
  decode_backend?: DecodeBackend | string | null;
  /** True once the registry confirms this camera has never completed a successful connection. */
  never_connected?: boolean;
  /** ISO timestamp of the last successful connection, or null if none has occurred. */
  last_ok_at?: string | null;
  /** ISO timestamp of the last connection probe (success or failure), or null if never probed. */
  last_probed_at?: string | null;
  /**
   * Unix epoch SECONDS of the last edge heartbeat, or null. Only populated by `GET /cameras`;
   * POST/PATCH/test responses always return null here — that is expected, not an error state.
   */
  last_heartbeat_at?: number | null;
  /** Seconds since the last edge heartbeat, or null. Same GET-only availability as last_heartbeat_at. */
  heartbeat_age_sec?: number | null;
};

export type CameraRegistry = {
  registry_version: number;
  cameras: Camera[];
};

export type HeartbeatStatus = 'online' | 'stale' | 'never_seen';

export type CameraHeartbeat = {
  camera_id: string;
  facility_id: string | null;
  status: HeartbeatStatus;
  last_heartbeat_at: number | null;
  age_sec: number | null;
  config_version: number | null;
};

/** Per-camera pose/skeleton overlay mode: no overlay, bed-exit skeleton, or fall-detection skeleton. */
export type OverlayMode = 'none' | 'bedexit' | 'fall';

export type RuntimeDecodeDiagnostics = {
  requested: string | null;
  selected: string | null;
  fallback_count: number | null;
  last_reason: string | null;
  updated_at_sec: number | null;
};

export type RuntimeLatencyDiagnostics = {
  first_attempt_samples: number | null;
  max_sec: number | null;
  since_sec: number | null;
};

export type RuntimeCameraDiagnostics = {
  camera_id: string;
  decode: RuntimeDecodeDiagnostics;
  measured_fps: number | null;
  latency: RuntimeLatencyDiagnostics | null;
};

/** Device-adaptive acceleration diagnostics (not CUDA-specific — backend may be any decode/inference device). */
export type RuntimeDeviceDiagnostics = {
  backend: string | null;
  available: boolean | null;
  device_name: string | null;
  captured_at_sec: number | null;
};

export type RuntimeWorkerDiagnostics = {
  alive: boolean | null;
  pid: number | null;
  started_at_sec: number | null;
};

export type RuntimeClipRecorder = {
  available: boolean | null;
  dropped_frames: number | null;
  dropped_events: number | null;
  failed_writes: number | null;
  finalized_clips: number | null;
  video_unavailable_clips: number | null;
  active_clips: number | null;
  encoder: string | null;
};

export type StatusSnapshot = {
  cameras: Record<string, CameraHeartbeat>;
  stale_after_sec: number | null;
  runtime: {
    cameras: Record<string, RuntimeCameraDiagnostics>;
    worker: RuntimeWorkerDiagnostics | null;
    device: RuntimeDeviceDiagnostics | null;
    clip_recorder: RuntimeClipRecorder | null;
    stale_after_sec: number | null;
  };
};

export type CameraInput = {
  label: string;
  rtsp_url: string;
};

export type CameraPatchInput = Partial<CameraInput> & {
  decode_backend?: DecodeBackend;
};

export type CameraTestResult = {
  ok: boolean;
  error_class?: 'timeout' | 'decode' | 'auth' | string;
  width?: number;
  height?: number;
};

export type SystemSnapshot = {
  updated_at?: string | null;
  backend: {
    configured: boolean;
    reachable: boolean | null;
    last_ok_at: string | null;
  };
  version: string | null;
  image_digests?: {
    ml_api: string | null;
    ml_worker: string | null;
  };
  storage?: {
    clip_store?: {
      total_bytes: number | null;
      used_bytes: number | null;
      used_pct: number | null;
    };
  };
};

export type Clip = {
  id: string;
  camera_id: string | null;
  camera_label: string;
  event_type: string;
  created_at: string | null;
  video_path: string;
  video_available: boolean;
  video_error: string | null;
};

export type HeartbeatRelayErrorClass = 'auth' | 'timeout' | 'unreachable';

export type HeartbeatRelayStatus = {
  enabled: boolean;
  last_success_at: string | null;
  last_error_class: HeartbeatRelayErrorClass | null;
  detail: string | null;
};

export type ConnectionView = {
  events_url: string | null;
  config_url: string | null;
  facility_id: string | null;
  facility_token_set: boolean;
  facility_token_masked: string | null;
  configured: boolean;
  reachable: boolean | null;
  last_ok_at: string | null;
  updated_at: string | null;
  /**
   * Optional (not just nullable): an older backend that hasn't shipped this field yet omits it
   * entirely. connectionNormalizer.normalizeConnectionView always fills it in from a live response,
   * so treat a missing value here as "talking to an old backend", not as a normalization bug.
   */
  heartbeat_relay?: HeartbeatRelayStatus;
};

/** Partial update body: an omitted field is left untouched; an explicit `null` clears it. */
export type ConnectionInput = {
  events_url?: string | null;
  config_url?: string | null;
  facility_id?: string | null;
  facility_token?: string | null;
};

export type ConnectionErrorClass = 'unconfigured' | 'invalid_url' | 'unreachable' | 'timeout' | 'auth';

export type ConnectionTestResult = {
  ok: boolean;
  error_class: ConnectionErrorClass | null;
  detail: string;
  probed_url: string | null;
};
