export type CameraStatus = 'online' | 'offline' | 'starting' | 'unknown';

export type DecodeBackend = 'auto' | 'nvdec' | 'opencv' | 'cpu';

export type Camera = {
  id: string;
  label: string;
  rtsp_url_masked: string;
  space_id: string | null;
  space_name: string | null;
  floor_name: string | null;
  backend_camera_id: string | null;
  status: CameraStatus;
  created_at: string | null;
  threshold?: number | null;
  domains?: Record<string, boolean> | null;
  bed_count?: number | null;
  night_start?: string | null;
  night_end?: string | null;
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
  /** Per-camera external-backend roster sync state. Only populated by `GET /cameras`. */
  sync?: CameraSync | null;
};

export type CameraSyncStatus = 'synced' | 'pending' | 'failed' | 'disabled';
export type CameraSyncErrorClass = 'unreachable' | 'timeout' | 'auth' | 'unconfigured';

export type CameraSync = {
  status: CameraSyncStatus;
  error_class: CameraSyncErrorClass | null;
  detail: string | null;
  last_ok_at: string | null;
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

export type RuntimeDecodeDiagnostics = {
  requested: string | null;
  selected: string | null;
  fallback_count: number | null;
  last_reason: string | null;
  updated_at_sec: number | null;
};

export type RuntimeCamera = {
  camera_id: string;
  decode: RuntimeDecodeDiagnostics;
  measured_fps?: number | null;
};

export type RuntimeGpuDiagnostics = {
  nvml_available: boolean | null;
  cuda_context_ok: boolean | null;
  driver_version: string | null;
  device_name: string | null;
  captured_at_sec: number | null;
};

export type RuntimeWorkerDiagnostics = {
  alive: boolean | null;
  pid: number | null;
  started_at_sec: number | null;
};

export type RuntimeLatencyDiagnostics = {
  first_attempt_samples: number | null;
  max_sec: number | null;
  since_sec: number | null;
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

export type RuntimeFacility = {
  facility_id: string;
  generation: number | null;
  seq: number | null;
  received_at: number | null;
  stale: boolean | null;
  cameras: RuntimeCamera[];
  clip_recorder: RuntimeClipRecorder | null;
  gpu?: RuntimeGpuDiagnostics | null;
  worker?: RuntimeWorkerDiagnostics | null;
  latency?: RuntimeLatencyDiagnostics | null;
};

export type StatusSnapshot = {
  cameras: Record<string, CameraHeartbeat>;
  stale_after_sec: number | null;
  runtime: {
    facilities: Record<string, RuntimeFacility>;
    stale_after_sec: number | null;
  };
};

export type CameraInput = {
  label: string;
  rtsp_url: string;
  space_id?: string;
};

export type CameraPatchInput = Partial<CameraInput> & {
  detectionSettings?: {
    threshold: number;
    domains: Record<string, boolean>;
    bedCount: number;
    nightWindow: { start: string; end: string };
  };
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

export type ClipLabel = 'TRUE_POSITIVE' | 'FALSE_POSITIVE' | 'UNREVIEWED';

export type Clip = {
  id: string;
  camera_id: string | null;
  camera_label: string;
  event_type: string;
  created_at: string | null;
  label: ClipLabel | null;
  reviewer?: string | null;
  reviewed_at?: string | null;
  reviewState?: 'unknown' | 'confirmed';
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

export type RosterSyncStatus = 'synced' | 'pending' | 'failed' | 'disabled';
export type RosterSyncErrorClass = 'unreachable' | 'timeout' | 'auth' | 'unconfigured';

export type RosterSyncResult = {
  status: RosterSyncStatus;
  error_class: RosterSyncErrorClass | null;
  detail: string | null;
  last_ok_at: string | null;
  next_retry_at: string | null;
  camera_count: number;
};
