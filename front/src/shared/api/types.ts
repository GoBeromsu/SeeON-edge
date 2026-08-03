export type CameraStatus = 'online' | 'offline' | 'starting' | 'unknown';

export type DecodeBackend = 'auto' | 'nvdec' | 'opencv' | 'cpu';

export type Camera = {
  id: string;
  label: string;
  rtsp_url_masked: string;
  /** Space-sync-owned floor name (external roster pull); read-only from the UI. See `floor`. */
  floor_name: string | null;
  /**
   * User-set floor override (issue #85): editable via the settings UI's floor chips, persisted on
   * the local registry record so it survives every space-sync re-pull untouched. Display precedence
   * is this field first, falling back to `floor_name`: `camera.floor ?? camera.floor_name`.
   */
  floor?: string | null;
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
  /** Auto-recognized bed area, or null when recognition has never succeeded ("인식 필요" in the UI). */
  bed_zone?: BedZone | null;
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

/** A single [x, y] vertex of a bed-zone polygon, in the coordinate space of image_width x image_height. */
export type BedZonePoint = [number, number];

/** Auto-recognized (YOLO segmentation) bed area for a camera. Persisted server-side, not user-drawn. */
export type BedZone = {
  polygon: BedZonePoint[];
  image_width: number;
  image_height: number;
  recognized_at: string;
};

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
  /** Initial user-set floor override (issue #85); omit to leave unset (falls back to floor_name). */
  floor?: string;
};

/**
 * Partial-update body: an omitted field is left untouched; for `floor`, an explicit `null` clears
 * the override back to the space-sync `floor_name` fallback (see cameraBody() in client.ts).
 * `floor` is redeclared here (rather than inherited via `Partial<CameraInput>`) so it can carry
 * `null` -- intersecting `Partial<CameraInput>`'s `floor?: string` with a `null`-inclusive override
 * would otherwise collapse to `string` only, silently losing the "clear" case.
 */
export type CameraPatchInput = Partial<Omit<CameraInput, 'floor'>> & {
  decode_backend?: DecodeBackend;
  floor?: string | null;
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
  /**
   * Clip duration in seconds. Optional (not just nullable): an older backend that hasn't shipped
   * this field yet omits it entirely from the manifest — treat a missing value as "unknown", not
   * as a normalization bug. normalizeClip always fills this in (as a value or null) from a live response.
   */
  duration_s?: number | null;
  /** Clip file size in bytes. Same optional/omitted-on-old-backend contract as duration_s — never fabricate a size for display. */
  size_bytes?: number | null;
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

/** GET/PUT /detection-settings domain keys — global, applied to every camera. */
export type DetectionDomainKey = 'fall' | 'bed_exit';

export type DetectionMode = 'always' | 'window';

export type DetectionDomainSetting = {
  on: boolean;
  mode: DetectionMode;
  /** HH:MM, only meaningful (and required by the form) when mode is "window". */
  start: string | null;
  end: string | null;
};

export type DetectionSettings = {
  domains: Record<DetectionDomainKey, DetectionDomainSetting>;
};

/** PUT body is a full replace of the same shape as the GET response. */
export type DetectionSettingsInput = DetectionSettings;

/** GET /clips/storage — usage + current selection under the CLIP_STORE_DIR mount root. */
export type ClipStorageInfo = {
  root: string;
  /** "" means the mount root itself; otherwise a relative subdirectory path. */
  selected_path: string;
  total_bytes: number | null;
  used_bytes: number | null;
  used_pct: number | null;
};

export type ClipStorageBrowseEntry = {
  name: string;
  path: string;
};

/** GET /clips/storage/browse?path= — one directory listing for the folder-explorer modal. */
export type ClipStorageBrowseResult = {
  path: string;
  parent: string | null;
  directories: ClipStorageBrowseEntry[];
};
