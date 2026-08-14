import { describe, expect, it } from 'vitest';
import { normalizeCamera, normalizeCameraRegistry, normalizeCameraResponse, normalizeCameraTestResult, normalizeClip, normalizeClipsResponse, normalizeConnectionView, normalizeStatusSnapshot, normalizeSystemSnapshot } from '@/shared/api/normalizers';

const validCameraResponse = {
  id: 'cam-1',
  label: '301호',
  rtsp_url_masked: 'rtsp://***@redacted-camera/stream',
  mapping_pending: false,
  status: 'online',
  decode_backend: null,
  created_at: null,
  floor_name: null,
};

const validClipManifest = {
  clip_id: 'clip-1',
  camera_id: 'cam-1',
  event_ref: 'event-1',
  event_type: null,
  started_at: '2026-07-21T00:00:00Z',
  duration_s: 4.2,
  codec: '',
  path: null,
  video_available: false,
  video_error: null,
  finalized: true,
};

describe('system normalization', () => {
  const validSystem = {
    updated_at: '2026-07-20T01:59:59.000Z',
    backend: { configured: true, reachable: true, last_ok_at: '2026-07-20T01:58:00.000Z' },
    version: '2026.07.20',
    image_digests: { ml_api: null, ml_worker: 'sha256:worker' },
    storage: { clip_store: { total_bytes: 1_000, used_bytes: 250, used_pct: 25 } },
  };

  it('preserves the server-produced system timestamp', () => {
    expect(normalizeSystemSnapshot(validSystem).updated_at).toBe('2026-07-20T01:59:59.000Z');
  });

  it.each([
    ['literal unknown', { ...validSystem, version: 'unknown' }],
  ])('keeps a %s version unavailable instead of inventing a value', (_case, payload) => {
    expect(normalizeSystemSnapshot(payload).version).toBeNull();
  });

  it('accepts only the nullable system leaves allowed by the backend contract', () => {
    expect(normalizeSystemSnapshot({
      ...validSystem,
      backend: { configured: false, reachable: null, last_ok_at: null },
      image_digests: { ml_api: null, ml_worker: null },
      storage: { clip_store: { total_bytes: null, used_bytes: null, used_pct: null } },
    })).toEqual({
      ...validSystem,
      backend: { configured: false, reachable: null, last_ok_at: null },
      image_digests: { ml_api: null, ml_worker: null },
      storage: { clip_store: { total_bytes: null, used_bytes: null, used_pct: null } },
    });
  });

  it.each([
    ['non-object envelope', null],
    ['missing backend', { ...validSystem, backend: undefined }],
    ['missing configured', { ...validSystem, backend: { reachable: true, last_ok_at: null } }],
    ['malformed reachable', { ...validSystem, backend: { configured: true, reachable: 'yes', last_ok_at: null } }],
    ['malformed last_ok_at', { ...validSystem, backend: { configured: true, reachable: true, last_ok_at: 123 } }],
    ['missing version', { ...validSystem, version: undefined }],
    ['missing image digest leaf', { ...validSystem, image_digests: { ml_api: null } }],
    ['missing clip store', { ...validSystem, storage: {} }],
    ['malformed byte count', { ...validSystem, storage: { clip_store: { total_bytes: '1000', used_bytes: 250, used_pct: 25 } } }],
    ['missing updated_at', { ...validSystem, updated_at: undefined }],
  ])('rejects a %s system success envelope', (_case, payload) => {
    expect(() => normalizeSystemSnapshot(payload)).toThrow('Invalid system response');
  });

  it('discards unsupported update and rollback history fields', () => {
    const normalized = normalizeSystemSnapshot({
      ...validSystem,
      update_history: [{ id: 'update-1', status: 'complete' }],
      rollback_history: [{ id: 'rollback-1', status: 'complete' }],
    });

    expect(normalized).not.toHaveProperty('update_history');
    expect(normalized).not.toHaveProperty('rollback_history');
  });
});

describe('camera connection test normalization', () => {
  it.each([
    [{ ok: true }, { ok: true }],
    [{ ok: false, error_class: 'timeout' }, { ok: false, error_class: 'timeout' }],
    [{ ok: true, error_class: null, width: null, height: null }, { ok: true }],
    [{ ok: true, width: 1920, height: 1080 }, { ok: true, width: 1920, height: 1080 }],
    // 이슈 #151: worker의 /probe에 닿지 못한 경우 -- error_class 없이 probe_unavailable만 실린다.
    [{ ok: false, probe_unavailable: true }, { ok: false, probe_unavailable: true }],
    [{ ok: false, probe_unavailable: false }, { ok: false }],
  ])('accepts a contract-valid result %#', (payload, expected) => {
    expect(normalizeCameraTestResult(payload)).toEqual(expected);
  });

  it.each([
    ['non-object', null],
    ['missing ok', {}],
    ['malformed ok', { ok: 'true' }],
    ['unknown error class', { ok: false, error_class: 'network' }],
    ['malformed nullable error class', { ok: false, error_class: 1 }],
    ['malformed probe_unavailable', { ok: false, probe_unavailable: 'true' }],
    ['fractional width', { ok: true, width: 1.5 }],
    ['malformed nullable height', { ok: true, height: '1080' }],
  ])('rejects a %s success envelope', (_case, payload) => {
    expect(() => normalizeCameraTestResult(payload)).toThrow('Invalid camera test response');
  });
});

describe('normalizeClip video availability', () => {
  it.each([
    ['missing', {}, '영상 제공 상태를 확인할 수 없습니다.'],
    ['null', { video_available: null }, '영상 제공 상태를 확인할 수 없습니다.'],
    ['string', { video_available: 'true' }, '영상 제공 상태를 확인할 수 없습니다.'],
    ['false', { video_available: false, video_error: 'encoder failed: /var/clips/clip.mp4' }, '저장된 영상을 사용할 수 없습니다.'],
    ['true', { video_available: true, video_error: 'stale backend detail' }, null],
  ])('normalizes a %s video_available value without inventing playable evidence', (_case, availability, expectedError) => {
    const clip = normalizeClip({ id: 'cam-1-clip', event_type: 'fall', ...availability });

    expect(clip?.video_available).toBe(_case === 'true');
    expect(clip?.video_error).toBe(expectedError);
    expect(clip?.video_error ?? '').not.toContain('/var/clips');
    expect(clip?.video_error ?? '').not.toContain('backend detail');
  });

  it('surfaces a sanitized unplayable clip when video_available is false', () => {
    const clip = normalizeClip({
      clip_id: 'cam-1-clip',
      event_type: 'fall',
      video_available: false,
      video_error: 'encoder failed: h264_nvenc',
    });
    expect(clip?.video_available).toBe(false);
    expect(clip?.video_error).toBe('저장된 영상을 사용할 수 없습니다.');
  });
});

describe('normalizeClip duration_s/size_bytes (null-tolerant, no fabrication)', () => {
  it('picks up duration_s and size_bytes when the manifest reports them', () => {
    const clip = normalizeClip({ id: 'clip-1', duration_s: 4.2, size_bytes: 8_400_000 });
    expect(clip?.duration_s).toBe(4.2);
    expect(clip?.size_bytes).toBe(8_400_000);
  });

  it('normalizes a missing or malformed duration_s/size_bytes to null instead of fabricating a value', () => {
    expect(normalizeClip({ id: 'clip-1' })?.duration_s).toBeNull();
    expect(normalizeClip({ id: 'clip-1' })?.size_bytes).toBeNull();
    expect(normalizeClip({ id: 'clip-1', duration_s: -1, size_bytes: -1 })?.duration_s).toBeNull();
    expect(normalizeClip({ id: 'clip-1', duration_s: -1, size_bytes: -1 })?.size_bytes).toBeNull();
    expect(normalizeClip({ id: 'clip-1', duration_s: 'four', size_bytes: 'big' })?.duration_s).toBeNull();
    expect(normalizeClip({ id: 'clip-1', duration_s: 'four', size_bytes: 'big' })?.size_bytes).toBeNull();
  });
});

describe('rtsp url redaction via normalizeCamera', () => {
  it('masks host and credentials for an rtsp url with credentials', () => {
    const camera = normalizeCamera({
      id: 'cam-1',
      rtsp_url: 'rtsp://operator:hunter2@camera.internal.example:554/stream',
    });

    expect(camera).not.toBeNull();
    const masked = camera!.rtsp_url_masked;
    expect(masked).toContain('redacted-camera');
    expect(masked).toContain('***');
    expect(masked).not.toContain('camera.internal.example');
    expect(masked).not.toContain('hunter2');
  });

  it('still redacts the host for an rtsp url without credentials', () => {
    const camera = normalizeCamera({
      id: 'cam-2',
      rtsp_url: 'rtsp://camera.internal.example:554/stream',
    });

    expect(camera).not.toBeNull();
    const masked = camera!.rtsp_url_masked;
    expect(masked).toContain('redacted-camera');
    expect(masked).not.toContain('camera.internal.example');
  });

  it('falls back to the Korean placeholder for an unparseable rtsp url', () => {
    const camera = normalizeCamera({
      id: 'cam-3',
      rtsp_url: 'not a valid rtsp url garbage',
    });

    expect(camera).not.toBeNull();
    expect(camera!.rtsp_url_masked).toBe('RTSP URL 비공개');
  });

  it('falls back to the Korean placeholder for an opaque-path rtsp url instead of leaking credentials', () => {
    const camera = normalizeCamera({
      id: 'cam-4',
      rtsp_url: 'rtsp:user:secret@camera.internal.example/live',
    });

    expect(camera).not.toBeNull();
    const masked = camera!.rtsp_url_masked;
    expect(masked).toBe('RTSP URL 비공개');
    expect(masked).not.toContain('secret');
    expect(masked).not.toContain('camera.internal.example');
  });
});
describe('camera roster normalization', () => {
  it('accepts the backend camera envelope, including an explicitly empty registry', () => {
    expect(normalizeCameraRegistry({ registry_version: 0, cameras: [] })).toEqual({ registry_version: 0, cameras: [] });
  });

  it.each([
    ['null', null],
    ['legacy array', []],
    ['missing version', { cameras: [] }],
    ['wrong version type', { registry_version: '1', cameras: [] }],
    ['missing cameras', { registry_version: 1 }],
    ['wrong cameras type', { registry_version: 1, cameras: {} }],
  ])('rejects a %s camera registry envelope', (_case, payload) => {
    expect(() => normalizeCameraRegistry(payload)).toThrow('Invalid camera registry response');
  });

  it.each([
    ['missing id', { ...validCameraResponse, id: undefined }],
    ['blank label', { ...validCameraResponse, label: ' ' }],
    ['malformed masked URL', { ...validCameraResponse, rtsp_url_masked: null }],
    ['unknown status', { ...validCameraResponse, status: 'stale' }],
    ['malformed mapping flag', { ...validCameraResponse, mapping_pending: 'false' }],
    ['malformed nullable floor', { ...validCameraResponse, floor_name: 301 }],
    ['malformed nullable floor override', { ...validCameraResponse, floor: '3층' }],
  ])('rejects an entry with a %s instead of dropping or defaulting it', (_case, camera) => {
    expect(() => normalizeCameraRegistry({ registry_version: 1, cameras: [validCameraResponse, camera] }))
      .toThrow('Invalid camera registry response');
  });

  it('preserves nullable floor metadata and does not synthesize created_at', () => {
    const camera = normalizeCamera({
      id: 'cam-1',
      rtsp_url_masked: 'rtsp://***',
      floorName: '3층',
    });

    expect(camera?.floor_name).toBe('3층');
    expect(camera?.created_at).toBeNull();
  });

  it('accepts a registry entry omitting the floor override key, defaulting it to null (backward compat)', () => {
    expect(() => normalizeCameraRegistry({ registry_version: 1, cameras: [validCameraResponse] })).not.toThrow();
    const { cameras } = normalizeCameraRegistry({ registry_version: 1, cameras: [validCameraResponse] });
    expect(cameras[0]?.floor).toBeNull();
  });

  it('keeps the user-set floor override independent of the read-only floor_name (issue #85 precedence)', () => {
    const camera = normalizeCamera({
      id: 'cam-1',
      rtsp_url_masked: 'rtsp://***',
      floor_name: '1층',
      floor: 3,
    });

    expect(camera?.floor_name).toBe('1층');
    expect(camera?.floor).toBe(3);
  });

  it('accepts a negative floor override (basement, issue #155)', () => {
    const camera = normalizeCamera({
      id: 'cam-1',
      rtsp_url_masked: 'rtsp://***',
      floor: -1,
    });

    expect(camera?.floor).toBe(-1);
  });
});

describe('camera mutation response normalization', () => {
  it('normalizes a valid backend CameraResponse and preserves nullable leaves', () => {
    expect(normalizeCameraResponse(validCameraResponse)).toMatchObject({
      id: 'cam-1', label: '301호', rtsp_url_masked: 'rtsp://***@redacted-camera/stream',
      status: 'online', decode_backend: null, created_at: null,
    });
  });

  it.each([
    ['missing label', { ...validCameraResponse, label: undefined }],
    ['missing masked URL', { ...validCameraResponse, rtsp_url_masked: undefined }],
    ['malformed status', { ...validCameraResponse, status: false }],
    ['malformed decode backend', { ...validCameraResponse, decode_backend: 1 }],
  ])('rejects a mutation response with %s', (_case, payload) => {
    expect(() => normalizeCameraResponse(payload)).toThrow('Invalid camera response');
  });
});

describe('clip list contract normalization', () => {
  it('accepts the explicitly empty backend clips envelope', () => {
    expect(normalizeClipsResponse({ clips: [] })).toEqual([]);
  });

  it.each([
    ['missing clip id', { ...validClipManifest, clip_id: undefined }],
    ['blank camera id', { ...validClipManifest, camera_id: ' ' }],
    ['missing event reference', { ...validClipManifest, event_ref: undefined }],
    ['malformed nullable event type', { ...validClipManifest, event_type: 1 }],
    ['missing start time', { ...validClipManifest, started_at: undefined }],
    ['negative duration', { ...validClipManifest, duration_s: -1 }],
    ['malformed codec', { ...validClipManifest, codec: null }],
    ['malformed nullable path', { ...validClipManifest, path: 1 }],
    ['missing video availability', { ...validClipManifest, video_available: undefined }],
    ['malformed nullable video error', { ...validClipManifest, video_error: false }],
    ['missing finalized flag', { ...validClipManifest, finalized: undefined }],
    ['malformed size_bytes', { ...validClipManifest, size_bytes: 'huge' }],
    ['negative size_bytes', { ...validClipManifest, size_bytes: -1 }],
  ])('rejects an entry with a %s instead of returning a partial list', (_case, clip) => {
    expect(() => normalizeClipsResponse({ clips: [validClipManifest, clip] })).toThrow('Invalid clips response');
  });

  it('accepts an entry that omits size_bytes entirely (older backend, null-tolerant)', () => {
    expect(() => normalizeClipsResponse({ clips: [validClipManifest] })).not.toThrow();
  });

  it('accepts an entry with an explicit null size_bytes and a numeric size_bytes alike', () => {
    expect(normalizeClipsResponse({ clips: [{ ...validClipManifest, size_bytes: null }] })[0].size_bytes).toBeNull();
    expect(normalizeClipsResponse({ clips: [{ ...validClipManifest, size_bytes: 1024 }] })[0].size_bytes).toBe(1024);
  });
});

describe('status normalization', () => {
  it('accepts the backend status envelope when both collections are explicitly empty', () => {
    expect(normalizeStatusSnapshot({ cameras: {}, runtime: { cameras: {} } })).toEqual({
      cameras: {},
      stale_after_sec: null,
      runtime: { cameras: {}, worker: null, device: null, clip_export_applied: { enabled: null, version: null, freshness: 'unknown' }, clip_recorder: null, stale_after_sec: null },
    });
  });

  it.each([
    ['null', null],
    ['missing cameras', { runtime: { cameras: {} } }],
    ['wrong cameras type', { cameras: [], runtime: { cameras: {} } }],
    ['missing runtime', { cameras: {} }],
    ['wrong runtime type', { cameras: {}, runtime: null }],
    ['missing runtime cameras', { cameras: {}, runtime: {} }],
    ['wrong runtime cameras type', { cameras: {}, runtime: { cameras: [] } }],
  ])('rejects a %s status envelope', (_case, payload) => {
    expect(() => normalizeStatusSnapshot(payload)).toThrow('Invalid status response');
  });

  it('preserves valid heartbeat and per-camera runtime diagnostics without converting epoch seconds', () => {
    expect(normalizeStatusSnapshot({
      cameras: {
        'worker/cam': {
          camera_id: 'worker/cam',
          facility_id: 'facility-a',
          status: 'online',
          last_heartbeat_at: 1_720_000_000.5,
          age_sec: 3.25,
          config_version: 7,
        },
      },
      stale_after_sec: 90,
      runtime: {
        cameras: {
          'worker/cam': {
            camera_id: 'worker/cam',
            measured_fps: 27.5,
            decode: { requested: 'auto', selected: 'nvdec', fallback_count: 1, last_reason: 'open_failed', updated_at_sec: 1_720_000_000 },
            latency: { first_attempt_samples: 3, max_sec: 0.75, since_sec: 1_720_000_000 },
            stale: true,
          },
        },
        worker: { alive: true, pid: 1234, started_at_sec: 1_719_999_000 },
        device: { backend: 'nvdec', available: true, device_name: 'NVIDIA Test GPU', captured_at_sec: 1_720_000_000 },
        clip_export_applied: { enabled: true, version: 7, freshness: 'fresh' },
        clip_recorder: { available: true, finalized_clips: 4 },
        stale_after_sec: 15,
      },
    })).toEqual({
      cameras: {
        'worker/cam': {
          camera_id: 'worker/cam',
          facility_id: 'facility-a',
          status: 'online',
          last_heartbeat_at: 1_720_000_000.5,
          age_sec: 3.25,
          config_version: 7,
        },
      },
      stale_after_sec: 90,
      runtime: {
        cameras: {
          'worker/cam': {
            camera_id: 'worker/cam',
            measured_fps: 27.5,
            decode: { requested: 'auto', selected: 'nvdec', fallback_count: 1, last_reason: 'open_failed', updated_at_sec: 1_720_000_000 },
            latency: { first_attempt_samples: 3, max_sec: 0.75, since_sec: 1_720_000_000 },
            stale: true,
          },
        },
        worker: { alive: true, pid: 1234, started_at_sec: 1_719_999_000 },
        device: { backend: 'nvdec', available: true, device_name: 'NVIDIA Test GPU', captured_at_sec: 1_720_000_000 },
        clip_export_applied: { enabled: true, version: 7, freshness: 'fresh' },
        clip_recorder: {
          available: true, dropped_frames: null, dropped_events: null, failed_writes: null,
          finalized_clips: 4, video_unavailable_clips: null, active_clips: null, encoder: null,
        },
        stale_after_sec: 15,
      },
    });
  });

  it('keeps null and partial runtime diagnostics available without inventing values', () => {
    expect(normalizeStatusSnapshot({
      cameras: {},
      runtime: {
        cameras: { 'cam-1': { camera_id: 'cam-1', measured_fps: 'fast' } },
        device: null,
        worker: { alive: 'yes', pid: 1.5 },
        clip_recorder: { available: 'yes' },
      },
    }).runtime).toMatchObject({
      device: null,
      worker: { alive: null, pid: null, started_at_sec: null },
      cameras: { 'cam-1': { camera_id: 'cam-1', measured_fps: null, latency: null, stale: null } },
    });
  });

  it('keeps partial records and degrades malformed fields to null/unknown', () => {
    expect(normalizeStatusSnapshot({
      cameras: {
        partial: { camera_id: 'partial', status: 'stale', age_sec: 'old' },
        malformed: { status: 'online' },
        never: { camera_id: 'never', status: 'never_seen' },
      },
      runtime: { cameras: { broken: { measured_fps: 'nope' } }, worker: null, device: null, clip_recorder: null },
    })).toEqual({
      cameras: {
        partial: { camera_id: 'partial', facility_id: null, status: 'stale', last_heartbeat_at: null, age_sec: null, config_version: null },
        never: { camera_id: 'never', facility_id: null, status: 'never_seen', last_heartbeat_at: null, age_sec: null, config_version: null },
      },
      stale_after_sec: null,
      runtime: {
        cameras: {},
        worker: null,
        device: null,
        clip_export_applied: { enabled: null, version: null, freshness: 'unknown' },
        clip_recorder: null,
        stale_after_sec: null,
      },
    });
  });
});

describe('connection freshness fields via normalizeCamera', () => {
  it('defaults never_connected to false and the timestamps to null when the backend has not shipped them yet', () => {
    const camera = normalizeCamera({ id: 'cam-1', rtsp_url_masked: 'rtsp://***' });
    expect(camera).toMatchObject({ never_connected: false, last_ok_at: null, last_probed_at: null });
  });

  it('preserves backend-provided freshness values', () => {
    const camera = normalizeCamera({
      id: 'cam-1',
      rtsp_url_masked: 'rtsp://***',
      never_connected: true,
      last_ok_at: '2026-07-20T00:00:00Z',
      last_probed_at: '2026-07-21T00:05:00Z',
    });
    expect(camera).toMatchObject({
      never_connected: true,
      last_ok_at: '2026-07-20T00:00:00Z',
      last_probed_at: '2026-07-21T00:05:00Z',
    });
  });

  it('accepts a camera registry entry that omits the freshness fields entirely', () => {
    expect(() => normalizeCameraRegistry({ registry_version: 1, cameras: [validCameraResponse] })).not.toThrow();
    const { cameras } = normalizeCameraRegistry({ registry_version: 1, cameras: [validCameraResponse] });
    expect(cameras[0]).toMatchObject({ never_connected: false, last_ok_at: null, last_probed_at: null });
  });

  it('accepts a camera registry entry with explicit freshness values', () => {
    const { cameras } = normalizeCameraRegistry({
      registry_version: 1,
      cameras: [{ ...validCameraResponse, never_connected: true, last_ok_at: null, last_probed_at: '2026-07-21T00:05:00Z' }],
    });
    expect(cameras[0]).toMatchObject({ never_connected: true, last_ok_at: null, last_probed_at: '2026-07-21T00:05:00Z' });
  });

  it.each([
    ['malformed never_connected', { ...validCameraResponse, never_connected: 'true' }],
    ['malformed last_ok_at', { ...validCameraResponse, last_ok_at: 1_720_000_000 }],
    ['malformed last_probed_at', { ...validCameraResponse, last_probed_at: false }],
  ])('rejects a registry entry with a %s instead of dropping or coercing it', (_case, camera) => {
    expect(() => normalizeCameraRegistry({ registry_version: 1, cameras: [camera] })).toThrow('Invalid camera registry response');
  });
});

describe('heartbeat freshness fields via normalizeCamera', () => {
  it('defaults last_heartbeat_at and heartbeat_age_sec to null when the backend has not shipped them yet', () => {
    const camera = normalizeCamera({ id: 'cam-1', rtsp_url_masked: 'rtsp://***' });
    expect(camera).toMatchObject({ last_heartbeat_at: null, heartbeat_age_sec: null });
  });

  it('preserves backend-provided heartbeat values, including fractional seconds', () => {
    const camera = normalizeCamera({
      id: 'cam-1',
      rtsp_url_masked: 'rtsp://***',
      last_heartbeat_at: 1_753_000_000.5,
      heartbeat_age_sec: 245.75,
    });
    expect(camera).toMatchObject({ last_heartbeat_at: 1_753_000_000.5, heartbeat_age_sec: 245.75 });
  });

  it('accepts a camera registry entry that omits the heartbeat fields entirely', () => {
    const { cameras } = normalizeCameraRegistry({ registry_version: 1, cameras: [validCameraResponse] });
    expect(cameras[0]).toMatchObject({ last_heartbeat_at: null, heartbeat_age_sec: null });
  });

  it('accepts a camera registry entry with explicit heartbeat values', () => {
    const { cameras } = normalizeCameraRegistry({
      registry_version: 1,
      cameras: [{ ...validCameraResponse, last_heartbeat_at: 1_753_000_000, heartbeat_age_sec: 12 }],
    });
    expect(cameras[0]).toMatchObject({ last_heartbeat_at: 1_753_000_000, heartbeat_age_sec: 12 });
  });

  it('treats a mutation response (POST/PATCH/test) with null heartbeat fields as normal, not an error', () => {
    expect(normalizeCameraResponse({ ...validCameraResponse, last_heartbeat_at: null, heartbeat_age_sec: null }))
      .toMatchObject({ last_heartbeat_at: null, heartbeat_age_sec: null });
  });

  it.each([
    ['malformed last_heartbeat_at', { ...validCameraResponse, last_heartbeat_at: '1753000000' }],
    ['malformed heartbeat_age_sec', { ...validCameraResponse, heartbeat_age_sec: 'stale' }],
  ])('rejects a registry entry with a %s instead of dropping or coercing it', (_case, camera) => {
    expect(() => normalizeCameraRegistry({ registry_version: 1, cameras: [camera] })).toThrow('Invalid camera registry response');
  });
});

describe('heartbeat relay status via normalizeConnectionView', () => {
  const validConnection = {
    events_url: 'https://backend.example.com/events',
    config_url: 'https://backend.example.com/config',
    facility_code: 'NH-7H2K9M4QXP',
    client_installation_ref: 'aa83ea3f-6e5f-4f45-a401-fb36c38835b6',
    facility_id: 'facility-42',
    edge_installation_id: 'c72bd9a7-3e04-47ba-a8cd-a56e54f98152',
    enrollment_generation: 3,
    facility_token_set: true,
    facility_token_masked: '****ab12',
    enrolled: true,
    configured: true,
    reachable: true,
    last_ok_at: '2026-08-01T00:00:00Z',
    updated_at: '2026-08-01T00:00:00Z',
  };

  it('preserves a healthy relay status from the backend', () => {
    const view = normalizeConnectionView({
      ...validConnection,
      heartbeat_relay: { enabled: true, last_success_at: '2026-08-02T00:00:00Z', last_error_class: null, detail: null },
    });
    expect(view.heartbeat_relay).toEqual({
      enabled: true,
      last_success_at: '2026-08-02T00:00:00Z',
      last_error_class: null,
      detail: null,
    });
  });

  it('preserves a failing relay status and its error class', () => {
    const view = normalizeConnectionView({
      ...validConnection,
      heartbeat_relay: { enabled: true, last_success_at: null, last_error_class: 'auth', detail: '인증 실패' },
    });
    expect(view.heartbeat_relay).toEqual({
      enabled: true,
      last_success_at: null,
      last_error_class: 'auth',
      detail: '인증 실패',
    });
  });

  it.each([
    ['missing entirely', undefined],
    ['non-object', 'enabled'],
    ['missing enabled', { last_success_at: null, last_error_class: null, detail: null }],
    ['malformed enabled', { enabled: 'true', last_success_at: null, last_error_class: null, detail: null }],
    ['malformed last_success_at', { enabled: true, last_success_at: 123, last_error_class: null, detail: null }],
    ['unknown last_error_class', { enabled: true, last_success_at: null, last_error_class: 'network', detail: null }],
    ['malformed nullable detail', { enabled: true, last_success_at: null, last_error_class: null, detail: 1 }],
  ])('defaults to the all-disabled relay status on a %s field, without rejecting the connection envelope', (_case, heartbeat_relay) => {
    const view = normalizeConnectionView({ ...validConnection, heartbeat_relay });
    expect(view.heartbeat_relay).toEqual({
      enabled: false,
      last_success_at: null,
      last_error_class: null,
      detail: null,
    });
  });

  it('still rejects a connection envelope with an unrelated malformed field, independent of the relay field', () => {
    expect(() => normalizeConnectionView({ ...validConnection, configured: 'yes' })).toThrow('Invalid connection response');
  });
});

describe('decode backend normalization via normalizeCamera', () => {
  it('maps a valid decode_backend value from the registry response', () => {
    const camera = normalizeCamera({ id: 'cam-1', rtsp_url: 'rtsp://camera.internal.example/stream', decode_backend: 'nvdec' });
    expect(camera?.decode_backend).toBe('nvdec');
  });

  it('defaults to null (auto) when decode_backend is absent or unset', () => {
    const camera = normalizeCamera({ id: 'cam-1', rtsp_url: 'rtsp://camera.internal.example/stream' });
    expect(camera?.decode_backend).toBeNull();
  });

  it('preserves an unknown string decode_backend value allowed by the backend contract', () => {
    const camera = normalizeCamera({ id: 'cam-1', rtsp_url: 'rtsp://camera.internal.example/stream', decode_backend: 'quicksync' });
    expect(camera?.decode_backend).toBe('quicksync');
  });
});
