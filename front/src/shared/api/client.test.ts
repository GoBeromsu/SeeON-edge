import { afterEach, describe, expect, it, vi } from 'vitest';
import { bedZoneRecognitionFailureDetail, browseClipStorage, cameraDuplicateDetail, cameraProbeFailureDetail, createCamera, fetchCameraOverlay, fetchCameras, fetchClipArtifacts, fetchClips, fetchClipStorage, fetchDetectionSettings, fetchRuntimeSettings, fetchStatus, fetchSystem, getApiBase, getCameraSnapshotUrl, getCameraStreamUrl, loginDashboard, logoutDashboard, recognizeBedZone, saveClipStorageLocation, saveConnection, saveDetectionSettings, saveRuntimeSettings, setCameraOverlay, testCamera, testConnection, updateCamera, updateCameraDecodeBackend } from '@/shared/api/client';
import { HttpError } from '@/shared/api/http';
import type { DetectionSettings } from '@/shared/api/client';

function clipManifest(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    clip_id: 'clip-1', camera_id: 'cam-1', event_ref: 'event-1', event_type: 'fall',
    started_at: '2026-07-06T00:00:00Z', duration_s: 4.2, codec: '', path: null,
    video_available: false, video_error: null, finalized: true, ...overrides,
  };
}

function cameraResponse(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id: 'cam-1', label: '301호', rtsp_url_masked: 'rtsp://***@redacted-camera/stream',
    mapping_pending: false, status: 'online',
    decode_backend: null, created_at: null, floor_name: null, ...overrides,
  };
}

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe('api client contracts', () => {
  it.each([
    ['cameras', fetchCameras, { registry_version: 0 }],
    ['status', fetchStatus, { cameras: {}, runtime: null }],
    ['clips', fetchClips, { clips: null }],
    ['system', fetchSystem, { backend: { configured: true } }],
    ['camera test', () => testCamera('cam-1'), { error_class: 'timeout' }],
  ])('rejects a contract-invalid %s success envelope', async (_name, load, payload) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => payload,
    }));

    await expect(load()).rejects.toThrow(/Invalid .* response/);
  });

  it('keeps valid empty list envelopes as successful empty data', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ registry_version: 0, cameras: [] }) })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ cameras: {}, runtime: { cameras: {} } }) })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ clips: [] }) });
    vi.stubGlobal('fetch', fetchMock);

    await expect(fetchCameras()).resolves.toEqual({ registry_version: 0, cameras: [] });
    await expect(fetchStatus()).resolves.toEqual({
      cameras: {},
      stale_after_sec: null,
      runtime: { cameras: {}, worker: null, device: null, clip_export_applied: { enabled: null, version: null, freshness: 'unknown' }, clip_recorder: null, stale_after_sec: null },
    });
    await expect(fetchClips()).resolves.toEqual([]);
  });

  it('serializes camera create and patch without a user-entered id', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => cameraResponse(),
    });
    vi.stubGlobal('fetch', fetchMock);

    await createCamera({ label: ' 301호 ', rtsp_url: ' rtsp://camera/stream ' });
    await updateCamera('cam-1', { label: ' 301호 A ', rtsp_url: ' rtsp://camera/a ' });

    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/v1/cameras', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ label: '301호', rtsp_url: 'rtsp://camera/stream' }),
    }));
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/v1/cameras/cam-1', expect.objectContaining({
      method: 'PATCH',
      body: JSON.stringify({ label: '301호 A', rtsp_url: 'rtsp://camera/a' }),
    }));
  });

  it('serializes enrollment as exactly code, token, and stable installation reference', async () => {
    const connection = {
      events_url: 'https://api.example/events', config_url: 'https://api.example/config',
      facility_code: 'NH-7H2K9M4QXP', client_installation_ref: 'aa83ea3f-6e5f-4f45-a401-fb36c38835b6',
      facility_id: 'facility-42', edge_installation_id: 'c72bd9a7-3e04-47ba-a8cd-a56e54f98152',
      enrollment_generation: 3, facility_token_set: true, facility_token_masked: '****ab12', enrolled: true,
      configured: true, reachable: true, last_ok_at: null, updated_at: null,
      heartbeat_relay: { enabled: false, last_success_at: null, last_error_class: null, detail: null },
    };
    const verified = { ok: true, error_class: null, detail: 'ok', facility_id: 'facility-42', edge_installation_id: connection.edge_installation_id, enrollment_generation: 3 };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => connection })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => verified });
    vi.stubGlobal('fetch', fetchMock);
    const input = { facility_code: 'NH-7H2K9M4QXP', facility_token: 'opaque-secret', client_installation_ref: connection.client_installation_ref };

    await saveConnection(input);
    await testConnection(input);

    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/v1/connection', expect.objectContaining({ method: 'PUT', body: JSON.stringify(input) }));
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/v1/connection/test', expect.objectContaining({ method: 'POST', body: JSON.stringify(input) }));
  });

  it.each([
    ['POST', () => createCamera({ label: '301호', rtsp_url: 'rtsp://camera/stream' }), cameraResponse({ label: undefined })],
    ['PATCH', () => updateCamera('cam-1', { label: '301호 A' }), cameraResponse({ decode_backend: false })],
  ])('rejects a malformed camera %s success response instead of fabricating mutation data', async (_method, mutate, payload) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => payload }));

    await expect(mutate()).rejects.toThrow('Invalid camera response');
  });

  it('omits force_register from the create body by default and includes it only when requested', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => cameraResponse() });
    vi.stubGlobal('fetch', fetchMock);

    await createCamera({ label: '301호', rtsp_url: 'rtsp://camera/stream' });
    await createCamera({ label: '301호', rtsp_url: 'rtsp://camera/stream' }, { forceRegister: true });

    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/v1/cameras', expect.objectContaining({
      body: JSON.stringify({ label: '301호', rtsp_url: 'rtsp://camera/stream' }),
    }));
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/v1/cameras', expect.objectContaining({
      body: JSON.stringify({ label: '301호', rtsp_url: 'rtsp://camera/stream', force_register: true }),
    }));
  });

  it('surfaces the 409 duplicate_camera body through requestJson so the caller can read it', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 409,
      json: async () => ({ detail: { error: 'duplicate_camera', existing_camera_id: 'cam-2', existing_label: '302호' } }),
    }));

    const error = await createCamera({ label: '301호', rtsp_url: 'rtsp://camera/stream' }).catch((caught: unknown) => caught);

    expect(error).toBeInstanceOf(HttpError);
    expect((error as HttpError).status).toBe(409);
    expect(cameraDuplicateDetail(error)).toEqual({ error: 'duplicate_camera', existing_camera_id: 'cam-2', existing_label: '302호' });
    expect(cameraProbeFailureDetail(error)).toBeUndefined();
  });

  it('surfaces the 422 probe_failed body through requestJson so the caller can read it', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({ detail: { error: 'probe_failed', error_class: 'auth' } }),
    }));

    const error = await createCamera({ label: '301호', rtsp_url: 'rtsp://camera/stream' }).catch((caught: unknown) => caught);

    expect(cameraProbeFailureDetail(error)).toEqual({ error: 'probe_failed', error_class: 'auth' });
    expect(cameraDuplicateDetail(error)).toBeUndefined();
  });

  it('treats a non-matching or body-less HTTP error as neither duplicate nor probe failure', async () => {
    expect(cameraDuplicateDetail(new HttpError(500))).toBeUndefined();
    expect(cameraProbeFailureDetail(new HttpError(500))).toBeUndefined();
    expect(cameraDuplicateDetail(new Error('network down'))).toBeUndefined();
    expect(cameraDuplicateDetail(new HttpError(409, { detail: { error: 'something_else' } }))).toBeUndefined();
    expect(cameraProbeFailureDetail(new HttpError(422, { detail: { error: 'something_else' } }))).toBeUndefined();
  });

  it('exposes the dashboard API base for operator-facing backend copy', () => {
    expect(getApiBase()).toBe('/api/v1');
  });

  it('routes requests, stream URLs, and clip URLs through the configured ML API base', async () => {
    vi.stubEnv('VITE_ML_API_BASE_URL', ' http://edge-ml-api.local:8000/api/v1/ ');
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ ok: true }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ clips: [clipManifest({ clip_id: 'clip/1', camera_id: 'cam/1', event_type: 'bed-exit' })] }),
      });
    vi.stubGlobal('fetch', fetchMock);

    await testCamera('cam/1');
    const clips = await fetchClips('cam/1');

    expect(getApiBase()).toBe('http://edge-ml-api.local:8000/api/v1');
    expect(fetchMock).toHaveBeenNthCalledWith(1, 'http://edge-ml-api.local:8000/api/v1/cameras/cam%2F1/test', expect.objectContaining({ method: 'POST' }));
    expect(fetchMock).toHaveBeenNthCalledWith(2, 'http://edge-ml-api.local:8000/api/v1/clips?camera_id=cam%2F1', expect.objectContaining({
      headers: expect.objectContaining({ Accept: 'application/json' }),
    }));
    expect(getCameraStreamUrl('cam/1')).toBe('http://edge-ml-api.local:8000/api/v1/streams/cam%2F1');
    expect(clips[0]?.video_path).toBe('http://edge-ml-api.local:8000/api/v1/clips/clip%2F1/video');
  });

  it('sends dashboard credentials only to the server session endpoint', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 204 });
    vi.stubGlobal('fetch', fetchMock);

    await loginDashboard('operator', 'secret');
    await logoutDashboard();

    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/v1/auth/session', expect.objectContaining({
      method: 'POST',
      credentials: 'same-origin',
      body: JSON.stringify({ username: 'operator', password: 'secret' }),
    }));
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/v1/auth/session', expect.objectContaining({
      method: 'DELETE',
      credentials: 'same-origin',
    }));
  });

  it('posts camera connection tests to the registered camera id endpoint', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ ok: true }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await testCamera('cam-1');

    expect(fetchMock).toHaveBeenCalledWith('/api/v1/cameras/cam-1/test', expect.objectContaining({ method: 'POST' }));
  });

  it('fetches the current per-camera overlay mode', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ mode: 'fall' }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await expect(fetchCameraOverlay('cam/1')).resolves.toBe('fall');

    expect(fetchMock).toHaveBeenCalledWith('/api/v1/streams/cam%2F1/pose', expect.objectContaining({ credentials: 'same-origin' }));
    expect(fetchMock.mock.calls[0]?.[1]?.method).toBeUndefined();
  });

  it('posts the requested overlay mode and returns the confirmed value', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ mode: 'none' }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await expect(setCameraOverlay('cam-1', 'none')).resolves.toBe('none');

    expect(fetchMock).toHaveBeenCalledWith('/api/v1/streams/cam-1/pose', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ mode: 'none' }),
    }));
  });

  it('rejects a contract-invalid overlay response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ mode: 'yes' }),
    }));

    await expect(fetchCameraOverlay('cam-1')).rejects.toThrow('Invalid overlay response');
  });

  it('normalizes clip video URLs without embedding credentials in the query string', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ clips: [clipManifest({ clip_id: 'clip/1', camera_label: '301호', video_path: '/ignored' })] }),
    });
    vi.stubGlobal('fetch', fetchMock);

    const clips = await fetchClips();

    expect(clips[0]?.video_path).toBe('/api/v1/clips/clip%2F1/video');
  });

  it('fetches clips scoped to the selected camera', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ clips: [] }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await fetchClips('cam/1');

    expect(fetchMock).toHaveBeenCalledWith('/api/v1/clips?camera_id=cam%2F1', expect.objectContaining({
      headers: expect.objectContaining({ Accept: 'application/json' }),
    }));
  });

  it('builds the real ml-api camera stream URL without a credential query', () => {
    expect(getCameraStreamUrl('cam/1')).toBe('/api/v1/streams/cam%2F1');
  });

  it('builds a camera snapshot URL without a credential query', () => {
    expect(getCameraSnapshotUrl('cam/1', 'refresh &=1')).toBe('/api/v1/streams/cam%2F1/snapshot?refresh=refresh%20%26%3D1');
  });

  it('normalizes the real ml-api clip manifest shape for bed-exit playback', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        clips: [{
          clip_id: 'clip-1',
          camera_id: 'cam-1',
          event_ref: '0:0',
          event_type: 'bed-exit',
          started_at: '2026-07-06T00:00:00Z',
          duration_s: 4.2,
          codec: 'mp4v',
          path: 'clips/clip-1/clip.mp4',
          finalized: true,
          video_available: true,
        }],
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    const clips = await fetchClips();

    expect(clips[0]).toMatchObject({
      id: 'clip-1',
      camera_id: 'cam-1',
      event_type: 'bed-exit',
      created_at: '2026-07-06T00:00:00Z',
      video_path: '/api/v1/clips/clip-1/video',
      video_available: true,
      video_error: null,
    });
  });

  it('patches only decode_backend when changing a camera decode backend', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => cameraResponse({ decode_backend: 'nvdec' }),
    });
    vi.stubGlobal('fetch', fetchMock);

    const updated = await updateCameraDecodeBackend('cam-1', 'nvdec');

    expect(fetchMock).toHaveBeenCalledWith('/api/v1/cameras/cam-1', expect.objectContaining({
      method: 'PATCH',
      body: JSON.stringify({ decode_backend: 'nvdec' }),
    }));
    expect(updated.decode_backend).toBe('nvdec');
  });

  it('recognizes a bed zone via a POST to the camera-scoped endpoint', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        bed_zone: {
          polygon: [[0, 0], [10, 0], [10, 10], [0, 10]],
          image_width: 1920,
          image_height: 1080,
          recognized_at: '2026-08-02T00:00:00Z',
        },
      }),
    });
    vi.stubGlobal('fetch', fetchMock);

    const bedZone = await recognizeBedZone('cam-1');

    expect(fetchMock).toHaveBeenCalledWith('/api/v1/cameras/cam-1/bed-zone/recognize', expect.objectContaining({
      method: 'POST',
    }));
    expect(bedZone).toEqual({
      polygon: [[0, 0], [10, 0], [10, 10], [0, 10]],
      image_width: 1920,
      image_height: 1080,
      recognized_at: '2026-08-02T00:00:00Z',
    });
  });

  it('rejects a malformed bed-zone recognition response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({ bed_zone: null }) }));

    await expect(recognizeBedZone('cam-1')).rejects.toThrow('Invalid bed-zone recognition response');
  });

  it('surfaces a 422 bed_not_found detail through bedZoneRecognitionFailureDetail, but not other errors', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({ detail: { error_class: 'bed_not_found' } }),
    }));

    const error = await recognizeBedZone('cam-1').catch((caught: unknown) => caught);

    expect(bedZoneRecognitionFailureDetail(error)).toEqual({ error_class: 'bed_not_found' });
    expect(bedZoneRecognitionFailureDetail(new HttpError(503))).toBeUndefined();
    expect(bedZoneRecognitionFailureDetail(new Error('network down'))).toBeUndefined();
  });

  it('fetches and saves detection settings with a full-replace PUT body', async () => {
    const settings: DetectionSettings = {
      domains: {
        fall: { on: true, mode: 'always', start: null, end: null },
        bed_exit: { on: true, mode: 'window', start: '21:00', end: '06:00' },
      },
    };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => settings });
    vi.stubGlobal('fetch', fetchMock);

    await expect(fetchDetectionSettings()).resolves.toEqual(settings);
    await saveDetectionSettings(settings);

    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/v1/detection-settings', expect.objectContaining({ credentials: 'same-origin' }));
    expect(fetchMock.mock.calls[0]?.[1]?.method).toBeUndefined();
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/v1/detection-settings', expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify(settings),
    }));
  });

  it('rejects a contract-invalid detection settings response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({ domains: null }) }));

    await expect(fetchDetectionSettings()).rejects.toThrow('Invalid detection settings response');
  });

    it('fetches and explicitly saves runtime clip export settings', async () => {
    const setting = { clip_export_enabled: false, version: 0 };
    const saved = { clip_export_enabled: true, version: 1 };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => setting })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => saved });
    vi.stubGlobal('fetch', fetchMock);

    await expect(fetchRuntimeSettings()).resolves.toEqual(setting);
    await expect(
      saveRuntimeSettings({ clip_export_enabled: true, expected_version: 0 }),
    ).resolves.toEqual(saved);

    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/v1/runtime-settings', expect.objectContaining({ credentials: 'same-origin' }));
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/v1/runtime-settings', expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify({ clip_export_enabled: true, expected_version: 0 }),
    }));
  });

  it('fetches clip storage info, browses a relative subdirectory, and saves a new location', async () => {
    const storageInfo = { mount_label: 'clip-store', selected_path: '', total_bytes: 100, used_bytes: 10, used_pct: 10 };
    // The browse endpoint's "" parent means "at the mount root" -- pickNullableString normalizes that to null.
    const browseResult = { path: 'cam-101a', parent: '', directories: [] };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => storageInfo })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => browseResult })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ ...storageInfo, selected_path: 'cam-101a' }) });
    vi.stubGlobal('fetch', fetchMock);

    await expect(fetchClipStorage()).resolves.toEqual(storageInfo);
    await expect(browseClipStorage('cam-101a')).resolves.toEqual({ ...browseResult, parent: null });
    await expect(saveClipStorageLocation('cam-101a')).resolves.toEqual({ ...storageInfo, selected_path: 'cam-101a' });

    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/v1/clips/storage', expect.objectContaining({ credentials: 'same-origin' }));
    expect(fetchMock.mock.calls[0]?.[1]?.method).toBeUndefined();
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/v1/clips/storage/browse?path=cam-101a', expect.objectContaining({ credentials: 'same-origin' }));
    expect(fetchMock.mock.calls[1]?.[1]?.method).toBeUndefined();
    expect(fetchMock).toHaveBeenNthCalledWith(3, '/api/v1/clips/storage/location', expect.objectContaining({
      method: 'PUT',
      body: JSON.stringify({ path: 'cam-101a' }),
    }));
  });

  it('browses the mount root without a query string when path is empty', async () => {
    const browseResult = { path: '', parent: null, directories: [] };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => browseResult });
    vi.stubGlobal('fetch', fetchMock);

    await browseClipStorage('');

    expect(fetchMock).toHaveBeenCalledWith('/api/v1/clips/storage/browse', expect.objectContaining({ credentials: 'same-origin' }));
    expect(fetchMock.mock.calls[0]?.[1]?.method).toBeUndefined();
  });

  it('rejects a contract-invalid clip storage response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({ selected_path: '' }) }));

    await expect(fetchClipStorage()).rejects.toThrow('Invalid clip storage response');
  });


  it('reads the slimmed clip artifacts contract', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ clip_id: 'clip/1', clean: 'AVAILABLE', snapshot: 'PENDING' }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await expect(fetchClipArtifacts('clip/1')).resolves.toEqual({
      clip_id: 'clip/1', clean: 'AVAILABLE', snapshot: 'PENDING',
    });
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/clips/clip%2F1/artifacts',
      expect.objectContaining({ credentials: 'same-origin' }),
    );
  });

  it('defaults an omitted snapshot artifact to null instead of inventing a state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true, status: 200, json: async () => ({ clip_id: 'clip-1', clean: 'UNAVAILABLE' }),
    }));

    await expect(fetchClipArtifacts('clip-1')).resolves.toEqual({
      clip_id: 'clip-1', clean: 'UNAVAILABLE', snapshot: null,
    });
  });

  it.each([
    [{ clip_id: 'clip-1', clean: 'CORRUPT' }],
    [{ clip_id: 'clip-1', clean: 'AVAILABLE', analysis: 'AVAILABLE', annotated: 'UNAVAILABLE', playback_view: 'annotated' }],
    [{ clip_id: '', clean: 'AVAILABLE' }],
    [{ clean: 'AVAILABLE' }],
  ])('rejects a retired or contract-invalid clip artifacts response %#', async (payload) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => payload }));

    await expect(fetchClipArtifacts('clip-1')).rejects.toThrow('Invalid clip artifacts response');
  });
});
