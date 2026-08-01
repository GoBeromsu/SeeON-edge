import { afterEach, describe, expect, it, vi } from 'vitest';
import { cameraDuplicateDetail, cameraMediaId, cameraProbeFailureDetail, clearClipLabelOverlay, createCamera, fetchCameras, fetchClips, fetchStatus, fetchSystem, getApiBase, getCameraSnapshotUrl, getCameraStreamUrl, labelClip, loginDashboard, logoutDashboard, testCamera, updateCamera, updateCameraDecodeBackend } from '@/shared/api/client';
import { HttpError } from '@/shared/api/http';

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
    space_id: null, backend_camera_id: null, mapping_pending: false, status: 'online',
    decode_backend: null, created_at: null, space_name: null, floor_name: null, ...overrides,
  };
}

afterEach(() => {
  clearClipLabelOverlay();
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
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ cameras: {}, runtime: { facilities: {} } }) })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ clips: [] }) });
    vi.stubGlobal('fetch', fetchMock);

    await expect(fetchCameras()).resolves.toEqual({ registry_version: 0, cameras: [] });
    await expect(fetchStatus()).resolves.toEqual({ cameras: {}, stale_after_sec: null, runtime: { facilities: {}, stale_after_sec: null } });
    await expect(fetchClips()).resolves.toEqual([]);
  });

  it('serializes camera create without a user-entered id or space_id', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => cameraResponse(),
    });
    vi.stubGlobal('fetch', fetchMock);

    await createCamera({ label: ' 301호 ', rtsp_url: ' rtsp://camera/stream ' });
    await updateCamera('cam-1', { label: ' 301호 A ', rtsp_url: ' rtsp://camera/a ', space_id: ' space-302 ' });

    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/v1/cameras', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ label: '301호', rtsp_url: 'rtsp://camera/stream' }),
    }));
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/v1/cameras/cam-1', expect.objectContaining({
      method: 'PATCH',
      body: JSON.stringify({ label: '301호 A', rtsp_url: 'rtsp://camera/a', space_id: 'space-302' }),
    }));
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

  it('keeps local camera identity separate from encoded backend media URLs', () => {
    const camera = { id: 'local/cam', backend_camera_id: 'worker/cam' };

    expect(cameraMediaId(camera)).toBe('worker/cam');
    expect(getCameraSnapshotUrl(cameraMediaId(camera), 'refresh &=1')).toBe('/api/v1/streams/worker%2Fcam/snapshot?refresh=refresh%20%26%3D1');
    expect(getCameraStreamUrl(cameraMediaId(camera))).toBe('/api/v1/streams/worker%2Fcam');
    expect(camera.id).toBe('local/cam');
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

  it('preserves clip metadata and reapplies a confirmed label overlay across polling', async () => {
    const manifest = clipManifest({ camera_id: 'worker-1', video_error: 'encoder failed' });
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ clips: [manifest] }) })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ clip_id: 'clip-1', label: 'TRUE_POSITIVE', reviewer: 'operator', reviewed_at: '2026-07-07T00:00:00Z' }) })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ clips: [manifest] }) });
    vi.stubGlobal('fetch', fetchMock);

    const original = (await fetchClips())[0]!;
    const labelled = await labelClip(original, 'TRUE_POSITIVE');
    const polled = (await fetchClips())[0]!;

    expect(labelled).toEqual(expect.objectContaining({
      id: 'clip-1', camera_id: 'worker-1', event_type: 'fall', created_at: '2026-07-06T00:00:00Z',
      video_available: false, video_error: '저장된 영상을 사용할 수 없습니다.', label: 'TRUE_POSITIVE', reviewer: 'operator',
      reviewed_at: '2026-07-07T00:00:00Z', reviewState: 'confirmed',
    }));
    expect(polled).toEqual(labelled);
  });

  it('preserves metadata for a partial explicit null-label response and a reload returns to unknown', async () => {
    const manifest = clipManifest({ clip_id: 'clip-null', camera_id: 'cam-a', event_type: 'bed-exit' });
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ clips: [manifest] }) })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ clip_id: 'clip-null', label: null, reviewer: 'operator', reviewed_at: '2026-07-07T00:00:00Z' }) })
      .mockResolvedValue({ ok: true, status: 200, json: async () => ({ clips: [manifest] }) });
    vi.stubGlobal('fetch', fetchMock);

    const original = (await fetchClips())[0]!;
    const cleared = await labelClip(original, 'UNREVIEWED');
    expect(cleared).toMatchObject({ camera_id: 'cam-a', event_type: 'bed-exit', label: null, reviewer: 'operator', reviewState: 'confirmed' });
    expect((await fetchClips())[0]).toMatchObject({ label: null, reviewer: 'operator', reviewState: 'confirmed' });

    clearClipLabelOverlay();
    expect((await fetchClips())[0]).toMatchObject({ label: null, reviewer: null, reviewed_at: null, reviewState: 'unknown' });
  });
});
