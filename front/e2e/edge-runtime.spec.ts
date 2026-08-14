import { mkdirSync } from 'node:fs';
import { expect, test, type Page, type Route } from '@playwright/test';

const qaDirectory = '/tmp/edge-runtime-architecture-qa';
const viewports = [
  { width: 1440, height: 900 },
  { width: 768, height: 1024 },
  { width: 390, height: 844 },
] as const;

function captureUnexpectedConsoleErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('console', (message) => {
    // An intentionally fulfilled 409 is a scenario assertion, not an application console fault.
    if (message.type() === 'error' && !/^Failed to load resource: the server responded with a status of 409/.test(message.text())) errors.push(message.text());
  });
  return errors;
}

function capturePageErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  return errors;
}

const cameras = {
  registry_version: 7,
  cameras: [{
    // Long masked RTSPS path reproduces QA F1 mobile overflow pressure (live settings @ 390×844).
    id: 'cam-1', backend_camera_id: 'cam-1', label: '서울 301호',
    rtsp_url_masked: 'rtsps://redacted-camera:322/Streaming/Channels/101',
    floor_name: null, status: 'online', created_at: '2026-08-01T00:00:00Z',
  }],
};

const clip = {
  clip_id: 'clip-1', camera_id: 'cam-1', event_ref: 'event-1', event_type: 'fall',
  started_at: '2026-08-02T03:12:00Z', duration_s: 12, codec: 'h264', path: null,
  video_available: false, thumbnail_available: false, video_error: '증거 파일이 손상되었습니다.', finalized: true,
};

const policy = {
  module_id: 'fall', module_version: 1, schema_id: 'fall-policy', schema_version: 1,
  source: 'camera-override', facility_revision_id: 4, camera_revision_id: 8,
  values: { threshold: 0.75 }, effective_policy_id: 'fall.v1:camera-override:8',
};

async function installOperatorBackend(page: Page): Promise<{ reviews: Array<Record<string, unknown>>; requests: Array<{ path: string; method: string; body: unknown }> }> {
  const reviews: Array<Record<string, unknown>> = [];
  const requests: Array<{ path: string; method: string; body: unknown }> = [];
  let policyApplyAttempts = 0;
  const derivativeStates: Record<string, string> = { still: 'NOT_REQUESTED', video: 'UNAVAILABLE' };
  await page.route('**/api/v1/**', async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const json = (body: unknown, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
    requests.push({ path, method: request.method(), body: request.postData() ? request.postDataJSON() : null });

    if (path.endsWith('/auth/session')) return json({});
    if (path.endsWith('/cameras')) return json(cameras);
    if (path.endsWith('/clips')) return json({ clips: [clip], pagination: { limit: 48, offset: 0, total: 1, has_more: false }, event_type_counts: { fall: 1 } });
    if (path.endsWith('/clips/clip-1/artifacts')) return json({ clip_id: 'clip-1', clean: 'CORRUPT', analysis: 'AVAILABLE', annotated: 'UNAVAILABLE', playback_view: 'clean', annotated_fallback_to_clean: true });
    if (path.endsWith('/clips/clip-1/analysis')) return json({ clip_id: 'clip-1', decision_trace_id: 'c'.repeat(64), module_qualified_id: 'fall.v1', policy_qualified_id: 'fall-policy.v1', effective_policy_id: 'd'.repeat(64), runtime_manifest_sha256: 'a'.repeat(64), reason: 'fall-onset', previous_state: 'clear', current_state: 'fall', triggered: true, track_id: 7, bed_id: null, values: [] });
    if (path.includes('/clips/clip-1/derivatives/')) {
      const kind = path.endsWith('/still') ? 'still' : 'video';
      if (request.method() === 'POST') derivativeStates[kind] = 'QUEUED';
      if (request.method() === 'DELETE') derivativeStates[kind] = 'CANCELLED';
      return json({ incident_id: 'incident-1', kind: kind.toUpperCase(), request_id: 'f'.repeat(64), state: derivativeStates[kind], reason: derivativeStates[kind] === 'UNAVAILABLE' ? '원본 증거가 없습니다.' : null, attempt_count: 1, render_backend: null, primary_clip_id: 'clip-1', decision_trace_id: 'c'.repeat(64), runtime_manifest_sha256: 'a'.repeat(64) }, 202);
    }
    if (path.endsWith('/incidents')) {
      return json({ incidents: [{
        incident_id: 'incident-1', edge_event_id: 'event-1', camera_id: 'cam-1', event_type: 'fall',
        detected_at: '2026-08-02T03:12:00Z', lifecycle_state: 'COMPLETE', revision: 3,
        failure_reason: null, runtime_manifest_sha256: 'a'.repeat(64), decision_trace_id: 'trace-한글',
        module_qualified_id: 'fall.v1', policy_qualified_id: 'fall-policy.v1', primary_clip_id: 'clip-1',
        primary_artifact_state: 'CORRUPT', snapshot_artifact_state: 'MISSING', derivative_state: 'UNAVAILABLE',
        event_delivery_state: 'DELIVERED', clip_publish_state: 'TRUNCATED', retention_state: 'RETAINED', review: null,
      }] });
    }
    if (path.endsWith('/incident-reviews/incident-1')) {
      const body = request.postDataJSON() as Record<string, unknown>;
      reviews.push(body);
      return json({
        incident_id: 'incident-1', edge_event_id: 'event-1', camera_id: 'cam-1', event_type: 'fall',
        detected_at: '2026-08-02T03:12:00Z', lifecycle_state: 'COMPLETE', revision: 3,
        failure_reason: null, runtime_manifest_sha256: 'a'.repeat(64), decision_trace_id: 'trace-한글',
        module_qualified_id: 'fall.v1', policy_qualified_id: 'fall-policy.v1', primary_clip_id: 'clip-1',
        primary_artifact_state: 'CORRUPT', snapshot_artifact_state: 'MISSING', derivative_state: 'UNAVAILABLE',
        event_delivery_state: 'DELIVERED', clip_publish_state: 'TRUNCATED', retention_state: 'RETAINED',
        review: { version: 1, disposition: body.disposition, reviewed_at: '2026-08-02T03:13:00Z', notes: null },
      });
    }
    if (path.endsWith('/detection-policies')) return json({
      activation_generation: 12,
      modules: [{ qualified_id: 'fall.v1', policy_qualified_id: 'fall-policy.v1', units: { threshold: 'score' } }],
      effective: { defaults: { fall: { ...policy, source: 'facility-default', camera_revision_id: null } }, cameras: { 'cam-1': { fall: policy } } },
      activations: [{ module_id: 'fall', module_version: 1, camera_id: 'cam-1', active_revision_id: 8, previous_revision_id: 7, activation_generation: 12, status: 'pending', refusal_reason: null }],
    });
    if (path.endsWith('/detection-policies/diff')) {
      const body = (request.postData() ? request.postDataJSON() : null) as Record<string, unknown> | null;
      return json({
        changed: true,
        current: policy,
        proposed: { ...policy, camera_revision_id: 0, values: body?.values ?? { threshold: 0.8 } },
        compared_payload: body,
        concurrency_token: typeof body?.camera_id === 'string' ? 8 : 4,
      });
    }
    if (path.endsWith('/detection-policies/apply')) {
      policyApplyAttempts += 1;
      return policyApplyAttempts === 1 ? json({ detail: 'stale revision' }, 409) : json({}, 202);
    }
    if (path.endsWith('/detection-policies/rollback')) return json({}, 202);
    if (path.endsWith('/detection-settings')) return json({ domains: { fall: { on: true, mode: 'always', start: null, end: null }, bed_exit: { on: true, mode: 'always', start: null, end: null } } });
    if (path.endsWith('/connection')) return json({ events_url: null, config_url: null, facility_code: 'SEOUL', client_installation_ref: null, facility_id: 'facility-1', edge_installation_id: 'edge-1', enrollment_generation: 12, facility_token_set: true, facility_token_masked: '***', enrolled: true, configured: true, reachable: true, last_ok_at: null, updated_at: null });
    if (path.endsWith('/status')) return json({ cameras: {}, stale_after_sec: null, runtime: { cameras: {}, worker: { alive: true, pid: 1, started_at_sec: 1 }, device: { backend: 'cpu', available: true, device_name: 'Intel CPU', captured_at_sec: 1 }, clip_recorder: null, clip_export_applied: { enabled: false, version: 0, freshness: 'fresh' }, stale_after_sec: null }, runtime_settings: { clip_export_enabled: false, version: 0 } });
    if (path.endsWith('/runtime-settings')) return json({ clip_export_enabled: false, version: 0 });
    if (path.endsWith('/system')) return json({ backend: { configured: true, reachable: true, last_ok_at: null }, version: 'test' });
    if (path.endsWith('/clips/storage')) return json({ root: '/evidence', selected_path: '', total_bytes: 1, used_bytes: 0, used_pct: 0 });
    return json({});
  });
  return { reviews, requests };
}

type SetupBackend = { requests: Array<{ path: string; method: string; body: unknown }> };

/** Stateful fixture: every visible wizard state is the server state returned after the action. */
async function installSetupBackend(page: Page): Promise<SetupBackend> {
  const requests: SetupBackend['requests'] = [];
  let enrolled = false;
  let synced = false;
  let syncAttempts = 0;
  let confirmAttempts = 0;
  const preview = {
    confirmation_id: 'preview-1', digest: 'b'.repeat(64), expires_at: '2099-08-02T03:12:00Z',
    snapshot_id: 'snapshot-1', client_revision: 7, server_revision: 9, cameras: 1, rooms: 0, floors: 0, confirmed: false,
  };
  const json = (route: Route, body: unknown, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
  await page.route('**/api/v1/**', async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const body = request.postData() ? request.postDataJSON() : null;
    requests.push({ path, method: request.method(), body });
    if (path.endsWith('/auth/session')) return json(route, {});
    if (path.endsWith('/connection') && request.method() === 'GET') return json(route, {
      events_url: 'https://hub.example.test', config_url: null, facility_code: enrolled ? 'NH-1234567890' : null,
      client_installation_ref: null, facility_id: enrolled ? 'facility-한글' : null, edge_installation_id: enrolled ? 'edge-1' : null,
      enrollment_generation: enrolled ? 1 : null, facility_token_set: enrolled, facility_token_masked: enrolled ? '***' : null,
      enrolled, configured: enrolled, reachable: enrolled, last_ok_at: null, updated_at: null,
    });
    if (path.endsWith('/connection') && request.method() === 'PUT') {
      if (body?.facility_code !== 'NH-1234567890' || body?.facility_token !== 'field-only-token') return json(route, { detail: 'bad request' }, 422);
      enrolled = true;
      return json(route, { events_url: 'https://hub.example.test', config_url: null, facility_code: 'NH-1234567890', client_installation_ref: body.client_installation_ref, facility_id: 'facility-한글', edge_installation_id: 'edge-1', enrollment_generation: 1, facility_token_set: true, facility_token_masked: '***', enrolled: true, configured: true, reachable: true, last_ok_at: null, updated_at: null });
    }
    if (path.endsWith('/connection/test')) return json(route, { ok: false, error_class: 'auth', detail: '토큰이 올바르지 않습니다.', facility_id: null, edge_installation_id: null, enrollment_generation: null });
    if (path.endsWith('/cameras/topology')) return json(route, { registry_version: 7, dirty_registry_version: synced ? null : 7, readiness_error: null, unmapped_camera_ids: [], floors: [{ edge_ref: 'floor-3', name: '3층', order_index: 3, rooms: [{ edge_ref: 'room-301', name: '301호', room_type: 'ROOM', capacity: 1, legacy_canonical_space_id: null, cameras: [{ edge_ref: 'cam-1', label: '서울 301호' }] }] }] });
    if (path.endsWith('/connection/topology-preview')) return json(route, { preview: synced ? preview : null });
    if (path.endsWith('/connection/sync-cameras')) {
      syncAttempts += 1;
      if (syncAttempts === 1) return json(route, { status: 'failed', error_class: 'conflict', detail: 'stale', last_ok_at: null, next_retry_at: null, camera_count: 1 });
      synced = true;
      return json(route, { status: 'synced', error_class: null, detail: null, last_ok_at: '2026-08-02T03:12:00Z', next_retry_at: null, camera_count: 1 });
    }
    if (path.endsWith('/connection/topology-preview/confirm')) {
      confirmAttempts += 1;
      if (confirmAttempts === 1) return json(route, { detail: 'conflict' }, 409);
      preview.confirmed = true;
      return json(route, { snapshot_id: 'snapshot-1', client_revision: 7, server_revision: 10 });
    }
    if (path.endsWith('/cameras')) return json(route, cameras);
    if (path.endsWith('/detection-policies')) return json(route, { activation_generation: 1, modules: [], effective: { defaults: {}, cameras: {} }, activations: [] });
    if (path.endsWith('/detection-settings')) return json(route, { domains: { fall: { on: true, mode: 'always', start: null, end: null }, bed_exit: { on: true, mode: 'always', start: null, end: null } } });
    if (path.endsWith('/status')) return json(route, { cameras: {}, stale_after_sec: null, runtime: { cameras: {}, worker: { alive: true, pid: 1, started_at_sec: 1 }, device: { backend: 'cpu', available: true, device_name: 'Intel CPU', captured_at_sec: 1 }, clip_recorder: null, clip_export_applied: { enabled: false, version: 0, freshness: 'fresh' }, stale_after_sec: null }, runtime_settings: { clip_export_enabled: false, version: 0 } });
    if (path.endsWith('/runtime-settings')) return json(route, { clip_export_enabled: false, version: 0 });
    if (path.endsWith('/system')) return json(route, { backend: { configured: true, reachable: true, last_ok_at: null }, version: 'test' });
    if (path.endsWith('/clips/storage')) return json(route, { root: '/evidence', selected_path: '', total_bytes: 1, used_bytes: 0, used_pct: 0 });
    return json(route, {});
  });
  return { requests };
}

test('three-step field setup keeps inputs local, exposes conflicts, and confirms the exact server snapshot', async ({ page }) => {
  const consoleErrors = captureUnexpectedConsoleErrors(page);
  const pageErrors = capturePageErrors(page);
  const backend = await installSetupBackend(page);

  await page.goto('/?page=settings');
  await expect(page.getByRole('heading', { name: '설정', exact: true })).toBeVisible();
  await expect(page.getByText('현장에서는 시설 코드와 토큰만 입력합니다.')).toBeVisible();
  await expect(page.getByLabel('시설 코드')).toBeVisible();
  await expect(page.getByLabel('시설 토큰')).toBeVisible();
  await page.getByLabel('시설 코드').fill('NH-1234567890');
  await page.getByLabel('시설 토큰').fill('field-only-token');
  await page.getByRole('button', { name: '등록 확인' }).click();
  await expect(page.getByRole('alert')).toContainText('토큰이 올바르지 않습니다.');
  await page.getByRole('button', { name: '등록 저장' }).click();
  await expect.poll(() => backend.requests.some((entry) => entry.path.endsWith('/connection') && entry.method === 'PUT')).toBe(true);
  expect(backend.requests.find((entry) => entry.path.endsWith('/connection') && entry.method === 'PUT')?.body).toMatchObject({ facility_code: 'NH-1234567890', facility_token: 'field-only-token' });

  await expect(page.getByText('서버 상태와 어긋났습니다. 화면을 새로고침한 뒤 다시 시도하세요.')).toBeVisible({ timeout: 1_000 }).catch(() => undefined);
  const sync = page.getByRole('button', { name: '카메라 동기화' });
  await expect(sync).toBeVisible();
  await sync.click();
  await expect(page.getByRole('alert')).toContainText('서버 상태와 어긋났습니다.');
  await sync.click();
  await expect(page.getByText('카메라 목록을 서버와 동기화했습니다.')).toBeVisible();

  await page.getByRole('button', { name: '변경 사항 확인 후 반영' }).click();
  await expect(page.getByRole('dialog', { name: '변경 제외 확정' })).toContainText('카메라');
  await page.getByRole('dialog').getByRole('button', { name: '제외 확정' }).click();
  await expect(page.getByRole('dialog')).toContainText('서버 상태가 그 사이 바뀌었습니다.');
  await page.getByRole('dialog').getByRole('button', { name: '제외 확정' }).click();
  await expect(page.getByText('변경 사항을 서버에 반영했습니다. (리비전 10)')).toBeVisible();
  const confirmation = backend.requests.filter((entry) => entry.path.endsWith('/connection/topology-preview/confirm')).at(-1);
  expect(confirmation?.body).toEqual({ confirmation_id: 'preview-1', digest: 'b'.repeat(64), client_revision: 7, server_revision: 9 });
  expect(consoleErrors).toEqual([]);
  expect(pageErrors).toEqual([]);
});

test('authenticated operator can inspect qualified policy, review evidence, and preserve Korean accessibility', async ({ page }) => {
  const consoleErrors = captureUnexpectedConsoleErrors(page);
  const pageErrors = capturePageErrors(page);
  const backend = await installOperatorBackend(page);

  await page.goto('/?page=events');
  await expect(page.getByRole('heading', { name: '이벤트' })).toBeVisible();
  await expect(page.getByText('중앙 인시던트')).toBeVisible();
  await page.getByRole('button', { name: /낙상/ }).last().click();
  await expect(page.getByRole('dialog')).toContainText('원본 / 스냅샷 / 파생');
  await expect(page.getByRole('dialog')).toContainText('CORRUPT / MISSING / UNAVAILABLE');
  await page.getByRole('button', { name: '실제 알림으로 검토' }).click();
  await expect.poll(() => backend.reviews.length).toBe(1);
  expect(backend.reviews[0]).toMatchObject({ expected_version: 0, disposition: 'TRUE_POSITIVE' });
  await page.getByRole('dialog').getByRole('button', { name: '닫기' }).click();
  await page.getByRole('button', { name: /서울 301호.*낙상/ }).click();
  await expect(page.getByRole('dialog', { name: /낙상/ })).toContainText('적용 실행 증명');
  await expect(page.getByRole('dialog')).toContainText('원본 손상됨 · 분석 사용 가능 · 주석 사용 불가');
  await page.getByRole('dialog').getByRole('button', { name: '요청', exact: true }).first().click();
  await expect(page.getByRole('dialog')).toContainText('주석 스틸: 대기 중');
  await page.getByRole('dialog').getByRole('button', { name: '취소' }).click();
  await expect(page.getByRole('dialog')).toContainText('주석 스틸: 취소됨');
  expect(backend.requests.filter((entry) => entry.path.endsWith('/derivatives/still')).map((entry) => entry.method)).toEqual(['POST', 'DELETE']);
  await page.getByRole('dialog').getByRole('button', { name: '닫기' }).click();

  await page.getByRole('button', { name: '설정', exact: true }).click();
  await expect(page.getByRole('heading', { name: '설정', exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: '클립 내보내기' })).toBeVisible();
  await expect(page.getByLabel('클립 내보내기 사용')).toBeVisible();
  await expect(page.getByText('활성 세대 12 · fall.v1 · fall-policy.v1')).toBeVisible();
  await page.getByLabel('적용 범위').selectOption('cam-1');
  await page.getByRole('button', { name: '차이 비교' }).click();
  await expect(page.getByText(/변경 확인:/)).toBeVisible();
  await page.getByRole('button', { name: '검토한 변경 적용' }).click();
  await expect(page.getByText('정책 변경이 충돌했거나 저장되지 않았습니다. 최신 값을 확인하세요.')).toBeVisible();
  await page.getByRole('button', { name: '차이 비교' }).click();
  await expect(page.getByText(/변경 확인:/)).toBeVisible();
  await page.getByRole('button', { name: '검토한 변경 적용' }).click();
  await expect(page.getByText('워커 재시작/적용 확인이 필요합니다.')).toBeVisible();
  const applied = backend.requests.filter((entry) => entry.path.endsWith('/detection-policies/apply'));
  expect(applied).toHaveLength(2);
  expect(applied.at(-1)?.body).toEqual({ module_id: 'fall', module_version: 1, schema_id: 'fall-policy', schema_version: 1, camera_id: 'cam-1', values: { threshold: 0.75 }, expected_revision_id: 8 });
  await page.getByRole('button', { name: '이전 개정으로 되돌리기' }).click();
  await expect.poll(() => backend.requests.some((entry) => entry.path.endsWith('/detection-policies/rollback'))).toBe(true);
  expect(backend.requests.find((entry) => entry.path.endsWith('/detection-policies/rollback'))?.body).toEqual({ module_id: 'fall', module_version: 1, camera_id: 'cam-1', expected_revision_id: 8 });
  await expect(page.getByText('이전 정책 개정으로 되돌리기를 요청했습니다. 적용 확인이 필요합니다.')).toBeVisible();

  mkdirSync(qaDirectory, { recursive: true });
  for (const viewport of viewports) {
    await page.setViewportSize(viewport);
    expect(page.viewportSize()).toEqual(viewport);
    await expect(page.locator('body')).toHaveJSProperty('scrollWidth', viewport.width);
    await page.screenshot({ path: `${qaDirectory}/settings-${viewport.width}x${viewport.height}.png`, fullPage: true });
  }
  await page.getByRole('button', { name: '이벤트', exact: true }).click();
  for (const viewport of viewports) {
    await page.setViewportSize(viewport);
    expect(page.viewportSize()).toEqual(viewport);
    await expect(page.locator('body')).toHaveJSProperty('scrollWidth', viewport.width);
    await page.screenshot({ path: `${qaDirectory}/events-${viewport.width}x${viewport.height}.png`, fullPage: true });
  }
  await page.keyboard.press('Tab');
  await expect(page.locator(':focus-visible')).toBeVisible();
  expect(consoleErrors).toEqual([]);
  expect(pageErrors).toEqual([]);
});
