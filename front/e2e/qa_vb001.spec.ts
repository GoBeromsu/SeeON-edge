import { expect, test } from '@playwright/test';

const requests: string[] = [];
const clip = {
  clip_id: 'clip-vb001', camera_id: 'cam-vb001', event_ref: 'event-vb001', event_type: 'fall',
  started_at: '2026-08-02T03:11:30Z', created_at: '2026-08-02T03:11:30Z',
  detected_at: '2026-08-02T03:12:00Z', duration_s: 12, codec: 'h264', path: null,
  video_available: false, thumbnail_available: false, video_error: 'QA_MEDIA_UNAVAILABLE',
  finalized: true, size_bytes: null, truncation_reasons: [],
};

test('VB001 events removes incidents, does not fetch scene, and displays detected_at', async ({ page }) => {
  await page.route('**/api/v1/**', async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    requests.push(path);
    const json = (body: object) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
    if (path.endsWith('/auth/session')) return json({});
    if (path.endsWith('/cameras')) return json({ registry_version: 1, cameras: [{ id: 'cam-vb001', backend_camera_id: 'cam-vb001', label: 'QA Camera', rtsp_url_masked: 'rtsp://masked', floor_name: null, status: 'online', created_at: '2026-08-01T00:00:00Z' }] });
    if (path.endsWith('/clips')) return json({ clips: [clip], pagination: { limit: 100, offset: 0, total: 1, has_more: false, next_cursor: null }, event_type_counts: { fall: 1 } });
    if (path.endsWith('/artifacts')) return json({ clip_id: 'clip-vb001', clean: 'UNAVAILABLE', snapshot: null });
    return json({});
  });

  await page.goto('/?page=events');
  await expect(page.getByRole('heading', { name: '이벤트' })).toBeVisible();
  await expect(page.getByText('중앙 인시던트')).toHaveCount(0);
  await page.locator('button[data-clip-id="clip-vb001"]').click();
  const dialog = page.getByRole('dialog', { name: /낙상/ });
  await expect(dialog).toBeVisible();
  const timeValue = dialog.locator('dt', { hasText: '시간' }).locator('xpath=following-sibling::dd[1]');
  await expect(timeValue).toHaveText('2026. 08. 02. 오후 12:12');
  expect(requests.some((path) => /\/scene$/.test(path))).toBeFalsy();
  await page.screenshot({ path: '/tmp/vb001-artifacts/vb001-events-dialog.png', fullPage: true });
});
