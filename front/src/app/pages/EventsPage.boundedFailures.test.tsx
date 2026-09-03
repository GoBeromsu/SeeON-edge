import { act } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  cameraRegistry,
  cleanupPages,
  clipManifest,
  clipRequestUrls,
  flush,
  keysetBody,
  renderPage,
  resetLocation,
} from '@/app/pages/EventsPage.testSupport';

const RETIRED_CONTROLS = ['증거 보기 선택', '파생 증거 제어', '적용 실행 증명'] as const;
const RETIRED_ROUTE_FRAGMENTS = ['/analysis', '/derivatives/', '/label', '/relay/analysis-traces'] as const;

function jsonResponse(body: unknown, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

function expectNoRetiredControl(): void {
  for (const label of RETIRED_CONTROLS) {
    expect(document.querySelector(`[aria-label="${label}"]`)).toBeNull();
  }
}

function expectNoRetiredRequest(fetchMock: ReturnType<typeof vi.fn>): void {
  const requested = fetchMock.mock.calls.map(([input]) => String(input));
  for (const fragment of RETIRED_ROUTE_FRAGMENTS) {
    expect(requested.filter((url) => url.includes(fragment))).toEqual([]);
  }
}

function findButton(root: ParentNode, label: string): HTMLButtonElement {
  const button = Array.from(root.querySelectorAll('button')).find((candidate) => candidate.textContent === label);
  if (!button) throw new Error(`missing button ${label}`);
  return button;
}

function typeConfirm(input: HTMLInputElement, value: string): void {
  act(() => {
    const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set;
    valueSetter?.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

afterEach(() => {
  cleanupPages();
  document.body.innerHTML = '';
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('EventsPage bounded failure states', () => {
  it('shows the bounded unavailable frame when stored media is 503 and never offers a retired view', async () => {
    resetLocation();
    const clip = clipManifest({
      clip_id: 'clip-503',
      event_ref: 'event-503',
      video_available: false,
      thumbnail_available: false,
      video_error: '저장된 영상을 사용할 수 없습니다.',
    });
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/cameras')) return Promise.resolve(jsonResponse(cameraRegistry));
      if (url.endsWith('/clips/clip-503/artifacts')) {
        return Promise.resolve(jsonResponse({ clip_id: 'clip-503', clean: 'UNAVAILABLE', snapshot: null }));
      }
      // Stored evidence reads fail closed while the local audit database is degraded.
      if (url.endsWith('/clips/clip-503/video')) return Promise.resolve(jsonResponse({ detail: 'unavailable' }, 503));
      if (url.includes('/clips')) {
        return Promise.resolve(jsonResponse(keysetBody([clip], 48, null)));
      }
      return Promise.reject(new Error(`unexpected fetch: ${url}`));
    });
    vi.stubGlobal('fetch', fetchMock);

    const { host } = await renderPage();
    act(() => (host.querySelector('button.rounded-card') as HTMLButtonElement).click());
    await flush();

    const dialog = document.querySelector('[role="dialog"]') as HTMLElement;
    expect(dialog.querySelector('video')).toBeNull();
    expect(dialog.querySelector('[data-testid="clip-modal-unavailable"]')).not.toBeNull();
    expect(dialog.querySelector('[data-testid="clip-artifact-status"]')).not.toBeNull();
    expectNoRetiredControl();
    expectNoRetiredRequest(fetchMock);
  });

  it('shows the bounded thumbnail placeholder when the thumbnail is missing but the clip still plays', async () => {
    resetLocation();
    const clip = clipManifest({ clip_id: 'clip-nothumb', event_ref: 'event-nothumb', thumbnail_available: false });
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/cameras')) return Promise.resolve(jsonResponse(cameraRegistry));
      if (url.endsWith('/artifacts')) {
        return Promise.resolve(jsonResponse({ clip_id: 'clip-nothumb', clean: 'AVAILABLE', snapshot: null }));
      }
      if (url.includes('/clips')) return Promise.resolve(jsonResponse(keysetBody([clip], 48, null)));
      return Promise.reject(new Error(`unexpected fetch: ${url}`));
    });
    vi.stubGlobal('fetch', fetchMock);
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();

    const { host } = await renderPage();

    // Placeholder, never a fabricated poster and never a thumbnail request.
    expect(host.querySelector('[data-testid="clip-thumbnail-available"]')).not.toBeNull();
    expect(host.querySelector('img')).toBeNull();
    expect(clipRequestUrls(fetchMock).filter((url) => url.endsWith('/thumbnail'))).toEqual([]);

    act(() => (host.querySelector('button.rounded-card') as HTMLButtonElement).click());
    await flush();

    expect(document.querySelectorAll('[role="dialog"] video')).toHaveLength(1);
    expectNoRetiredControl();
    expectNoRetiredRequest(fetchMock);
  });


  it('keeps playback and the delete control when the worker holds the deletion', async () => {
    resetLocation();
    const clip = clipManifest({ clip_id: 'clip-held', event_ref: 'event-held' });
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/cameras')) return Promise.resolve(jsonResponse(cameraRegistry));
      if (url.endsWith('/artifacts')) {
        return Promise.resolve(jsonResponse({ clip_id: 'clip-held', clean: 'AVAILABLE', snapshot: null }));
      }
      if (url.endsWith('/clips/clip-held') && init?.method === 'DELETE') {
        return Promise.resolve(jsonResponse({ clip_id: 'clip-held', status: 'HELD' }, 202));
      }
      if (url.includes('/clips')) return Promise.resolve(jsonResponse(keysetBody([clip], 48, null)));
      return Promise.reject(new Error(`unexpected fetch: ${url}`));
    });
    vi.stubGlobal('fetch', fetchMock);
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();

    const { host } = await renderPage();
    act(() => (host.querySelector('button.rounded-card') as HTMLButtonElement).click());
    await flush();
    act(() => findButton(document.body, '클립 삭제').click());
    // Located by the confirm input's stable id, never by the dialog's rendered copy.
    const confirmInput = document.querySelector('#delete-confirm-input') as HTMLInputElement;
    const confirmDialog = confirmInput.closest('[role="dialog"]') as HTMLElement;
    typeConfirm(confirmInput, 'clip-held');
    await act(async () => { findButton(confirmDialog, '삭제').click(); });
    await flush();

    expect(document.querySelector('[data-testid="clip-delete-status"]')).not.toBeNull();
    expect(document.querySelectorAll('[role="dialog"] video')).toHaveLength(1);
    expect(findButton(document.body, '클립 삭제')).toBeDefined();
    expect(host.querySelectorAll('button.rounded-card')).toHaveLength(1);
    expectNoRetiredControl();
    expectNoRetiredRequest(fetchMock);
  });

  it('bounds the cursor boundary: the last keyset page disables 다음 and never retries the same cursor', async () => {
    resetLocation();
    const clips = Array.from({ length: 60 }, (_, index) => clipManifest({
      clip_id: `clip-${String(index + 1).padStart(3, '0')}`,
      event_ref: `event-${index + 1}`,
      started_at: '2026-08-02T03:12:00Z',
    }));
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/cameras')) return Promise.resolve(jsonResponse(cameraRegistry));
      const params = new URL(url, 'http://localhost').searchParams;
      return Promise.resolve(jsonResponse(keysetBody(clips, Number(params.get('limit')), params.get('cursor'))));
    });
    vi.stubGlobal('fetch', fetchMock);

    const { host } = await renderPage();
    act(() => (host.querySelector('button[aria-label="다음 페이지"]') as HTMLButtonElement).click());
    await flush();

    const nextButton = host.querySelector('button[aria-label="다음 페이지"]') as HTMLButtonElement;
    expect(host.querySelectorAll('button.rounded-card')).toHaveLength(12);
    expect(nextButton.disabled).toBe(true);
    const cursors = clipRequestUrls(fetchMock)
      .map((url) => new URL(url, 'http://localhost').searchParams.get('cursor'))
      .filter((value): value is string => value !== null);
    expect(new Set(cursors).size).toBe(cursors.length);
    expectNoRetiredRequest(fetchMock);
  });
});
