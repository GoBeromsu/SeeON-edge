import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { recognizeBedZone, saveBedZone, type BedZone } from '@/shared/api/client';
import { BedZoneRecognitionPanel } from '@/shared/ui/BedZoneRecognitionPanel';

vi.mock('@/shared/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return { ...actual, recognizeBedZone: vi.fn(), saveBedZone: vi.fn() };
});

const savedZone: BedZone = {
  regions: [{ id: 'saved', polygon: [[0, 0], [100, 0], [100, 100]], origin: 'manual' }],
  image_width: 1920,
  image_height: 1080,
  recognized_at: '2026-08-01T00:00:00Z',
};
const candidateZone: BedZone = {
  regions: [{ id: 'candidate', polygon: [[10, 10], [200, 10], [200, 200]], origin: 'model' }],
  image_width: 1280,
  image_height: 720,
  recognized_at: '2026-08-02T00:00:00Z',
};
const roots = new Set<Root>();

function render(zone: BedZone | null = savedZone, onSaved = vi.fn(), onCancel = vi.fn()) {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  roots.add(root);
  act(() => root.render(<BedZoneRecognitionPanel cameraId="cam-1" bedZone={zone} onSaved={onSaved} onCancel={onCancel} />));
  return { host, root, onSaved, onCancel };
}

function action(host: HTMLElement, label: string): HTMLButtonElement {
  return host.querySelector(`button[aria-label="${label}"]`) as HTMLButtonElement;
}

async function flush(): Promise<void> {
  await act(async () => { await Promise.resolve(); await Promise.resolve(); await Promise.resolve(); });
}

beforeEach(() => {
  vi.mocked(recognizeBedZone).mockReset();
  vi.mocked(saveBedZone).mockReset();
});

afterEach(() => {
  act(() => roots.forEach((root) => root.unmount()));
  roots.clear();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  document.body.innerHTML = '';
});

describe('BedZoneRecognitionPanel', () => {
  it('initializes the editor from saved regions and exposes icon-only explicit actions', () => {
    const { host } = render();
    expect(host.querySelector('[data-region-id="saved"]')).not.toBeNull();
    expect(host.querySelector('svg[viewBox="0 0 1920 1080"]')).not.toBeNull();
    expect(action(host, '자동 인식').textContent).toBe('자동 인식');
    expect(action(host, '저장').textContent).toBe('저장');
    expect(action(host, '취소').textContent).toBe('취소');
    expect(host.textContent).toContain('자동으로 찾거나 침대 모서리를 직접 지정하세요.');
    expect(host.textContent).toContain('침대 영역 1개');
  });

  it('uses snapshot natural dimensions when no saved geometry exists', () => {
    const { host } = render(null);
    const image = host.querySelector('img[alt="카메라 영상"]') as HTMLImageElement;
    expect(image.className).toContain('h-auto');
    expect(image.parentElement?.className).not.toContain('event-media-frame');
    Object.defineProperties(image, {
      naturalWidth: { value: 1440 },
      naturalHeight: { value: 810 },
    });
    act(() => image.dispatchEvent(new Event('load', { bubbles: true })));
    expect(host.querySelector('svg[viewBox="0 0 1440 810"]')).not.toBeNull();
    expect(action(host, '저장').disabled).toBe(false);
  });

  it('uses inverse confidence for sensitivity and treats recognition as an unsaved candidate', async () => {
    vi.mocked(recognizeBedZone).mockResolvedValue(candidateZone);
    const { host, onSaved } = render();
    const sensitivity = host.querySelector('input[aria-label="인식 민감도"]') as HTMLInputElement;
    act(() => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
      setter?.call(sensitivity, '0.9');
      sensitivity.dispatchEvent(new Event('input', { bubbles: true }));
    });

    await act(async () => action(host, '자동 인식').click());

    expect(recognizeBedZone).toHaveBeenCalledWith('cam-1', expect.closeTo(0.1));
    expect(host.querySelector('[data-region-id="candidate"]')).not.toBeNull();
    expect(onSaved).not.toHaveBeenCalled();
    expect(host.querySelector('[aria-label="침대 영역 후보 준비됨"]')).not.toBeNull();
  });

  it('persists only on explicit save and then reports the actual response', async () => {
    vi.mocked(saveBedZone).mockResolvedValue(savedZone);
    const { host, onSaved } = render();

    await act(async () => action(host, '저장').click());

    expect(saveBedZone).toHaveBeenCalledWith('cam-1', {
      regions: savedZone.regions,
      image_width: 1920,
      image_height: 1080,
    });
    expect(onSaved).toHaveBeenCalledWith(savedZone);
  });

  it('can delete the last region and explicitly save an empty clear', async () => {
    vi.mocked(saveBedZone).mockResolvedValue(null);
    const { host, onSaved } = render();
    act(() => (host.querySelector('[data-region-id="saved"] polygon') as SVGPolygonElement).dispatchEvent(new MouseEvent('click', { bubbles: true })));
    act(() => action(host, '선택 영역 삭제').click());

    await act(async () => action(host, '저장').click());

    expect(saveBedZone).toHaveBeenCalledWith('cam-1', { regions: [], image_width: 1920, image_height: 1080 });
    expect(onSaved).toHaveBeenCalledWith(null);
  });

  it('disables save while a polygon draft is incomplete and restores it when cancelled', () => {
    const { host } = render();
    expect(action(host, '저장').disabled).toBe(false);
    act(() => action(host, '직접 그리기').click());
    expect(action(host, '저장').disabled).toBe(true);
    expect(host.textContent).toContain('모서리를 찍은 뒤 영역 완료를 누르세요.');
    const editor = host.querySelector('svg[aria-label="침대 영역 편집 캔버스"]') as SVGSVGElement;
    act(() => editor.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Escape' })));
    expect(action(host, '저장').disabled).toBe(false);
  });

  it('disables save after a vertex edit makes a committed polygon invalid', () => {
    const { host } = render();
    const vertex = host.querySelector('circle[aria-label="영역 saved 꼭짓점 1"]') as SVGCircleElement;
    act(() => vertex.dispatchEvent(new FocusEvent('focusin', { bubbles: true })));
    const x = host.querySelector('input[aria-label="꼭짓점 X"]') as HTMLInputElement;
    act(() => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
      setter?.call(x, '100');
      x.dispatchEvent(new Event('input', { bubbles: true }));
    });
    expect(action(host, '저장').disabled).toBe(true);
  });

  it('cancels without persistence', () => {
    const { host, onCancel } = render();
    act(() => action(host, '취소').click());
    expect(onCancel).toHaveBeenCalledOnce();
    expect(saveBedZone).not.toHaveBeenCalled();
  });

  it('ignores a stale recognition response after camera change', async () => {
    let resolve!: (zone: BedZone) => void;
    vi.mocked(recognizeBedZone).mockReturnValue(new Promise((done) => { resolve = done; }));
    const { host, root } = render();
    act(() => action(host, '자동 인식').click());
    act(() => root.render(<BedZoneRecognitionPanel cameraId="cam-2" bedZone={null} onSaved={vi.fn()} onCancel={vi.fn()} />));
    await act(async () => resolve(candidateZone));
    expect(host.querySelector('[data-region-id="candidate"]')).toBeNull();
  });

  it('does not report a pending save after the popup unmounts', async () => {
    let resolve!: (zone: BedZone | null) => void;
    vi.mocked(saveBedZone).mockReturnValue(new Promise((done) => { resolve = done; }));
    const { host, root, onSaved } = render();
    act(() => action(host, '저장').click());
    act(() => root.unmount());
    roots.delete(root);
    await act(async () => resolve(savedZone));
    expect(onSaved).not.toHaveBeenCalled();
  });

  it('shows accessible status icons for request failures without visible helper prose', async () => {
    vi.mocked(recognizeBedZone).mockRejectedValue(new Error('offline'));
    const { host } = render();
    await flush();
    await act(async () => action(host, '자동 인식').click());
    expect(host.querySelector('[role="alert"][aria-label="침대 영역 인식 실패"]')).not.toBeNull();
    expect(host.textContent).toContain('침대 영역 인식 실패');
    expect(host.textContent).toContain('다시 시도');
    expect(host.textContent).toContain('직접 그리기도 사용할 수 있습니다.');
    await act(async () => action(host, '다시 시도').click());
    expect(recognizeBedZone).toHaveBeenCalledTimes(2);
  });
});
