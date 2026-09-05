import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { BedRegion } from '@/shared/api/client';
import { BedZonePolygonEditor } from '@/shared/ui/BedZonePolygonEditor';

const roots = new Set<Root>();
const initial: BedRegion[] = [{
  id: 'model-bed',
  polygon: [[100, 100], [400, 100], [400, 300]],
  origin: 'model',
}];

function render(regions: readonly BedRegion[] = [], imageWidth = 1000, imageHeight = 500, onChange = vi.fn()) {
  const host = document.createElement('div');
  host.style.position = 'relative';
  document.body.append(host);
  const root = createRoot(host);
  roots.add(root);
  act(() => root.render(
    <BedZonePolygonEditor
      regions={regions}
      imageWidth={imageWidth}
      imageHeight={imageHeight}
      onChange={onChange}
    />,
  ));
  const svg = host.querySelector('svg[aria-label="침대 영역 편집 캔버스"]') as SVGSVGElement;
  vi.spyOn(svg, 'getBoundingClientRect').mockReturnValue({
    left: 10, top: 20, width: 500, height: 250, right: 510, bottom: 270, x: 10, y: 20, toJSON: () => ({}),
  });
  return { host, root, svg, onChange };
}

function button(host: HTMLElement, label: string): HTMLButtonElement {
  return host.querySelector(`button[aria-label="${label}"]`) as HTMLButtonElement;
}

function clickCanvas(svg: SVGSVGElement, clientX: number, clientY: number): void {
  act(() => svg.dispatchEvent(new MouseEvent('click', { bubbles: true, clientX, clientY })));
}

function press(svg: SVGSVGElement, key: string, shiftKey = false): void {
  act(() => svg.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key, shiftKey })));
}

function pointerEvent(type: string, pointerId: number, clientX: number, clientY: number): Event {
  const event = new Event(type, { bubbles: true });
  Object.defineProperties(event, {
    pointerId: { value: pointerId },
    clientX: { value: clientX },
    clientY: { value: clientY },
  });
  return event;
}

beforeEach(() => {
  vi.stubGlobal('crypto', {
    getRandomValues: (bytes: Uint8Array) => {
      bytes.fill(0);
      bytes[15] = 1;
      return bytes;
    },
  });
});

afterEach(() => {
  act(() => roots.forEach((root) => root.unmount()));
  roots.clear();
  document.body.innerHTML = '';
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('BedZonePolygonEditor', () => {
  it('scales canvas clicks into image coordinates and commits only after explicit close', () => {
    const onChange = vi.fn();
    const { host, svg } = render([], 1000, 500, onChange);

    act(() => button(host, '직접 그리기').click());
    clickCanvas(svg, 60, 70);
    clickCanvas(svg, 260, 70);
    clickCanvas(svg, 260, 220);

    expect(onChange).not.toHaveBeenCalled();
    expect(button(host, '영역 완료').disabled).toBe(false);
    act(() => button(host, '영역 완료').click());

    expect(onChange).toHaveBeenCalledWith([{
      id: '00000000-0000-4000-8000-000000000001',
      polygon: [[100, 100], [500, 100], [500, 400]],
      origin: 'manual',
    }]);
  });

  it('rejects duplicate and collinear drafts and can cancel the last point', () => {
    const onChange = vi.fn();
    const { host, svg } = render([], 1000, 500, onChange);
    act(() => button(host, '직접 그리기').click());
    clickCanvas(svg, 60, 70);
    clickCanvas(svg, 160, 70);
    clickCanvas(svg, 260, 70);

    expect(button(host, '영역 완료').disabled).toBe(true);
    expect(host.querySelectorAll('[data-draft-point]')).toHaveLength(3);
    act(() => button(host, '마지막 점 취소').click());
    expect(host.querySelectorAll('[data-draft-point]')).toHaveLength(2);
    expect(onChange).not.toHaveBeenCalled();
  });

  it('selects and deletes one existing region without a duplicate row action', () => {
    const onChange = vi.fn();
    const { host } = render([...initial, { id: 'manual-bed', polygon: [[20, 20], [40, 20], [40, 40]], origin: 'manual' }], 1000, 500, onChange);

    act(() => (host.querySelector('[data-region-id="model-bed"] polygon') as SVGPolygonElement).dispatchEvent(new MouseEvent('click', { bubbles: true })));
    act(() => button(host, '선택 영역 삭제').click());

    expect(onChange).toHaveBeenCalledWith([{ id: 'manual-bed', polygon: [[20, 20], [40, 20], [40, 40]], origin: 'manual' }]);
  });

  it('offers numeric keyboard editing, clips coordinates, and marks model regions manual', () => {
    const onChange = vi.fn();
    const { host } = render(initial, 1000, 500, onChange);
    const vertex = host.querySelector('circle[aria-label="영역 model-bed 꼭짓점 1"]') as SVGCircleElement;
    act(() => vertex.dispatchEvent(new FocusEvent('focusin', { bubbles: true })));

    const x = host.querySelector('input[aria-label="꼭짓점 X"]') as HTMLInputElement;
    expect(x).not.toBeNull();
    act(() => {
      const valueSetter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
      valueSetter?.call(x, '1200');
      x.dispatchEvent(new Event('input', { bubbles: true }));
    });

    expect(onChange).toHaveBeenCalledWith([{
      id: 'model-bed',
      polygon: [[1000, 100], [400, 100], [400, 300]],
      origin: 'manual',
    }]);
  });

  it('captures touch-style vertex drags, scales them, and clips them to image bounds', () => {
    const onChange = vi.fn();
    const { host, svg } = render(initial, 1000, 500, onChange);
    const vertex = host.querySelector('circle[aria-label="영역 model-bed 꼭짓점 1"]') as SVGCircleElement;
    const setPointerCapture = vi.fn();
    Object.defineProperty(vertex, 'setPointerCapture', { value: setPointerCapture });

    act(() => vertex.dispatchEvent(pointerEvent('pointerdown', 7, 60, 70)));
    act(() => svg.dispatchEvent(pointerEvent('pointermove', 7, 900, -100)));
    act(() => svg.dispatchEvent(pointerEvent('pointerup', 7, 900, -100)));

    expect(setPointerCapture).toHaveBeenCalledWith(7);
    expect(onChange).toHaveBeenCalledWith([{
      id: 'model-bed',
      polygon: [[1000, 0], [400, 100], [400, 300]],
      origin: 'manual',
    }]);
  });

  it('resets an incomplete draft when reference dimensions change', () => {
    const onChange = vi.fn();
    const { host, root, svg } = render([], 1000, 500, onChange);
    act(() => button(host, '직접 그리기').click());
    clickCanvas(svg, 60, 70);
    expect(host.querySelector('[data-draft-region]')).not.toBeNull();

    act(() => root.render(<BedZonePolygonEditor regions={[]} imageWidth={640} imageHeight={480} onChange={onChange} />));

    expect(host.querySelector('[data-draft-region]')).toBeNull();
    expect(onChange).not.toHaveBeenCalled();
  });

  it('reports an incomplete draft as invalid and Escape cancellation restores validity', () => {
    const validity = vi.fn();
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    roots.add(root);
    act(() => root.render(<BedZonePolygonEditor regions={initial} imageWidth={1000} imageHeight={500} onChange={vi.fn()} onDraftValidityChange={validity} />));
    const svg = host.querySelector('svg[aria-label="침대 영역 편집 캔버스"]') as SVGSVGElement;

    expect(validity).toHaveBeenLastCalledWith(true);
    act(() => button(host, '직접 그리기').click());
    expect(validity).toHaveBeenLastCalledWith(false);
    act(() => svg.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Escape' })));
    expect(validity).toHaveBeenLastCalledWith(true);
  });

  it('creates and explicitly closes a valid polygon using only the keyboard', () => {
    const onChange = vi.fn();
    const { host, svg } = render([], 2, 2, onChange);

    act(() => button(host, '직접 그리기').click());
    expect(document.activeElement).toBe(svg);
    expect(svg.tabIndex).toBe(0);
    expect(document.getElementById(svg.getAttribute('aria-describedby') ?? '')?.textContent).toContain('Enter');

    press(svg, 'Enter');
    press(svg, 'ArrowLeft', true);
    press(svg, ' ');
    press(svg, 'ArrowUp', true);
    press(svg, 'Enter');

    expect(host.querySelectorAll('[data-draft-point]')).toHaveLength(3);
    expect(button(host, '영역 완료').disabled).toBe(false);
    act(() => button(host, '영역 완료').click());
    expect(onChange).toHaveBeenCalledWith([{
      id: '00000000-0000-4000-8000-000000000001',
      polygon: [[1, 1], [0, 1], [0, 0]],
      origin: 'manual',
    }]);
  });

  it('enforces region and vertex limits and disables all edits when requested', () => {
    const eight = Array.from({ length: 8 }, (_, index): BedRegion => ({
      id: `bed-${index}`,
      polygon: [[0, 0], [10, 0], [10, 10]],
      origin: 'manual',
    }));
    const limited = render(eight);
    expect(button(limited.host, '직접 그리기').disabled).toBe(true);

    const vertexLimited = render([]);
    act(() => button(vertexLimited.host, '직접 그리기').click());
    for (let index = 0; index < 17; index += 1) {
      clickCanvas(vertexLimited.svg, 20 + index * 10, 30 + (index % 2) * 20);
    }
    expect(vertexLimited.host.querySelectorAll('[data-draft-point]')).toHaveLength(16);

    const disabledHost = document.createElement('div');
    document.body.append(disabledHost);
    const disabledRoot = createRoot(disabledHost);
    roots.add(disabledRoot);
    act(() => disabledRoot.render(<BedZonePolygonEditor regions={initial} imageWidth={1000} imageHeight={500} onChange={vi.fn()} disabled />));
    expect(button(disabledHost, '직접 그리기').disabled).toBe(true);
    expect(button(disabledHost, '선택 영역 삭제').disabled).toBe(true);
  });
});
