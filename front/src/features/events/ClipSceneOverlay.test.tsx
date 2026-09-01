import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';
import { ClipSceneOverlay } from '@/features/events/ClipSceneOverlay';
import type { ClipScene, ClipSceneFrame } from '@/shared/api/types';

const roots = new Set<ReturnType<typeof createRoot>>();
afterEach(() => { act(() => roots.forEach((root) => root.unmount())); roots.clear(); document.body.replaceChildren(); });

const scene = {
  source_dimensions: [640, 480],
  style: { palette: { bed: [1, 2, 3], danger: [4, 5, 6], neutral: [7, 8, 9], person: [10, 11, 12], pose: [13, 14, 15], pose_dot: [16, 17, 18] }, skeleton: { edges: [[0, 1]] }, z_order: { bed: 10, decision: 40, person: 20 } },
} as ClipScene;
const frame = { t: 0, p: 0, q: 0, sd: false, dc: [], bd: [{ b: [1, 2, 3, 4], c: 1, i: 0, pg: [], pv: '', sm: '', ct: [] }], ps: [{ b: [10, 20, 30, 40], c: 1, i: 0, tr: null, k: [[0, null, null, .8], [1, 30, 40, .5]] }], lb: [{ x: 4, y: 5, t: 'late', c: [9, 8, 7], z: 9 }, { x: 1, y: 2, t: 'early', c: [6, 5, 4], z: 1 }] } as ClipSceneFrame;

describe('ClipSceneOverlay', () => {
  it('uses received colors and z order, skips null-coordinate keypoints, and preserves source viewBox', () => {
    const host = document.createElement('div'); document.body.append(host); const root = createRoot(host); roots.add(root);
    act(() => root.render(<ClipSceneOverlay scene={scene} frame={frame} />));
    const overlay = host.querySelector('svg')!;
    expect(overlay.getAttribute('viewBox')).toBe('0 0 640 480');
    expect(overlay.getAttribute('preserveAspectRatio')).toBe('none');
    expect(host.querySelectorAll('rect')[1]?.getAttribute('stroke')).toBe('rgb(10 11 12)');
    expect([...host.querySelectorAll('text')].map((node) => node.textContent)).toEqual(['early', 'late']);
    expect(host.querySelectorAll('circle')).toHaveLength(1);
    expect(host.querySelectorAll('line')).toHaveLength(0);
    const [bedBox, personBox] = host.querySelectorAll('rect');
    expect(bedBox.getAttribute('x')).toBe('1');
    expect(bedBox.getAttribute('y')).toBe('2');
    expect(bedBox.getAttribute('width')).toBe('2');
    expect(bedBox.getAttribute('height')).toBe('2');
    expect(personBox.getAttribute('x')).toBe('10');
    expect(personBox.getAttribute('y')).toBe('20');
    expect(personBox.getAttribute('width')).toBe('20');
    expect(personBox.getAttribute('height')).toBe('20');
  });

  it('falls back to the bed box when a bed polygon is empty', () => {
    const host = document.createElement('div'); document.body.append(host); const root = createRoot(host); roots.add(root);
    act(() => root.render(<ClipSceneOverlay scene={scene} frame={{ ...frame, bd: [{ ...frame.bd[0], pg: [] }] }} />));
    const bedBox = host.querySelector('rect')!;
    expect(bedBox.getAttribute('width')).toBe('2');
    expect(bedBox.getAttribute('height')).toBe('2');
  });
});
