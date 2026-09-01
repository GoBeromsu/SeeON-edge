import { act, useEffect } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';
import { useEventsLocation } from '@/features/events/useEventsLocation';
import type { Clip } from '@/shared/api/types';

function resetLocation(search = '?page=events'): void {
  window.history.replaceState(null, '', `/${search}`);
}

function clip(id: string, eventType: string): Clip {
  return {
    id,
    camera_id: 'cam-1',
    camera_label: '301호',
    event_type: eventType,
    created_at: '2026-08-02T03:12:00Z',
    video_path: `/api/v1/clips/${id}/video`,
    video_available: true,
    thumbnail_available: false,
    video_error: null,
    scene_available: false,
    scene_frame_count: null,
  };
}

type Harness = ReturnType<typeof useEventsLocation>;

function mount(clips: Clip[], status: 'idle' | 'loading' | 'success' | 'error' = 'success'): { host: HTMLDivElement; root: Root; get current(): Harness } {
  let current!: Harness;
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  function Component(): JSX.Element {
    current = useEventsLocation();
    useEffect(() => {
      current.validate(status, clips, [...new Set(clips.map((item) => item.event_type))]);
    }, [clips, current.validate, status]);
    return <span>{current.eventType ?? ''}|{current.clipId ?? ''}</span>;
  }
  act(() => root.render(<Component />));
  return { host, root, get current() { return current; } };
}

afterEach(() => {
  document.body.innerHTML = '';
  resetLocation();
});

describe('useEventsLocation', () => {
  it('reads the initial event/clip filter from the URL', () => {
    resetLocation('?page=events&event=fall&clip=clip-1');
    const view = mount([clip('clip-1', 'fall')]);

    expect(view.current.eventType).toBe('fall');
    expect(view.current.clipId).toBe('clip-1');
    act(() => view.root.unmount());
  });

  it('setEventType pushes the event filter into the URL', () => {
    resetLocation('?page=events');
    const view = mount([clip('clip-1', 'fall'), clip('clip-2', 'bed-exit')]);

    act(() => view.current.setEventType('bed-exit'));

    expect(view.current.eventType).toBe('bed-exit');
    expect(window.location.search).toContain('event=bed-exit');
    act(() => view.root.unmount());
  });

  it('setEventType(undefined) clears the filter back to "전체"', () => {
    resetLocation('?page=events&event=fall');
    const view = mount([clip('clip-1', 'fall')]);

    act(() => view.current.setEventType(undefined));

    expect(view.current.eventType).toBeUndefined();
    expect(window.location.search).not.toContain('event=');
    act(() => view.root.unmount());
  });

  it('openClip/closeClip drive the clip modal via the `clip` query param', () => {
    resetLocation('?page=events');
    const view = mount([clip('clip-1', 'fall')]);

    act(() => view.current.openClip('clip-1'));
    expect(view.current.clipId).toBe('clip-1');
    expect(window.location.search).toContain('clip=clip-1');

    act(() => view.current.closeClip());
    expect(view.current.clipId).toBeUndefined();
    expect(window.location.search).not.toContain('clip=');
    act(() => view.root.unmount());
  });

  it('preserves an off-page clip id until metadata confirms it is missing', () => {
    resetLocation('?page=events&clip=stale-clip');
    const view = mount([clip('clip-1', 'fall')]);

    expect(view.current.clipId).toBe('stale-clip');
    act(() => view.current.discardClip());

    expect(view.current.clipId).toBeUndefined();
    expect(window.location.search).not.toContain('clip=stale-clip');
    act(() => view.root.unmount());
  });

  it('keeps a clip id in the URL while the clip list has not loaded yet (no premature stripping)', () => {
    resetLocation('?page=events&clip=clip-1');
    const view = mount([], 'loading');

    expect(view.current.clipId).toBe('clip-1');
    act(() => view.root.unmount());
  });
});
