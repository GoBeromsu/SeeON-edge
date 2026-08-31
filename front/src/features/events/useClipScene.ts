import { useEffect, useMemo, useState } from 'react';
import { fetchClipScene } from '@/shared/api/client';
import type { Clip, ClipScene, ClipSceneFrame } from '@/shared/api/types';

const MAX_FRAME_DRIFT_MS = 120;

type ClipSceneState = { scene: ClipScene | null; frames: ClipSceneFrame[] };

/** Fetches the immutable sidecar only while the corresponding evidence modal is open. */
export function useClipScene(clip: Clip | null, open: boolean, mediaTimeMs: number): ClipSceneState & { frame: ClipSceneFrame | null } {
  const [state, setState] = useState<ClipSceneState>({ scene: null, frames: [] });
  const clipId = clip?.id;
  const available = clip?.scene_available === true;

  useEffect(() => {
    const controller = new AbortController();
    setState({ scene: null, frames: [] });
    if (!open || !clipId || !available) return () => controller.abort();
    void fetchClipScene(clipId, controller.signal).then((scene) => {
      if (controller.signal.aborted || scene === null) return;
      setState({ scene, frames: [...scene.frames].sort((left, right) => left.t - right.t) });
    }).catch((error: unknown) => {
      if (!controller.signal.aborted && !(error instanceof DOMException && error.name === 'AbortError')) {
        setState({ scene: null, frames: [] });
      }
    });
    return () => controller.abort();
  }, [available, clipId, open]);

  const frame = useMemo(() => closestSceneFrameAtOrBefore(state.frames, mediaTimeMs), [mediaTimeMs, state.frames]);
  return { ...state, frame };
}

export function closestSceneFrameAtOrBefore(frames: readonly ClipSceneFrame[], mediaTimeMs: number): ClipSceneFrame | null {
  let low = 0;
  let high = frames.length - 1;
  let candidate: ClipSceneFrame | null = null;
  while (low <= high) {
    const middle = low + Math.floor((high - low) / 2);
    const frame = frames[middle];
    if (frame.t <= mediaTimeMs) {
      candidate = frame;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  return candidate !== null && mediaTimeMs - candidate.t <= MAX_FRAME_DRIFT_MS ? candidate : null;
}
