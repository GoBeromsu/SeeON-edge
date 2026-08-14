import { useCallback, useEffect, useState } from 'react';
import { fetchClipMetadata } from '@/shared/api/clipPagination';
import { HttpError } from '@/shared/api/http';
import type { Clip } from '@/shared/api/types';

export type ClipMetadataStatus = 'idle' | 'loading' | 'success' | 'invalid' | 'error';

type StoredClipMetadataState = {
  readonly clipId: string | null;
  readonly status: ClipMetadataStatus;
  readonly clip: Clip | null;
};

export type ClipMetadataState = StoredClipMetadataState & {
  readonly retry: () => void;
};

type ClipMetadataSource = {
  readonly clipId: string | undefined;
  readonly pageClips: readonly Clip[];
  readonly completeClips: readonly Clip[] | null;
  readonly pageReady: boolean;
  /** When set, skip metadata validation/fetch for this id (accepted or in-flight deletion). */
  readonly suppressClipId?: string | null;
};

export function useClipMetadata({
  clipId,
  pageClips,
  completeClips,
  pageReady,
  suppressClipId = null,
}: ClipMetadataSource): ClipMetadataState {
  const suppressed = clipId !== undefined && suppressClipId !== null && clipId === suppressClipId;
  const pageClip = clipId && !suppressed
    ? pageClips.find((clip) => clip.id === clipId) ?? completeClips?.find((clip) => clip.id === clipId) ?? null
    : null;
  const [state, setState] = useState<StoredClipMetadataState>({ clipId: null, status: 'idle', clip: null });
  const [retryKey, setRetryKey] = useState(0);
  const retry = useCallback(() => setRetryKey((current) => current + 1), []);

  useEffect(() => {
    if (!clipId || suppressed || pageClip || !pageReady) return;
    const controller = new AbortController();
    setState({ clipId, status: 'loading', clip: null });
    void fetchClipMetadata(clipId, controller.signal).then(
      (clip) => {
        if (!controller.signal.aborted) setState({ clipId, status: 'success', clip });
      },
      (error: unknown) => {
        if (controller.signal.aborted) return;
        if (error instanceof HttpError && (error.status === 400 || error.status === 404)) {
          setState({ clipId, status: 'invalid', clip: null });
          return;
        }
        setState({ clipId, status: 'error', clip: null });
      },
    );
    return () => controller.abort();
  }, [clipId, pageClip, pageReady, retryKey, suppressed]);

  if (!clipId) return { clipId: null, status: 'idle', clip: null, retry };
  // Accepted/in-flight deletion must not fall through to invalid/error validation.
  if (suppressed) return { clipId, status: 'idle', clip: null, retry };
  if (pageClip) return { clipId, status: 'success', clip: pageClip, retry };
  if (!pageReady) return { clipId, status: 'loading', clip: null, retry };
  if (state.clipId !== clipId) return { clipId, status: 'loading', clip: null, retry };
  return { ...state, retry };
}
