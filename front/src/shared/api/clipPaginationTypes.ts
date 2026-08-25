import type { Clip } from '@/shared/api/types';

export type ClipPageQuery = {
  readonly cameraId?: string;
  readonly eventType?: string;
  readonly limit: number;
  /** Opaque backend keyset cursor. Absent selects the newest page. */
  readonly cursor?: string;
};

export type ClipPagination = {
  readonly limit: number;
  readonly total: number;
  readonly has_more: boolean;
  /** Opaque `(started_at, clip_id)` keyset cursor for the next page; null on the last page. */
  readonly next_cursor: string | null;
};

export type ClipPage = {
  readonly clips: readonly Clip[];
  readonly pagination: ClipPagination;
  readonly event_type_counts: Readonly<Record<string, number>>;
  readonly complete_clips: readonly Clip[] | null;
};
