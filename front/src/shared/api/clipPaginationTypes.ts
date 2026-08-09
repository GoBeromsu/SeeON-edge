import type { Clip } from '@/shared/api/types';

export type ClipPageQuery = {
  readonly cameraId?: string;
  readonly eventType?: string;
  readonly limit: number;
  readonly offset: number;
};

export type ClipPagination = {
  readonly limit: number;
  readonly offset: number;
  readonly total: number;
  readonly has_more: boolean;
};

export type ClipPage = {
  readonly clips: readonly Clip[];
  readonly pagination: ClipPagination;
  readonly event_type_counts: Readonly<Record<string, number>>;
  readonly complete_clips: readonly Clip[] | null;
};
