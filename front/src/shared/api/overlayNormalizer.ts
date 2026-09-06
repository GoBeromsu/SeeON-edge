import { isRecord } from '@/shared/api/normalizerFields';
import type { OverlaySelection } from '@/shared/api/types';

export function normalizeOverlaySelection(value: unknown): OverlaySelection {
  const keys = isRecord(value) ? Object.keys(value) : [];
  if (
    !isRecord(value)
    || keys.length !== 2
    || !keys.includes('person')
    || !keys.includes('bed')
    || typeof value.person !== 'boolean'
    || typeof value.bed !== 'boolean'
  ) {
    throw new Error('Invalid overlay response');
  }
  return { person: value.person, bed: value.bed };
}
