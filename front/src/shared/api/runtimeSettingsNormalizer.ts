import { isRecord, pickNonNegativeNumber } from '@/shared/api/normalizerFields';
import type { RuntimeSettings } from '@/shared/api/types';

export function normalizeRuntimeSettings(value: unknown): RuntimeSettings {
  if (!isRecord(value) || typeof value.clip_export_enabled !== 'boolean') {
    throw new Error('Invalid runtime settings response');
  }
  const version = pickNonNegativeNumber(value, ['version']);
  if (version === null || !Number.isInteger(version)) {
    throw new Error('Invalid runtime settings response');
  }
  return { clip_export_enabled: value.clip_export_enabled, version };
}
