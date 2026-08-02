export {
  normalizeCamera,
  normalizeCameraRegistry,
  normalizeCameraResponse,
  normalizeCameraTestResult,
} from '@/shared/api/cameraNormalizer';
export { normalizeClip, normalizeClipsResponse } from '@/shared/api/clipNormalizer';
export {
  normalizeConnectionTestResult,
  normalizeConnectionView,
  normalizeRosterSyncResult,
} from '@/shared/api/connectionNormalizer';
export { isRecord } from '@/shared/api/normalizerFields';
export { normalizeStatusSnapshot } from '@/shared/api/statusNormalizer';
export { normalizeSystemSnapshot } from '@/shared/api/systemNormalizer';
