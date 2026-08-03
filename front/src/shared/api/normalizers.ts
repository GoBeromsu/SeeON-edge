export {
  normalizeBedZoneRecognitionResponse,
  normalizeCamera,
  normalizeCameraRegistry,
  normalizeCameraResponse,
  normalizeCameraTestResult,
} from '@/shared/api/cameraNormalizer';
export { normalizeClip, normalizeClipsResponse } from '@/shared/api/clipNormalizer';
export { normalizeClipStorageBrowse, normalizeClipStorageInfo } from '@/shared/api/clipStorageNormalizer';
export {
  normalizeConnectionTestResult,
  normalizeConnectionView,
} from '@/shared/api/connectionNormalizer';
export { normalizeDetectionSettings } from '@/shared/api/detectionSettingsNormalizer';
export { isRecord } from '@/shared/api/normalizerFields';
export { normalizeStatusSnapshot } from '@/shared/api/statusNormalizer';
export { normalizeSystemSnapshot } from '@/shared/api/systemNormalizer';
